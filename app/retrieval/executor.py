"""
Retrieval executor: runs each step of the plan deterministically — no LLM.
- document: embed the step query, run Chroma similarity search, return top-k chunks.
- structured: load CSV for each registered structured source, run a simple pandas filter/head and format as text.
- arxiv: call arXiv API with step query, parse Atom, return title+abstract as chunks.
Production: timeout on OpenAI, per-step error handling so one failure does not abort the whole plan.
"""
import asyncio
from pathlib import Path

from app.arxiv_client import fetch_query
from app.config import get_settings
from app.llm_client import get_embedding_client
from app.db import connection_scope, init_schema, get_sources_by_type
from app.logging_config import get_logger
from app.models import RetrievedChunk

logger = get_logger(__name__)


def get_embedding(text: str, client, model: str) -> list[float]:
    if hasattr(client, "embed_query"):
        return client.embed_query(text)
    r = client.embeddings.create(input=[text], model=model)
    return r.data[0].embedding


def _chroma_query(query: str, top_k: int, settings) -> list[RetrievedChunk]:
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        coll = client.get_or_create_collection("documents")
        emb_client = get_embedding_client()
        emb = get_embedding(query, emb_client, settings.openai_embedding_model)
        results = coll.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                meta = (results["metadatas"][0] or [{}])[i] if results["metadatas"] else {}
                dist = (results["distances"][0] or [0])[i] if results.get("distances") else 0
                score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
                doc = (results["documents"][0] or [""])[i] if results["documents"] else ""
                out.append(
                    RetrievedChunk(
                        source_id=meta.get("source_id", "unknown"),
                        source_type="document",
                        content=doc or "",
                        score=score,
                        metadata=meta,
                    )
                )
        return out
    except Exception as e:
        logger.warning("chroma_query_failed", extra={"error": type(e).__name__})
        return []


def _structured_query(query: str, settings) -> list[RetrievedChunk]:
    """MVP: load each structured table and return head rows as context. No NL→SQL yet."""
    try:
        import pandas as pd
        with connection_scope(settings.db_path) as conn:
            init_schema(conn)
            sources = get_sources_by_type(conn, "structured")
        chunks = []
        for src in sources:
            try:
                path_str = src.metadata.get("table_path")
                if not path_str:
                    continue
                path = Path(path_str)
                if not path.exists():
                    continue
                df = pd.read_csv(path, nrows=20, on_bad_lines="skip")
                text = f"Columns: {list(df.columns)}\n\n" + df.to_string()
                chunks.append(
                    RetrievedChunk(
                        source_id=src.source_id,
                        source_type="structured",
                        content=text,
                        score=0.9,
                        metadata={"title": src.title},
                    )
                )
            except Exception as e:
                logger.warning("structured_row_failed", extra={"source_id": src.source_id, "error": type(e).__name__})
        return chunks
    except Exception as e:
        logger.warning("structured_query_failed", extra={"error": type(e).__name__})
        return []


def _arxiv_query(query: str, settings) -> list[RetrievedChunk]:
    """Live arXiv API call: search_query, parse Atom, return one chunk per entry (title + summary)."""
    try:
        max_results = getattr(settings, "arxiv_max_results", 15)
        timeout = getattr(settings, "arxiv_timeout", 30.0)
        entries = fetch_query(
            search_query=query or "all:all",
            start=0,
            max_results=max_results,
            timeout=timeout,
        )
        out = []
        for e in entries:
            source_id = e.arxiv_id.replace("/", "_")
            content = f"Title: {e.title}\n\nAbstract: {e.summary}"
            out.append(
                RetrievedChunk(
                    source_id=source_id,
                    source_type="arxiv",
                    content=content,
                    score=0.95,
                    metadata={"arxiv_id": e.arxiv_id, "link_abs": e.link_abs, "authors": e.authors},
                )
            )
        return out
    except Exception as e:
        logger.warning("arxiv_query_failed", extra={"error": type(e).__name__})
        return []


class RetrievalExecutor:
    def __init__(self):
        self.settings = get_settings()

    async def execute_step(self, source_type: str, query: str) -> list[RetrievedChunk]:
        if source_type == "document":
            return await asyncio.to_thread(_chroma_query, query, self.settings.top_k_docs, self.settings)
        if source_type == "structured":
            return await asyncio.to_thread(_structured_query, query, self.settings)
        if source_type == "arxiv":
            return await asyncio.to_thread(_arxiv_query, query, self.settings)
        return []

    async def execute_plan(self, steps: list) -> list[RetrievedChunk]:
        """Run each step concurrently and concatenate results. Dedupe by (source_id, content)."""
        tasks = [self.execute_step(step.source_type, step.query) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_chunks: list[RetrievedChunk] = []
        seen = set()
        
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning("executor_step_failed_concurrently", extra={"error": str(res), "source_type": steps[i].source_type})
                continue
            for chunk in res:
                key = (chunk.source_id, chunk.content[:100])
                if key not in seen:
                    seen.add(key)
                    all_chunks.append(chunk)
        return all_chunks
