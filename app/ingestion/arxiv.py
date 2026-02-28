"""
Ingest arXiv papers: fetch via API (search_query or id_list), store metadata in SQLite,
and add title+abstract to Chroma so they are searchable with other documents.
Uses app.arxiv_client for HTTP and Atom parsing.
"""
from datetime import datetime
from pathlib import Path

from app.arxiv_client import fetch_query, fetch_with_paging, ArxivEntry
from app.config import get_settings
from app.db import connection_scope, init_schema, upsert_source, increment_data_version, get_sources_by_type
from app.ingestion.base import BaseIngester
from app.models import SourceMetadata


def _sanitize_arxiv_id(arxiv_id: str) -> str:
    """Use as source_id: replace / with _ so it is a safe identifier."""
    return arxiv_id.replace("/", "_").strip()


class ArxivIngester(BaseIngester):
    source_type = "arxiv"

    def __init__(self):
        self.settings = get_settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)

    def _get_chroma(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=str(self.settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection("documents", metadata={"description": "Document chunks for RAG"})

    def _embed_and_upsert(self, entries: list[ArxivEntry]) -> None:
        """One chunk per paper: title + summary. Embed and upsert to Chroma."""
        from app.llm_client import get_embedding_client
        client = get_embedding_client()
        coll = self._get_chroma()
        for e in entries:
            content = f"{e.title}\n\n{e.summary}"
            source_id = _sanitize_arxiv_id(e.arxiv_id)
            chunk_id = f"arxiv_{source_id}"
            try:
                if hasattr(client, "embed_query"):
                    emb = client.embed_query(content)
                else:
                    resp = client.embeddings.create(
                        input=[content],
                        model=self.settings.openai_embedding_model,
                    )
                    emb = resp.data[0].embedding
            except Exception:
                continue
            coll.upsert(
                ids=[chunk_id],
                embeddings=[emb],
                documents=[content],
                metadatas=[{"source_id": source_id, "source_type": "arxiv", "arxiv_id": e.arxiv_id}],
            )

    def ingest(
        self,
        path: Path | None = None,
        *,
        search_query: str = "all:all",
        id_list: str | None = None,
        max_results: int = 20,
        title: str | None = None,
        **kwargs,
    ) -> SourceMetadata:
        """
        Ingest from arXiv API. path is ignored.
        Use search_query (e.g. "ti:electron", "all:machine+learning") and/or id_list (comma-separated IDs).
        max_results caps how many papers to fetch (paging with delay if needed).
        """
        if id_list and id_list.strip():
            entries = fetch_query(
                search_query or "all:all",
                id_list=id_list.strip(),
                start=0,
                max_results=min(max_results, 2000),
                timeout=getattr(self.settings, "arxiv_timeout", 30.0),
            )
        else:
            entries = fetch_with_paging(
                search_query or "all:all",
                max_total=max_results,
                per_request=min(20, max_results),
                delay_seconds=getattr(self.settings, "arxiv_request_delay_seconds", 3.0),
                timeout=getattr(self.settings, "arxiv_timeout", 30.0),
            )
        if not entries:
            raise ValueError("No arXiv entries returned; check search_query or id_list.")

        # Batch source: one metadata row for this ingestion run, or one per paper?
        # We do one per paper so source_id is stable and we can cite individual papers.
        with connection_scope(self.settings.db_path) as conn:
            init_schema(conn)
            for e in entries:
                source_id = _sanitize_arxiv_id(e.arxiv_id)
                meta = SourceMetadata(
                    source_id=source_id,
                    source_type=self.source_type,
                    title=e.title[:500] if e.title else "",
                    metadata={
                        "arxiv_id": e.arxiv_id,
                        "authors": e.authors,
                        "link_abs": e.link_abs,
                        "link_pdf": e.link_pdf,
                        "published": e.published,
                        "updated": e.updated,
                        "primary_category": e.primary_category,
                    },
                    ingested_at=datetime.utcnow().isoformat() + "Z",
                )
                upsert_source(conn, meta)
            increment_data_version(conn)

        self._embed_and_upsert(entries)
        # Return metadata for the "batch" as the first entry (for CLI feedback)
        first = entries[0]
        return SourceMetadata(
            source_id=_sanitize_arxiv_id(first.arxiv_id),
            source_type=self.source_type,
            title=first.title[:500] if first.title else "",
            metadata={"ingested_count": len(entries), "search_query": search_query},
            ingested_at=datetime.utcnow().isoformat() + "Z",
        )

    def list_ingested(self) -> list[SourceMetadata]:
        with connection_scope(self.settings.db_path) as conn:
            init_schema(conn)
            out = get_sources_by_type(conn, self.source_type)
        return out
