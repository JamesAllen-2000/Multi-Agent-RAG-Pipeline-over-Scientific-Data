"""
Ingest unstructured documents (PDF, text): chunk, embed with OpenAI, store in Chroma.
What this does: (1) Extract text from PDF, (2) Split into overlapping chunks,
(3) Call OpenAI embeddings API, (4) Upsert into Chroma with source_id in metadata.
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from app.config import get_settings
from app.db import connection_scope, get_sources_by_type, init_schema, upsert_source, increment_data_version
from app.ingestion.base import BaseIngester
from app.models import SourceMetadata


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks with overlap. No LLM — deterministic."""
    if not text or not text.strip():
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start = end - overlap if overlap < chunk_size else end
    return chunks


def get_embedding(text: str, client, model: str) -> list[float]:
    """Single embedding call."""
    if hasattr(client, "embed_query"):
        return client.embed_query(text)
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


class DocumentIngester(BaseIngester):
    source_type = "document"

    def __init__(self):
        self.settings = get_settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)

    def _get_chroma(self):
        """Lazy import Chroma to avoid loading at module level. Chroma stores vectors and allows similarity search."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=str(self.settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document chunks for RAG"},
        )

    def ingest(self, path: Path, title: str | None = None, **kwargs) -> SourceMetadata:
        if not path.exists():
            raise FileNotFoundError(str(path))
        raw_text = ""
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            raw_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        elif path.suffix.lower() in (".txt", ".md"):
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported document type: {path.suffix}")

        source_id = hashlib.sha256(path.resolve().as_posix().encode()).hexdigest()[:16]
        chunks = chunk_text(
            raw_text,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No text chunks extracted")

        from app.llm_client import get_embedding_client
        client = get_embedding_client()
        coll = self._get_chroma()
        ids = [f"{source_id}_{i}" for i in range(len(chunks))]
        if hasattr(client, "embed_documents"):
            embeddings = client.embed_documents(chunks)
        else:
            embeddings = [
                get_embedding(c, client, self.settings.openai_embedding_model)
                for c in chunks
            ]
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source_id": source_id, "chunk_idx": i} for i in range(len(chunks))],
        )

        meta = SourceMetadata(
            source_id=source_id,
            source_type=self.source_type,
            title=title or path.name,
            metadata={"path": str(path), "chunk_count": len(chunks)},
            ingested_at=datetime.utcnow().isoformat() + "Z",
        )
        with connection_scope(self.settings.db_path) as conn:
            init_schema(conn)
            upsert_source(conn, meta)
            increment_data_version(conn)
        return meta

    def list_ingested(self) -> list[SourceMetadata]:
        with connection_scope(self.settings.db_path) as conn:
            init_schema(conn)
            out = get_sources_by_type(conn, self.source_type)
        return out
