"""
Ingest Kaggle datasets: download via kagglehub.dataset_download(slug), then ingest each file.
- CSV → structured (copy to data/tables, register with metadata dataset_slug).
- PDF / txt / md → document (chunk, embed, Chroma, register with metadata dataset_slug).
One parent source_type "kaggle" records the dataset; child sources are document/structured with dataset_slug in metadata.
"""
import re
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.db import (
    connection_scope,
    init_schema,
    upsert_source,
    increment_data_version,
    get_sources_by_type,
    update_source_metadata,
)
from app.ingestion.base import BaseIngester
from app.ingestion.documents import DocumentIngester
from app.ingestion.structured import StructuredIngester
from app.logging_config import get_logger
from app.models import SourceMetadata

logger = get_logger(__name__)

# File types we ingest from a Kaggle dataset
KAGGLE_STRUCTURED_SUFFIXES = (".csv",)
KAGGLE_DOCUMENT_SUFFIXES = (".pdf", ".txt", ".md")


def _sanitize_slug(slug: str) -> str:
    """Safe source_id from dataset slug (e.g. Cornell-University/arxiv -> Cornell-University_arxiv)."""
    return re.sub(r"[^\w\-.]", "_", slug.strip()).strip("_") or "kaggle"


class KaggleIngester(BaseIngester):
    """
    Ingest a Kaggle dataset by slug (e.g. Cornell-University/arxiv).
    Uses kagglehub.dataset_download(); requires Kaggle API credentials for private datasets.
    """
    source_type = "kaggle"

    def __init__(self):
        self.settings = get_settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._doc_ingester = DocumentIngester()
        self._struct_ingester = StructuredIngester()

    def ingest(
        self,
        path: Path | None = None,
        *,
        dataset_slug: str,
        title: str | None = None,
        **kwargs,
    ) -> SourceMetadata:
        """
        Download dataset via kagglehub and ingest supported files.
        path is ignored; use dataset_slug (e.g. "Cornell-University/arxiv").
        """
        try:
            import kagglehub
        except ImportError:
            raise ImportError("kagglehub is required for Kaggle ingestion; pip install kagglehub") from None

        slug = dataset_slug.strip()
        if not slug or "/" not in slug:
            raise ValueError("dataset_slug must be of form 'owner/dataset' (e.g. Cornell-University/arxiv)")

        download_path = kagglehub.dataset_download(slug)
        root = Path(download_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset path is not a directory: {root}")

        ingested_ids: list[str] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            suf = file_path.suffix.lower()
            try:
                if suf in KAGGLE_STRUCTURED_SUFFIXES:
                    meta = self._struct_ingester.ingest(file_path, title=file_path.name)
                    ingested_ids.append(meta.source_id)
                    with connection_scope(self.settings.db_path) as conn:
                        update_source_metadata(conn, meta.source_id, {"dataset_slug": slug})
                elif suf in KAGGLE_DOCUMENT_SUFFIXES:
                    meta = self._doc_ingester.ingest(file_path, title=file_path.name)
                    ingested_ids.append(meta.source_id)
                    with connection_scope(self.settings.db_path) as conn:
                        update_source_metadata(conn, meta.source_id, {"dataset_slug": slug})
            except Exception as e:
                logger.warning(
                    "kaggle_file_skip",
                    extra={"path": str(file_path), "error": type(e).__name__},
                )

        if not ingested_ids:
            raise ValueError(f"No supported files (.csv, .pdf, .txt, .md) found under {root}")

        # Register parent "kaggle" source
        source_id = _sanitize_slug(slug)
        parent_meta = SourceMetadata(
            source_id=source_id,
            source_type=self.source_type,
            title=title or slug,
            metadata={
                "dataset_slug": slug,
                "download_path": str(root),
                "ingested_files": ingested_ids,
                "ingested_count": len(ingested_ids),
            },
            ingested_at=datetime.utcnow().isoformat() + "Z",
        )
        with connection_scope(self.settings.db_path) as conn:
            init_schema(conn)
            upsert_source(conn, parent_meta)
            increment_data_version(conn)

        return parent_meta

    def list_ingested(self) -> list[SourceMetadata]:
        with connection_scope(self.settings.db_path) as conn:
            init_schema(conn)
            out = get_sources_by_type(conn, self.source_type)
        return out
