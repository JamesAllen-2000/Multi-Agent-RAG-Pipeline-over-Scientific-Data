"""
Abstract interface for ingestion. Each source type has one ingester.
We use this so the pipeline can support multiple source types without magic.
"""
from abc import ABC, abstractmethod
from pathlib import Path

from app.models import SourceMetadata


class BaseIngester(ABC):
    """Ingest one source type: parse, chunk/validate, persist to vector DB and/or metadata store."""

    source_type: str = "base"

    @abstractmethod
    def ingest(self, path: Path, **kwargs) -> SourceMetadata:
        """Ingest from path (file or dir). Returns source metadata."""
        ...

    @abstractmethod
    def list_ingested(self) -> list[SourceMetadata]:
        """List already ingested sources of this type (from metadata store)."""
        ...
