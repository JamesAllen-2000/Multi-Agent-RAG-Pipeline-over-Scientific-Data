"""
Ingest structured data (CSV): validate schema, copy under data dir, store metadata in SQLite.
Retrieval will query via pandas (read CSV, filter/aggregate). No embedding — structured is queried deterministically.
"""
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.db import connection_scope, init_schema, upsert_source, increment_data_version, get_sources_by_type
from app.ingestion.base import BaseIngester
from app.models import SourceMetadata


class StructuredIngester(BaseIngester):
    source_type = "structured"

    def __init__(self):
        self.settings = get_settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.data_dir / "tables").mkdir(parents=True, exist_ok=True)

    def ingest(self, path: Path, title: str | None = None, **kwargs) -> SourceMetadata:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if path.suffix.lower() != ".csv":
            raise ValueError("Structured ingester supports only CSV for MVP")

        df = pd.read_csv(path, nrows=0)
        columns = list(df.columns)
        # Copy to stable location under data/tables
        source_id = hashlib.sha256(path.resolve().as_posix().encode()).hexdigest()[:16]
        dest = self.settings.data_dir / "tables" / f"{source_id}.csv"
        shutil.copy2(path, dest)

        meta = SourceMetadata(
            source_id=source_id,
            source_type=self.source_type,
            title=title or path.name,
            metadata={
                "path": str(path),
                "table_path": str(dest),
                "columns": columns,
                "schema": {c: "string" for c in columns},
            },
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
