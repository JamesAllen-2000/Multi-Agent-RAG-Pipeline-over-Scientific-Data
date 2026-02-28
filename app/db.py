"""
SQLite store for source metadata and data version. No ORM — plain sqlite3.
Used by ingestion (write) and retrieval (read). Single DB file path from config.
Production: context manager for connections, timeout on connect.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.models import SourceMetadata

# Default timeout for busy wait (seconds)
DB_TIMEOUT = 5.0


def get_connection(db_path: Path, timeout: float = DB_TIMEOUT) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connection_scope(db_path: Path, timeout: float = DB_TIMEOUT) -> Iterator[sqlite3.Connection]:
    """Ensure connection is closed and commits on success. Use in ingestion/retrieval."""
    conn = get_connection(db_path, timeout=timeout)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            title TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}',
            ingested_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS data_version (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO data_version (key, value) VALUES ('global', 1);
    """)
    conn.commit()


def upsert_source(conn: sqlite3.Connection, meta: SourceMetadata) -> None:
    import json
    conn.execute(
        "INSERT OR REPLACE INTO sources (source_id, source_type, title, metadata, ingested_at) VALUES (?, ?, ?, ?, ?)",
        (meta.source_id, meta.source_type, meta.title, json.dumps(meta.metadata), meta.ingested_at),
    )
    conn.commit()


def get_sources_by_type(conn: sqlite3.Connection, source_type: str) -> list[SourceMetadata]:
    import json
    rows = conn.execute(
        "SELECT source_id, source_type, title, metadata, ingested_at FROM sources WHERE source_type = ?",
        (source_type,),
    ).fetchall()
    return [
        SourceMetadata(
            source_id=r["source_id"],
            source_type=r["source_type"],
            title=r["title"] or "",
            metadata=json.loads(r["metadata"] or "{}"),
            ingested_at=r["ingested_at"] or "",
        )
        for r in rows
    ]


def get_all_sources(conn: sqlite3.Connection) -> list[SourceMetadata]:
    import json
    rows = conn.execute(
        "SELECT source_id, source_type, title, metadata, ingested_at FROM sources"
    ).fetchall()
    return [
        SourceMetadata(
            source_id=r["source_id"],
            source_type=r["source_type"],
            title=r["title"] or "",
            metadata=json.loads(r["metadata"] or "{}"),
            ingested_at=r["ingested_at"] or "",
        )
        for r in rows
    ]


def get_data_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM data_version WHERE key = 'global'").fetchone()
    return int(row["value"]) if row else 1


def increment_data_version(conn: sqlite3.Connection) -> int:
    conn.execute("UPDATE data_version SET value = value + 1 WHERE key = 'global'")
    conn.commit()
    return get_data_version(conn)


def update_source_metadata(conn: sqlite3.Connection, source_id: str, extra: dict) -> None:
    """Merge extra keys into existing source metadata. No-op if source_id missing."""
    import json
    row = conn.execute("SELECT metadata FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    if not row:
        return
    meta = json.loads(row["metadata"] or "{}")
    meta.update(extra)
    conn.execute("UPDATE sources SET metadata = ? WHERE source_id = ?", (json.dumps(meta), source_id))
    conn.commit()


def get_structured_table_path(conn: sqlite3.Connection, source_id: str) -> str | None:
    """Stored metadata may contain 'table_path' for structured sources."""
    import json
    row = conn.execute("SELECT metadata FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    if not row:
        return None
    meta = json.loads(row["metadata"] or "{}")
    return meta.get("table_path")
