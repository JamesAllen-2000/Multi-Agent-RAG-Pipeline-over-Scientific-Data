"""
Readiness checks: LLM (Groq or OpenAI), embeddings (OpenAI), SQLite, Chroma. No secrets in check output.
"""
from pathlib import Path

from app.config import get_settings
from app.llm_client import has_llm_configured
from app.logging_config import get_logger

logger = get_logger(__name__)


def check_ready() -> tuple[bool, dict[str, str]]:
    """
    Returns (ok, details). details keys: llm, embeddings, db, chroma.
    Values are "ok" or an error message (no secrets).
    """
    settings = get_settings()
    details: dict[str, str] = {}

    # LLM: Groq or OpenAI key (for planner + reasoning)
    details["llm"] = "ok" if has_llm_configured() else "not_configured"

    # Embeddings: OpenAI only (for document/vector retrieval)
    if not (settings.openai_api_key and settings.openai_api_key.strip()):
        details["embeddings"] = "not_configured"
    else:
        details["embeddings"] = "ok"

    # SQLite: create dir and open connection
    try:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = __import__("sqlite3").connect(str(settings.db_path), timeout=2.0)
        conn.execute("SELECT 1")
        conn.close()
        details["db"] = "ok"
    except Exception as e:
        details["db"] = f"error: {type(e).__name__}"
        logger.warning("readiness_db_failed", extra={"error": type(e).__name__})

    # Chroma: open persistent client (heartbeat or list_collections as fallback)
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        if hasattr(client, "heartbeat"):
            client.heartbeat()
        else:
            client.list_collections()
        details["chroma"] = "ok"
    except Exception as e:
        details["chroma"] = f"error: {type(e).__name__}"
        logger.warning("readiness_chroma_failed", extra={"error": type(e).__name__})

    # Ready if LLM and db and chroma are ok (embeddings optional for arxiv/structured-only use)
    ok = details["llm"] == "ok" and details["db"] == "ok" and details["chroma"] == "ok"
    return ok, details
