"""
Structured logging for production. No secrets in log messages.
"""
import logging
import sys
from contextvars import ContextVar

# Request ID for tracing (set by middleware)
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class StructuredFormatter(logging.Formatter):
    """Format as level + message + optional extra (request_id, duration_ms, etc.)."""

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_ctx.get() or getattr(record, "request_id", "")
        parts = [
            self.formatTime(record, self.datefmt),
            record.levelname,
            record.name,
        ]
        if rid:
            parts.append(f"request_id={rid}")
        parts.append(record.getMessage())
        # LogRecord stores extra keys as attributes
        for key in ("log_level", "steps", "chunks", "duration_ms", "abstained", "error", "attempt", "source_id"):
            if hasattr(record, key):
                parts.append(f"{key}={getattr(record, key)}")
        return " | ".join(str(p) for p in parts)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with structured formatter. Idempotent."""
    level_value = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            h.setFormatter(StructuredFormatter())
            h.setLevel(level_value)
        root.setLevel(level_value)
        return
    root.setLevel(level_value)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
