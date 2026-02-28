"""
Pydantic models for API and internal data. No framework magic — plain request/response shapes.
"""
from typing import Any

from pydantic import BaseModel, Field


# --- Ingestion ---
class SourceMetadata(BaseModel):
    source_id: str
    source_type: str  # "document" | "structured"
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: str = ""


# --- Retrieval planning (LLM output) ---
class RetrievalStep(BaseModel):
    """One retrieval action: where to search and what query."""

    source_type: str  # "document" | "structured" | "arxiv"
    query: str
    reason: str = ""


class RetrievalPlan(BaseModel):
    """Structured plan from retrieval planner LLM."""

    steps: list[RetrievalStep] = Field(default_factory=list)


# --- Retrieved evidence ---
class RetrievedChunk(BaseModel):
    source_id: str
    source_type: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Query API ---
class LatencyBreakdown(BaseModel):
    planning_ms: float = 0.0
    retrieval_ms: float = 0.0
    reasoning_ms: float = 0.0
    total_ms: float = 0.0


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class CitedSource(BaseModel):
    source_id: str
    source_type: str
    excerpt: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[CitedSource] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    abstained: bool = False
    warning: str = ""
