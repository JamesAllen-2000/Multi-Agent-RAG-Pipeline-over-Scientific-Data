"""
FastAPI app: single /query endpoint. Runs plan → retrieve → reason, returns answer with sources and latency.
Production: request IDs, structured logging, bounded concurrency, timeouts, deep health, safe error responses.
"""
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.llm_client import has_llm_configured
from app.logging_config import configure_logging, get_logger, request_id_ctx
from app.models import (
    QueryRequest,
    QueryResponse,
    LatencyBreakdown,
    CitedSource,
)
from app.retrieval.planner import RetrievalPlanner
from app.retrieval.executor import RetrievalExecutor
from app.reasoning.agent import ReasoningAgent

logger = get_logger(__name__)

import asyncio

# Bounded concurrency: limit in-flight pipeline runs
_concurrency_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _concurrency_semaphore
    if _concurrency_semaphore is None:
        _concurrency_semaphore = asyncio.Semaphore(get_settings().max_concurrent_queries)
    return _concurrency_semaphore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and validate config on startup."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("startup", extra={"log_level": settings.log_level})
    if not has_llm_configured():
        logger.warning("LLM not configured (set GROQ_API_KEY or OPENAI_API_KEY); /query will return 503")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Multi-Agent RAG Pipeline",
    description="Query API: answer with sources and latency breakdown.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a request_id to each request for tracing. No PII in the ID."""
    rid = request.headers.get("X-Request-ID")
    if not rid or not rid.strip():
        rid = str(uuid.uuid4())[:8]
    request_id_ctx.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled errors; log with request_id, return 503 without leaking internals."""
    rid = request_id_ctx.get() or ""
    logger.exception("unhandled_exception", extra={"request_id": rid})
    return JSONResponse(
        status_code=503,
        content={"detail": "Service temporarily unavailable.", "request_id": rid},
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Run the full pipeline: plan retrieval → execute → reason → return answer + sources + latency."""
    settings = get_settings()
    if not has_llm_configured():
        logger.warning("query_rejected_no_llm_key")
        raise HTTPException(status_code=503, detail="Service misconfigured: set GROQ_API_KEY or OPENAI_API_KEY.")

    sem = _get_semaphore()
    if sem.locked():
        logger.warning("query_rejected_concurrency_limit")
        raise HTTPException(status_code=503, detail="Too many concurrent queries; try again shortly.")

    async with sem:
        return await _run_pipeline(req, settings)


async def _run_pipeline(req: QueryRequest, settings) -> QueryResponse:
    t_total_start = time.perf_counter()
    rid = request_id_ctx.get() or ""

    try:
        # 1) Planning
        t0 = time.perf_counter()
        planner = RetrievalPlanner()
        plan = await planner.plan(req.question)
        t_plan = (time.perf_counter() - t0) * 1000
        logger.info(
            "planning_done",
            extra={"request_id": rid, "steps": len(plan.steps), "duration_ms": round(t_plan, 2)},
        )

        # 2) Retrieval
        t0 = time.perf_counter()
        executor = RetrievalExecutor()
        chunks = await executor.execute_plan(plan.steps)
        t_retrieval = (time.perf_counter() - t0) * 1000
        logger.info(
            "retrieval_done",
            extra={"request_id": rid, "chunks": len(chunks), "duration_ms": round(t_retrieval, 2)},
        )

        # 3) Reasoning
        t0 = time.perf_counter()
        agent = ReasoningAgent()
        answer, cited_ids, abstained = await agent.run(req.question, chunks)
        t_reasoning = (time.perf_counter() - t0) * 1000
        logger.info(
            "reasoning_done",
            extra={"request_id": rid, "abstained": abstained, "duration_ms": round(t_reasoning, 2)},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("pipeline_error", extra={"request_id": rid})
        raise HTTPException(
            status_code=503,
            detail="Query failed; please try again.",
        ) from e

    t_total = (time.perf_counter() - t_total_start) * 1000

    # Build cited sources from chunks
    id_to_excerpt = {}
    for c in chunks:
        if c.source_id not in id_to_excerpt:
            id_to_excerpt[c.source_id] = (c.content[:200] + "..." if len(c.content) > 200 else c.content, c.score)
    sources = [
        CitedSource(
            source_id=sid,
            source_type=next((c.source_type for c in chunks if c.source_id == sid), "unknown"),
            excerpt=id_to_excerpt.get(sid, ("", 0))[0],
            score=id_to_excerpt.get(sid, (0, 0))[1],
        )
        for sid in cited_ids
    ]
    if not sources and chunks:
        seen = set()
        for c in chunks:
            if c.source_id not in seen:
                seen.add(c.source_id)
                sources.append(
                    CitedSource(
                        source_id=c.source_id,
                        source_type=c.source_type,
                        excerpt=c.content[:200] + "..." if len(c.content) > 200 else c.content,
                        score=c.score,
                    )
                )

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency=LatencyBreakdown(
            planning_ms=round(t_plan, 2),
            retrieval_ms=round(t_retrieval, 2),
            reasoning_ms=round(t_reasoning, 2),
            total_ms=round(t_total, 2),
        ),
        abstained=abstained,
        warning="Insufficient evidence; answer may be incomplete." if abstained else "",
    )


@app.get("/health")
def health():
    """Liveness: is the process up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: can we serve traffic (API key set, DB and Chroma reachable)."""
    from app.health import check_ready
    ok, details = check_ready()
    if ok:
        return {"status": "ready", "checks": details}
    return JSONResponse(status_code=503, content={"status": "not_ready", "checks": details})
