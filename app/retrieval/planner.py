"""
Retrieval planner: one LLM call that outputs a structured plan (which sources, what query per source).

Default path uses direct OpenAI-compatible API calls.
Optional path uses a thin LangChain model wrapper when USE_LANGCHAIN=true.
"""
import asyncio
import json
import re
import time

from pydantic import ValidationError

from app.config import get_settings
from app.llm_client import get_chat_client, get_chat_model
from app.logging_config import get_logger
from app.models import RetrievalPlan, RetrievalStep

logger = get_logger(__name__)


PLANNER_SYSTEM = """You are a retrieval planner for a scientific RAG system. You have access to three source types:
- document: unstructured text (papers, reports already ingested) - use semantic search; your "query" is the search string.
- structured: tabular data (CSV) - use for numbers, tables, datasets; your "query" describes what to look up.
- arxiv: live search on arXiv.org (title, abstract, author, etc.); your "query" is an arXiv search query (e.g. "ti:electron", "all:machine learning", "au:smith"). Use for recent papers or when the question is about published research.

Given a research question, output a JSON object with a "steps" array. Each step has:
- source_type: "document", "structured", or "arxiv"
- query: the search query or lookup description for that source
- reason: one sentence why this step helps answer the question

Use at least one step. Prefer "arxiv" when the question is about finding papers or recent research. Output ONLY valid JSON, no markdown."""

PLANNER_USER_TMPL = """Research question: {question}

Available source types: document, structured, arxiv.

Output your retrieval plan as JSON: {{"steps": [{{"source_type": "...", "query": "...", "reason": "..."}}, ...]}}"""


def _fallback_plan(question: str) -> RetrievalPlan:
    """When LLM output is invalid, return a single document step so the pipeline can still run."""
    return RetrievalPlan(steps=[RetrievalStep(source_type="document", query=question, reason="fallback")])


def _normalize_text_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content or "").strip()


class RetrievalPlanner:
    def __init__(self):
        self.settings = get_settings()
        self._client = get_chat_client()
        self._model = get_chat_model()

    async def _call_direct(self, user: str) -> str:
        for attempt in range(self.settings.openai_max_retries + 1):
            try:
                resp = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self._model,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(
                    "planner_attempt_failed",
                    extra={"attempt": attempt + 1, "error": type(e).__name__},
                )
                if attempt < self.settings.openai_max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    raise
        raise RuntimeError("planner_direct_failed")

    async def _call_langchain(self, user: str) -> RetrievalPlan:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.langchain_client import get_langchain_chat_model
        from app.models import RetrievalPlan

        llm = get_langchain_chat_model().with_structured_output(RetrievalPlan)
        for attempt in range(self.settings.openai_max_retries + 1):
            try:
                plan = await llm.ainvoke([
                    SystemMessage(content=PLANNER_SYSTEM),
                    HumanMessage(content=user),
                ])
                return plan
            except Exception as e:
                logger.warning(
                    "planner_attempt_failed",
                    extra={"attempt": attempt + 1, "error": type(e).__name__},
                )
                if attempt < self.settings.openai_max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    raise
        raise RuntimeError("planner_langchain_failed")

    async def plan(self, question: str) -> RetrievalPlan:
        """Call LLM once; parse JSON into RetrievalPlan; validate steps. Timeout and retries applied."""
        user = PLANNER_USER_TMPL.format(question=question)
        if self.settings.use_langchain:
            try:
                plan = await self._call_langchain(user)
            except Exception:
                logger.warning("planner_langchain_using_fallback")
                return _fallback_plan(question)
        else:
            try:
                raw = await self._call_direct(user)
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)
                plan = RetrievalPlan(**data)
            except (json.JSONDecodeError, ValidationError):
                logger.warning("planner_parse_failed")
                return _fallback_plan(question)

        allowed = {"document", "structured", "arxiv"}
        plan.steps = [s for s in plan.steps if s.source_type in allowed]
        if not plan.steps:
            return _fallback_plan(question)
        return plan
