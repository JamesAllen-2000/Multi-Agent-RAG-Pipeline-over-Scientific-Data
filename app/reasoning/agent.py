"""
Reasoning agent: synthesizes an answer from retrieved evidence only; may call tools (e.g. calculator).

Default path uses direct OpenAI-compatible tool-calling.
Optional path uses LangChain model wrapper when USE_LANGCHAIN=true.
"""
import asyncio
import json
import re

from app.config import get_settings
from app.llm_client import get_chat_client, get_chat_model
from app.models import RetrievedChunk
from app.reasoning.tools import calculator


REASONING_SYSTEM = """You are a scientific reasoning agent. You must answer the user's question using ONLY the provided evidence. Every claim must be traceable to a cited source (by source_id). If the evidence is insufficient to answer, say so clearly and do not guess. You have a calculator tool: use it for any arithmetic (e.g. converting units, summing numbers from the evidence). Do not make up numbers."""

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Input a string with numbers and + - * / ** ( ). Use for arithmetic on retrieved numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2.5 * 10' or '(100 + 50) / 2'"},
            },
            "required": ["expression"],
        },
    },
}


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source {c.source_id}]\n{c.content}")
    return "\n\n---\n\n".join(parts)


def _run_tool(name: str, arguments: dict) -> str:
    if name == "calculator":
        return calculator(arguments.get("expression", ""))
    return "Unknown tool"


def _extract_citations(text: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"\[Source\s+([\w\-.]+)\]", text or "", re.I):
        out.add(m.group(1))
    return out


class ReasoningAgent:
    def __init__(self):
        self.settings = get_settings()
        self._client = get_chat_client()
        self._model = get_chat_model()

    async def _run_direct(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str], bool]:
        evidence_text = _format_evidence(chunks)
        user_content = f"Evidence:\n\n{evidence_text}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        max_tool_rounds = 5
        cited_ids: set[str] = set()

        for _ in range(max_tool_rounds):
            resp = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
                tools=[CALCULATOR_TOOL],
                tool_choice="auto",
                temperature=0,
            )
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            cited_ids.update(_extract_citations(content))

            if not (msg.tool_calls and len(msg.tool_calls) > 0):
                return (content or "I could not produce an answer from the evidence."), list(cited_ids), False

            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _run_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return "Reasoning was cut off after multiple tool rounds.", list(cited_ids), True

    async def _run_langchain(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str], bool]:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langchain_core.tools import tool

        from app.langchain_client import get_langchain_chat_model

        @tool("calculator")
        def calculator_tool(expression: str) -> str:
            """Evaluate arithmetic expression with + - * / ** and parentheses only."""
            return calculator(expression)

        llm = get_langchain_chat_model().bind_tools([calculator_tool])
        evidence_text = _format_evidence(chunks)
        user_content = f"Evidence:\n\n{evidence_text}\n\nQuestion: {question}"
        messages = [
            SystemMessage(content=REASONING_SYSTEM),
            HumanMessage(content=user_content),
        ]

        max_tool_rounds = 5
        cited_ids: set[str] = set()

        for _ in range(max_tool_rounds):
            ai_msg = await llm.ainvoke(messages)
            content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content or "")
            cited_ids.update(_extract_citations(content))
            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                return (content.strip() or "I could not produce an answer from the evidence."), list(cited_ids), False

            messages.append(ai_msg)
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tool_call_id = tc.get("id", "")
                result = await asyncio.to_thread(_run_tool, name, args)
                messages.append(ToolMessage(content=result, tool_call_id=tool_call_id, name=name or "calculator"))

        return "Reasoning was cut off after multiple tool rounds.", list(cited_ids), True

    async def run(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str], bool]:
        """Returns (answer, list of source_ids cited, abstained)."""
        if not chunks:
            return "No evidence was retrieved. I cannot answer without sources.", [], True

        if self.settings.use_langchain:
            return await self._run_langchain(question, chunks)
        return await self._run_direct(question, chunks)
