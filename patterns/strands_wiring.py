"""Drop-in wiring: the context gate in a Strands agent's path.

`ContextGate` is framework-agnostic. This is the piece that puts it into a
Strands agent without you editing the agent's logic, as an InterventionHandler:

    from strands import Agent
    from patterns.strands_wiring import ContextGateIntervention

    agent = Agent(
        model=model,
        tools=[search_memory],
        interventions=[ContextGateIntervention(gate, retriever_tool="search_memory")],
    )

Two interception points, both earning their place:

`after_tool_call`  the retriever has returned candidates and they are about to
                   enter the conversation. Gate them here and the model never
                   sees what did not survive.

`before_model_call`  inject the grounding level, so the agent's tone matches its
                     evidence instead of always sounding certain.

The retriever tool must return JSON of the form `{"chunks": [{"id", "text"}]}`.
Anything else is passed through untouched, so this is safe to attach to an agent
whose other tools return whatever they like.
"""
from __future__ import annotations

import json
from typing import Any

from strands import InterventionHandler
from strands.hooks import AfterToolCallEvent, BeforeModelCallEvent
from strands.interventions import Guide, Proceed, Transform

from .context_relevance import ContextGate, GateReport, as_chunks

GROUNDING_GUIDANCE = {
    "partial": (
        "The reference notes only partially cover this question. Answer what they "
        "support and say explicitly which part you cannot answer from them."
    ),
    "none": (
        "Nothing in memory answers this question. Say so plainly rather than "
        "guessing from the notes you were given."
    ),
}


class ContextGateIntervention(InterventionHandler):
    """Score retrieved chunks with Decider 2B before they reach the model."""

    name = "context-gate"

    def __init__(
        self,
        gate: ContextGate,
        *,
        retriever_tool: str,
        keep: int | None = 3,
        verbose: bool = True,
    ) -> None:
        self.gate = gate
        self.retriever_tool = retriever_tool
        self.keep = keep
        self.verbose = verbose
        self.last_report: GateReport | None = None

    # -- the gate itself ----------------------------------------------------

    def after_tool_call(self, event: AfterToolCallEvent, **kwargs: Any):
        tool_use = getattr(event, "tool_use", None) or {}
        if tool_use.get("name") != self.retriever_tool:
            return Proceed()

        chunks = _parse_chunks(getattr(event, "result", None))
        if not chunks:
            return Proceed()

        query = _query_of(event) or tool_use.get("input", {}).get("query", "")
        report = self.gate.report(query, as_chunks(chunks), k=self.keep)
        self.last_report = report

        if self.verbose:
            print(f"[{self.name}] {report.trace()}")

        def apply(e: Any) -> None:
            kept = [{"id": r.id, "text": r.text} for r in report.kept]
            payload = {"chunks": kept, "grounding": report.grounding}
            if not kept:
                payload["near_misses"] = [
                    {"id": r.id, "relevance": round(r.relevance, 2)}
                    for r in report.near_misses
                ]
            e.result["content"] = [{"text": json.dumps(payload)}]

        dropped = len(chunks) - len(report.kept)
        return Transform(apply=apply, reason=f"context gate dropped {dropped} of {len(chunks)}")

    # -- tone follows the evidence ------------------------------------------

    def before_model_call(self, event: BeforeModelCallEvent, **kwargs: Any):
        report, self.last_report = self.last_report, None
        if report is None:
            return Proceed()
        if guidance := GROUNDING_GUIDANCE.get(report.grounding):
            return Guide(feedback=guidance)
        return Proceed()


def _parse_chunks(result: Any) -> list[dict]:
    """Pull `{"chunks": [...]}` out of a ToolResult, or give up quietly."""
    if not isinstance(result, dict):
        return []
    for block in result.get("content") or []:
        text = block.get("text") if isinstance(block, dict) else None
        if not text:
            continue
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            continue
        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if isinstance(chunks, list) and all(
            isinstance(c, dict) and "id" in c and "text" in c for c in chunks
        ):
            return chunks
    return []


def _query_of(event: Any) -> str:
    """The user's last message, which is what the chunks have to answer."""
    agent = getattr(event, "agent", None)
    for message in reversed(getattr(agent, "messages", []) or []):
        if message.get("role") != "user":
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("text"):
                return block["text"]
    return ""
