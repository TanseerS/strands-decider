"""Reusable Decider 2B decision pattern: #context-memory

Score which memory or context chunks are relevant before injecting them.

Retrieval gives you candidates ranked by embedding similarity, which measures
*aboutness*, not *usefulness*. A note about unexpected card charges is highly
similar to a question about failed payouts -- same vocabulary, same domain, wrong
answer. Injecting it costs tokens and invites the generative model to use it.

This gate asks Decider 2B a typed question per candidate and keeps only the chunks
that earn their place. It returns a calibrated score and a reason for each decision,
so the agent can explain its own context window.

    from patterns import ContextGate
    from strands_decider.infer import load_engine

    gate = ContextGate(load_engine(CHECKPOINT, device="mps"))
    kept = gate.select(user_question, candidate_chunks, k=3)
    prompt = build_prompt(user_question, [c.text for c in kept])

Drop-in notes
-------------
* Any object with `.ask(state, questions) -> response` works as the engine.
* `threshold` is on a calibrated probability, so 0.6 means the same thing across
  different corpora. That is the point of using a calibrated model here.
* `min_confidence` abstains rather than guessing, and defaults to 0.0 -- off.
  Abstention sounds like the safe default and measured badly: on the bundled eval
  a floor of 0.35 cut recall from 0.92 to 0.58 and bought no extra precision,
  because the chunks it suppressed were ones the threshold already handled
  correctly. Raise it to ~0.10 only if a wrong chunk costs you more than a
  missing one. Sweep it on your own data with sweep.py rather than trusting
  either default.
* To wire into a Strands agent, call `select()` inside an InterventionHandler
  before the model call, or inside your retriever's post-processing step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from strands_decider.schema import NoulQuestion, Question, ScoreQuestion

# The typed question set. `answers_it` is the decision; the other two axes are
# diagnostic and make the gate's reasoning legible in logs and demos.
RELEVANCE_QUESTIONS: dict[str, Question] = {
    "answers_it": NoulQuestion(
        instructions=(
            "Does the reference note contain information that actually answers "
            "the user's question?"
        )
    ),
    "same_topic": NoulQuestion(
        instructions=(
            "Is the reference note about the same underlying problem as the "
            "user's question, rather than merely a related one?"
        )
    ),
    "specificity": ScoreQuestion(
        instructions="How specific and actionable is the reference note?",
        criteria=["vague background", "general guidance", "a concrete procedure"],
    ),
}

STATE_TEMPLATE = (
    "USER QUESTION:\n{query}\n\n"
    "REFERENCE NOTE:\n{chunk}\n\n"
    "Decide whether this reference note should be given to an assistant that is "
    "about to answer the user's question."
)


class Chunk(Protocol):
    id: str
    text: str


@dataclass(frozen=True)
class GateResult:
    """One gate decision, carrying its own explanation."""

    id: str
    text: str
    keep: bool
    relevance: float
    confidence: float
    axes: dict[str, float]
    reason: str

    def __str__(self) -> str:
        verdict = "KEEP" if self.keep else "DROP"
        return f"{verdict} {self.id} relevance={self.relevance:.2f} {self.reason}"


@dataclass(frozen=True)
class GateReport:
    """What the gate did, in a form the agent can act on and a human can read.

    `grounding` is the agent-visible consequence of calibration. Because
    `answers_it` is a calibrated probability rather than a relative score, the
    strength of the best surviving chunk is meaningful on its own, so the agent
    can decide how confidently to speak instead of always sounding certain.
    """

    kept: tuple[GateResult, ...]
    dropped: tuple[GateResult, ...]

    @property
    def grounding(self) -> str:
        if not self.kept:
            return "none"
        best = max(r.relevance for r in self.kept)
        return "strong" if best >= 0.7 else "partial"

    @property
    def near_misses(self) -> tuple[GateResult, ...]:
        """The best of what was discarded -- what the agent almost used."""
        return tuple(sorted(self.dropped, key=lambda r: -r.relevance)[:3])

    def trace(self) -> str:
        """A memory trace in plain words. Every line is a decision with a number."""
        lines = [f"grounding: {self.grounding}"]
        for r in sorted(self.kept, key=lambda r: -r.relevance):
            lines.append(
                f"  used    {r.id}  {r.relevance:.2f}  "
                f"({_specificity_word(r)}, {r.reason})"
            )
        for r in self.near_misses:
            lines.append(f"  ignored {r.id}  {r.relevance:.2f}  ({r.reason})")
        return "\n".join(lines)


def _specificity_word(r: GateResult) -> str:
    s = r.axes.get("specificity", 1.0)
    if s >= 1.5:
        return "a concrete procedure"
    if s >= 0.75:
        return "general guidance"
    return "vague background"


class ContextGate:
    """Decide which candidate chunks are worth putting in the prompt."""

    def __init__(
        self,
        engine: Any,
        *,
        threshold: float = 0.45,
        min_confidence: float = 0.0,
        questions: dict[str, Question] | None = None,
    ) -> None:
        self.engine = engine
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.questions = questions or RELEVANCE_QUESTIONS

    def score_one(self, query: str, chunk: Chunk) -> GateResult:
        state = STATE_TEMPLATE.format(query=query, chunk=chunk.text)
        answers = self.engine.ask(state, self.questions).answers

        axes: dict[str, float] = {}
        for name, a in answers.items():
            axes[name] = float(a.noul) if a.type == "noul" else float(a.score)

        relevance = axes["answers_it"]
        # A noul carries no separate confidence field; distance from 0.5 is the
        # honest equivalent -- 0.5 is maximum doubt.
        confidence = abs(relevance - 0.5) * 2

        if confidence < self.min_confidence:
            keep, reason = False, f"abstained, confidence {confidence:.2f}"
        elif relevance < self.threshold:
            keep, reason = False, f"does not answer it ({relevance:.2f})"
        elif axes.get("same_topic", 1.0) < 0.5:
            keep, reason = False, f"adjacent topic only ({axes['same_topic']:.2f})"
        else:
            keep, reason = True, f"answers it ({relevance:.2f})"

        return GateResult(
            id=chunk.id,
            text=chunk.text,
            keep=keep,
            relevance=relevance,
            confidence=confidence,
            axes=axes,
            reason=reason,
        )

    def score(self, query: str, chunks: Iterable[Chunk]) -> list[GateResult]:
        return [self.score_one(query, c) for c in chunks]

    def select(
        self, query: str, chunks: Iterable[Chunk], k: int | None = None
    ) -> list[GateResult]:
        """Gate the candidates, best first. `k` caps how many survive."""
        kept = [r for r in self.score(query, chunks) if r.keep]
        kept.sort(key=lambda r: -r.relevance)
        return kept[:k] if k else kept

    def report(
        self, query: str, chunks: Iterable[Chunk], k: int | None = None
    ) -> GateReport:
        """Like `select`, but keeps the rejections so the agent can explain itself."""
        results = self.score(query, chunks)
        kept = sorted([r for r in results if r.keep], key=lambda r: -r.relevance)
        if k:
            spill = kept[k:]
            kept = kept[:k]
        else:
            spill = []
        dropped = [r for r in results if not r.keep] + spill
        return GateReport(kept=tuple(kept), dropped=tuple(dropped))


def as_chunks(rows: Sequence[dict]) -> list[Chunk]:
    """Adapt plain dicts with 'id' and 'text' to the Chunk protocol."""

    @dataclass(frozen=True)
    class _C:
        id: str
        text: str

    return [_C(id=r["id"], text=r["text"]) for r in rows]
