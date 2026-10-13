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
* `min_confidence` abstains rather than guessing. An uncertain chunk is dropped,
  which is the safe direction for context injection.
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


class ContextGate:
    """Decide which candidate chunks are worth putting in the prompt."""

    def __init__(
        self,
        engine: Any,
        *,
        threshold: float = 0.6,
        min_confidence: float = 0.35,
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


def as_chunks(rows: Sequence[dict]) -> list[Chunk]:
    """Adapt plain dicts with 'id' and 'text' to the Chunk protocol."""

    @dataclass(frozen=True)
    class _C:
        id: str
        text: str

    return [_C(id=r["id"], text=r["text"]) for r in rows]
