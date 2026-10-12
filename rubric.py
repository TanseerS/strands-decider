"""The rubric: each entry becomes one named, calibrated dimension.

A conventional embedding gives you ~1536 opaque floats. This gives you one float
per question below, each in [0, 1], each with a name you chose and a calibration
the model was trained to honour. Everything in analyze.py is possible only because
the axes mean something.

Keep questions independent. Two axes that ask the same thing in different words
double-count that trait in every distance calculation downstream.
"""
from __future__ import annotations

from strands_decider.schema import NoulQuestion, Question, ScoreQuestion

RUBRIC: dict[str, Question] = {
    "billing": NoulQuestion(
        instructions="Is this about payments, payouts, invoices, refunds or charges?"
    ),
    "security": NoulQuestion(
        instructions="Does this raise a security or unauthorised-access concern?"
    ),
    "bug": NoulQuestion(
        instructions="Is the customer reporting that something is broken or malfunctioning?"
    ),
    "presales": NoulQuestion(
        instructions="Is this a pre-purchase question about plans, pricing or capabilities?"
    ),
    "money_at_stake": NoulQuestion(
        instructions="Is a specific sum of the customer's money at risk or in dispute?"
    ),
    "repeat_contact": NoulQuestion(
        instructions="Does the customer indicate they have raised this before without resolution?"
    ),
    "angry": NoulQuestion(
        instructions="Is the customer's tone frustrated or angry?"
    ),
    "churn_risk": NoulQuestion(
        instructions="Does the customer hint they may cancel or switch to a competitor?"
    ),
    "urgency": ScoreQuestion(
        instructions="How urgently does this need a response?",
        criteria=["whenever", "this week", "today", "right now"],
    ),
    "blast_radius": ScoreQuestion(
        instructions="How many people does this problem affect?",
        criteria=["just this customer", "their whole team", "many customers"],
    ),
    "effort": ScoreQuestion(
        instructions="How much work is resolving this likely to take?",
        criteria=["a quick reply", "some investigation", "an engineering fix"],
    ),
}

AXES: list[str] = list(RUBRIC)


def levels(axis: str) -> int:
    """Number of ordinal levels for a score axis; 2 for a yes/no axis."""
    q = RUBRIC[axis]
    return len(q.criteria) if isinstance(q, ScoreQuestion) else 2


def normalise(axis: str, answer) -> float:
    """Map any answer onto [0, 1] so axes are mutually comparable.

    noul  -> the probability itself.
    score -> expected level index divided by (levels - 1); read_score returns an
             expectation over level indices, so a 4-level scale lands in [0, 3].
    """
    if answer.type == "noul":
        return float(answer.noul)
    return float(answer.score) / (levels(axis) - 1)
