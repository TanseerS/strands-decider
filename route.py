"""Ticket routing with Strands Decider 2B.

The decider does not generate text -- it scores a fixed set of options and
returns a calibrated probability for each one.
"""
from __future__ import annotations

import time

import torch

from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"

# criteria maps each option to a description; "" leaves it undescribed.
TEAMS = {
    "billing": "payments, payouts, invoices, refunds, subscription charges",
    "sales": "pricing, quotes, plans, discounts, new business",
    "retail": "products, stock, sizing, shipping, store operations",
}

TICKETS = [
    "Help! My payouts have been failing for 3 days!",
    "Do you offer volume discounts for 500+ seats?",
    "The size chart on the product page is wrong.",
]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = auto_device()
    t0 = time.perf_counter()
    engine = load_engine(CHECKPOINT, device=device)
    print(f"loaded on {device} in {time.perf_counter() - t0:.1f}s\n")

    question = ChoiceQuestion(
        instructions="Which team should handle this?", criteria=TEAMS
    )

    for ticket in TICKETS:
        t0 = time.perf_counter()
        response = engine.ask(ticket, {"team": question})
        ms = (time.perf_counter() - t0) * 1000

        answer = response.answers["team"]
        ranked = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])
        spread = "  ".join(f"{opt}={p:.3f}" for opt, p in ranked)

        print(f"{ticket}")
        print(f"  -> {answer.choice}  (confidence {answer.confidence:.3f}, {ms:.0f}ms)")
        print(f"     {spread}\n")


if __name__ == "__main__":
    main()
