"""Does asking N questions cost ~1 state encoding, or N?"""
import time

import torch

from strands_decider.infer import load_engine
from strands_decider.schema import NoulQuestion

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"

STATE = (
    "Customer writes: our nightly payout batch has failed three days running. "
    "Finance is escalating. We moved 40k GBP of volume last month and are "
    "considering alternatives if this is not fixed before the weekend."
) * 4  # a longer state makes the prefix cost dominate

QS = [
    "Is this urgent?", "Is this about billing?", "Is the customer angry?",
    "Is there churn risk?", "Does this need a human?", "Is this a bug report?",
    "Is money at stake?", "Is this a security issue?",
]


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    engine = load_engine(CHECKPOINT, device=device)
    print(f"device={device}  state_chars={len(STATE)}\n")

    def timed(n: int, reps: int = 3) -> float:
        qs = {f"q{i}": NoulQuestion(instructions=QS[i]) for i in range(n)}
        engine.ask(STATE, qs)  # warm
        best = min(
            (time.perf_counter() - t0)
            for _ in range(reps)
            for t0 in [time.perf_counter()]
            if engine.ask(STATE, qs) or True
        )
        return best * 1000

    one = timed(1)
    print(f" 1 question : {one:7.0f} ms")
    for n in (2, 4, 8):
        t = timed(n)
        print(f"{n:2d} questions: {t:7.0f} ms   ({t/one:4.2f}x of 1q, {t/n:6.0f} ms/question)")

    print("\nSeparate calls, 8 questions one at a time:")
    qs = [NoulQuestion(instructions=q) for q in QS]
    engine.ask(STATE, {"q": qs[0]})
    t0 = time.perf_counter()
    for i, q in enumerate(qs):
        engine.ask(STATE, {f"q{i}": q})
    sep = (time.perf_counter() - t0) * 1000
    print(f" 8 separate : {sep:7.0f} ms   ({sep/8:6.0f} ms/question)")


if __name__ == "__main__":
    main()
