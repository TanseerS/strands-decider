"""Why does the gate reject kb03?

kb03 answers "we were charged twice for the same annual plan" and the gate
scores it 0.27. This isolates the cause: is it the way the question is phrased,
the way the note is written, or the way I worded the axis?

    python probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from patterns.context_relevance import STATE_TEMPLATE
from strands_decider.infer import load_engine
from strands_decider.schema import NoulQuestion

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"

NOTES = {r["id"]: r["text"] for r in
         (json.loads(l) for l in Path("data/memory.jsonl").read_text().splitlines() if l.strip())}

ORIGINAL = "Does the reference note contain information that actually answers the user's question?"

PHRASINGS = [
    "we were charged twice for the same annual plan",
    "we were charged twice",
    "we got a duplicate charge",
    "there is a duplicate charge on our account",
    "why did a payment get charged twice?",
    "our card was billed two times for one annual subscription",
]

AXIS_WORDINGS = {
    "original": ORIGINAL,
    "explains": "Does the reference note explain the situation the user is describing?",
    "useful": "Would this reference note be useful to someone answering the user's question?",
    "relevant": "Is the reference note relevant to the user's question?",
    "same_problem": "Is the reference note about the same problem the user is describing?",
}


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score(engine, query: str, chunk: str, instructions: str) -> float:
    state = STATE_TEMPLATE.format(query=query, chunk=chunk)
    ans = engine.ask(state, {"q": NoulQuestion(instructions=instructions)}).answers["q"]
    return float(ans.noul)


def main() -> None:
    engine = load_engine(CHECKPOINT, device=auto_device())
    kb03 = NOTES["kb03"]

    print("1. Is it the question's phrasing?  (kb03, original axis wording)\n")
    for q in PHRASINGS:
        print(f"   {score(engine, q, kb03, ORIGINAL):.2f}  {q}")

    print("\n2. Is it the note?  (original question, every note scored)\n")
    q0 = PHRASINGS[0]
    ranked = sorted(((score(engine, q0, txt, ORIGINAL), i) for i, txt in NOTES.items()),
                    reverse=True)
    for s, i in ranked[:6]:
        mark = " <- the correct note" if i == "kb03" else ""
        print(f"   {s:.2f}  {i}  {NOTES[i][:58]}...{mark}")

    print("\n3. Is it my axis wording?  (kb03 + original question)\n")
    for name, instructions in AXIS_WORDINGS.items():
        print(f"   {score(engine, q0, kb03, instructions):.2f}  {name:<13} \"{instructions[:62]}...\"")


if __name__ == "__main__":
    main()
