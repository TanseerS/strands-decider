"""Hunt for queries where the ungated agent answers wrongly.

'Saves tokens' is a weak claim; both answers being correct makes the gate look
optional. The interesting case is where an irrelevant-but-similar note actually
misleads the generative model. This runs several queries both ways in one model
load so the two answers can be compared side by side.

    python contrast.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3
import torch

from agent import BEDROCK_MODEL, BEDROCK_REGION, ask_bedrock, retrieve
from patterns import ContextGate
from patterns.context_relevance import as_chunks
from strands_decider.infer import load_engine

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"

# Each query has a plausible decoy in the store that could explain it wrongly.
QUERIES = [
    ("we were charged twice for the same annual plan", "kb03", "kb02 blames a plan upgrade"),
    ("my card was charged on a date I did not expect", "kb02", "kb03 blames a gateway retry"),
    ("our payout totals are slightly off every month", "kb05", "kb01 blames an expired mandate"),
    ("the refund has not reached our customer yet", "kb04", "kb03 blames a duplicate charge"),
    ("our nightly payout batch keeps failing", "kb01", "kb02/kb03 are charge notes"),
]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=6,
                    help="how many notes the retriever hands over; raise it to "
                         "test whether gating matters more as decoys multiply")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path("data/memory.jsonl").read_text().splitlines() if l.strip()]
    engine = load_engine(CHECKPOINT, device=auto_device())
    gate = ContextGate(engine)
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    for query, gold, decoy in QUERIES:
        print("=" * 78)
        print(f"Q: {query}")
        print(f"   correct note: {gold}   |   decoy risk: {decoy}\n")

        cands = retrieve(query, rows, args.candidates)
        t0 = time.perf_counter()
        verdicts = gate.score(query, as_chunks(cands))
        ms = (time.perf_counter() - t0) * 1000
        kept = sorted([v for v in verdicts if v.keep], key=lambda v: -v.relevance)

        print(f"   gate ({ms:.0f}ms): kept {[v.id for v in kept] or 'nothing'}"
              f"   dropped {[v.id for v in verdicts if not v.keep]}")

        if kept:
            g, gt = ask_bedrock(client, query, [v.text for v in kept])
        else:
            g, gt = "(gate kept nothing -- the agent would decline)", 0
        u, ut = ask_bedrock(client, query, [c["text"] for c in cands])

        print(f"\n   GATED ({gt} tok):\n     {g.strip()[:420]}")
        print(f"\n   UNGATED ({ut} tok):\n     {u.strip()[:420]}\n")


if __name__ == "__main__":
    main()
