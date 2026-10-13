"""A support agent whose context window is chosen by Strands Decider 2B.

Pattern: #context-memory

Retrieval proposes; Decider 2B disposes. A cheap lexical retriever pulls candidate
notes from memory, Decider 2B scores each one with a typed question and keeps only
those that actually answer the question, and Amazon Bedrock writes the reply from
the surviving notes.

The interesting part is the decoys. Ask about failed payouts and a similarity
search happily returns notes about unexpected card charges and duplicate charges:
same vocabulary, same domain, wrong answer. The gate drops them, and says why.

    python agent.py "our nightly payout batch keeps failing"
    python agent.py --compare "our nightly payout batch keeps failing"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import boto3
import torch

from patterns import ContextGate, GateReport, as_chunks
from strands_decider.infer import load_engine

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"
BEDROCK_MODEL = "us.amazon.nova-lite-v1:0"
BEDROCK_REGION = "us-east-1"

BASE_SYSTEM = (
    "You are a support engineer. Answer the customer's question using only the "
    "reference notes provided. If the notes do not answer it, say so plainly "
    "rather than guessing. Be concise and concrete."
)

# Decider 2B returns a calibrated probability, not a relative score, so the
# strength of the surviving evidence is meaningful on its own. That lets the
# agent match its tone to its evidence instead of always sounding certain.
GROUNDING_STYLE = {
    "strong": "The notes directly answer this. Answer plainly and confidently.",
    "partial": (
        "The notes only partially cover this. Answer what they support, and say "
        "explicitly which part of the question you cannot answer from them."
    ),
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "our", "my", "we", "i", "it",
    "to", "of", "and", "for", "on", "in", "at", "that", "this", "with", "has",
    "have", "been", "keeps", "does", "do", "not", "but", "they", "them", "from",
}


def _stems(text: str) -> set[str]:
    """Crude 5-char stemming so payout/payouts and failing/failures collide."""
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


def retrieve(query: str, rows: list[dict], n: int) -> list[dict]:
    """Deliberately naive lexical overlap -- it is what the gate has to clean up.

    Returns the top n by overlap without a relevance floor, which is exactly how
    a top-k vector search behaves: it always hands back k candidates, relevant or
    not. Supplying that floor is the gate's job.
    """
    terms = _stems(query)
    scored = [(len(terms & _stems(r["text"])), r) for r in rows]
    scored.sort(key=lambda kv: -kv[0])
    return [r for _, r in scored[:n]]


def ask_bedrock(
    client, query: str, notes: list[str], grounding: str = "strong"
) -> tuple[str, int]:
    context = "\n\n".join(f"- {n}" for n in notes) or "(no reference notes)"
    msg = f"Reference notes:\n{context}\n\nCustomer question:\n{query}"
    system = BASE_SYSTEM
    if style := GROUNDING_STYLE.get(grounding):
        system = f"{BASE_SYSTEM} {style}"
    r = client.converse(
        modelId=BEDROCK_MODEL,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": msg}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.2},
    )
    text = r["output"]["message"]["content"][0]["text"]
    return text, r["usage"]["inputTokens"]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", default=[])
    ap.add_argument("--memory", type=Path, default=Path("data/memory.jsonl"))
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--compare", action="store_true",
                    help="also answer with the ungated context, to show the difference")
    args = ap.parse_args()

    query = " ".join(args.query) or "our nightly payout batch keeps failing"
    rows = [json.loads(l) for l in args.memory.read_text().splitlines() if l.strip()]

    candidates = retrieve(query, rows, args.candidates)
    if not candidates:
        sys.exit("retriever found nothing")

    print(f"Q: {query}\n")
    print(f"retriever proposed {len(candidates)}: {' '.join(c['id'] for c in candidates)}\n")

    engine = load_engine(CHECKPOINT, device=auto_device())
    gate = ContextGate(engine, threshold=args.threshold)

    # Scored and printed one at a time: each decision is independent, and seeing
    # them land as they are made is the whole point of a gate you can inspect.
    print("Decider 2B gate:")
    t0 = time.perf_counter()
    verdicts = []
    for chunk in as_chunks(candidates):
        v = gate.score_one(query, chunk)
        verdicts.append(v)
        mark = "keep" if v.keep else "drop"
        print(f"  [{mark}] {v.id}  answers_it={v.axes['answers_it']:.2f} "
              f"same_topic={v.axes['same_topic']:.2f} "
              f"specificity={v.axes['specificity']:.2f}  -- {v.reason}",
              flush=True)
    gate_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  {gate_ms:.0f}ms for {len(verdicts)} chunks "
          f"({gate_ms / len(verdicts):.0f}ms each)\n")

    kept_all = sorted([v for v in verdicts if v.keep], key=lambda v: -v.relevance)
    report = GateReport(
        kept=tuple(kept_all[: args.keep]),
        dropped=tuple([v for v in verdicts if not v.keep] + kept_all[args.keep :]),
    )

    print("Memory trace:")
    for line in report.trace().splitlines():
        print(f"  {line}")
    print()

    if report.grounding == "none":
        # Declining silently hides the gate's own mistakes. Show what it almost
        # used, so a wrong rejection is visible instead of looking like an
        # empty memory store.
        print("Nothing in memory answers this, so the agent declines rather than")
        print("inventing an answer. The closest it looked at:\n")
        for v in report.near_misses:
            print(f"  {v.id}  {v.relevance:.2f}  {v.text[:88]}...")
        return

    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    kept = list(report.kept)
    answer, tokens = ask_bedrock(
        client, query, [v.text for v in kept], grounding=report.grounding
    )

    print(f"Bedrock ({BEDROCK_MODEL}) answered from {' '.join(v.id for v in kept)}, "
          f"grounding {report.grounding}, {tokens} input tokens:\n")
    print(answer.strip())

    if args.compare:
        ungated, utokens = ask_bedrock(
            client, query, [c["text"] for c in candidates], grounding="strong"
        )
        print(f"\n{'=' * 70}")
        print(f"Ungated: all {len(candidates)} chunks, {utokens} input tokens "
              f"({utokens - tokens:+d} vs gated, {100 * (utokens - tokens) / utokens:.0f}% saved)\n")
        print(ungated.strip())


if __name__ == "__main__":
    main()
