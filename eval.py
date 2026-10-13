"""Does the gate actually help? A labelled eval over the memory store.

Each query has exactly one correct note. The retriever returns the top k with no
relevance floor, so its precision is bounded at 1/k by construction. The question
is whether the gate raises precision without destroying recall -- and what it does
when it is unsure.

    python eval.py [--candidates 6] [--threshold 0.6]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from agent import retrieve
from patterns import ContextGate
from patterns.context_relevance import as_chunks
from strands_decider.infer import load_engine

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", type=Path, default=Path("data/memory.jsonl"))
    ap.add_argument("--eval", type=Path, default=Path("data/eval.jsonl"))
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--min-confidence", type=float, default=0.35)
    ap.add_argument("--scores", type=Path, default=Path("data/eval_scores.json"),
                    help="cache of raw per-chunk axes, so thresholds can be swept offline")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.memory.read_text().splitlines() if l.strip()]
    cases = [json.loads(l) for l in args.eval.read_text().splitlines() if l.strip()]

    engine = load_engine(CHECKPOINT, device=auto_device())
    gate = ContextGate(engine, threshold=args.threshold,
                       min_confidence=args.min_confidence)
    cache: list[dict] = []

    r_tp = r_fp = r_fn = 0
    g_tp = g_fp = g_fn = 0
    chars_in = chars_out = 0
    abstained = 0
    misses: list[str] = []
    t0 = time.perf_counter()

    for case in cases:
        gold = set(case["gold"])
        cands = retrieve(case["query"], rows, args.candidates)
        cand_ids = {c["id"] for c in cands}

        r_tp += len(cand_ids & gold)
        r_fp += len(cand_ids - gold)
        r_fn += len(gold - cand_ids)

        verdicts = gate.score(case["query"], as_chunks(cands))
        cache.append({
            "query": case["query"],
            "gold": sorted(gold),
            "chunks": [
                {"id": v.id, "chars": len(v.text), "axes": v.axes} for v in verdicts
            ],
        })
        kept = {v.id for v in verdicts if v.keep}
        abstained += sum(1 for v in verdicts if "abstained" in v.reason)

        g_tp += len(kept & gold)
        g_fp += len(kept - gold)
        g_fn += len(gold - kept)

        chars_in += sum(len(c["text"]) for c in cands)
        chars_out += sum(len(v.text) for v in verdicts if v.keep)

        hit = "ok " if gold <= kept else "MISS"
        if gold > kept:
            lost = next((v for v in verdicts if v.id in gold), None)
            misses.append(
                f"  {case['query'][:52]:<52} wanted {sorted(gold)[0]} "
                f"({lost.reason if lost else 'not retrieved'})"
            )
        print(f"[{hit}] {case['query'][:56]:<56} kept {sorted(kept) or '-'}")

    took = time.perf_counter() - t0
    n_chunks = len(cases) * args.candidates

    print(f"\n{len(cases)} queries, {args.candidates} candidates each, "
          f"threshold {args.threshold}")
    print(f"{took:.0f}s total, {took / n_chunks * 1000:.0f}ms per chunk decision\n")

    rp, rr, rf = prf(r_tp, r_fp, r_fn)
    gp, gr, gf = prf(g_tp, g_fp, g_fn)
    print(f"{'':<12}{'precision':>10}{'recall':>9}{'F1':>8}")
    print(f"{'retriever':<12}{rp:>10.2f}{rr:>9.2f}{rf:>8.2f}")
    print(f"{'+ gate':<12}{gp:>10.2f}{gr:>9.2f}{gf:>8.2f}")
    print(f"\ncontext shrunk {chars_in} -> {chars_out} chars "
          f"({100 * (chars_in - chars_out) / chars_in:.0f}% less)")
    print(f"gate abstained on {abstained}/{n_chunks} chunk decisions")

    args.scores.write_text(json.dumps(cache, indent=1))
    print(f"\nraw scores cached to {args.scores} -- sweep with: python sweep.py")

    if misses:
        print(f"\nrecall lost on {len(misses)}:")
        print("\n".join(misses))


if __name__ == "__main__":
    main()
