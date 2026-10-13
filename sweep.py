"""Find the gate's operating point without re-running the model.

eval.py caches the raw axes for every (query, chunk) pair, so the threshold and
abstention floor can be swept offline in milliseconds. The default settings were
guesses; this is how you replace them with a choice.

    python eval.py          # once, to build the cache
    python sweep.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(cases: list[dict], threshold: float, min_conf: float) -> dict:
    tp = fp = fn = 0
    kept_chars = total_chars = 0
    abstained = 0

    for case in cases:
        gold = set(case["gold"])
        kept = set()
        for c in case["chunks"]:
            rel = c["axes"]["answers_it"]
            conf = abs(rel - 0.5) * 2
            total_chars += c["chars"]
            if conf < min_conf:
                abstained += 1
                continue
            if rel < threshold:
                continue
            if c["axes"].get("same_topic", 1.0) < 0.5:
                continue
            kept.add(c["id"])
            kept_chars += c["chars"]
        tp += len(kept & gold)
        fp += len(kept - gold)
        fn += len(gold - kept)

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "threshold": threshold, "min_conf": min_conf,
        "precision": p, "recall": r, "f1": f,
        "kept_pct": 100 * kept_chars / total_chars if total_chars else 0.0,
        "abstained": abstained,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("data/eval_scores.json"))
    args = ap.parse_args()
    cases = json.loads(args.scores.read_text())

    grid = [
        (t / 100, m / 100)
        for t in (40, 45, 50, 55, 60, 65)
        for m in (0, 10, 20, 35)
    ]
    rows = [evaluate(cases, t, m) for t, m in grid]
    rows.sort(key=lambda r: (-r["f1"], -r["precision"]))

    print(f"{'thresh':>7}{'minconf':>9}{'prec':>7}{'recall':>8}{'F1':>7}"
          f"{'context':>9}{'abstain':>9}")
    for r in rows:
        print(f"{r['threshold']:>7.2f}{r['min_conf']:>9.2f}{r['precision']:>7.2f}"
              f"{r['recall']:>8.2f}{r['f1']:>7.2f}{r['kept_pct']:>8.0f}%"
              f"{r['abstained']:>9d}")

    best = rows[0]
    print(f"\nbest F1: threshold {best['threshold']:.2f}, "
          f"min_confidence {best['min_conf']:.2f} -> "
          f"P {best['precision']:.2f} / R {best['recall']:.2f} / F1 {best['f1']:.2f}, "
          f"keeping {best['kept_pct']:.0f}% of context")
if __name__ == "__main__":
    main()
