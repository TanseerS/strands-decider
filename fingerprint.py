"""Turn documents into interpretable embeddings.

Reads JSONL ({"id", "text", ...}), scores every document against the rubric in a
single pass per document, and writes one CSV of values and one of confidences.

Usage:
    python fingerprint.py [--corpus data/tickets.jsonl] [--out data]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from rubric import AXES, RUBRIC, normalise
from strands_decider.infer import load_engine

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_corpus(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/tickets.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    docs = read_corpus(args.corpus)
    device = args.device or auto_device()

    t0 = time.perf_counter()
    engine = load_engine(CHECKPOINT, device=device)
    print(f"loaded on {device} in {time.perf_counter() - t0:.1f}s")
    print(f"{len(docs)} documents x {len(AXES)} axes\n")

    values: list[dict] = []
    confs: list[dict] = []
    total = 0.0

    for i, doc in enumerate(docs, 1):
        t0 = time.perf_counter()
        answers = engine.ask(doc["text"], RUBRIC).answers
        ms = (time.perf_counter() - t0) * 1000
        total += ms

        row = {"id": doc["id"], "period": doc.get("period", "")}
        crow = dict(row)
        for axis in AXES:
            a = answers[axis]
            row[axis] = round(normalise(axis, a), 4)
            # noul answers carry no separate confidence; distance from 0.5 is the
            # closest honest equivalent, rescaled to [0, 1].
            crow[axis] = round(
                abs(float(a.noul) - 0.5) * 2 if a.type == "noul" else a.confidence, 4
            )
        values.append(row)
        confs.append(crow)

        top = sorted(
            ((axis, row[axis]) for axis in AXES), key=lambda kv: -kv[1]
        )[:3]
        summary = " ".join(f"{k}={v:.2f}" for k, v in top)
        print(f"[{i:2d}/{len(docs)}] {doc['id']}  {ms:5.0f}ms  {summary}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fingerprints", values), ("confidence", confs)):
        path = args.out / f"{name}.csv"
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "period", *AXES])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {path}")

    print(f"\n{total / len(docs):.0f} ms/document, {total / 1000:.1f}s total")


if __name__ == "__main__":
    main()
