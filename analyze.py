"""Everything an embedding gives you, except each answer explains itself.

Clusters, neighbours, anomalies and drift are all ordinary vector operations. The
difference is that every axis has a name, so each result can be reported in terms
of the traits responsible for it rather than as a bare distance.

Usage:
    python analyze.py [--k 4] [--neighbours t16]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rubric import AXES


def load(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["id"] for r in rows]
    periods = [r["period"] for r in rows]
    X = np.array([[float(r[a]) for a in AXES] for r in rows])
    return ids, periods, X


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    """k-means++ init, fixed seed, so runs are reproducible."""
    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None] - np.array(centres)) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(X), 1 / len(X))
        centres.append(X[rng.choice(len(X), p=probs)])
    C = np.array(centres)

    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = np.argmin(((X[:, None] - C) ** 2).sum(-1), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                C[j] = X[labels == j].mean(axis=0)
    return labels


def describe(centroid: np.ndarray, baseline: np.ndarray, n: int = 3) -> str:
    """Name a group by the axes where it departs most from the corpus average."""
    delta = centroid - baseline
    order = np.argsort(-np.abs(delta))[:n]
    return ", ".join(
        f"{AXES[i]} {'+' if delta[i] > 0 else '-'}{abs(delta[i]):.2f}" for i in order
    )


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprints", type=Path, default=Path("data/fingerprints.csv"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--neighbours", default=None, help="document id to explain")
    args = ap.parse_args()

    ids, periods, X = load(args.fingerprints)
    mean = X.mean(axis=0)
    index = {d: i for i, d in enumerate(ids)}

    section(f"{args.k} clusters, each named by what makes it distinctive")
    labels = kmeans(X, args.k)
    for j in range(args.k):
        members = [ids[i] for i in range(len(ids)) if labels[i] == j]
        if not members:
            continue
        print(f"\ncluster {j}  ({len(members)}): {describe(X[labels == j].mean(axis=0), mean)}")
        print(f"  {' '.join(members)}")

    section("Most unusual documents, and the axis responsible")
    dist = np.linalg.norm(X - mean, axis=1)
    for i in np.argsort(-dist)[:5]:
        axis = int(np.argmax(np.abs(X[i] - mean)))
        print(
            f"  {ids[i]}  distance {dist[i]:.2f}   driven by "
            f"{AXES[axis]}={X[i][axis]:.2f} (corpus {mean[axis]:.2f})"
        )

    if args.neighbours:
        if args.neighbours not in index:
            raise SystemExit(f"unknown id {args.neighbours!r}")
        q = index[args.neighbours]
        section(f"Nearest neighbours of {args.neighbours}, with the reason")
        d = np.linalg.norm(X - X[q], axis=1)
        for i in np.argsort(d)[1:4]:
            # Rank by shared *presence*: two documents both scoring 0.2 on an axis
            # agree, but that agreement says nothing about why they are alike.
            both = (X[i] + X[q]) / 2
            shared = np.argsort(-(both * (1 - np.abs(X[i] - X[q]))))[:3]
            agree = ", ".join(f"{AXES[a]}~{both[a]:.2f}" for a in shared)
            print(f"  {ids[i]}  distance {d[i]:.2f}   both high on {agree}")

    groups = sorted({p for p in periods if p})
    if len(groups) == 2:
        a, b = groups
        A = X[[i for i, p in enumerate(periods) if p == a]]
        B = X[[i for i, p in enumerate(periods) if p == b]]
        section(f"Drift {a} -> {b}: which trait moved, not just that something did")
        delta = B.mean(axis=0) - A.mean(axis=0)
        for i in np.argsort(-np.abs(delta)):
            if abs(delta[i]) < 0.05:
                continue
            arrow = "up" if delta[i] > 0 else "down"
            print(
                f"  {AXES[i]:<15} {A.mean(axis=0)[i]:.2f} -> {B.mean(axis=0)[i]:.2f}"
                f"   {arrow} {abs(delta[i]):.2f}"
            )


if __name__ == "__main__":
    main()
