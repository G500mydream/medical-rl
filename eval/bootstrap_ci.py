"""Paired bootstrap CI for two scored JSONL files.

Usage:
    python eval/bootstrap_ci.py baseline_scored.jsonl candidate_scored.jsonl

Both files need `id` and `score`. Reports candidate - baseline.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def load_scores(path: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            scores[str(row.get("id", i))] = float(row["score"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = load_scores(args.baseline)
    cand = load_scores(args.candidate)
    common = sorted(set(base) & set(cand))
    if len(common) < 10:
        raise SystemExit(f"only {len(common)} shared ids; need >= 10")

    a = np.array([base[k] for k in common], dtype=np.float64)
    b = np.array([cand[k] for k in common], dtype=np.float64)
    diff = b - a

    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.n_boot, dtype=np.float64)
    n = len(common)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()

    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two_sided = 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))

    report = {
        "n": n,
        "baseline_mean": float(a.mean()),
        "candidate_mean": float(b.mean()),
        "diff": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": min(1.0, p_two_sided),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
