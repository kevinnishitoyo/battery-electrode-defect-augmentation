#!/usr/bin/env python3
"""Aggregate per-seed experiment metrics into mean and standard deviation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "exact_match_accuracy",
    "hamming_loss",
    "micro_f1",
    "macro_f1",
    "surface_crack_f1",
    "delamination_f1",
    "pinhole_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "summary.csv"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.results_dir.glob("*/seed_*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-seed metrics found under {args.results_dir}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    missing = {"method", "seed", *METRICS}.difference(runs.columns)
    if missing:
        raise ValueError(f"Metrics files are missing columns: {sorted(missing)}")
    grouped = runs.groupby("method", sort=False)
    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "runs", grouped.size())
    summary = summary.reset_index().sort_values("macro_f1_mean", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Saved summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
