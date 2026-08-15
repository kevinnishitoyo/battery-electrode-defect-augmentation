#!/usr/bin/env python3
"""Run one or more reproducible multilabel classifier experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from battery_defects import METHODS, run_experiment, validate_experiment_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="baseline")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override configured epochs")
    parser.add_argument(
        "--run-name",
        help="Optional result namespace, for example 'smoke' for test runs",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing run")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Evaluate validation only; use this for smoke tests",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files and split isolation without training",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = validate_experiment_data(PROJECT_ROOT)
    print("Data validation passed:", summary)
    if args.validate_only:
        return 0
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        config["epochs"] = args.epochs
    methods = METHODS if args.method == "all" else (args.method,)
    if args.all_seeds:
        seeds = [int(seed) for seed in config["seeds"]]
    else:
        seeds = [args.seed if args.seed is not None else int(config["seeds"][0])]
    for method in methods:
        for seed in seeds:
            run_experiment(
                method=method,
                seed=seed,
                config=config,
                project_root=PROJECT_ROOT,
                run_name=args.run_name,
                force=args.force,
                evaluate_test=not args.skip_test,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
