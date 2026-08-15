#!/usr/bin/env python3
"""Run paired five-seed comparisons and create final result figures."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
EXPECTED_SEEDS = {42, 123, 456, 789, 2026}
METHOD_NAMES = {
    "baseline": "Baseline",
    "weighted": "Weighted BCE",
    "oversampling": "Oversampling",
    "vae_augmented": "VAE Augmentation",
    "gan_augmented": "GAN Augmentation",
    "vae_oversampling": "VAE + Oversampling",
}
METHOD_ORDER = [
    "baseline",
    "weighted",
    "vae_augmented",
    "gan_augmented",
    "vae_oversampling",
    "oversampling",
]
CLASS_METRICS = [
    ("surface_crack_f1", "Surface Crack"),
    ("delamination_f1", "Delamination"),
    ("pinhole_f1", "Pinhole"),
]


def load_runs() -> pd.DataFrame:
    paths = sorted(RESULTS_DIR.glob("*/seed_*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-seed metrics found under {RESULTS_DIR}")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {
        "method",
        "seed",
        "macro_f1",
        "surface_crack_f1",
        "delamination_f1",
        "pinhole_f1",
    }
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Metrics are missing columns: {sorted(missing)}")
    if set(runs["method"]) != set(METHOD_NAMES):
        raise ValueError(
            "Expected methods "
            f"{sorted(METHOD_NAMES)}, found {sorted(set(runs['method']))}"
        )
    for method, method_runs in runs.groupby("method"):
        seeds = set(method_runs["seed"])
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"{method} has seeds {sorted(seeds)}, expected {sorted(EXPECTED_SEEDS)}"
            )
        if method_runs["seed"].duplicated().any():
            raise ValueError(f"{method} contains duplicate seed results")
    return runs


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided paired randomization test over all 2^n sign flips."""
    observed = abs(differences.mean())
    permuted = [
        abs((differences * np.asarray(signs)).mean())
        for signs in product((-1, 1), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def bootstrap_mean_interval(
    differences: np.ndarray, samples: int = 20_000
) -> tuple[float, float]:
    """Seeded percentile bootstrap interval for the paired mean difference."""
    rng = np.random.default_rng(42)
    indices = rng.integers(
        0, len(differences), size=(samples, len(differences))
    )
    means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm family-wise correction while preserving the original row order."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running_max = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted.tolist()


def paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    pivot = runs.pivot(index="seed", columns="method", values="macro_f1")
    rows = []
    for method in METHOD_ORDER:
        if method == "oversampling":
            continue
        differences = (pivot["oversampling"] - pivot[method]).to_numpy()
        lower, upper = bootstrap_mean_interval(differences)
        rows.append(
            {
                "comparison": f"Oversampling - {METHOD_NAMES[method]}",
                "mean_macro_f1_difference": differences.mean(),
                "difference_std": differences.std(ddof=1),
                "bootstrap_ci_95_lower": lower,
                "bootstrap_ci_95_upper": upper,
                "oversampling_wins": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
                "other_method_wins": int((differences < 0).sum()),
                "exact_permutation_pvalue": exact_sign_flip_pvalue(differences),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_adjusted_pvalue"] = holm_adjust(
        result["exact_permutation_pvalue"].tolist()
    )
    return result.sort_values("mean_macro_f1_difference", ascending=False)


def plot_macro_summary(runs: pd.DataFrame) -> None:
    summary = runs.groupby("method")["macro_f1"].agg(["mean", "std"])
    summary = summary.sort_values("mean", ascending=False)
    labels = [METHOD_NAMES[method] for method in summary.index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, summary["mean"], yerr=summary["std"], capsize=5)
    ax.set_ylabel("Macro F1 (mean ± SD)")
    ax.set_ylim(0.88, 0.98)
    ax.set_title("Five-seed multilabel classifier comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "macro_f1_comparison.png", dpi=200)
    plt.close(fig)


def plot_per_class_summary(runs: pd.DataFrame) -> None:
    means = runs.groupby("method")[[metric for metric, _ in CLASS_METRICS]].mean()
    ordering = (
        runs.groupby("method")["macro_f1"].mean().sort_values(ascending=False).index
    )
    means = means.loc[ordering]
    x = np.arange(len(means))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, (metric, label) in enumerate(CLASS_METRICS):
        ax.bar(x + (index - 1) * width, means[metric], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_NAMES[method] for method in means.index], rotation=25
    )
    ax.set_ylabel("Mean F1")
    ax.set_ylim(0.84, 1.0)
    ax.set_title("Per-class F1 across five seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "per_class_f1_comparison.png", dpi=200)
    plt.close(fig)


def plot_seed_results(runs: pd.DataFrame) -> None:
    pivot = runs.pivot(index="seed", columns="method", values="macro_f1")
    x = np.arange(len(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for seed, row in pivot.iterrows():
        ax.plot(
            x,
            row[METHOD_ORDER],
            marker="o",
            linewidth=1.3,
            alpha=0.8,
            label=f"Seed {seed}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_NAMES[method] for method in METHOD_ORDER], rotation=25
    )
    ax.set_ylabel("Macro F1")
    ax.set_title("Paired macro F1 by training seed")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "macro_f1_by_seed.png", dpi=200)
    plt.close(fig)


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    comparisons = paired_comparisons(runs)
    comparisons.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)
    plot_macro_summary(runs)
    plot_per_class_summary(runs)
    plot_seed_results(runs)
    print(comparisons.round(4).to_string(index=False))
    print(f"Saved statistics to {RESULTS_DIR / 'paired_comparisons.csv'}")
    print(f"Saved figures to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
