"""Generate the project README from experiment artifacts.

Run from any directory:

    python notebooks/multilabel/generate_readme.py

Use ``--check`` in CI to verify that README.md is current without changing it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "multilabel"
DEFAULT_OUTPUT = PROJECT_ROOT / "README.md"
SUMMARY_FILE = PROJECT_ROOT / "results" / "summary.csv"
PAIRED_FILE = PROJECT_ROOT / "results" / "paired_comparisons.csv"
LABEL_COLUMNS = ("Surface_Crack", "Delamination", "Pinhole")
METHOD_NAMES = {
    "baseline": "Baseline",
    "weighted": "Weighted BCE",
    "oversampling": "Oversampling",
    "vae_augmented": "VAE Augmentation",
    "gan_augmented": "GAN Augmentation",
    "vae_oversampling": "VAE + Oversampling",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact is missing: {path.relative_to(PROJECT_ROOT)}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_summary(name: str) -> dict[str, int | str]:
    rows = read_rows(SPLIT_DIR / f"{name}.csv")
    groups = {row["frame_group"] for row in rows}
    summary: dict[str, int | str] = {
        "name": name.capitalize(),
        "images": len(rows),
        "groups": len(groups),
        "multilabel": sum(
            sum(int(row[label]) for label in LABEL_COLUMNS) > 1 for row in rows
        ),
    }
    for label in LABEL_COLUMNS:
        summary[label] = sum(int(row[label]) for row in rows)
    return summary


def score(row: dict[str, str], column: str) -> float:
    return float(row[column])


def fmt(value: str) -> str:
    return f"{float(value):.4f}"


def fmt_mean_std(row: dict[str, str], metric: str) -> str:
    return f"{fmt(row[f'{metric}_mean'])} ± {fmt(row[f'{metric}_std'])}"


def build_readme() -> str:
    splits = [split_summary(name) for name in ("train", "val", "test")]
    metrics = read_rows(SUMMARY_FILE)
    best_macro = max(metrics, key=lambda row: score(row, "macro_f1_mean"))
    best_exact = max(
        metrics, key=lambda row: score(row, "exact_match_accuracy_mean")
    )
    oversampling = next(row for row in metrics if row["method"] == "oversampling")
    vae_oversampling = next(
        row for row in metrics if row["method"] == "vae_oversampling"
    )
    synthetic_only = [
        row for row in metrics if row["method"] in {"vae_augmented", "gan_augmented"}
    ]
    best_synthetic_only = max(
        synthetic_only, key=lambda row: score(row, "macro_f1_mean")
    )
    synthetic_delta = score(
        best_synthetic_only, "macro_f1_mean"
    ) - score(oversampling, "macro_f1_mean")
    combined_delta = score(
        vae_oversampling, "macro_f1_mean"
    ) - score(oversampling, "macro_f1_mean")
    total_images = sum(int(split["images"]) for split in splits)
    total_groups = sum(int(split["groups"]) for split in splits)
    paired = read_rows(PAIRED_FILE)

    split_lines = [
        "| Split | Images | Source frames | Surface Crack | Delamination | Pinhole | Multilabel images |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in splits:
        split_lines.append(
            f"| {split['name']} | {split['images']} | {split['groups']} | "
            f"{split['Surface_Crack']} | {split['Delamination']} | "
            f"{split['Pinhole']} | {split['multilabel']} |"
        )

    result_lines = [
        "| Method | Runs | Exact match | Micro F1 | Macro F1 | Surface Crack F1 | Delamination F1 | Pinhole F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        metrics, key=lambda item: score(item, "macro_f1_mean"), reverse=True
    ):
        result_lines.append(
            f"| {METHOD_NAMES[row['method']]} | {row['runs']} | "
            f"{fmt_mean_std(row, 'exact_match_accuracy')} | "
            f"{fmt_mean_std(row, 'micro_f1')} | "
            f"{fmt_mean_std(row, 'macro_f1')} | "
            f"{fmt_mean_std(row, 'surface_crack_f1')} | "
            f"{fmt_mean_std(row, 'delamination_f1')} | "
            f"{fmt_mean_std(row, 'pinhole_f1')} |"
        )

    comparison_text = (
        f"The strongest synthetic-only arm, "
        f"{METHOD_NAMES[best_synthetic_only['method']]}, trails ordinary "
        f"oversampling by {abs(synthetic_delta):.4f} mean macro F1. Adding VAE "
        f"samples to oversampling also trails ordinary oversampling by "
        f"{abs(combined_delta):.4f}, largely because its mean Delamination F1 "
        f"falls to {fmt(vae_oversampling['delamination_f1_mean'])}."
    )

    paired_lines = [
        "| Comparison | Mean macro-F1 difference | Bootstrap 95% CI | Seed wins | Exact p | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        paired_lines.append(
            f"| {row['comparison']} | "
            f"{fmt(row['mean_macro_f1_difference'])} | "
            f"[{fmt(row['bootstrap_ci_95_lower'])}, "
            f"{fmt(row['bootstrap_ci_95_upper'])}] | "
            f"{row['oversampling_wins']}/5 | "
            f"{fmt(row['exact_permutation_pvalue'])} | "
            f"{fmt(row['holm_adjusted_pvalue'])} |"
        )

    return f"""# Battery Electrode Defect Augmentation

Synthetic-data augmentation for multilabel lithium-ion battery electrode coating defect classification using a ResNet-18 classifier, conditional VAE, and conditional GAN.

> This README is generated from the frozen split manifests and metric CSVs. Do not edit its result tables manually; run `python notebooks/multilabel/generate_readme.py` instead.

## Research question

Do synthetic minority-defect images improve classification beyond strong non-generative controls such as class weighting and random oversampling?

The three independent targets are Surface Crack, Delamination, and Pinhole. Images can contain more than one recognized defect. Unclassified images and rows without a recognized defect are excluded.

## Experimental protocol

- Source-frame groups are disjoint across train, validation, and test splits.
- Validation data selects checkpoints; test images remain real and are used only for final metrics.
- Synthetic images are added only to the training set.
- All classifier arms use ResNet-18, Adam, and the same five seeds: 42, 123, 456, 789, and 2026.
- Training runs for at most 30 epochs with patience-5 early stopping on validation macro F1.
- Per-class decision thresholds are selected on validation data and then frozen for real-only test evaluation.
- Primary metrics are macro F1, micro F1, per-class F1, exact-match accuracy, and Hamming loss.

The current frozen dataset contains {total_images} usable images from {total_groups} source frames.

{chr(10).join(split_lines)}

## Experimental arms

| Method | Change from baseline |
|---|---|
| Baseline | Pretrained ResNet-18 with unweighted BCE loss |
| Weighted BCE | Up-weights minority positive labels in the loss |
| Oversampling | Samples training images using inverse label frequency |
| VAE Augmentation | Adds conditional-VAE minority samples to training |
| GAN Augmentation | Adds conditional-GAN minority samples to training |
| VAE + Oversampling | Adds VAE samples and applies weighted random sampling |

## Five-seed results

Values are mean ± sample standard deviation across five seeds. Every method uses the same frozen source-frame splits and evaluation protocol.

{chr(10).join(result_lines)}

The highest mean macro F1 is {fmt(best_macro['macro_f1_mean'])} from {METHOD_NAMES[best_macro['method']]}. The highest mean exact-match accuracy is {fmt(best_exact['exact_match_accuracy_mean'])} from {METHOD_NAMES[best_exact['method']]}. {comparison_text}

The main finding is that ordinary random oversampling is the strongest overall method. Learned synthetic augmentation does not improve macro F1 beyond this simpler non-generative control.

## Paired analysis

Because every method uses the same five seeds, macro-F1 differences can be paired by seed. Positive differences favor oversampling.

{chr(10).join(paired_lines)}

Oversampling wins four or five of the five paired seeds against every comparator. However, with only five pairs the exact sign-flip test has coarse resolution, and no comparison remains significant after Holm correction. The result therefore supports a consistent positive effect for oversampling, but not a definitive formal significance claim.

## Figures

![Five-seed macro F1 comparison](results/figures/macro_f1_comparison.png)

![Per-class F1 comparison](results/figures/per_class_f1_comparison.png)

![Paired macro F1 by training seed](results/figures/macro_f1_by_seed.png)

### Qualitative examples

![Real battery-electrode defect examples](results/figures/real_defect_examples.png)

![Real versus conditional VAE and GAN samples](results/figures/vae_gan_comparison.png)

The learned generators capture broad electrode color and texture, but the VAE samples are visibly blurred and many GAN samples lack the distinct conditioned defect structures present in real images. This qualitative gap is consistent with synthetic augmentation failing to outperform ordinary oversampling.

## Repository layout

```text
battery-electrode-defect-augmentation/
├── configs/experiment.json               # shared seeds and hyperparameters
├── data/
│   ├── raw/archive/classification/       # local labels and real images
│   ├── processed/multilabel/             # frozen splits and metric CSVs
│   └── synthetic/                        # generated images and metadata
├── models/multilabel/                    # local checkpoints
├── results/                              # per-seed metrics and summaries
├── scripts/
│   ├── run_experiments.py                # validation and multi-seed training CLI
│   ├── summarize_results.py              # mean and standard-deviation tables
│   ├── analyze_results.py                # paired tests and result figures
│   └── plot_qualitative_examples.py       # real/VAE/GAN example grids
├── src/battery_defects/                  # shared experiment pipeline
├── tests/                                # data, training-protocol, and analysis checks
└── notebooks/multilabel/
    ├── 01_multilabel_exploration.ipynb
    ├── 02_multilabel_preparation.ipynb
    ├── 03_multilabel_baseline.ipynb
    ├── 04_multilabel_weighted.ipynb
    ├── 05_multilabel_oversampling.ipynb
    ├── 06_conditional_vae.ipynb
    ├── 07_vae_generation.ipynb
    ├── 08_vae_augmented_classifier.ipynb
    ├── 09_conditional_gan.ipynb
    ├── 10_gan_augmented_classifier.ipynb
    ├── 12_vae_oversampling.ipynb
    ├── 11_multilabel_comparison.ipynb
    ├── generative_models.py
    ├── multilabel_utils.py
    └── generate_readme.py
```

## Data setup

Place the private dataset under:

```text
data/raw/archive/classification/
├── labels.csv
└── images/
```

The preparation notebook creates group-separated manifests under `data/processed/multilabel/`.

## Run the experiments

Validate all image paths and confirm that source-frame groups are isolated:

```bash
python scripts/run_experiments.py --validate-only
```

Run a one-epoch smoke test in a separate result namespace:

```bash
python scripts/run_experiments.py --method baseline --seed 42 --epochs 1 --run-name smoke --skip-test
```

Run every method across the five configured seeds:

```bash
python scripts/run_experiments.py --method all --all-seeds
python scripts/summarize_results.py
python scripts/analyze_results.py
```

Regenerate the qualitative image grids:

```bash
python scripts/plot_qualitative_examples.py
```

The notebooks remain useful for exploration and generator training. Their execution order is documented in [`notebooks/multilabel/README.md`](notebooks/multilabel/README.md).

After metrics change, regenerate this README:

```bash
python notebooks/multilabel/generate_readme.py
```

Check that it is current without rewriting it:

```bash
python notebooks/multilabel/generate_readme.py --check
```

Run the automated tests:

```bash
python -m pytest -q
```

## Limitations and next steps

- Five seeds quantify training variability, but the results still come from one fixed dataset split.
- Compare several synthetic-data quantities against a sample-budget-matched oversampling control.
- Inspect real/generated grids, diversity, and nearest neighbours before claiming synthetic quality.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="README path (default: project-root README.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the output does not match generated content",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content = build_readme()
    output = args.output.resolve()

    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != content:
            print(f"README is stale: {output}", file=sys.stderr)
            return 1
        print(f"README is current: {output}")
        return 0

    output.write_text(content, encoding="utf-8")
    print(f"Generated {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
