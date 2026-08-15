#!/usr/bin/env python3
"""Create deterministic real and synthetic defect-example grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "archive" / "classification" / "images"
)
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "multilabel" / "train.csv"
VAE_FILE = PROJECT_ROOT / "data" / "synthetic" / "multilabel_vae" / "metadata.csv"
GAN_FILE = PROJECT_ROOT / "data" / "synthetic" / "multilabel_gan" / "metadata.csv"
LABEL_COLUMNS = ["Surface_Crack", "Delamination", "Pinhole"]
REAL_CONDITIONS = [
    ("Surface Crack", (1, 0, 0)),
    ("Delamination", (0, 1, 0)),
    ("Pinhole", (0, 0, 1)),
    ("Crack + Pinhole", (1, 0, 1)),
    ("Crack + Delamination", (1, 1, 0)),
    ("All three", (1, 1, 1)),
]
GENERATOR_CONDITIONS = REAL_CONDITIONS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_rows(
    dataframe: pd.DataFrame,
    condition: tuple[int, int, int],
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    mask = np.ones(len(dataframe), dtype=bool)
    for column, expected in zip(LABEL_COLUMNS, condition):
        mask &= dataframe[column].to_numpy() == expected
    candidates = dataframe.loc[mask]
    if len(candidates) < count:
        raise ValueError(
            f"Condition {condition} has {len(candidates)} rows; {count} required"
        )
    indices = rng.choice(candidates.index.to_numpy(), size=count, replace=False)
    return dataframe.loc[indices]


def image_path(row: pd.Series, synthetic: bool) -> Path:
    if synthetic:
        return PROJECT_ROOT / str(row["image_path"])
    return REAL_IMAGE_DIR / str(row["file_name"])


def show_image(axis: plt.Axes, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        axis.imshow(image.convert("RGB"))
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_real_examples(
    train: pd.DataFrame, output_dir: Path, rng: np.random.Generator
) -> Path:
    examples_per_condition = 4
    fig, axes = plt.subplots(
        len(REAL_CONDITIONS),
        examples_per_condition,
        figsize=(10, 13),
        squeeze=False,
    )
    for row_index, (label, condition) in enumerate(REAL_CONDITIONS):
        selected = select_rows(train, condition, examples_per_condition, rng)
        for column_index, (_, sample) in enumerate(selected.iterrows()):
            show_image(axes[row_index, column_index], image_path(sample, False))
        axes[row_index, 0].set_ylabel(
            label, rotation=0, ha="right", va="center", labelpad=18, fontsize=10
        )
    fig.suptitle("Real battery-electrode defect examples", fontsize=16)
    fig.tight_layout(rect=(0.12, 0, 1, 0.97), h_pad=0.5, w_pad=0.3)
    output = output_dir / "real_defect_examples.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_generator_comparison(
    train: pd.DataFrame,
    vae: pd.DataFrame,
    gan: pd.DataFrame,
    output_dir: Path,
    rng: np.random.Generator,
) -> Path:
    sources = (("Real", train, False), ("VAE", vae, True), ("GAN", gan, True))
    examples_per_source = 3
    total_columns = len(sources) * examples_per_source
    fig, axes = plt.subplots(
        len(GENERATOR_CONDITIONS),
        total_columns,
        figsize=(16, 10),
        squeeze=False,
    )
    for row_index, (label, condition) in enumerate(GENERATOR_CONDITIONS):
        for source_index, (source_name, dataframe, synthetic) in enumerate(sources):
            selected = select_rows(dataframe, condition, examples_per_source, rng)
            for example_index, (_, sample) in enumerate(selected.iterrows()):
                column = source_index * examples_per_source + example_index
                show_image(axes[row_index, column], image_path(sample, synthetic))
                if row_index == 0 and example_index == 1:
                    axes[row_index, column].set_title(source_name, fontsize=13, pad=10)
        axes[row_index, 0].set_ylabel(
            label, rotation=0, ha="right", va="center", labelpad=18, fontsize=10
        )
        for boundary in (3, 6):
            axes[row_index, boundary].spines["left"].set_visible(True)
            axes[row_index, boundary].spines["left"].set_color("#888888")
            axes[row_index, boundary].spines["left"].set_linewidth(1.5)
    fig.suptitle("Real versus conditional VAE and GAN samples", fontsize=17)
    fig.tight_layout(rect=(0.10, 0, 1, 0.96), h_pad=0.5, w_pad=0.25)
    output = output_dir / "vae_gan_comparison.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_FILE)
    vae = pd.read_csv(VAE_FILE)
    gan = pd.read_csv(GAN_FILE)
    rng = np.random.default_rng(args.seed)
    real_output = plot_real_examples(train, output_dir, rng)
    generator_output = plot_generator_comparison(
        train, vae, gan, output_dir, rng
    )
    print(f"Saved {real_output}")
    print(f"Saved {generator_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
