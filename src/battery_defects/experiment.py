"""Shared data, training, threshold-selection, and evaluation pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

LABEL_COLUMNS = ["Surface_Crack", "Delamination", "Pinhole"]
CLASS_NAMES = ["Surface Crack", "Delamination", "Pinhole"]
METHODS = (
    "baseline",
    "weighted",
    "oversampling",
    "vae_augmented",
    "gan_augmented",
    "vae_oversampling",
)
SYNTHETIC_METADATA = {
    "vae_augmented": "data/synthetic/multilabel_vae/metadata.csv",
    "gan_augmented": "data/synthetic/multilabel_gan/metadata.csv",
    "vae_oversampling": "data/synthetic/multilabel_vae/metadata.csv",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classifier_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


class MultilabelDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        project_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.project_root = project_root
        self.real_image_dir = (
            project_root / "data" / "raw" / "archive" / "classification" / "images"
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        if _as_bool(row.get("is_synthetic", False)):
            image_path = self.project_root / str(row["image_path"])
        else:
            image_path = self.real_image_dir / str(row["file_name"])
        image = self.transform(Image.open(image_path).convert("RGB"))
        labels = torch.tensor(
            row[LABEL_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        return image, labels


def load_splits(project_root: Path) -> dict[str, pd.DataFrame]:
    split_dir = project_root / "data" / "processed" / "multilabel"
    return {
        name: pd.read_csv(split_dir / f"{name}.csv")
        for name in ("train", "val", "test")
    }


def validate_experiment_data(project_root: Path) -> dict[str, int]:
    splits = load_splits(project_root)
    required = {"file_name", "frame_group", *LABEL_COLUMNS}
    for name, frame in splits.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
        if frame[LABEL_COLUMNS].isna().any().any():
            raise ValueError(f"{name}.csv contains missing labels")

    groups = {name: set(frame["frame_group"]) for name, frame in splits.items()}
    if not groups["train"].isdisjoint(groups["val"]):
        raise ValueError("Source-frame leakage between train and validation")
    if not groups["train"].isdisjoint(groups["test"]):
        raise ValueError("Source-frame leakage between train and test")
    if not groups["val"].isdisjoint(groups["test"]):
        raise ValueError("Source-frame leakage between validation and test")

    real_image_dir = (
        project_root / "data" / "raw" / "archive" / "classification" / "images"
    )
    missing_images = [
        name
        for frame in splits.values()
        for name in frame["file_name"]
        if not (real_image_dir / str(name)).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} real split images are missing; first: {missing_images[0]}"
        )

    for relative_path in SYNTHETIC_METADATA.values():
        metadata_path = project_root / relative_path
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing synthetic metadata: {relative_path}")
        metadata = pd.read_csv(metadata_path)
        missing = {"image_path", "is_synthetic", *LABEL_COLUMNS}.difference(
            metadata.columns
        )
        if missing:
            raise ValueError(f"{relative_path} is missing columns: {sorted(missing)}")
        missing_synthetic = [
            path
            for path in metadata["image_path"]
            if not (project_root / str(path)).is_file()
        ]
        if missing_synthetic:
            raise FileNotFoundError(
                f"{len(missing_synthetic)} synthetic images are missing from "
                f"{relative_path}; first: {missing_synthetic[0]}"
            )

    return {
        "train_images": len(splits["train"]),
        "validation_images": len(splits["val"]),
        "test_images": len(splits["test"]),
        "source_frames": sum(len(value) for value in groups.values()),
    }


def build_training_frame(
    method: str, splits: dict[str, pd.DataFrame], project_root: Path
) -> pd.DataFrame:
    train = splits["train"].copy()
    train["is_synthetic"] = False
    train["image_path"] = ""
    metadata_path = SYNTHETIC_METADATA.get(method)
    if metadata_path is not None:
        synthetic = pd.read_csv(project_root / metadata_path)
        train = pd.concat([train, synthetic], ignore_index=True, sort=False)
    return train


def create_loaders(
    method: str,
    splits: dict[str, pd.DataFrame],
    project_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    transform = classifier_transform()
    train = build_training_frame(method, splits, project_root)
    train_dataset = MultilabelDataset(train, project_root, transform)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(seed)

    if method in {"oversampling", "vae_oversampling"}:
        frequencies = train[LABEL_COLUMNS].mean().clip(lower=1e-12)
        weights = train[LABEL_COLUMNS].mul(1.0 / frequencies).max(axis=1)
        sampler = WeightedRandomSampler(
            torch.tensor(weights.to_numpy(), dtype=torch.double),
            num_samples=len(train),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **common)
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **common
        )

    val_loader = DataLoader(
        MultilabelDataset(splits["val"], project_root, transform), **common
    )
    test_loader = DataLoader(
        MultilabelDataset(splits["test"], project_root, transform), **common
    )
    return train_loader, val_loader, test_loader, train


def create_model(pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_COLUMNS))
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_images = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)
    return total_loss / total_images


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_list: list[torch.Tensor] = []
    probabilities_list: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            labels_list.append(labels.int())
            probabilities_list.append(torch.sigmoid(logits).cpu())
    return (
        torch.cat(labels_list).numpy(),
        torch.cat(probabilities_list).numpy(),
    )


def select_thresholds(
    labels: np.ndarray, probabilities: np.ndarray, grid: list[float]
) -> np.ndarray:
    thresholds = []
    for column in range(labels.shape[1]):
        candidates = []
        for threshold in grid:
            predictions = (probabilities[:, column] >= threshold).astype(int)
            value = f1_score(labels[:, column], predictions, zero_division=0)
            candidates.append((value, -abs(threshold - 0.5), threshold))
        thresholds.append(max(candidates)[2])
    return np.asarray(thresholds)


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict[str, float]:
    predictions = (probabilities >= thresholds).astype(int)
    metrics = {
        "exact_match_accuracy": accuracy_score(labels, predictions),
        "hamming_loss": hamming_loss(labels, predictions),
        "micro_f1": f1_score(labels, predictions, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }
    for index, name in enumerate(CLASS_NAMES):
        key = name.lower().replace(" ", "_") + "_f1"
        metrics[key] = f1_score(
            labels[:, index], predictions[:, index], zero_division=0
        )
    return metrics


def select_evaluation_loader(
    evaluate_test: bool, val_loader: DataLoader, test_loader: DataLoader
) -> tuple[DataLoader, str]:
    """Choose the permitted final loader and label its evaluation split."""
    if evaluate_test:
        return test_loader, "test"
    return val_loader, "validation"


def _criterion(method: str, train: pd.DataFrame, device: torch.device) -> nn.Module:
    if method != "weighted":
        return nn.BCEWithLogitsLoss()
    positive = train[LABEL_COLUMNS].sum().clip(lower=1)
    negative = len(train) - positive
    pos_weight = torch.tensor(
        (negative / positive).to_numpy(), dtype=torch.float32, device=device
    )
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def run_experiment(
    method: str,
    seed: int,
    config: dict[str, Any],
    project_root: Path,
    run_name: str | None = None,
    force: bool = False,
    evaluate_test: bool = True,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    set_seed(seed)
    device = get_device()
    splits = load_splits(project_root)
    output_root = project_root / "results"
    model_root = project_root / "models" / "multiseed"
    if run_name:
        output_root = output_root / run_name
        model_root = model_root / run_name
    output_dir = output_root / method / f"seed_{seed}"
    metrics_filename = "metrics.csv" if evaluate_test else "validation_metrics.csv"
    metrics_path = output_dir / metrics_filename
    if metrics_path.exists() and not force:
        raise FileExistsError(
            f"Run already exists: {metrics_path}. Use --force to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_root / method / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pth"

    train_loader, val_loader, test_loader, train = create_loaders(
        method=method,
        splits=splits,
        project_root=project_root,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        seed=seed,
    )
    model = create_model(bool(config["pretrained"])).to(device)
    criterion = _criterion(method, train, device)
    learning_rate = float(config["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    grid = [float(value) for value in config["threshold_grid"]]
    history = []
    best_score = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        val_labels, val_probabilities = predict(model, val_loader, device)
        thresholds = select_thresholds(val_labels, val_probabilities, grid)
        val_metrics = calculate_metrics(val_labels, val_probabilities, thresholds)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"{method} seed={seed} epoch={epoch}/{config['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_score + float(config["min_delta"]):
            best_score = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_macro_f1": best_score,
                    "thresholds": thresholds.tolist(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["patience"]):
                print(f"Early stopping after epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    thresholds = np.asarray(checkpoint["thresholds"], dtype=float)
    evaluation_loader, evaluation_split = select_evaluation_loader(
        evaluate_test, val_loader, test_loader
    )
    evaluation_labels, evaluation_probabilities = predict(
        model, evaluation_loader, device
    )
    metrics = calculate_metrics(
        evaluation_labels, evaluation_probabilities, thresholds
    )
    metrics.update(
        {
            "method": method,
            "seed": seed,
            "device": str(device),
            "best_epoch": int(checkpoint["epoch"]),
            "validation_macro_f1": float(checkpoint["validation_macro_f1"]),
            "training_images": len(train),
            "evaluation_split": evaluation_split,
        }
    )

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    (output_dir / "thresholds.json").write_text(
        json.dumps(dict(zip(LABEL_COLUMNS, thresholds.tolist())), indent=2) + "\n",
        encoding="utf-8",
    )
    run_config = dict(config)
    run_config.update(
        {
            "method": method,
            "seed": seed,
            "device": str(device),
            "learning_rate": learning_rate,
            "run_name": run_name,
            "evaluate_test": evaluate_test,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved metrics to {metrics_path}")
    return metrics_path
