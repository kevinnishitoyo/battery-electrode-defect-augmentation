import sys
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from battery_defects.experiment import (
    LABEL_COLUMNS,
    build_training_frame,
    calculate_metrics,
    create_loaders,
    load_splits,
    select_evaluation_loader,
    select_thresholds,
    validate_experiment_data,
)


@pytest.mark.data
def test_complete_dataset_and_split_isolation():
    summary = validate_experiment_data(PROJECT_ROOT)
    assert summary == {
        "train_images": 1503,
        "validation_images": 302,
        "test_images": 300,
        "source_frames": 359,
    }


def test_source_frame_groups_are_disjoint():
    splits = load_splits(PROJECT_ROOT)
    groups = {
        name: set(frame["frame_group"])
        for name, frame in splits.items()
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])

@pytest.mark.data
def test_synthetic_images_are_added_only_to_training():
    splits = load_splits(PROJECT_ROOT)
    augmented = build_training_frame("vae_augmented", splits, PROJECT_ROOT)
    assert len(augmented) > len(splits["train"])
    assert augmented["is_synthetic"].astype(str).str.lower().eq("true").any()
    assert "is_synthetic" not in splits["val"].columns
    assert "is_synthetic" not in splits["test"].columns


def test_baseline_contains_no_synthetic_images():
    splits = load_splits(PROJECT_ROOT)
    baseline = build_training_frame("baseline", splits, PROJECT_ROOT)
    assert len(baseline) == len(splits["train"])
    assert not baseline["is_synthetic"].any()


def test_validation_and_test_loaders_are_sequential():
    splits = load_splits(PROJECT_ROOT)
    train_loader, val_loader, test_loader, _ = create_loaders(
        method="baseline",
        splits=splits,
        project_root=PROJECT_ROOT,
        batch_size=32,
        num_workers=0,
        seed=42,
    )
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)


def test_oversampling_uses_weighted_sampler():
    splits = load_splits(PROJECT_ROOT)
    train_loader, _, _, _ = create_loaders(
        method="oversampling",
        splits=splits,
        project_root=PROJECT_ROOT,
        batch_size=32,
        num_workers=0,
        seed=42,
    )
    assert isinstance(train_loader.sampler, WeightedRandomSampler)


def test_thresholds_are_selected_from_supplied_validation_grid():
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]]
    )
    thresholds = select_thresholds(labels, probabilities, [0.3, 0.5, 0.8])
    np.testing.assert_allclose(thresholds, [0.5, 0.5])


def test_metrics_are_perfect_for_perfect_predictions():
    labels = np.array([[1, 0, 1], [0, 1, 0]])
    probabilities = labels.astype(float)
    metrics = calculate_metrics(labels, probabilities, np.array([0.5] * 3))
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["hamming_loss"] == 0.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    for label in LABEL_COLUMNS:
        key = label.lower() + "_f1"
        assert metrics[key] == 1.0


def test_smoke_mode_selects_validation_instead_of_test():
    validation_loader = object()
    test_loader = object()
    selected, split = select_evaluation_loader(
        False, validation_loader, test_loader
    )
    assert selected is validation_loader
    assert split == "validation"
    selected, split = select_evaluation_loader(
        True, validation_loader, test_loader
    )
    assert selected is test_loader
    assert split == "test"
