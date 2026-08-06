"""Shared, beginner-friendly utilities for the multilabel experiments."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)


LABEL_COLUMNS = ["Surface_Crack", "Delamination", "Pinhole"]
CLASS_NAMES = ["Surface Crack", "Delamination", "Pinhole"]


def set_seed(seed=42):
    """Set the random seeds used by Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    """Use Apple MPS when available, otherwise use the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def classifier_transform():
    """ImageNet preprocessing required by pretrained ResNet-18."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class MultilabelDataset(Dataset):
    """Load real or synthetic images and return a three-value label vector."""

    def __init__(self, dataframe, project_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.real_image_dir = (
            self.project_root
            / "data"
            / "raw"
            / "archive"
            / "classification"
            / "images"
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        if bool(row.get("is_synthetic", False)):
            image_path = self.project_root / row["image_path"]
        else:
            image_path = self.real_image_dir / row["file_name"]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[LABEL_COLUMNS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        return image, labels


def create_resnet18(number_of_labels=3):
    """Create a pretrained ResNet-18 with independent multilabel outputs."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, number_of_labels)
    return model


def run_classifier_epoch(model, loader, criterion, device, optimizer=None):
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    number_of_images = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        number_of_images += batch_size

    return total_loss / number_of_images


def train_classifier(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    checkpoint_path,
    epochs=5,
):
    """Train and save the checkpoint with the lowest validation loss."""
    best_validation_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = run_classifier_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss = run_classifier_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

    return history


def predict_multilabel(model, loader, device, threshold=0.5):
    """Return true labels, probabilities, and thresholded predictions."""
    model.eval()
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            batch_probabilities = torch.sigmoid(logits).cpu()
            true_labels.append(labels.int())
            probabilities.append(batch_probabilities)

    true_labels = torch.cat(true_labels).numpy()
    probabilities = torch.cat(probabilities).numpy()
    predictions = (probabilities >= threshold).astype(int)
    return true_labels, probabilities, predictions


def calculate_multilabel_metrics(true_labels, predictions, model_name):
    """Calculate metrics suitable for multilabel classification."""
    report = classification_report(
        true_labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    return {
        "Model": model_name,
        "Exact Match Accuracy": accuracy_score(true_labels, predictions),
        "Hamming Loss": hamming_loss(true_labels, predictions),
        "Micro F1": f1_score(
            true_labels, predictions, average="micro", zero_division=0
        ),
        "Macro F1": f1_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "Surface Crack F1": report["Surface Crack"]["f1-score"],
        "Delamination F1": report["Delamination"]["f1-score"],
        "Pinhole F1": report["Pinhole"]["f1-score"],
    }
