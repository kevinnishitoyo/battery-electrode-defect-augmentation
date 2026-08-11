# Multilabel Coating Defect Classification

This folder contains the recommended multilabel version of the project. The
existing notebooks in the parent folder remain as the original single-label
experiment.

The targets are three independent binary labels:

- Surface Crack
- Delamination
- Pinhole

Rows marked `unclassified` and rows without a recognized defect are excluded.
Images may retain one, two, or all three recognized defects.

## Notebook order

1. `01_multilabel_exploration.ipynb`
2. `02_multilabel_preparation.ipynb`
3. `03_multilabel_baseline.ipynb`
4. `04_multilabel_weighted.ipynb`
5. `05_multilabel_oversampling.ipynb`
6. `06_conditional_vae.ipynb`
7. `07_vae_generation.ipynb`
8. `08_vae_augmented_classifier.ipynb`
9. `09_conditional_gan.ipynb`
10. `10_gan_augmented_classifier.ipynb`
11. `12_vae_oversampling.ipynb`
12. `11_multilabel_comparison.ipynb`

Run the comparison notebook last so it includes every completed experiment.

## Generate the project README

The top-level README is generated from the frozen split manifests and metric
CSVs, so its dataset and result tables stay synchronized with the experiments:

```bash
python notebooks/multilabel/generate_readme.py
```

Use `--check` in continuous integration to fail when the README is stale:

```bash
python notebooks/multilabel/generate_readme.py --check
```

## Evaluation rules

- Source-frame groups never cross train, validation, and test subsets.
- Validation selects checkpoints; the test set is used only for final metrics.
- Synthetic images are added only to training.
- Classifier experiments use the same pretrained ResNet-18, seed, optimizer,
  epoch count, and threshold.
- `BCEWithLogitsLoss` receives raw logits. Sigmoid is used only for prediction.

Primary metrics are micro F1, macro F1, per-class F1, Hamming loss, and exact
match accuracy.
