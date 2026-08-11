# Battery Electrode Defect Augmentation

Synthetic-data augmentation for multilabel lithium-ion battery electrode coating defect classification using a ResNet-18 classifier, conditional VAE, and conditional GAN.

> This README is generated from the frozen split manifests and metric CSVs. Do not edit its result tables manually; run `python notebooks/multilabel/generate_readme.py` instead.

## Research question

Do synthetic minority-defect images improve classification beyond strong non-generative controls such as class weighting and random oversampling?

The three independent targets are Surface Crack, Delamination, and Pinhole. Images can contain more than one recognized defect. Unclassified images and rows without a recognized defect are excluded.

## Experimental protocol

- Source-frame groups are disjoint across train, validation, and test splits.
- Validation data selects checkpoints; test images remain real and are used only for final metrics.
- Synthetic images are added only to the training set.
- All classifier arms use ResNet-18, seed 42, five preliminary epochs, Adam, and a 0.5 decision threshold.
- Primary metrics are macro F1, micro F1, per-class F1, exact-match accuracy, and Hamming loss.

The current frozen dataset contains 2105 usable images from 359 source frames.

| Split | Images | Source frames | Surface Crack | Delamination | Pinhole | Multilabel images |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1503 | 255 | 1365 | 146 | 358 | 345 |
| Val | 302 | 53 | 274 | 31 | 72 | 70 |
| Test | 300 | 51 | 276 | 26 | 73 | 70 |

## Experimental arms

| Method | Change from baseline |
|---|---|
| Baseline | Pretrained ResNet-18 with unweighted BCE loss |
| Weighted BCE | Up-weights minority positive labels in the loss |
| Oversampling | Samples training images using inverse label frequency |
| VAE Augmentation | Adds conditional-VAE minority samples to training |
| GAN Augmentation | Adds conditional-GAN minority samples to training |
| VAE + Oversampling | Adds VAE samples and applies weighted random sampling |

## Preliminary results

All values below come from one five-epoch run and should not be treated as confidence-tested final results.

| Method | Exact match | Micro F1 | Macro F1 | Surface Crack F1 | Delamination F1 | Pinhole F1 |
|---|---:|---:|---:|---:|---:|---:|
| VAE + Oversampling | 0.9333 | 0.9667 | 0.9366 | 0.9819 | 0.8936 | 0.9342 |
| Oversampling | 0.9233 | 0.9663 | 0.9362 | 0.9816 | 0.8936 | 0.9333 |
| Baseline | 0.9300 | 0.9664 | 0.9292 | 0.9856 | 0.8800 | 0.9220 |
| Weighted BCE | 0.8800 | 0.9458 | 0.9129 | 0.9727 | 0.8980 | 0.8679 |
| GAN Augmentation | 0.9267 | 0.9647 | 0.8847 | 0.9909 | 0.7317 | 0.9315 |
| VAE Augmentation | 0.9033 | 0.9545 | 0.8816 | 0.9760 | 0.7317 | 0.9371 |

The current highest macro F1 is 0.9366 from VAE + Oversampling. The highest exact-match accuracy is 0.9333 from VAE + Oversampling. VAE + Oversampling differs from ordinary oversampling by only +0.0004 macro F1. This is too small to interpret from one seed; the two methods should currently be treated as tied.

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
│   └── summarize_results.py              # mean and standard-deviation tables
├── src/battery_defects/                  # shared experiment pipeline
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

## Limitations and next steps

- Repeat every classifier arm across at least five seeds and report mean and standard deviation.
- Use identical early-stopping rules and a larger training budget for every arm.
- Tune per-class thresholds on validation data, then freeze them before final test evaluation.
- Compare several synthetic-data quantities against a sample-budget-matched oversampling control.
- Inspect real/generated grids, diversity, and nearest neighbours before claiming synthetic quality.
- Add automated tests for split leakage, dataset routing, threshold selection, and metric aggregation.
