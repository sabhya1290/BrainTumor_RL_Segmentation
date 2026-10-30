# Reinforcement Learning–Driven Refinement of Deep Semantic Segmentation for Brain Tumor Delineation in MRI

## Project Overview

This project investigates the use of Reinforcement Learning (RL) to refine deep-learning-based semantic segmentation of brain tumors in MRI scans.

The initial segmentation model is a U-Net trained on the BraTS2020 dataset. The long-term objective is to investigate whether an RL-based refinement strategy can improve segmentation quality.

## Current Progress

- [x] Dataset inspection and preprocessing
- [x] Volume-grouped training, validation, and test splits
- [x] U-Net architecture implementation
- [x] Quick-subset baseline training
- [x] Full-dataset baseline training
- [x] Initial validation evaluation
- [ ] Best-checkpoint evaluation
- [ ] Reinforcement Learning agent implementation
- [ ] RL-based segmentation refinement
- [ ] Final test-set evaluation

## Dataset

Dataset: BraTS2020 Training Data

Kaggle source: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

The MRI dataset is not included in this repository because of its size.

Download the dataset separately and place the extracted files under the expected `data/raw/` directory.

## Project Structure

```text
BrainTumor_RL_Segmentation/
├── data/
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── models/
├── results/
│   └── baseline_metrics.txt
├── src/
│   ├── dataset.py
│   ├── unet.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   └── ...
├── .gitignore
├── README.md
└── requirements.txt
```

## Environment

- Python 3.13.4
- PyTorch 2.14.1
- CUDA 12.6
- NVIDIA GeForce RTX 2050 (4 GB VRAM)

## Baseline Model

Architecture: U-Net

Training dataset: 45,725 slices

Validation dataset: 5,735 slices

Test dataset: 5,735 slices

Epochs: 10

Batch size: 2

Optimizer: Adam

Learning rate: 0.001

### Initial Validation Results

| Mask Channel | Dice Score |
|---|---:|
| Channel 0 | 0.5253 |
| Channel 1 | 0.6605 |
| Channel 2 | 0.7210 |
| Macro Average | 0.6356 |

These are preliminary baseline results, not final test-set results.

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd BrainTumor_RL_Segmentation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install a PyTorch build compatible with your system's CUDA setup if needed.

## Training

After downloading and preparing the dataset:

```bash
python src/train_baseline.py
```

## Evaluation

```bash
python src/evaluate_baseline.py
```

## Important Notes

- MRI data and large model checkpoints are excluded from Git.
- Dataset split manifests are included for reproducibility.
- The current implementation is a U-Net baseline.
- Reinforcement Learning refinement is planned as a subsequent development stage.

## Disclaimer

This project is for academic and research purposes only. It is not intended for clinical diagnosis or medical decision-making.
