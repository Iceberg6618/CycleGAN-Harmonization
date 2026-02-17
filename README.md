# MRI Harmonization using CycleGAN

A CycleGAN-based framework for harmonizing medical images across different vendors (e.g., SIEMENS → Philips). This project enables domain translation of MRI/CT scans without paired supervision, leveraging adversarial training and cycle-consistency losses.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)

---

## Overview

Medical images acquired from different vendors exhibit domain-specific artifacts and appearance variations due to hardware, calibration, and imaging protocols. This project addresses **vendor harmonization**—translating images from one vendor's domain to another while preserving anatomical structure and clinical utility.

**Key Innovation**: Combines multiple loss terms (adversarial, cycle-consistency, identity, and original matching) in a symmetric CycleGAN architecture to ensure bidirectional and faithful harmonization.

---

## Features

- **Symmetric CycleGAN Architecture**: Two paired generators (A↔B) and discriminators for bidirectional translation
- **Multi-loss Training**: Adversarial + Cycle Consistency + Identity + Original Matching losses
- **Paired Dataset Support**: Loads pre-aligned image pairs (vendor1 vs vendor2) from pickle files
- **Evaluation Metrics**: SSIM and PSNR computed during evaluation to track harmonization quality
- **Checkpoint Management**: Periodic model saving and loss/metric visualization
- **NIfTI Inference**: Slice-wise inference on 3D medical volumes (NIfTI format)
- **Flexible Configuration**: Command-line arguments for easy hyperparameter tuning

---

## Project Structure

```
.
├── train.py                           # Training entrypoint
├── inference_nii.py                   # Inference on 3D NIfTI volumes
├── README.md                          # This file
│
├── dataset/
│   ├── __init__.py
│   ├── dataset.py                     # Paired_Dataset class for loading pickle-based image pairs
│   └── generate_paired_dataset.py     # Script to create paired dataset from raw images
│
├── model/
│   ├── __init__.py
│   ├── generator.py                   # ResNet-style generators (9 residual blocks)
│   └── discriminator.py               # PatchGAN-style discriminators
│
└── utils/
    ├── __init__.py
    ├── arguments.py                   # CLI argument parser
    ├── trainer.py                     # CycleGANTrainer: main training loop and evaluation
    └── utils.py                       # Helper functions (SSIM, PSNR, LR scheduling)
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- nibabel (for NIfTI format support)
- scikit-image (for SSIM metric)
- pandas, matplotlib (for logging and visualization)
- tqdm (for progress bars)

### Setup

```bash
# Clone the repository
git clone https://github.com/Iceberg6618/CycleGAN-Harmonization.git
cd github_upload

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision nibabel scikit-image pandas matplotlib tqdm
```

---

## Dataset Preparation

### Expected Format

The `Paired_Dataset` expects a directory structure with **pickle files** containing paired images:

```
dataset/
├── trainset/
│   ├── subject_001/
│   │   ├── slice_001.pkl
│   │   ├── slice_002.pkl
│   │   └── ...
│   ├── subject_002/
│   │   └── ...
│   └── ...
├── evalset/
│   └── (same structure as trainset)
└── testset/
    ├── SIEMENS/
    │   ├── subj_001.nii.gz
    │   └── ...
    └── Philips/
        └── ...
```

Each pickle file should contain a dictionary:

```python
{
    'SIEMENS': <numpy array (H, W, 1)>,  # Vendor A image
    'Philips': <numpy array (H, W, 1)>   # Vendor B image
}
```

## Usage

### Training

#### Basic Training

```bash
python train.py -ex MyExperiment --data_root ./dataset/Harmonization_dataset
```

This command:
1. Loads training and evaluation datasets from `./dataset/Harmonization_dataset/{trainset,evalset}`
2. Creates output directory `./results/MyExperiment/`
3. Initializes two generators and discriminators
4. Runs the training loop for `--total_epoch` epochs (default: 50)

#### Advanced Training with Custom Hyperparameters

```bash
python train.py \
    -ex HarmonizationV2 \
    --data_root ./dataset/Harmonization_dataset \
    --vendor1 SIEMENS \
    --vendor2 Philips \
    --total_epoch 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --ccl_weight 5.0 \
    --identity_weight 1.0 \
    --oml_weight 5.0 \
    --decay_epoch 10 \
    --save_freq 5 \
    --log_freq 100 \
    --sample_save_freq 40 \
    --random_seed 42
```

**Key Arguments** (see `utils/arguments.py` for full list):

| Argument | Default | Description |
|----------|---------|-------------|
| `-ex, --experiment` | *required* | Experiment name; used for output folder naming. |
| `--data_root` | `./dataset/Harmonization_dataset` | Path to dataset root. |
| `--total_epoch` | 50 | Total number of training epochs. |
| `--batch_size` | 16 | Batch size for training. |
| `--learning_rate` | 2e-4 | Initial learning rate for optimizers. |
| `--ccl_weight` | 5.0 | Weight of cycle-consistency loss. |
| `--identity_weight` | 1.0 | Weight of identity loss. |
| `--oml_weight` | 5.0 | Weight of original matching loss. |
| `--decay_epoch` | 4 | Epoch interval for LR decay. |
| `--save_freq` | 1 | Epoch frequency to save model checkpoints. |
| `--log_freq` | 200 | Iteration frequency to log training metrics. |
| `--lower_bound` | 0.1 | Lower intensity percentile for data normalization. |
| `--upper_bound` | 0.9 | Upper intensity percentile for data normalization. |

#### Output Directory

After training, the `./results/MyExperiment/` folder contains:

```
results/MyExperiment/
├── arguments.txt                # Persisted CLI arguments
├── loss/
│   ├── loss.csv               # Epoch-wise losses and metrics
│   └── loss.png               # Training/eval loss curves and SSIM/PSNR plots
├── model/
│   ├── netG_S_to_P_epoch_0.pth
│   ├── netG_P_to_S_epoch_0.pth
│   ├── netD_S_epoch_0.pth
│   ├── netD_P_epoch_0.pth
│   └── ... (checkpoints for other epochs)
└── sample/
    ├── epoch_0/
    │   ├── iter_0.png         # Sample visualization: A, A→B, A→B→A, B, B→A, B→A→B
    │   └── ...
    └── epoch_5/
        └── ...
```

---

### Inference

#### Inference on 3D NIfTI Volumes

Use `inference_nii.py` to harmonize 3D medical volumes (SIEMENS → Philips):

1. **Edit configuration** in `inference_nii.py`:

```python
target_experiment = 'MyExperiment'  # Must match a trained experiment folder
best_epochs = {
    'SIEMENS_to_Philips': 50  # Checkpoint epoch to use
}
```

2. **Uncomment and run** the main call:

```python
if __name__=='__main__':
    data_dir = './dataset/Harmonization_dataset/testset/SIEMENS'
    save_dir = './results/MyExperiment/testset/SIEMENS_to_Philips'
    main(data_dir, save_dir, src_vendor='SIEMENS', trg_vendor='Philips')
```

Or run directly in Python:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from inference_nii import main
main('./dataset/Harmonization_dataset/testset/SIEMENS', 
     './results/MyExperiment/testset/SIEMENS_to_Philips',
     src_vendor='SIEMENS', trg_vendor='Philips')
"
```

#### Inference Details

- **Input**: NIfTI files (`.nii` or `.nii.gz`) in `data_dir`
- **Processing**: Splits 3D volume into axial slices, runs generator on each slice, re-stacks to 3D
- **Output**: Harmonized NIfTI files saved to `save_dir` with naming convention `{subject_id}_harmonized.nii.gz`
- **Slice Detection**: Automatically detects axial axis from NIfTI affine (looks for S/I orientation codes)

---

