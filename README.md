# MRI Harmonization using CycleGAN

A CycleGAN-based framework for harmonizing medical images across different vendors (e.g., SIEMENS → Philips). This project enables domain translation of MRI/CT scans without paired supervision, leveraging adversarial training and cycle-consistency losses.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Loss Functions & Architecture](#loss-functions--architecture)
- [Configuration](#configuration)
- [Output & Results](#output--results)
- [Citation](#citation)

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
git clone <repository-url>
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

### Creating the Dataset

Use `dataset/generate_paired_dataset.py` to convert raw images (2D slices or 3D volumes) into paired pickle files:

```bash
python dataset/generate_paired_dataset.py \
    --vendor1_dir /path/to/SIEMENS/images \
    --vendor2_dir /path/to/Philips/images \
    --output_dir ./dataset/Harmonization_dataset \
    --split_ratio 0.8
```

(See the script for detailed argument documentation.)

---

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

## Loss Functions & Architecture

### Generators: ResNet-style Image-to-Image Translation

Each generator consists of:

1. **Initial Conv**: 7×7 convolution with reflection padding (input → 64 channels)
2. **Downsampling**: 2 stride-2 convolutions to extract features (64 → 128 → 256 channels)
3. **Residual Blocks**: 9 residual blocks (with skip connections) to refine features
4. **Upsampling**: 2 transpose convolutions to restore spatial resolution
5. **Synthesis**: 7×7 convolution with tanh activation (output in [-1, 1])

### Discriminators: PatchGAN-style

Each discriminator:

1. Stacks 4 convolutional blocks with instance normalization
2. Outputs a feature map that is globally average-pooled to produce a single scalar score
3. Trained to distinguish real images (label=1) from generated images (label=0)

### Loss Terms

During training, the generator loss is:

$$\mathcal{L}_G = \mathcal{L}_{adv} + \lambda_{ccl} \mathcal{L}_{ccl} + \lambda_{id} \mathcal{L}_{id} + \lambda_{oml} \mathcal{L}_{oml}$$

1. **Adversarial Loss** ($\mathcal{L}_{adv}$): BCEWithLogits. Encourages generators to fool discriminators.
   - $\mathcal{L}_{adv} = \text{BCE}(D_B(G_{A \to B}(A)), 1) + \text{BCE}(D_A(G_{B \to A}(B)), 1)$

2. **Cycle Consistency Loss** ($\mathcal{L}_{ccl}$): L1. Ensures $G_{B \to A}(G_{A \to B}(A)) \approx A$.
   - $\mathcal{L}_{ccl} = |G_{B \to A}(G_{A \to B}(A)) - A| + |G_{A \to B}(G_{B \to A}(B)) - B|$

3. **Identity Loss** ($\mathcal{L}_{id}$): L1. Encourages $G_{A \to B}(A) \approx A$ if input is already in domain B.
   - $\mathcal{L}_{id} = |G_{A \to B}(A) - A| + |G_{B \to A}(B) - B|$

4. **Original Matching Loss** ($\mathcal{L}_{oml}$): L1. Direct pairing supervision (exploits paired data).
   - $\mathcal{L}_{oml} = |G_{A \to B}(A) - B| + |G_{B \to A}(B) - A|$

Discriminator loss (standard adversarial):

$$\mathcal{L}_D = \text{BCE}(D(real), 1) + \text{BCE}(D(fake), 0)$$

### Weighting Strategy

The loss weights (default: `ccl=5, id=1, oml=5`) balance three competing objectives:

- **High OML weight**: Leverages paired data for fidelity
- **Moderate CCL weight**: Ensures cycle consistency as an unsupervised constraint
- **Low ID weight**: Prevents over-fitting; can be set to 0 for pure unpaired training

---

## Configuration

### Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--batch_size` | 16 | Larger batch → more stable updates but higher memory usage |
| `--learning_rate` | 2e-4 | Lower LR → slower convergence, potentially better quality |
| `--decay_epoch` | 4 | LR decay starts at this epoch; decay continues every `decay_epoch` epochs |
| `--ccl_weight` | 5.0 | Higher → stronger cycle consistency; lower → more direct pairing focus |
| `--oml_weight` | 5.0 | Higher → more faithful pairing; lower → more flexible unsupervised learning |
| `--identity_weight` | 1.0 | Can be set to 0 to disable; useful for domain-agnostic features |
| `--lower_bound`, `--upper_bound` | 0.1, 0.9 | Data normalization range; clips intensity to [0.1*max, 0.9*max] |
| `--num_workers` | 1 | Number of DataLoader workers; increase for faster data loading (if CPU permits) |

### Recommendation for Different Scenarios

- **Paired Data (both vendors available)**: Keep `oml_weight` high (5.0), use cycle loss for consistency.
- **Unpaired/Semi-paired Data**: Reduce `oml_weight` (1.0-2.0), increase `ccl_weight` (5.0+).
- **Memory Constraints**: Reduce `batch_size` and `num_workers`.
- **High-Resolution Images**: Reduce `batch_size` or use gradient accumulation.

---

## Output & Results

### Training Metrics

- **Total Loss**: Sum of all losses (generators + discriminators)
- **Adversarial Loss**: Per-batch adversarial loss
- **Cycle Loss**: L1 distance between cycled images and originals
- **Identity Loss**: L1 distance between identity-mapped images and originals
- **Matching Loss**: L1 distance between generated images and paired targets
- **Discriminator Loss**: Average of both discriminators' losses

### Evaluation Metrics

Computed during evaluation on the eval dataset:

- **SSIM (Structural Similarity Index)**: Higher is better (range: -1 to 1, typically 0.5-1.0)
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (dB); PSNR=∞ means identical images

### Visualization

- `loss/loss.csv`: Complete loss history per epoch (importable to spreadsheet/analysis tools)
- `loss/loss.png`: Multi-panel plot showing:
  - Train vs eval curves for each loss component
  - Eval SSIM and PSNR trends
- `sample/epoch_{n}/iter_{i}.png`: Grid of 6 columns (A, A→B, A→B→A, B, B→A, B→A→B) for visual inspection

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{harmonization_cyclegan_2024,
  title={Medical Image Harmonization using CycleGAN},
  author={Your Name},
  year={2024},
  url={<repository-url>}
}
```

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `--batch_size` (e.g., 8 or 4)
   - Reduce image resolution
   - Set `--num_workers 0` to reduce memory overhead

2. **Training loss not decreasing**
   - Check data normalization (ensure pixel values are in expected range)
   - Reduce `--learning_rate`
   - Increase `--ccl_weight` or `--oml_weight` to stabilize training

3. **NIfTI inference axis detection fails**
   - Ensure your NIfTI files have a valid affine matrix
   - Manually specify the axis in `inference_nii.py` if needed

4. **No improvement in eval metrics**
   - Increase training epochs or reduce `--decay_epoch` for longer learning
   - Verify dataset quality and pairing correctness
   - Experiment with different loss weight combinations

### Debug Mode

Add verbose logging by modifying `trainer.py`:

```python
if i % self.args.log_freq == 0:
    print(f"[Epoch {epoch}, Iter {i}] Loss breakdown:")
    print(f"  ADV: {loss_adv.item():.4f}, CCL: {loss_ccl.item():.4f}")
    print(f"  ID: {loss_identity.item():.4f}, OML: {loss_oml.item():.4f}")
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes with clear messages
4. Submit a pull request

---

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

