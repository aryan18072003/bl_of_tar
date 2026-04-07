# Bilevel Learning for Task-Adapted Regularizer (BL-TAR)

A research codebase for **bilevel optimization**–based image reconstruction, where a reconstruction operator and its regularizer are jointly optimized with a downstream task (classification or segmentation). The framework learns to reconstruct images from corrupted/incomplete measurements in a way that maximally benefits the end task, rather than optimizing reconstruction quality in isolation.

## Overview

The project formulates image reconstruction as a bilevel optimization problem:

- **Inner (lower-level) problem:** Given regularizer parameters **θ**, solve for the reconstructed image **w\*** by minimizing a data-fidelity term ‖y − A(w)‖² plus a regularizer R_θ(w).
- **Outer (upper-level) problem:** Optimize **θ** (and/or a task network) to minimize a task-specific loss (e.g., cross-entropy for classification, Dice-BCE for segmentation) evaluated on the reconstructed images.

Hypergradients are computed via the **HOAG (Hyper-gradient via One-step Approximation with Gradient)** algorithm, which uses conjugate gradient to implicitly invert the inner Hessian.

## Project Structure

```
bl_of_tar/
├── bl_medical/                # Reference HOAG bilevel optimizer (TV & FoE regularizers)
│   ├── hoag/                   #   Core HOAG algorithm (L-BFGS-B inner solver, CG for hypergradients)
│   ├── exp_tv/                 #   Experiment: TV regularizer for CT spleen segmentation
│   ├── exp_foe/                #   Experiment: Fields of Experts regularizer for MRI cardiac segmentation
│   └── vis.py                  #   Visualization utilities
│
├── bl_icnn/                    # HOAG with learned ICNN-based convex regularizer
│   ├── convex_models.py        #   ICNN, Sparsifying Filter Bank (SFB), L2net architectures
│   ├── train_icnn.py           #   Pre-train ICNN via Adversarial Convex Regularization (ACR)
│   ├── physics.py              #   Physics operators + inner loss with ICNN regularizer
│   ├── hoag.py                 #   HOAG adapted for ICNN (inner solver + hypergradient step)
│   ├── hoag_utils.py           #   Hessian-vector products & conjugate gradient
│   └── main.py                 #   Main experiment: bilevel training with ICNN regularizer
│
├── task_adapted_recon_mnist/   # Bilevel task-adapted reconstruction on MNIST
│   ├── model.py                #   ReconstructionNet (UNet), TaskNet (ResNet-18), JointModel
│   ├── train.py                #   Training: sequential, end-to-end, joint, upper/lower bound
│   ├── dataset.py              #   MNIST with Gaussian blur degradation
│   ├── evaluate.py             #   Evaluation and metrics
│   └── config.py               #   Hyperparameters and paths
│
├── task_adapted_recon_medical/ # Bilevel task-adapted reconstruction on medical data
│   ├── model.py                #   UNet for segmentation
│   ├── train.py                #   Training strategies for CT/MRI segmentation
│   ├── dataset.py              #   Medical Segmentation Decathlon loader (NIfTI)
│   └── config.py               #   CT/MRI configuration
│
├── data_driven_convex_regularization-main/  # ACR pre-training pipeline
│   ├── convex_models.py        #   Original ICNN architecture
│   ├── train_convex_reg.py     #   ACR training (WGAN-GP style)
│   ├── eval_convex_reg.py      #   Evaluation of learned regularizer
│   └── mayo_utils.py           #   Mayo CT data utilities
│
├── mnist_bilevel_learning/     # Earlier bilevel experiments on MNIST
│   ├── mnist_tv/               #   MNIST with TV regularizer
│   └── mnist_foe/              #   MNIST with FoE regularizer
│
├── data_medical/               # Medical imaging datasets (MSD format)
└── data_mnist/                 # MNIST data
```

## Modules

### `bl_medical` — Reference HOAG Bilevel Optimizer

The core bilevel optimization engine. The `hoag/` library implements HOAG with L-BFGS-B for the inner optimization and conjugate gradient for implicit hypergradient computation. Two experiment directories demonstrate different regularizers:

| Experiment | Regularizer | Learnable Parameters (θ) | Modality | Task |
|---|---|---|---|---|
| `exp_tv/` | Total Variation | 2 (λ, ε) | CT | Spleen segmentation |
| `exp_foe/` | Fields of Experts | 136 (weights + 5×5 filters) | MRI | Cardiac segmentation |

Both share the same bilevel structure: the inner problem reconstructs images from undersampled measurements, while the outer problem trains a UNet segmenter on the reconstructed outputs using Dice-BCE loss.

### `bl_icnn` — Learned Convex Regularizer via ICNN

Replaces hand-crafted regularizers with a **data-driven convex prior** composed of three components:

- **ICNN** — Input-Convex Neural Network with non-negative weight constraints
- **SFB** — Sparsifying Filter Bank with learned convolutional kernels
- **L2net** — Learnable L2 penalty

These are pre-trained via ACR (adversarial training: R(clean) − R(A†(y)) + gradient penalty), then frozen. Only the scalar mixing weights θ = [log λ_icnn, log λ_sfb, log λ_l2] are optimized at the upper level.

### `task_adapted_recon_mnist` — MNIST Classification

Demonstrates five training strategies for task-adapted reconstruction:

1. **Upper Bound** — Task network trained on clean images
2. **Lower Bound** — Task network trained on clean, evaluated on A†(y)
3. **Sequential** — Train reconstruction, then train task network
4. **End-to-End** — Fine-tune both networks jointly with task loss only
5. **Joint** — Simultaneous training with c·MSE + (1−c)·CE loss

### `task_adapted_recon_medical` — Medical Image Segmentation

Extends the MNIST framework to CT (Task09_Spleen) and MRI (Task02_Heart) from the Medical Segmentation Decathlon, with UNet-based segmentation.

## Tech Stack

Python, PyTorch, DeepInverse, MONAI, NumPy, SciPy, torchvision, NiBabel, Matplotlib, Jupyter Notebook

## Getting Started

### Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU (recommended)

### Installation

```bash
pip install torch torchvision monai deepinv nibabel matplotlib scipy numpy
```

### Usage

**1. Pre-train the ICNN regularizer (optional, for `bl_icnn`):**
```bash
cd bl_icnn
python train_icnn.py
```

**2. Run bilevel optimization with TV regularizer:**
```bash
cd bl_medical/exp_tv
python main.py
```

**3. Run bilevel optimization with ICNN regularizer:**
```bash
cd bl_icnn
python main.py
```

**4. Run MNIST task-adapted reconstruction:**
```bash
cd task_adapted_recon_mnist
python main.py
```

### Data

- **MNIST** — Downloaded automatically via torchvision
- **Medical data** — Place [Medical Segmentation Decathlon](http://medicaldecathlon.com/) datasets under `data_medical/ct_data/` (e.g., `Task09_Spleen/`) in NIfTI format

## Key References

- Crockett et al., *Bilevel Learning of the Group Lasso Structure* (HOAG algorithm)
- Mukherjee et al., *Learned Convex Regularizers for Inverse Problems* (ACR / ICNN training)
- Adler & Öktem, *Deep Bayesian Inversion* (task-adapted reconstruction)
