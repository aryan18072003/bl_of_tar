"""
Utility helpers for Task-Adapted Reconstruction.

Provides:
  - set_seed           : reproducibility
  - save_checkpoint    : persist model + optimiser state
  - load_checkpoint    : restore model weights
  - AverageMeter       : running mean / count tracker
  - plot_reconstructions : side-by-side visualisation helper
"""

import os
import random

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: torch.nn.Module, path: str, optimizer=None, epoch=None, extra=None):
    """Save model (and optionally optimiser) state to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra is not None:
        state.update(extra)
    torch.save(state, path)
    print(f"[checkpoint] saved -> {path}")


def load_checkpoint(model: torch.nn.Module, path: str, device: str = "cpu"):
    """Load model weights from *path*.  Returns the full state dict."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print(f"[checkpoint] loaded <- {path}")
    return state


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics tracker
# ═══════════════════════════════════════════════════════════════════════════

class AverageMeter:
    """Computes and stores the running average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_reconstructions(images_dict: dict, save_path: str, title: str = ""):
    """Plot a row of named images side by side and save to *save_path*.

    Args:
        images_dict: ``{name: tensor}`` where each tensor is
            ``(1, H, W)`` or ``(H, W)``.
        save_path: File path to save the figure.
        title: Optional super-title.
    """
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, images_dict.items()):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {save_path}")
