"""
Configuration for the task-adapted reconstruction pipeline.

All settings are defined here in one place.
Change `mode` to switch between training strategies.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ---------- Which mode to run ----------
    # Options: "sequential", "end_to_end", "joint", "upper_bound", "lower_bound"
    mode: str = "joint"

    # ---------- Data ----------
    img_size: int = 28          # MNIST image size
    n_channels: int = 1         # greyscale
    num_classes: int = 10       # digits 0-9
    data_root: str = "./data_mnist"
    subset_size: Optional[int] = None   # None = use full dataset

    # ---------- Blur Physics ----------
    blur_sigma: float = 3     # std-dev of Gaussian blur kernel
    noise_sigma: float = 0.01   # additive Gaussian noise level

    # ---------- Network architecture ----------
    recon_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    task_channels: List[int] = field(default_factory=lambda: [32, 64, 128])

    # ---------- Training ----------
    recon_lr: float = 1e-3
    recon_epochs: int = 20
    recon_batch_size: int = 256

    task_lr: float = 1e-3
    task_epochs: int = 80
    task_batch_size: int = 256

    joint_lr: float = 1e-3
    joint_epochs: int = 60
    joint_batch_size: int = 256
    c: float = 0.5              # joint loss = c * recon_loss + (1-c) * task_loss

    # ---------- General ----------
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0        # must be 0 on Windows (deepinv lambda can't be pickled)
    save_dir: str = "./results"

    def __post_init__(self):
        valid = {"sequential", "end_to_end", "joint", "upper_bound", "lower_bound"}
        if self.mode not in valid:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from {valid}")
