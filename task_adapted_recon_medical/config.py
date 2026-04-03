"""
Configuration for the medical imaging task-adapted reconstruction pipeline.

Supports both CT (Tomography) and MRI modalities.
Task: binary segmentation (background / organ).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    mode: str = "upper_bound"  #"upper_bound", "lower_bound", "sequential", "end_to_end", "joint"
    modality: str = "mri"

    # ---------- Data ----------
    img_size: int = 128
    n_channels: int = 1
    num_classes: int = 2          
    data_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_medical")
    ct_task_name: str = "Task09_Spleen"
    mri_task_name: str = "Task02_Heart"
    subset_size: Optional[int] = None

    # ---------- Physics ----------
    acceleration: int = 4
    noise_sigma: float = 0.5
    center_frac: float = 0.08

    # ---------- Network architecture ----------
    recon_channels: List[int] = field(default_factory=lambda: [16, 32, 64])
    task_channels: List[int] = field(default_factory=lambda: [16, 32, 64])

    # ---------- Training ----------
    recon_lr: float = 1e-3
    recon_epochs: int = 15
    recon_batch_size: int = 16

    task_lr: float = 5e-3
    task_epochs: int = 30
    task_batch_size: int = 16
    
    joint_lr: float = 1e-3
    joint_epochs: int = 35
    joint_batch_size: int = 16
    c: float = 0.5              # joint loss = c * recon_loss + (1-c) * seg_loss

    # ---------- General ----------
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0
    save_dir: str = "./results_medical"

    def __post_init__(self):
        valid_modes = {"sequential", "end_to_end", "joint", "upper_bound", "lower_bound"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from {valid_modes}")
        valid_modalities = {"ct", "mri"}
        if self.modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{self.modality}'. Choose from {valid_modalities}")
        # MRI uses 2-channel complex tensors (real + imaginary)
        if self.modality == "mri":
            self.n_channels = 2
