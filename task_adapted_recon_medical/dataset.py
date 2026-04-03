"""
Dataset loading for medical imaging task-adapted reconstruction.

Adapted from dataset_new.py and physics.py:
- Lazy loading via (img_path, lbl_path, slice_idx) references
- Raw HU values (no [0,1] normalization at dataset level)
- Physics operator from physics.py factory
- Returns (sinogram, image, mask) tuples
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from PIL import Image
import deepinv as dinv

from task_adapted_recon_medical.config import Config


# ---------- Physics operator factory (from physics.py) ----------

def build_physics(config):
    """Build the forward physics operator based on modality."""
    device = config.device
    modality = config.modality.upper()

    if modality == "CT":
        if config.acceleration == 1:
            num_views = 180
        else:
            num_views = int(180 / config.acceleration)

        angles = torch.linspace(0, 180, num_views).to(device)

        physics = dinv.physics.Tomography(
            angles=angles,
            img_width=config.img_size,
            circle=False,
            device=device,
            normalize=True,
            noise_model=dinv.physics.GaussianNoise(sigma=config.noise_sigma),
        )

    elif modality == "MRI":
        img_size = config.img_size
        mask = torch.zeros((1, img_size, img_size))

        pad = (img_size - int(img_size * config.center_frac) + 1) // 2
        width = max(1, int(img_size * config.center_frac))
        mask[:, :, pad:pad + width] = 1.0

        num_keep = int(img_size / config.acceleration)
        all_cols = np.arange(img_size)
        kept_cols = np.where(mask[0, 0, :].cpu().numpy() == 1)[0]
        zero_cols = np.setdiff1d(all_cols, kept_cols)

        if len(zero_cols) > 0 and (num_keep - len(kept_cols) > 0):
            chosen = np.random.choice(zero_cols, num_keep - len(kept_cols), replace=False)
            mask[:, :, chosen] = 1.0

        mask = mask.to(device)
        physics = dinv.physics.MRI(
            mask=mask,
            img_size=(1, img_size, img_size),
            device=device,
            normalize=True,
            noise_model=dinv.physics.GaussianNoise(sigma=config.noise_sigma),
        )

    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return physics

# ---------- Dataset (adapted from dataset_new.py) ----------

class MedicalSegDataset(Dataset):
    """2D medical image segmentation dataset.

    Lazily loads slices from NIfTI volumes.
    Each sample returns (sinogram, image, mask):
        - sinogram: physics.A(image), shape (1, H_sino, W_sino)
        - image: raw image slice, shape (1, H, W)
        - mask: segmentation mask, shape (1, H, W)
    """

    def __init__(self, config, physics, train=True):
        self.img_size = config.img_size
        self.physics = physics
        self.device = config.device
        self.modality = config.modality.lower()
        self.slices = []

        # Resolve data path
        if self.modality == "ct":
            data_subdir = os.path.join(config.data_root, "ct_data")
            task_name = config.ct_task_name
        else:
            data_subdir = os.path.join(config.data_root, "mri_data")
            task_name = config.mri_task_name

        # Check standard and nested paths (from dataset_new.py)
        path_standard = os.path.join(data_subdir, task_name)
        path_nested = os.path.join(data_subdir, task_name, task_name)

        if os.path.exists(os.path.join(path_standard, "imagesTr")):
            self.task_path = path_standard
        elif os.path.exists(os.path.join(path_nested, "imagesTr")):
            self.task_path = path_nested
        else:
            raise ValueError(
                f"Could not find 'imagesTr' folder.\n"
                f"Checked:\n  1. {path_standard}\n  2. {path_nested}"
            )

        self.img_dir = os.path.join(self.task_path, "imagesTr")
        self.lbl_dir = os.path.join(self.task_path, "labelsTr")

        img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.nii.gz")))
        lbl_files = sorted(glob.glob(os.path.join(self.lbl_dir, "*.nii.gz")))

        if len(img_files) == 0:
            raise ValueError(f"No data found in {self.img_dir}. Check paths!")

        print(f"Scanning {len(img_files)} volumes for Task: {task_name}...")
        print(f"  -> Source: {self.img_dir}")

        count = 0
        for img_p, lbl_p in zip(img_files, lbl_files):
            if config.subset_size and count >= config.subset_size:
                break
            try:
                nii_lbl = nib.load(lbl_p)
                data_lbl = nib.as_closest_canonical(nii_lbl).get_fdata()
                for i in range(data_lbl.shape[2]):
                    if config.subset_size and count >= config.subset_size:
                        break
                    if np.max(data_lbl[:, :, i]) > 0:
                        self.slices.append((img_p, lbl_p, i))
                        count += 1
            except Exception as e:
                print(f"Skipping corrupt file {img_p}: {e}")

        print(f"  -> Valid Slices Found: {len(self.slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_p, lbl_p, s_idx = self.slices[idx]

        img = nib.as_closest_canonical(nib.load(img_p)).dataobj[..., s_idx]
        lbl = nib.as_closest_canonical(nib.load(lbl_p)).dataobj[..., s_idx]

        # Image: resize
        img_t = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        img_t = F.interpolate(img_t, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        img_t = img_t.squeeze(0)  # (1, img_size, img_size)

        # Label: binary mask, nearest interpolation
        lbl_pil = Image.fromarray(((lbl > 0) * 255).astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )
        lbl_np = np.array(lbl_pil) / 255.0
        lbl_t = torch.from_numpy(lbl_np).float().unsqueeze(0)  # (1, img_size, img_size)

        # Generate sinogram on-the-fly
        with torch.no_grad():
            if self.modality == "mri":
                # deepinv MRI expects 2-channel complex input: (B, 2, H, W)
                x_complex = torch.cat([img_t, torch.zeros_like(img_t)], dim=0)  # (2, H, W)
                x_batch = x_complex.unsqueeze(0).to(self.device)  # (1, 2, H, W)
                sino = self.physics.A(x_batch).squeeze(0).cpu()   # (2, H, W) complex k-space
            else:
                x = img_t.unsqueeze(0).to(self.device)            # (1, 1, H, W)
                sino = self.physics.A(x).squeeze(0).cpu()         # (1, H_sino, W_sino)

        return sino, img_t, lbl_t


def get_dataloaders(config, physics):
    """Build train, val, test DataLoaders.

    Training set is split 80/10/10 into train/val/test.
    """
    full_ds = MedicalSegDataset(config, physics)

    # Split: 80% train, 10% val, 10% test
    n = len(full_ds)
    n_test = max(1, int(0.1 * n))
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed),
    )

    bs = config.recon_batch_size
    kw = dict(batch_size=bs, num_workers=config.num_workers,
              pin_memory=(config.device != "cpu"))

    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
