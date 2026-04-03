"""
Cached dataset for medical imaging task-adapted reconstruction.

All slices are preloaded into GPU memory during __init__ for fast training.
Sinograms are precomputed once (not on every access).
~65 MB for 1051 slices at 128x128 — trivial for H100.
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


# ---------- Physics operator factory ----------

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


# ---------- Cached Dataset ----------

class MedicalSegDataset(Dataset):
    """2D medical image segmentation dataset with full in-memory caching.

    All slices are loaded into memory during __init__.
    Sinograms are precomputed once (batch-wise on GPU for speed).
    __getitem__ is a simple tensor index — no I/O, no computation.
    """

    def __init__(self, config, physics, train=True):
        self.img_size = config.img_size
        self.physics = physics
        self.device = config.device
        self.modality = config.modality.lower()

        # Resolve data path
        if self.modality == "ct":
            data_subdir = os.path.join(config.data_root, "ct_data")
            task_name = config.ct_task_name
        else:
            data_subdir = os.path.join(config.data_root, "mri_data")
            task_name = config.mri_task_name

        path_standard = os.path.join(data_subdir, task_name)
        path_nested = os.path.join(data_subdir, task_name, task_name)

        if os.path.exists(os.path.join(path_standard, "imagesTr")):
            task_path = path_standard
        elif os.path.exists(os.path.join(path_nested, "imagesTr")):
            task_path = path_nested
        else:
            raise ValueError(
                f"Could not find 'imagesTr' folder.\n"
                f"Checked:\n  1. {path_standard}\n  2. {path_nested}"
            )

        img_dir = os.path.join(task_path, "imagesTr")
        lbl_dir = os.path.join(task_path, "labelsTr")

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.nii.gz")))

        if len(img_files) == 0:
            raise ValueError(f"No data found in {img_dir}. Check paths!")

        print(f"Scanning {len(img_files)} volumes for Task: {task_name}...")
        print(f"  -> Source: {img_dir}")

        # --- Step 1: Load all valid slices into CPU memory ---
        images_list = []
        masks_list = []

        count = 0
        for img_p, lbl_p in zip(img_files, lbl_files):
            if config.subset_size and count >= config.subset_size:
                break
            try:
                nii_img = nib.as_closest_canonical(nib.load(img_p))
                nii_lbl = nib.as_closest_canonical(nib.load(lbl_p))
                data_img = nii_img.get_fdata()
                data_lbl = nii_lbl.get_fdata()

                for i in range(data_lbl.shape[2]):
                    if config.subset_size and count >= config.subset_size:
                        break
                    if np.max(data_lbl[:, :, i]) > 0:
                        # Image
                        img_slice = data_img[:, :, i].astype(np.float32)
                        img_t = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0)
                        img_t = F.interpolate(img_t, size=(self.img_size, self.img_size),
                                              mode='bilinear', align_corners=False)
                        img_t = img_t.squeeze(0)  # (1, H, W)

                        # Mask
                        lbl_slice = data_lbl[:, :, i]
                        lbl_pil = Image.fromarray(((lbl_slice > 0) * 255).astype(np.uint8)).resize(
                            (self.img_size, self.img_size), Image.NEAREST
                        )
                        lbl_t = torch.from_numpy(np.array(lbl_pil) / 255.0).float().unsqueeze(0)

                        images_list.append(img_t)
                        masks_list.append(lbl_t)
                        count += 1
            except Exception as e:
                print(f"Skipping corrupt file {img_p}: {e}")

        n = len(images_list)
        print(f"  -> Valid Slices Found: {n}")

        # Stack into contiguous tensors on CPU
        self.images = torch.stack(images_list)  # (N, 1, H, W)
        self.masks = torch.stack(masks_list)    # (N, 1, H, W)

        # --- Step 2: Precompute sinograms in batches on GPU ---
        print("  -> Precomputing sinograms...")
        sino_list = []
        batch_sz = 64  # process 64 at a time to avoid OOM
        with torch.no_grad():
            for start in range(0, n, batch_sz):
                end = min(start + batch_sz, n)
                batch_imgs = self.images[start:end].to(self.device)

                if self.modality == "mri":
                    batch_imgs = torch.cat([batch_imgs, torch.zeros_like(batch_imgs)], dim=1)

                sinos = self.physics.A(batch_imgs).cpu()
                sino_list.append(sinos)

        self.sinograms = torch.cat(sino_list, dim=0)  # (N, C, H_sino, W_sino)
        print(f"  -> Caching complete. Images: {self.images.shape}, "
              f"Sinograms: {self.sinograms.shape}, "
              f"Memory: {(self.images.nbytes + self.sinograms.nbytes + self.masks.nbytes) / 1024 / 1024:.1f} MB")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.sinograms[idx], self.images[idx], self.masks[idx]


def get_dataloaders(config, physics):
    """Build train, val, test DataLoaders."""
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
    kw = dict(batch_size=bs, num_workers=0,
              pin_memory=False)

    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )
