"""
Cached MSD Dataset — all slices preloaded into memory for fast training.
Drop-in replacement: `from dataset_new import MSDDataset`
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image


class MSDDataset(Dataset):
    """MSD dataset with full in-memory caching.

    All slices loaded once during __init__.
    __getitem__ is a simple tensor index — zero I/O.
    """

    def __init__(self, root_dir, task_name, img_size=128, modality="CT", subset_size=None):

        path_standard = os.path.join(root_dir, task_name)
        path_nested = os.path.join(root_dir, task_name, task_name)

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

        # --- Preload all valid slices into memory ---
        images_list = []
        masks_list = []

        count = 0
        for img_p, lbl_p in zip(img_files, lbl_files):
            if subset_size and count >= subset_size:
                break
            try:
                nii_img = nib.as_closest_canonical(nib.load(img_p))
                nii_lbl = nib.as_closest_canonical(nib.load(lbl_p))
                data_img = nii_img.get_fdata()
                data_lbl = nii_lbl.get_fdata()

                for i in range(data_lbl.shape[2]):
                    if subset_size and count >= subset_size:
                        break
                    if np.max(data_lbl[:, :, i]) > 0:
                        # Image
                        img_slice = data_img[:, :, i].astype(np.float32)
                        img_t = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0)
                        img_t = F.interpolate(img_t, size=(img_size, img_size),
                                              mode='bilinear', align_corners=False)
                        img_t = img_t.squeeze(0)  # (1, H, W)

                        # Mask
                        lbl_slice = data_lbl[:, :, i]
                        lbl_pil = Image.fromarray(((lbl_slice > 0) * 255).astype(np.uint8)).resize(
                            (img_size, img_size), Image.NEAREST
                        )
                        lbl_t = torch.from_numpy(np.array(lbl_pil) / 255.0).float().unsqueeze(0)

                        images_list.append(img_t)
                        masks_list.append(lbl_t)
                        count += 1
            except Exception as e:
                print(f"Skipping corrupt file {img_p}: {e}")

        n = len(images_list)
        print(f"  -> Valid Slices Found: {n}")

        # Stack into contiguous tensors
        self.images = torch.stack(images_list)  # (N, 1, H, W)
        self.masks = torch.stack(masks_list)    # (N, 1, H, W)

        mem_mb = (self.images.nbytes + self.masks.nbytes) / 1024 / 1024
        print(f"  -> Cached in memory: {mem_mb:.1f} MB")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]
