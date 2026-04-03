"""
Models for the medical imaging task-adapted reconstruction pipeline.

Uses MONAI UNet for both reconstruction and segmentation networks.
- ReconstructionNet: A†(y) -> image (residual refinement on pseudo-inverse)
- SegmentationNet: image -> binary segmentation mask (sigmoid output)
- JointModel: y -> A†(y) -> recon -> segmentation
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet

def norm(img):
    img = torch.clamp(img, min=-150, max=250)
    img = (img + 150) / 400.0
    return img

def norm_z_score(img):
    mean = img.mean()
    std = img.std()
    if std > 0:
        img = (img - mean) / std
    else:
        img = torch.zeros_like(img)
    return img


# Reconstruction Network: A†(y) -> image (residual refinement)
class ReconstructionNet(nn.Module):
    def __init__(self, in_channels=1, channels=None, output_size=None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        self.unet = MonaiUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            strides=[2] * (len(channels) - 1),
            num_res_units=2,
        )

    def forward(self, x):
        # Residual learning: input is A†(y), network learns the correction
        return x + self.unet(x)


# Segmentation Network: image -> binary mask (sigmoid output)
class SegmentationNet(nn.Module):
    def __init__(self, in_channels=1, channels=None, num_classes=2, output_size=None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        self.unet = MonaiUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=channels,
            strides=[2] * (len(channels) - 1),
            num_res_units=2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.unet(x))  # (B, 1, H, W) in [0, 1]


# Joint Model: y -> A†(y) -> recon -> segmentation
class JointModel(nn.Module):
    def __init__(self, physics, config):
        super().__init__()
        self.physics = physics
        self.modality = config.modality.lower()
        # Recon net always works on 1-channel images:
        # CT: 1-ch raw → norm → recon
        # MRI: 2-ch complex → magnitude (1-ch) → norm → recon
        self.recon_net = ReconstructionNet(
            in_channels=1, channels=config.recon_channels,
        )
        # Segmentation always operates on 1-channel images
        self.seg_net = SegmentationNet(
            in_channels=1, channels=config.task_channels,
            num_classes=config.num_classes,
        )

    def forward(self, y):
        # Apply A†(y), normalize based on modality, then refine with recon_net
        a_dag = self.physics.A_dagger(y)
        if self.modality == "mri":
            # MRI: extract magnitude from 2-channel complex, then z-score normalize
            magnitude = torch.sqrt(a_dag[:, 0:1, :, :]**2 + a_dag[:, 1:2, :, :]**2 + 1e-8)
            a_dag_n = norm_z_score(magnitude)
        else:
            a_dag_n = norm(a_dag)
        x_hat = self.recon_net(a_dag_n)
        seg = self.seg_net(x_hat)
        return x_hat, seg
