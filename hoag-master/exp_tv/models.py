import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet


class UNet(nn.Module):
    """Segmentation network using MONAI UNet (matches task_adapted_recon_medical)."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        channels = [16, 32, 64]
        self.unet = MonaiUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=[2] * (len(channels) - 1),
            num_res_units=2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.unet(x))