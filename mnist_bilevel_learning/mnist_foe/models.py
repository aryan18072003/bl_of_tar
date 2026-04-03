import torch
import torch.nn as nn
from torchvision.models import resnet18


class TaskNet(nn.Module):
    """MNIST classifier using ResNet-18 (adapted for 1-channel 28x28 input)."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()
        self.net = resnet18(weights=None, num_classes=num_classes)
        # Adapt first conv for 1-channel input (MNIST) + smaller kernel for 28x28
        self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove aggressive downsampling (no maxpool for small images)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)
