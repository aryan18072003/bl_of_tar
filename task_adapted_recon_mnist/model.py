import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet
from torchvision.models import resnet18


#  Reconstruction Network  (A†_θ : learns refinement on top of A†(y))
class ReconstructionNet(nn.Module):

    def __init__(self, in_channels: int = 1, channels: list = None, output_size: int = None):
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


#  Task Network  (T_θ : X → D)  —  MNIST Classifier (ResNet-18)
class TaskNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()
        self.net = resnet18(weights=None, num_classes=num_classes)
        # Adapt first conv for 1-channel input (MNIST) + smaller kernel for 28x28
        self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove aggressive downsampling (no maxpool for small images)
        self.net.maxpool = nn.Identity()

    def forward(self, x):
        return self.net(x)


#  Joint Model
class JointModel(nn.Module):
    def __init__(self, physics, recon_net=None, task_net=None, config=None):
        super().__init__()
        self.physics = physics

        if recon_net is not None:
            self.recon_net = recon_net
        else:
            self.recon_net = ReconstructionNet(
                in_channels=config.n_channels,
                channels=config.recon_channels,
            )

        if task_net is not None:
            self.task_net = task_net
        else:
            self.task_net = TaskNet(
                in_channels=config.n_channels,
                channels=config.task_channels,
                num_classes=config.num_classes,
                img_size=config.img_size,
            )

    def forward(self, y):
        # Apply A†(y) first, then refine with recon_net
        a_dag = self.physics.A_dagger(y)
        x_hat = self.recon_net(a_dag)
        logits = self.task_net(x_hat)
        return x_hat, logits
