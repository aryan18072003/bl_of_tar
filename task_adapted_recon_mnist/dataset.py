"""
Dataset module.

- BlurMNISTDataset: wraps MNIST and generates blurred images on-the-fly
- build_physics:    creates the Gaussian blur forward operator (deepinv.physics.BlurFFT)
- get_dataloaders:  returns train / val / test DataLoaders
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import deepinv as dinv
from deepinv.physics.blur import gaussian_blur

from task_adapted_recon_mnist.config import Config


class BlurMNISTDataset(Dataset):
    """Each sample returns (blurred, image, label).

    blurred = physics(image)  -- noisy Gaussian-blurred measurement.
    Generated on-the-fly so we don't store all blurred images in memory.
    """

    def __init__(self, config: Config, physics, train=True):
        super().__init__()
        self.physics = physics
        self.device = config.device

        self.dataset = torchvision.datasets.MNIST(
            root=config.data_root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),  # -> [0,1], shape (1,28,28)
        )

        # Use only a subset if configured
        if config.subset_size is not None:
            n = min(config.subset_size, len(self.dataset))
            self.dataset = Subset(self.dataset, list(range(n)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # image: (1, 28, 28)

        # Generate blurred image: add batch dim -> physics -> remove batch dim
        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)
            y = self.physics(x).squeeze(0).cpu()

        return y, image, label


def build_physics(config: Config):
    """Create the Gaussian blur forward operator."""
    kernel = gaussian_blur(
        sigma=(config.blur_sigma, config.blur_sigma),
    )
    return dinv.physics.Blur(
        filter=kernel,
        padding="circular",
        device=config.device,
        noise_model=dinv.physics.GaussianNoise(sigma=config.noise_sigma),
    )


def get_dataloaders(config: Config, physics):
    """Build train, val, test DataLoaders.

    Combines MNIST train (60k) + test (10k) = 70k, then splits
    by 75:15:15 ratio  →  50,000 train / 10,000 val / 10,000 test.
    """
    ds_train = BlurMNISTDataset(config, physics, train=True)
    ds_test  = BlurMNISTDataset(config, physics, train=False)

    # Merge into one pool
    full_ds = torch.utils.data.ConcatDataset([ds_train, ds_test])

    # 75 : 15 : 15  ratio split
    total = len(full_ds)
    ratio_sum = 75 + 15 + 15
    n_train = int(total * 75 / ratio_sum)     # 50 000
    n_val   = int(total * 15 / ratio_sum)     # 10 000
    n_test  = total - n_train - n_val         # 10 000

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # Pick batch size for this mode
    bs_map = {
        "sequential": config.recon_batch_size,
        "end_to_end": config.task_batch_size,
        "joint": config.joint_batch_size,
    }
    bs = bs_map.get(config.mode, config.task_batch_size)

    kw = dict(batch_size=bs, num_workers=config.num_workers,
              pin_memory=(config.device != "cpu"))

    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))
