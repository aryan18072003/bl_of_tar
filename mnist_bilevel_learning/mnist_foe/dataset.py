"""
Dataset module.

- BlurMNISTDataset: wraps MNIST and generates blurred images on-the-fly
- get_dataloaders:  returns train / val / test DataLoaders
"""

import torch
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms


class BlurMNISTDataset(Dataset):
    """Each sample returns (blurred, image, label).

    blurred = physics(image)  -- noisy Gaussian-blurred measurement.
    Generated on-the-fly so we don't store all blurred images in memory.
    """

    def __init__(self, physics, root_dir="./data",
                 train=True, img_size=28, subset_size=None, device="cpu"):
        super().__init__()
        self.physics = physics
        self.device = device

        self.dataset = torchvision.datasets.MNIST(
            root=root_dir,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),  # -> [0,1], shape (1, img_size, img_size)
            ]),
        )

        if subset_size is not None:
            n = min(subset_size, len(self.dataset))
            self.dataset = Subset(self.dataset, list(range(n)))

        print(f"MNIST {'Train' if train else 'Test'}: {len(self.dataset)} samples, "
              f"resized to {img_size}x{img_size}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # image: (1, img_size, img_size)

        # Generate blurred image on-the-fly
        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)
            y = self.physics(x).squeeze(0).cpu()

        return y, image, label
