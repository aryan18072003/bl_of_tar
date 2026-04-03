import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DiceBCELoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Dice
        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # BCE
        bce_loss = torch.nn.functional.binary_cross_entropy(preds_flat, targets_flat)

        return (0.8*dice_loss) + (0.2*bce_loss)


def dice_score(preds, targets, smooth=1.0):
    preds_bin = (preds > 0.5).float()
    preds_flat = preds_bin.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth + 1e-8)
    return dice.mean().item() * 100.0


def save_checkpoint(model, path, optimizer=None, epoch=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    torch.save(state, path)
    print(f"[checkpoint] saved -> {path}")


def plot_samples(images, masks, preds, recons, save_path, n=4):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = min(n, len(images))
    has_recon = recons is not None and len(recons) > 0

    cols = 4 if has_recon else 3
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3, n * 3))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()
        msk = masks[i].squeeze().cpu().numpy()
        pred = preds[i].squeeze().cpu().numpy()

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(msk, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        if has_recon:
            rec = recons[i].cpu()
            if rec.dim() == 3 and rec.shape[0] == 2:
                rec = torch.sqrt(rec[0] ** 2 + rec[1] ** 2).numpy()
            else:
                rec = rec.squeeze().numpy()
            axes[i, 3].imshow(rec, cmap="gray")
            axes[i, 3].set_title("Reconstruction")
            axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[plot] saved -> {save_path}")
