
import os
import torch
import torch.nn as nn

from task_adapted_recon_medical.config import Config
from task_adapted_recon_medical.utils import AverageMeter, dice_score, plot_samples

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

def apply_norm(img, modality):
    if modality == "ct":
        return norm(img)
    else:
        return norm_z_score(img)

def to_magnitude(x):
    return torch.sqrt(x[:, 0:1, :, :]**2 + x[:, 1:2, :, :]**2 + 1e-8)


def compute_psnr(x_hat, x, max_val=1.0):
    mse = torch.mean((x_hat - x) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


@torch.no_grad()
def evaluate_model(model, physics, test_loader, config, mode="sequential", modality="ct"):
    device = config.device
    save_dir = os.path.join(config.save_dir, mode)

    if isinstance(model, tuple):
        recon_net, seg_net = model
        recon_net.eval()
        seg_net.eval()
    else:
        model.eval()

    dice_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()

    # For plotting samples
    sample_images, sample_masks, sample_preds, sample_recons = [], [], [], []
    sample_saved = False

    for sinogram, image, mask in test_loader:
        sinogram = sinogram.to(device)
        image = image.to(device)
        mask = mask.to(device)

        if mode == "sequential":
            recon_net, seg_net = model
            a_dag = physics.A_dagger(sinogram)
            if modality == "mri":
                a_dag_n = apply_norm(to_magnitude(a_dag), modality)
            else:
                a_dag_n = apply_norm(a_dag, modality)
            x_hat = recon_net(a_dag_n)
            logits = seg_net(x_hat)

        elif mode == "end_to_end":
            a_dag = physics.A_dagger(sinogram)
            if modality == "mri":
                a_dag_n = apply_norm(to_magnitude(a_dag), modality)
            else:
                a_dag_n = apply_norm(a_dag, modality)
            x_hat = recon_net(a_dag_n)
            logits = seg_net(x_hat)

        elif mode == "joint":
            x_hat, logits = model(sinogram)

        elif mode == "upper_bound":
            image_norm = apply_norm(image, modality)
            logits = model(image_norm)
            x_hat = None

        elif mode == "lower_bound":
            a_dag = physics.A_dagger(sinogram)
            if modality == "mri":
                a_dag_n = apply_norm(to_magnitude(a_dag), modality)
            else:
                a_dag_n = apply_norm(a_dag, modality)
            x_hat = recon_net(a_dag_n)
            logits = seg_net(x_hat)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        dice_meter.update(dice_score(logits, mask), mask.size(0))

        if x_hat is not None:
            # For MSE/PSNR, convert to magnitude for MRI
            x_hat_1ch = x_hat
            image_n = apply_norm(image, modality)
            mse_val = nn.functional.mse_loss(x_hat_1ch, image_n).item()
            mse_meter.update(mse_val, image.size(0))
            psnr_meter.update(compute_psnr(x_hat_1ch, image_n), image.size(0))

        # Collect samples for plotting
        if not sample_saved:
            pred = (logits > 0.5).float()  # (B, 1, H, W)
            n = min(4, image.size(0))
            sample_images.extend([image[i] for i in range(n)])
            sample_masks.extend([mask[i] for i in range(n)])
            sample_preds.extend([pred[i] for i in range(n)])
            if x_hat is not None:
                sample_recons.extend([x_hat[i] for i in range(n)])
            if len(sample_images) >= 4:
                plot_samples(
                    sample_images[:4], sample_masks[:4],
                    sample_preds[:4],
                    sample_recons[:4] if sample_recons else None,
                    os.path.join(save_dir, "samples.png"),
                )
                sample_saved = True

    # Print results
    results = {"dice": dice_meter.avg}
    print(f"\n{'—' * 50}")
    print(f"[Evaluation - {mode}]")
    print(f"  Dice Score : {dice_meter.avg:.2f}%")
    if mse_meter.count > 0:
        results["mse"] = mse_meter.avg
        results["psnr"] = psnr_meter.avg
        print(f"  MSE        : {mse_meter.avg:.6f}")
        print(f"  PSNR       : {psnr_meter.avg:.2f} dB")
    print(f"{'—' * 50}\n")

    return results
