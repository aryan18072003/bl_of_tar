"""
Evaluation module.

Computes accuracy, MSE, PSNR on the test set and saves a sample image.
"""

import os
import torch
import torch.nn as nn

from task_adapted_recon_mnist.config import Config
from task_adapted_recon_mnist.utils import AverageMeter, plot_reconstructions


def compute_psnr(x_hat, x, max_val=1.0):
    mse = torch.mean((x_hat - x) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def evaluate_model(model, physics, test_loader, config, mode="joint"):
    """Evaluate a trained model on the test set.

    Handles all 5 modes:
        sequential    - (recon_net, task_net) tuple, blurred -> recon -> classify
        end_to_end    - blurred -> model -> logits
        joint         - blurred -> model -> (reconstruction, logits)
        upper_bound   - clean image -> classifier
        lower_bound   - pseudo-inverse -> classifier (no learned recon)
    """
    device = config.device
    acc_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    mse_loss = nn.MSELoss()

    # Unpack sequential / end-to-end / lower-bound tuple
    if mode in ("sequential", "end_to_end", "lower_bound") and isinstance(model, tuple):
        recon_net, task_net = model
        recon_net.eval()
        task_net.eval()
    else:
        model.eval()

    sample_saved = False

    with torch.no_grad():
        for blurred, image, label in test_loader:
            blurred = blurred.to(device)
            image = image.to(device)
            label = label.to(device)
            pseudo_inv = physics.A_dagger(blurred)

            if mode == "sequential":
                x_hat = recon_net(physics.A_dagger(blurred))
                logits = task_net(x_hat)

            elif mode == "end_to_end":
                x_hat = recon_net(physics.A_dagger(blurred))
                logits = task_net(x_hat)

            elif mode == "joint":
                x_hat, logits = model(blurred)

            elif mode == "upper_bound":
                logits = model(image)
                x_hat = None

            elif mode == "lower_bound":
                x_hat = recon_net(physics.A_dagger(blurred))
                logits = task_net(x_hat)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # ---------- metrics ----------

            preds = logits.argmax(dim=1)
            acc_meter.update(
                (preds == label).float().mean().item() * 100.0,
                label.size(0),
            )

            if x_hat is not None:
                mse_meter.update(mse_loss(x_hat, image).item(), image.size(0))
                psnr_meter.update(compute_psnr(x_hat, image), image.size(0))

            # ---------- save one sample image ----------

            if not sample_saved:
                vis = {
                    "Blurred": blurred[0, 0],
                    "Pseudo-Inverse": pseudo_inv[0, 0],
                    "Ground Truth": image[0, 0],
                }
                if x_hat is not None:
                    vis["Reconstruction"] = x_hat[0, 0]

                plot_reconstructions(
                    vis,
                    save_path=os.path.join(config.save_dir, mode, "samples.png"),
                    title=f"Pred: {preds[0].item()}  |  True: {label[0].item()}",
                )
                sample_saved = True

    # ---------- print results ----------

    results = {"accuracy": acc_meter.avg}
    if mse_meter.count > 0:
        results["mse"] = mse_meter.avg
        results["psnr"] = psnr_meter.avg

    print(f"\n{'-' * 50}")
    print(f"[Evaluation - {mode}]")
    print(f"  Accuracy : {results['accuracy']:.2f}%")
    if "mse" in results:
        print(f"  MSE      : {results['mse']:.6f}")
        print(f"  PSNR     : {results['psnr']:.2f} dB")
    print(f"{'-' * 50}\n")

    return results
