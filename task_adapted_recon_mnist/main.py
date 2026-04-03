import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from task_adapted_recon_mnist.config import Config
from task_adapted_recon_mnist.dataset import build_physics, get_dataloaders
from task_adapted_recon_mnist.train import (train_sequential, train_end_to_end, train_joint,
                                      train_upper_bound, train_lower_bound)
from task_adapted_recon_mnist.evaluate import evaluate_model
from task_adapted_recon_mnist.utils import set_seed


class Tee:
    """Duplicate stdout to both console and a log file."""
    def __init__(self, filepath, mode="w"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, mode, encoding="utf-8")
        self.console = sys.stdout

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    config = Config()
    log_path = os.path.join(config.save_dir, "train.log")
    tee = Tee(log_path)
    sys.stdout = tee

    if config.device == "cuda" and not torch.cuda.is_available():
        print("[info] CUDA not available -- falling back to CPU")
        config.device = "cpu"

    set_seed(config.seed)

    print("=" * 60)
    print(f"  Task-Adapted Reconstruction -- mode: {config.mode}")
    print(f"  Device: {config.device}  |  Subset: {config.subset_size or 'full'}")
    print("=" * 60)

    physics = build_physics(config)
    train_loader, val_loader, test_loader = get_dataloaders(config, physics)

    print(f"  Train samples : {len(train_loader.dataset)}")
    print(f"  Val samples   : {len(val_loader.dataset)}")
    print(f"  Test samples  : {len(test_loader.dataset)}")

    if config.mode == "sequential":
        model = train_sequential(config, physics, train_loader, val_loader)
        evaluate_model(model, physics, test_loader, config, mode="sequential")

    elif config.mode == "end_to_end":
        model = train_end_to_end(config, physics, train_loader, val_loader)
        evaluate_model(model, physics, test_loader, config, mode="end_to_end")

    elif config.mode == "joint":
        model = train_joint(config, physics, train_loader, val_loader)
        evaluate_model(model, physics, test_loader, config, mode="joint")

    elif config.mode == "upper_bound":
        model = train_upper_bound(config, physics, train_loader, val_loader)
        evaluate_model(model, physics, test_loader, config, mode="upper_bound")

    elif config.mode == "lower_bound":
        from task_adapted_recon_mnist.model import ReconstructionNet, TaskNet

        # Load recon_net from sequential Phase 1
        recon_net = ReconstructionNet(
            in_channels=config.n_channels, channels=config.recon_channels,
        ).to(config.device)
        recon_ckpt = torch.load(
            os.path.join(config.save_dir, "sequential", "recon_best.pt"),
            map_location=config.device,
        )
        recon_net.load_state_dict(recon_ckpt["model_state_dict"])

        # Load task_net from upper bound
        task_net = TaskNet(
            in_channels=config.n_channels, channels=config.task_channels,
            num_classes=config.num_classes, img_size=config.img_size,
        ).to(config.device)
        task_ckpt = torch.load(
            os.path.join(config.save_dir, "upper_bound", "upper_best.pt"),
            map_location=config.device,
        )
        task_net.load_state_dict(task_ckpt["model_state_dict"])

        model = (recon_net, task_net)
        evaluate_model(model, physics, test_loader, config, mode="lower_bound")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
