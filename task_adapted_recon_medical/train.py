import os
import torch
import torch.nn as nn
import torch.optim as optim

from task_adapted_recon_medical.config import Config
from task_adapted_recon_medical.model import ReconstructionNet, SegmentationNet, JointModel
from task_adapted_recon_medical.utils import AverageMeter, DiceBCELoss, dice_score, save_checkpoint

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
    """Apply modality-appropriate normalization."""
    if modality == "ct":
        return norm(img)
    else:  # mri
        return norm_z_score(img)

def to_magnitude(x):
    """Extract single-channel magnitude from 2-channel complex tensor (B,2,H,W) -> (B,1,H,W)."""
    return torch.sqrt(x[:, 0:1, :, :]**2 + x[:, 1:2, :, :]**2 + 1e-8)


# 1. Sequential

def train_sequential(config, physics, train_loader, val_loader,modality):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "sequential")
    mse = nn.MSELoss()
    seg_loss_fn = DiceBCELoss()

    # --- Phase 1: Train reconstruction network ---
    print("\n" + "=" * 60)
    print("Sequential Phase 1 - Reconstruction Network (A†(y) -> x, MSE)")
    print("=" * 60)

    recon_net = ReconstructionNet(
        in_channels=1, channels=config.recon_channels,
    ).to(device)
    recon_opt = optim.Adam(recon_net.parameters(), lr=config.recon_lr)

    best_val_loss = float("inf")
    for ep in range(1, config.recon_epochs + 1):
        recon_net.train()
        train_loss = AverageMeter()
        for sinogram, image, mask in train_loader:
            sinogram, image = sinogram.to(device), image.to(device)
            a_dag = physics.A_dagger(sinogram)
            if modality == "mri":
                # For MRI recon: train on magnitude
                image_n = apply_norm(image, modality)
                a_dag_n = apply_norm(to_magnitude(a_dag), modality)
            else:
                image_n = apply_norm(image, modality)
                a_dag_n = apply_norm(a_dag, modality)
            x_hat = recon_net(a_dag_n)
            loss = mse(x_hat, image_n)
            recon_opt.zero_grad()
            loss.backward()
            recon_opt.step()
            train_loss.update(loss.item(), image.size(0))

        recon_net.eval()
        val_loss = AverageMeter()
        with torch.no_grad():
            for sinogram, image, mask in val_loader:
                sinogram, image = sinogram.to(device), image.to(device)
                a_dag = physics.A_dagger(sinogram)
                if modality == "mri":
                    image_n = apply_norm(image, modality)
                    a_dag_n = apply_norm(to_magnitude(a_dag), modality)
                else:
                    image_n = apply_norm(image, modality)
                    a_dag_n = apply_norm(a_dag, modality)
                x_hat = recon_net(a_dag_n)
                val_loss.update(mse(x_hat, image_n).item(), image.size(0))

        print(f"  [Recon] Epoch {ep}/{config.recon_epochs}  "
              f"loss={train_loss.avg:.6f}  val_loss={val_loss.avg:.6f}")
        if val_loss.avg < best_val_loss:
            best_val_loss = val_loss.avg
            save_checkpoint(recon_net, os.path.join(ckpt_dir, "recon_best.pt"),
                            optimizer=recon_opt, epoch=ep)

    # --- Phase 2: Train segmentation network on reconstructed images ---
    print("\n" + "=" * 60)
    print("Sequential Phase 2 - Segmentation Network (Dice+BCE)")
    print("=" * 60)

    recon_net.eval()
    seg_net = SegmentationNet(
        in_channels=1, channels=config.task_channels,
        num_classes=config.num_classes,
    ).to(device)

    # Load pre-trained upper bound segmentation net as initialization
    ub_path = os.path.join(config.save_dir, "upper_bound", "upper_best.pt")
    if os.path.isfile(ub_path):
        ub_ckpt = torch.load(ub_path, map_location=device)
        seg_net.load_state_dict(ub_ckpt["model_state_dict"])
        print(f"  [info] Loaded upper bound weights from {ub_path}")
    else:
        print(f"  [warn] Upper bound checkpoint not found at {ub_path}, training from scratch")

    seg_opt = optim.Adam(seg_net.parameters(), lr=config.task_lr)

    best_dice = 0.0
    for ep in range(1, config.task_epochs + 1):
        seg_net.train()
        tr_loss, tr_dice = AverageMeter(), AverageMeter()
        for sinogram, image, mask in train_loader:
            sinogram, mask = sinogram.to(device), mask.to(device)
            with torch.no_grad():
                a_dag = physics.A_dagger(sinogram)
                if modality == "mri":
                    a_dag_n = apply_norm(to_magnitude(a_dag), modality)
                else:
                    a_dag_n = apply_norm(a_dag, modality)
                x_hat = recon_net(a_dag_n)
            logits = seg_net(x_hat)
            loss = seg_loss_fn(logits, mask)
            seg_opt.zero_grad()
            loss.backward()
            seg_opt.step()
            tr_loss.update(loss.item(), mask.size(0))
            tr_dice.update(dice_score(logits, mask), mask.size(0))

        seg_net.eval()
        va_loss, va_dice = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for sinogram, image, mask in val_loader:
                sinogram, mask = sinogram.to(device), mask.to(device)
                a_dag = physics.A_dagger(sinogram)
                if modality == "mri":
                    a_dag_n = apply_norm(to_magnitude(a_dag), modality)
                else:
                    a_dag_n = apply_norm(a_dag, modality)
                x_hat = recon_net(a_dag_n)
                logits = seg_net(x_hat)
                va_loss.update(seg_loss_fn(logits, mask).item(), mask.size(0))
                va_dice.update(dice_score(logits, mask), mask.size(0))

        print(f"  [Seg]   Epoch {ep}/{config.task_epochs}  "
              f"loss={tr_loss.avg:.4f}  dice={tr_dice.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_dice={va_dice.avg:.1f}%")
        if va_dice.avg > best_dice:
            best_dice = va_dice.avg
            save_checkpoint(seg_net, os.path.join(ckpt_dir, "seg_best.pt"),
                            optimizer=seg_opt, epoch=ep)

    print("\n[Sequential] Done.")
    return recon_net, seg_net


# 2. End-to-End  (C = 0 in the joint loss)
#  Pre-trained recon_net (from sequential Phase 1) + seg_net (from upper bound),
#  then fine-tuned end-to-end with only Dice+BCE loss (gradients flow through both).

def train_end_to_end(config, physics, train_loader, val_loader, modality):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "end_to_end")
    seg_loss_fn = DiceBCELoss()

    # ---- Build recon_net (pre-trained from sequential Phase 1) ----
    recon_net = ReconstructionNet(
        in_channels=1, channels=config.recon_channels,
        output_size=config.img_size,
    ).to(device)

    recon_ckpt_path = os.path.join(config.save_dir, "sequential", "recon_best.pt")
    if os.path.isfile(recon_ckpt_path):
        ckpt = torch.load(recon_ckpt_path, map_location=device)
        recon_net.load_state_dict(ckpt["model_state_dict"])
        print(f"  [info] Loaded pre-trained recon_net from {recon_ckpt_path}")
    else:
        print(f"  [warn] Recon checkpoint not found at {recon_ckpt_path}, training from scratch")

    # ---- Build seg_net (initialised from upper-bound segmentation) ----
    seg_net = SegmentationNet(
        in_channels=1, channels=config.task_channels,
        num_classes=config.num_classes,
    ).to(device)

    ub_path = os.path.join(config.save_dir, "upper_bound", "upper_best.pt")
    if os.path.isfile(ub_path):
        ckpt = torch.load(ub_path, map_location=device)
        seg_net.load_state_dict(ckpt["model_state_dict"])
        print(f"  [info] Loaded upper bound seg_net from {ub_path}")
    else:
        print(f"  [warn] Upper bound checkpoint not found at {ub_path}, training from scratch")

    print("\n" + "=" * 60)
    print("End-to-End Training  (C=0: Dice+BCE only, gradients through recon+seg)")
    print("=" * 60)

    # Optimise both networks jointly
    optimizer = optim.Adam(
        list(recon_net.parameters()) + list(seg_net.parameters()),
        lr=config.task_lr,
    )

    best_dice = 0.0
    for ep in range(1, config.task_epochs + 1):
        recon_net.train()
        seg_net.train()
        tr_loss, tr_dice = AverageMeter(), AverageMeter()
        for sinogram, image, mask in train_loader:
            sinogram, mask = sinogram.to(device), mask.to(device)
            a_dag = physics.A_dagger(sinogram)
            if modality == "mri":
                a_dag_n = apply_norm(to_magnitude(a_dag), modality)
            else:
                a_dag_n = apply_norm(a_dag, modality)
            x_hat = recon_net(a_dag_n)
            logits = seg_net(x_hat)
            loss = seg_loss_fn(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss.update(loss.item(), mask.size(0))
            tr_dice.update(dice_score(logits, mask), mask.size(0))

        recon_net.eval()
        seg_net.eval()
        va_loss, va_dice = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for sinogram, image, mask in val_loader:
                sinogram, mask = sinogram.to(device), mask.to(device)
                a_dag = physics.A_dagger(sinogram)
                if modality == "mri":
                    a_dag_n = apply_norm(to_magnitude(a_dag), modality)
                else:
                    a_dag_n = apply_norm(a_dag, modality)
                x_hat = recon_net(a_dag_n)
                logits = seg_net(x_hat)
                va_loss.update(seg_loss_fn(logits, mask).item(), mask.size(0))
                va_dice.update(dice_score(logits, mask), mask.size(0))

        print(f"  [E2E]   Epoch {ep}/{config.task_epochs}  "
              f"loss={tr_loss.avg:.4f}  dice={tr_dice.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_dice={va_dice.avg:.1f}%")
        if va_dice.avg > best_dice:
            best_dice = va_dice.avg
            save_checkpoint(recon_net, os.path.join(ckpt_dir, "recon_best.pt"),
                            optimizer=optimizer, epoch=ep)
            save_checkpoint(seg_net, os.path.join(ckpt_dir, "seg_best.pt"),
                            optimizer=optimizer, epoch=ep)

    print("\n[End-to-End] Done.")
    return recon_net, seg_net


# 3. Joint

def train_joint(config, physics, train_loader, val_loader, modality):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "joint")
    mse = nn.MSELoss()
    seg_loss_fn = DiceBCELoss()
    c = config.c

    model = JointModel(physics=physics, config=config).to(device)

    print("\n" + "=" * 60)
    print("Joint Training (Recon + Segmentation)")
    print("=" * 60)

    optimizer = optim.Adam(model.parameters(), lr=config.joint_lr)

    # ---- Resume from previous joint checkpoint if available ----
    joint_ckpt_path = os.path.join(ckpt_dir, "joint_best.pt")
    best_dice = 0.0

    _try_load_sequential_weights(model, config, device)

    for ep in range(1, config.joint_epochs + 1):
        model.train()
        tr_recon, tr_seg, tr_dice = AverageMeter(), AverageMeter(), AverageMeter()
        for sinogram, image, mask in train_loader:
            sinogram, image, mask = sinogram.to(device), image.to(device), mask.to(device)
            x_hat, seg_logits = model(sinogram)
            image_n = apply_norm(image, modality)
            x_hat_1ch = x_hat
            recon_loss = mse(x_hat_1ch, image_n)
            seg_loss = seg_loss_fn(seg_logits, mask)
            loss = c * recon_loss + (1 - c) * seg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_recon.update(recon_loss.item(), image.size(0))
            tr_seg.update(seg_loss.item(), image.size(0))
            tr_dice.update(dice_score(seg_logits, mask), mask.size(0))

        model.eval()
        va_recon, va_seg, va_dice = AverageMeter(), AverageMeter(), AverageMeter()
        with torch.no_grad():
            for sinogram, image, mask in val_loader:
                sinogram, image, mask = sinogram.to(device), image.to(device), mask.to(device)
                x_hat, seg_logits = model(sinogram)
                image_n = apply_norm(image, modality)
                x_hat_1ch = x_hat
                va_recon.update(mse(x_hat_1ch, image_n).item(), image.size(0))
                va_seg.update(seg_loss_fn(seg_logits, mask).item(), mask.size(0))
                va_dice.update(dice_score(seg_logits, mask), mask.size(0))

        print(f"  [Joint] Epoch {ep}/{config.joint_epochs}  "
              f"recon={tr_recon.avg:.6f}  seg={tr_seg.avg:.4f}  dice={tr_dice.avg:.1f}%  "
              f"val_dice={va_dice.avg:.1f}%")
        if va_dice.avg > best_dice:
            best_dice = va_dice.avg
            save_checkpoint(model, os.path.join(ckpt_dir, "joint_best.pt"),
                            optimizer=optimizer, epoch=ep)

    print("\n[Joint] Done.")
    return model


# 4. Upper Bound

def train_upper_bound(config, physics, train_loader, val_loader, modality):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "upper_bound")
    seg_loss_fn = DiceBCELoss()

    print("\n" + "=" * 60)
    print("Upper Bound - Segmentation on clean images")
    print("=" * 60)

    seg_net = SegmentationNet(
        in_channels=1, channels=config.task_channels,
        num_classes=config.num_classes,
    ).to(device)
    optimizer = optim.Adam(seg_net.parameters(), lr=config.task_lr)

    best_dice = 0.0
    for ep in range(1, config.task_epochs + 1):
        seg_net.train()
        tr_loss, tr_dice = AverageMeter(), AverageMeter()
        for sinogram, image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            if modality == "ct":
                image = norm(image)
            else:
                image = norm_z_score(image)
            logits = seg_net(image)
            loss = seg_loss_fn(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss.update(loss.item(), mask.size(0))
            tr_dice.update(dice_score(logits, mask), mask.size(0))

        seg_net.eval()
        va_loss, va_dice = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for sinogram, image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                if modality == "ct":
                    image = norm(image)
                else:
                    image = norm_z_score(image)
                logits = seg_net(image)
                va_loss.update(seg_loss_fn(logits, mask).item(), mask.size(0))
                va_dice.update(dice_score(logits, mask), mask.size(0))

        print(f"  [Upper] Epoch {ep}/{config.task_epochs}  "
              f"loss={tr_loss.avg:.4f}  dice={tr_dice.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_dice={va_dice.avg:.1f}%")
        if va_dice.avg > best_dice:
            best_dice = va_dice.avg
            save_checkpoint(seg_net, os.path.join(ckpt_dir, "upper_best.pt"),
                            optimizer=optimizer, epoch=ep)

    print("\n[Upper Bound] Done.")
    return seg_net


# 5. Lower Bound

def train_lower_bound(config, physics, train_loader, val_loader, modality):
    device = config.device

    print("\n" + "=" * 60)
    print("Lower Bound - Loading upper bound model (evaluate on FBP)")
    print("=" * 60)

    seg_net = SegmentationNet(
        in_channels=1, channels=config.task_channels,
        num_classes=config.num_classes,
    ).to(device)

    ub_path = os.path.join(config.save_dir, "upper_bound", "upper_best.pt")
    if os.path.isfile(ub_path):
        ckpt = torch.load(ub_path, map_location=device)
        seg_net.load_state_dict(ckpt["model_state_dict"])
        print(f"  [info] Loaded upper bound weights from {ub_path}")
    else:
        raise FileNotFoundError(
            f"Upper bound checkpoint not found at {ub_path}. "
            f"Run upper_bound mode first."
        )

    print("\n[Lower Bound] Done (no training needed).")
    return seg_net


# ---- Helper ----

def _try_load_sequential_weights(model, config, device):
    recon_path = os.path.join(config.save_dir, "sequential", "recon_best.pt")
    seg_path = os.path.join(config.save_dir, "sequential", "seg_best.pt")
    if os.path.isfile(recon_path) and os.path.isfile(seg_path):
        print("[info] Loading pre-trained sequential weights...")
        try:
            model.recon_net.load_state_dict(
                torch.load(recon_path, map_location=device)["model_state_dict"]
            )
            model.seg_net.load_state_dict(
                torch.load(seg_path, map_location=device)["model_state_dict"]
            )
            print("[info] Sequential weights loaded successfully.")
        except RuntimeError as e:
            print(f"[warn] Could not load sequential weights (modality mismatch?): {e}")
            print("[info] Training joint model from scratch.")
