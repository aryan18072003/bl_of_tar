import os
import torch
import torch.nn as nn
import torch.optim as optim

from task_adapted_recon_mnist.config import Config
from task_adapted_recon_mnist.model import ReconstructionNet, TaskNet, JointModel
from task_adapted_recon_mnist.utils import AverageMeter, save_checkpoint


def _accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item() * 100.0


# ── 1. Sequential ────────────────────────────────────────────────────────────
#  Phase 1: train recon network with (blurred, image) pairs
#            blurred -> recon_net -> x_hat, loss = MSE(x_hat, image)
#  Phase 2: train classifier on reconstructed images
#            blurred -> recon_net -> task_net -> logits, loss = CE
#  At inference: blurred -> recon_net -> task_net

def train_sequential(config, physics, train_loader, val_loader):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "sequential")
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    # ---- Phase 1: Reconstruction (A†(y) -> image) ----
    print("\n" + "=" * 60)
    print("Sequential - Phase 1: Reconstruction (A†(y) -> image)")
    print("=" * 60)

    recon_net = ReconstructionNet(
        in_channels=config.n_channels, channels=config.recon_channels,
    ).to(device)

    recon_ckpt_path = os.path.join(ckpt_dir, "recon_best.pt")
    # if os.path.isfile(recon_ckpt_path):
    #     # Skip training — load existing checkpoint
    #     recon_ckpt = torch.load(recon_ckpt_path, map_location=device)
    #     recon_net.load_state_dict(recon_ckpt["model_state_dict"])
    #     print(f"  [info] Loaded existing recon checkpoint from {recon_ckpt_path}, skipping Phase 1 training.")
    # else:
    recon_opt = optim.Adam(recon_net.parameters(), lr=config.recon_lr)

    best_val = float("inf")
    for ep in range(1, config.recon_epochs + 1):
        recon_net.train()
        train_loss = AverageMeter()
        for blurred, image, label in train_loader:
            blurred, image = blurred.to(device), image.to(device)
            a_dag = physics.A_dagger(blurred)
            x_hat = recon_net(a_dag)
            loss = mse(x_hat, image)
            recon_opt.zero_grad()
            loss.backward()
            recon_opt.step()
            train_loss.update(loss.item(), blurred.size(0))

        recon_net.eval()
        val_loss = AverageMeter()
        with torch.no_grad():
            for blurred, image, label in val_loader:
                blurred, image = blurred.to(device), image.to(device)
                a_dag = physics.A_dagger(blurred)
                x_hat = recon_net(a_dag)
                val_loss.update(mse(x_hat, image).item(), blurred.size(0))

        print(f"  [Recon] Epoch {ep}/{config.recon_epochs}  "
                f"train={train_loss.avg:.6f}  val={val_loss.avg:.6f}")
        if val_loss.avg < best_val:
            best_val = val_loss.avg
            save_checkpoint(recon_net, recon_ckpt_path,
                            optimizer=recon_opt, epoch=ep)

    # ---- Phase 2: Classifier on reconstructed images ----
    print("\n" + "=" * 60)
    print("Sequential - Phase 2: Classifier (on reconstructed images)")
    print("=" * 60)

    recon_net.eval()  

    task_net = TaskNet(
        in_channels=config.n_channels, channels=config.task_channels,
        num_classes=config.num_classes, img_size=config.img_size
    ).to(device)

    # Load pre-trained upper bound classifier as initialization
    ub_path = os.path.join(config.save_dir, "upper_bound", "upper_best.pt")
    if os.path.isfile(ub_path):
        ub_ckpt = torch.load(ub_path, map_location=device)
        task_net.load_state_dict(ub_ckpt["model_state_dict"])
        print(f"  [info] Loaded upper bound weights from {ub_path}")
    else:
        print(f"  [warn] Upper bound checkpoint not found at {ub_path}, training from scratch")

    task_opt = optim.Adam(task_net.parameters(), lr=config.task_lr)

    best_acc = 0.0
    for ep in range(1, config.task_epochs + 1):
        task_net.train()
        tr_loss, tr_acc = AverageMeter(), AverageMeter()
        for blurred, image, label in train_loader:
            blurred, label = blurred.to(device), label.to(device)
            with torch.no_grad():
                a_dag = physics.A_dagger(blurred)
                img_recon = recon_net(a_dag)
            logits = task_net(img_recon)          
            loss = ce(logits, label)
            task_opt.zero_grad()
            loss.backward()
            task_opt.step()
            tr_loss.update(loss.item(), label.size(0))
            tr_acc.update(_accuracy(logits, label), label.size(0))

        task_net.eval()
        va_loss, va_acc = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for blurred, image, label in val_loader:
                blurred, label = blurred.to(device), label.to(device)
                a_dag = physics.A_dagger(blurred)
                img_recon = recon_net(a_dag)
                logits = task_net(img_recon)
                va_loss.update(ce(logits, label).item(), label.size(0))
                va_acc.update(_accuracy(logits, label), label.size(0))

        print(f"  [Task]  Epoch {ep}/{config.task_epochs}  "
              f"loss={tr_loss.avg:.4f}  acc={tr_acc.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_acc={va_acc.avg:.1f}%")
        if va_acc.avg > best_acc:
            best_acc = va_acc.avg
            save_checkpoint(task_net, os.path.join(ckpt_dir, "task_best.pt"),
                            optimizer=task_opt, epoch=ep)

    print("\n[Sequential] Done.")
    return recon_net, task_net


# 2. End-to-End  (C = 0 in the joint loss)
#  Pre-trained recon_net (from sequential Phase 1) + task_net,
#  then fine-tuned end-to-end with only CE loss (gradients flow through both).

def train_end_to_end(config, physics, train_loader, val_loader):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "end_to_end")
    ce = nn.CrossEntropyLoss()

    # ---- Build recon_net (pre-trained from sequential Phase 1) ----
    recon_net = ReconstructionNet(
        in_channels=config.n_channels, channels=config.recon_channels,
    ).to(device)

    recon_ckpt_path = os.path.join(config.save_dir, "sequential", "recon_best.pt")
    if os.path.isfile(recon_ckpt_path):
        ckpt = torch.load(recon_ckpt_path, map_location=device)
        recon_net.load_state_dict(ckpt["model_state_dict"])
        print(f"  [info] Loaded pre-trained recon_net from {recon_ckpt_path}")
    else:
        print(f"  [warn] Recon checkpoint not found at {recon_ckpt_path}, training from scratch")

    # ---- Build task_net (initialised from upper-bound classifier) ----
    task_net = TaskNet(
        in_channels=config.n_channels, channels=config.task_channels,
        num_classes=config.num_classes, img_size=config.img_size,
    ).to(device)

    ub_path = os.path.join(config.save_dir, "upper_bound", "upper_best.pt")
    if os.path.isfile(ub_path):
        ckpt = torch.load(ub_path, map_location=device)
        task_net.load_state_dict(ckpt["model_state_dict"])
        print(f"  [info] Loaded upper bound task_net from {ub_path}")
    else:
        print(f"  [warn] Upper bound checkpoint not found at {ub_path}, training from scratch")

    print("\n" + "=" * 60)
    print("End-to-End Training  (C=0: CE loss only, gradients through recon+task)")
    print("=" * 60)

    # Optimise both networks jointly
    optimizer = optim.Adam(
        list(recon_net.parameters()) + list(task_net.parameters()),
        lr=config.task_lr,
    )

    best_acc = 0.0
    for ep in range(1, config.task_epochs + 1):
        recon_net.train()
        task_net.train()
        tr_loss, tr_acc = AverageMeter(), AverageMeter()
        for blurred, image, label in train_loader:
            blurred, label = blurred.to(device), label.to(device)
            a_dag = physics.A_dagger(blurred)
            x_hat = recon_net(a_dag)
            logits = task_net(x_hat)
            loss = ce(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss.update(loss.item(), label.size(0))
            tr_acc.update(_accuracy(logits, label), label.size(0))

        recon_net.eval()
        task_net.eval()
        va_loss, va_acc = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for blurred, image, label in val_loader:
                blurred, label = blurred.to(device), label.to(device)
                a_dag = physics.A_dagger(blurred)
                x_hat = recon_net(a_dag)
                logits = task_net(x_hat)
                va_loss.update(ce(logits, label).item(), label.size(0))
                va_acc.update(_accuracy(logits, label), label.size(0))

        print(f"  [E2E]   Epoch {ep}/{config.task_epochs}  "
              f"loss={tr_loss.avg:.4f}  acc={tr_acc.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_acc={va_acc.avg:.1f}%")
        if va_acc.avg > best_acc:
            best_acc = va_acc.avg
            save_checkpoint(recon_net, os.path.join(ckpt_dir, "recon_best.pt"),
                            optimizer=optimizer, epoch=ep)
            save_checkpoint(task_net, os.path.join(ckpt_dir, "task_best.pt"),
                            optimizer=optimizer, epoch=ep)

    print("\n[End-to-End] Done.")
    return recon_net, task_net


# ── 3. Joint ─────────────────────────────────────────────────────────────────
#  blurred -> JointModel -> (x_hat, logits)
#  loss = c * MSE(x_hat, image) + (1-c) * CE(logits, label)

def train_joint(config, physics, train_loader, val_loader):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "joint")
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    c = config.c

    model = JointModel(physics=physics, config=config).to(device)

    print("\n" + "=" * 60)
    print("Joint Training")
    print("=" * 60)

    optimizer = optim.Adam(model.parameters(), lr=config.joint_lr)

    # ---- Resume from previous joint checkpoint if available ----
    joint_ckpt_path = os.path.join(ckpt_dir, "joint_best.pt")
    best_acc = 0.0

    # Always train from scratch (or from sequential weights)
    _try_load_sequential_weights(model, config, device)

    for ep in range(1, config.joint_epochs + 1):
        model.train()
        tr_loss, tr_acc = AverageMeter(), AverageMeter()
        for blurred, image, label in train_loader:
            blurred, image, label = blurred.to(device), image.to(device), label.to(device)
            x_hat, logits = model(blurred)
            loss = c * mse(x_hat, image) + (1 - c) * ce(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss.update(loss.item(), label.size(0))
            tr_acc.update(_accuracy(logits, label), label.size(0))

        model.eval()
        va_loss, va_acc = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for blurred, image, label in val_loader:
                blurred, image, label = blurred.to(device), image.to(device), label.to(device)
                x_hat, logits = model(blurred)
                loss = c * mse(x_hat, image) + (1 - c) * ce(logits, label)
                va_loss.update(loss.item(), label.size(0))
                va_acc.update(_accuracy(logits, label), label.size(0))

        print(f"  [Joint] Epoch {ep}/{config.joint_epochs}  "
              f"loss={tr_loss.avg:.4f}  acc={tr_acc.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_acc={va_acc.avg:.1f}%")
        if va_acc.avg > best_acc:
            best_acc = va_acc.avg
            save_checkpoint(model, os.path.join(ckpt_dir, "joint_best.pt"),
                            optimizer=optimizer, epoch=ep)

    print("\n[Joint] Done.")
    return model


# 4. Upper Bound 

def train_upper_bound(config, physics, train_loader, val_loader):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "upper_bound")
    ce = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("Upper Bound - Classifier on Clean Images")
    print("=" * 60)

    task_net = TaskNet(
        in_channels=config.n_channels, channels=config.task_channels,
        num_classes=config.num_classes, img_size=config.img_size
    ).to(device)
    optimizer = optim.Adam(task_net.parameters(), lr=config.task_lr)

    best_acc = 0.0
    for ep in range(1, config.task_epochs + 1):
        task_net.train()
        tr_loss, tr_acc = AverageMeter(), AverageMeter()
        for blurred, image, label in train_loader:
            image, label = image.to(device), label.to(device)
            logits = task_net(image)
            loss = ce(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss.update(loss.item(), label.size(0))
            tr_acc.update(_accuracy(logits, label), label.size(0))

        task_net.eval()
        va_loss, va_acc = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for blurred, image, label in val_loader:
                image, label = image.to(device), label.to(device)
                logits = task_net(image)
                va_loss.update(ce(logits, label).item(), label.size(0))
                va_acc.update(_accuracy(logits, label), label.size(0))

        print(f"  [Upper] Epoch {ep}/{config.task_epochs}  "
              f"loss={tr_loss.avg:.4f}  acc={tr_acc.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_acc={va_acc.avg:.1f}%")
        if va_acc.avg > best_acc:
            best_acc = va_acc.avg
            save_checkpoint(task_net, os.path.join(ckpt_dir, "upper_best.pt"),
                            optimizer=optimizer, epoch=ep)

    print("\n[Upper Bound] Done.")
    return task_net


# 5. Lower Bound
#  Step 1: Train classifier on clean images (same as upper bound)
#  Step 2: At evaluation, pass blurred -> A_dagger -> trained task_net

def train_lower_bound(config, physics, train_loader, val_loader):
    device = config.device
    ckpt_dir = os.path.join(config.save_dir, "lower_bound")
    ce = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print("Lower Bound - Train classifier on clean images")
    print("=" * 60)

    task_net = TaskNet(
        in_channels=config.n_channels, channels=config.task_channels,
        num_classes=config.num_classes, img_size=config.img_size
    ).to(device)
    optimizer = optim.Adam(task_net.parameters(), lr=config.task_lr)

    best_acc = 0.0
    for ep in range(1, config.task_epochs + 1):
        task_net.train()
        tr_loss, tr_acc = AverageMeter(), AverageMeter()
        for blurred, image, label in train_loader:
            image, label = image.to(device), label.to(device)
            logits = task_net(image)
            loss = ce(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss.update(loss.item(), label.size(0))
            tr_acc.update(_accuracy(logits, label), label.size(0))

        task_net.eval()
        va_loss, va_acc = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for blurred, image, label in val_loader:
                image, label = image.to(device), label.to(device)
                logits = task_net(image)
                va_loss.update(ce(logits, label).item(), label.size(0))
                va_acc.update(_accuracy(logits, label), label.size(0))

        print(f"  [Lower] Epoch {ep}/{config.task_epochs}  "
              f"loss={tr_loss.avg:.4f}  acc={tr_acc.avg:.1f}%  "
              f"val_loss={va_loss.avg:.4f}  val_acc={va_acc.avg:.1f}%")
        if va_acc.avg > best_acc:
            best_acc = va_acc.avg
            save_checkpoint(task_net, os.path.join(ckpt_dir, "lower_best.pt"),
                            optimizer=optimizer, epoch=ep)

    print("\n[Lower Bound] Done.")
    return task_net


# ── Helper ────────────────────────────────────────────────────────────────────

def _try_load_sequential_weights(model, config, device):
    recon_path = os.path.join(config.save_dir, "sequential", "recon_best.pt")
    task_path = os.path.join(config.save_dir, "sequential", "task_best.pt")
    if os.path.isfile(recon_path) and os.path.isfile(task_path):
        print("[info] Loading pre-trained sequential weights...")
        model.recon_net.load_state_dict(
            torch.load(recon_path, map_location=device)["model_state_dict"]
        )
        model.task_net.load_state_dict(
            torch.load(task_path, map_location=device)["model_state_dict"]
        )
