import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split, ConcatDataset

from models import TaskNet
from dataset import BlurMNISTDataset
from physics import build_physics, inner_loss_func

from hoag import HOAGState, hoag_step, solve_inner_problem


class Config:
    DATA_ROOT = "../../data_mnist"
    OUTPUT_DIR = "./results_mnist_tv"
    UPPER_BOUND_CKPT = "../../task_adapted_recon_mnist/results/upper_bound/upper_best.pt"

    SUBSET_SIZE = None
    SEED = 42
    IMG_SIZE = 28
    BATCH_SIZE = 256

    BLUR_SIGMA = 3
    NOISE_SIGMA = 0.01

    INNER_STEPS = 400
    INNER_LR = 0.05

    EPOCHS_HOAG = 15
    EPOCHS_JOINT = 15
    LR_MODEL = 1e-3
    LR_THETA = 0.05

    HOAG_EPSILON_TOL_INIT = 1e-3
    HOAG_TOLERANCE_DECREASE = 'exponential'
    HOAG_DECREASE_FACTOR = 0.9
    HOAG_CG_MAX_ITER = 50

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_progress(epoch, batch, total_batches, loss, theta, info=""):
    reg_val = torch.exp(theta[0]).item()
    eps_val = torch.exp(theta[1]).item()
    sys.stdout.write(f"\r[{info}] Ep {epoch+1} | Batch {batch+1}/{total_batches} | "
                     f"Loss: {loss:.4f} | Reg: {reg_val:.5f} | Smooth: {eps_val:.5f}")
    sys.stdout.flush()


def validate(model, val_loader, physics_op, theta=None, steps=0, mode="clean"):
    model.eval()
    correct = 0
    total = 0

    for y, img, label in val_loader:
        y, img, label = y.to(Config.DEVICE), img.to(Config.DEVICE), label.to(Config.DEVICE)

        if mode == "clean":
            x_in = img

        elif mode == "noisy":
            with torch.no_grad():
                x_recon = physics_op.A_dagger(y).clamp(0, 1)
                x_in = x_recon

        elif mode == "hoag":
            w = physics_op.A_dagger(y).detach().clone().clamp(0, 1)
            w.requires_grad_(True)
            optimizer_inner = torch.optim.Adam([w], lr=Config.INNER_LR)
            scheduler_inner = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_inner, T_max=steps, eta_min=Config.INNER_LR * 0.01)
            with torch.enable_grad():
                for _ in range(steps):
                    optimizer_inner.zero_grad()
                    loss = inner_loss_func(w, theta, y, physics_op)
                    loss.backward()
                    optimizer_inner.step()
                    scheduler_inner.step()
            x_recon = w.detach()
            x_in = x_recon

        with torch.no_grad():
            logits = model(x_in)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return correct / total if total > 0 else 0.0


def run_experiment():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Starting MNIST Experiment (HOAG + TV Regularizer)")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    physics = build_physics(Config.IMG_SIZE, Config.BLUR_SIGMA, Config.NOISE_SIGMA, Config.DEVICE)

    ds_train = BlurMNISTDataset(
        physics=physics,
        root_dir=Config.DATA_ROOT, train=True, img_size=Config.IMG_SIZE,
        subset_size=Config.SUBSET_SIZE, device=Config.DEVICE)
    ds_test = BlurMNISTDataset(
        physics=physics,
        root_dir=Config.DATA_ROOT, train=False, img_size=Config.IMG_SIZE,
        device=Config.DEVICE)

    full_ds = ConcatDataset([ds_train, ds_test])
    total = len(full_ds)
    ratio_sum = 75 + 15 + 15
    n_train = int(total * 75 / ratio_sum)
    n_val = int(total * 15 / ratio_sum)
    n_test = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(Config.SEED))

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    loss_fn = nn.CrossEntropyLoss()
    results = {}
    dummy_theta = torch.tensor([-10.0, -10.0])

    # PHASE 1: Upper Bound
    print("\n--- PHASE 1: Upper Bound ---")
    ckpt_path = Config.UPPER_BOUND_CKPT
    model_upper = TaskNet().to(Config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_upper.load_state_dict(ckpt["model_state_dict"])
    else:
        model_upper.load_state_dict(ckpt)
    model_upper.eval()
    print(f"Loaded upper bound from {ckpt_path}")

    results['Upper Bound'] = validate(model_upper, test_loader, physics, theta=dummy_theta, mode="clean")
    print(f"Upper Bound (Clean): {results['Upper Bound']:.4f}")

    # PHASE 2: Lower Bound
    print("\n--- PHASE 2: Lower Bound ---")
    results['Lower Bound'] = validate(model_upper, test_loader, physics, theta=dummy_theta, mode="noisy")
    print(f"Lower Bound (Noisy): {results['Lower Bound']:.4f}")

    # PHASE 3: Approach 1 (HOAG - Theta Only)
    print("\n--- PHASE 3: Approach 1 (HOAG - Optimizing Theta Only) ---")

    model_fixed = TaskNet().to(Config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_fixed.load_state_dict(ckpt["model_state_dict"])
    else:
        model_fixed.load_state_dict(ckpt)
    model_fixed.eval()
    for p in model_fixed.parameters():
        p.requires_grad = False

    theta = torch.tensor([-1.0, -4.0], device=Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)

    hoag_state = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )

    path_hoag = os.path.join(Config.OUTPUT_DIR, "hoag_theta.pth")

    for ep in range(Config.EPOCHS_HOAG):
        for i, (y, img, label) in enumerate(train_loader):
            y, label = y.to(Config.DEVICE), label.to(Config.DEVICE)

            hyper_grad, val_loss_value, w_star = hoag_step(
                theta=theta, y=y, physics_op=physics,
                model=model_fixed, loss_fn=loss_fn, label=label,
                inner_loss_fn=inner_loss_func, state=hoag_state,
                inner_lr=Config.INNER_LR, inner_steps=Config.INNER_STEPS,
                cg_max_iter=Config.HOAG_CG_MAX_ITER, verbose=0
            )

            opt_theta.zero_grad()
            theta.grad = hyper_grad
            torch.nn.utils.clip_grad_norm_([theta], max_norm=1.0)
            opt_theta.step()

            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 1 (HOAG)")

        hoag_state.decrease_tolerance()
        print(f"  [epsilon_tol: {hoag_state.epsilon_tol:.2e}]")
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state.epsilon_tol}, path_hoag)

    results['Approach 1'] = validate(model_fixed, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    print(f"Approach 1 Score: {results['Approach 1']:.4f}")

    # PHASE 4: Approach 2 (Joint Learning)
    print("\n--- PHASE 4: Approach 2 (Joint Learning - Theta + Classifier) ---")

    model_joint = TaskNet().to(Config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model_joint.load_state_dict(ckpt["model_state_dict"])
    else:
        model_joint.load_state_dict(ckpt)
    print(f"Classifier initialized from: {ckpt_path}")
    opt_model = torch.optim.Adam(model_joint.parameters(), lr=Config.LR_MODEL)

    theta = torch.tensor([-1.0, -4.0], device=Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)

    hoag_state_joint = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )

    path_joint = os.path.join(Config.OUTPUT_DIR, "model_joint.pth")
    path_theta_joint = os.path.join(Config.OUTPUT_DIR, "theta_joint.pth")

    for ep in range(Config.EPOCHS_JOINT):
        for i, (y, img, label) in enumerate(train_loader):
            y, label = y.to(Config.DEVICE), label.to(Config.DEVICE)

            w_star, _ = solve_inner_problem(
                w_init=physics.A_dagger(y).detach().clone().clamp(0, 1),
                theta=theta, y=y, physics_op=physics,
                inner_loss_fn=inner_loss_func, state=hoag_state_joint,
                lr=Config.INNER_LR, max_steps=Config.INNER_STEPS, verbose=0
            )

            w_fixed = w_star.detach().clone().requires_grad_(False)

            model_joint.train()
            opt_model.zero_grad()
            loss_model = loss_fn(model_joint(w_fixed), label)
            loss_model.backward()
            opt_model.step()

            model_joint.eval()

            hyper_grad, val_loss_value, w_star = hoag_step(
                theta=theta, y=y, physics_op=physics,
                model=model_joint, loss_fn=loss_fn, label=label,
                inner_loss_fn=inner_loss_func, state=hoag_state_joint,
                inner_lr=Config.INNER_LR, inner_steps=Config.INNER_STEPS,
                cg_max_iter=Config.HOAG_CG_MAX_ITER, verbose=0
            )

            opt_theta.zero_grad()
            theta.grad = hyper_grad
            torch.nn.utils.clip_grad_norm_([theta], max_norm=1.0)
            opt_theta.step()

            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 2 (Joint)")

        hoag_state_joint.decrease_tolerance()
        print(f"  [epsilon_tol: {hoag_state_joint.epsilon_tol:.2e}]")
        torch.save(model_joint.state_dict(), path_joint)
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state_joint.epsilon_tol}, path_theta_joint)

    results['Approach 2'] = validate(model_joint, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")

    print("\n=== FINAL RESULTS (MNIST + TV) ===")
    print(f"1. Upper Bound (Clean):  {results['Upper Bound']:.4f}")
    print(f"2. Lower Bound (Noisy):  {results['Lower Bound']:.4f}")
    print(f"3. Approach 1 (HOAG):    {results['Approach 1']:.4f}")
    print(f"4. Approach 2 (Joint):   {results['Approach 2']:.4f}")

    results_path = os.path.join(Config.OUTPUT_DIR, "final_results.txt")
    with open(results_path, "w") as f:
        f.write("=== FINAL RESULTS (MNIST + TV Regularizer) ===\n")
        f.write(f"Blur sigma: {Config.BLUR_SIGMA} | Noise: {Config.NOISE_SIGMA}\n")
        f.write(f"HOAG: tol_init={Config.HOAG_EPSILON_TOL_INIT}, "
                f"schedule={Config.HOAG_TOLERANCE_DECREASE}, "
                f"decay={Config.HOAG_DECREASE_FACTOR}\n")
        f.write(f"Epochs: HOAG={Config.EPOCHS_HOAG}, "
                f"Joint={Config.EPOCHS_JOINT} | Inner Steps: {Config.INNER_STEPS}\n\n")
        f.write(f"1. Upper Bound (Clean):      {results['Upper Bound']:.4f}\n")
        f.write(f"2. Lower Bound (Noisy):      {results['Lower Bound']:.4f}\n")
        f.write(f"3. Approach 1 (HOAG theta):  {results['Approach 1']:.4f}\n")
        f.write(f"4. Approach 2 (HOAG joint):  {results['Approach 2']:.4f}\n")
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    run_experiment()
