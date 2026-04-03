"""
Visualization script for MNIST bilevel learning.

Generates reconstructed images for a selected digit label:
  x              - original clean image
  y              - blurred noisy measurement
  A_dagger_y     - pseudo-inverse (naive reconstruction)
  sequential     - sequential (recon then classify)
  end_to_end     - end-to-end trained recon+classifier
  joint          - joint-trained recon+classifier
  approach_1_foe - HOAG FoE, theta only (fixed classifier)
  approach_2_foe - HOAG FoE, joint theta+classifier

All images saved as PNG in ./visualization/
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ---- Add parent paths so we can import from sibling directories ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # source code root (for imports)
ROOT_DIR = "/mnt/hdd/mallika/_aa/results_mnist"      # checkpoints/results root
ROOT_DIR2 = "/mnt/hdd/mallika/_aa"      # checkpoints/results root

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "task_adapted_recon_mnist"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "mnist_foe"))


# ==========================================
#  CONFIG
# ==========================================
class Config:
    TARGET_LABEL = 7              # Which digit to visualize
    NUM_SAMPLES = 5               # How many samples of that digit
    IMG_SIZE = 28
    BLUR_SIGMA = 3.0
    NOISE_SIGMA = 0.01
    SEED = 42

    # Checkpoint paths (relative to mnist_bilevel_learning/)
    SEQ_RECON_CKPT   = os.path.join(ROOT_DIR, "sequential", "recon_best.pt")
    E2E_RECON_CKPT   = os.path.join(ROOT_DIR, "end_to_end", "recon_best.pt")
    JOINT_CKPT       = os.path.join(ROOT_DIR, "joint", "joint_best.pt")

    FOE_A1_THETA     = os.path.join(ROOT_DIR2, "results_mnist_foe", "hoag_theta_foe.pth")
    FOE_A2_MODEL     = os.path.join(ROOT_DIR2,"results_mnist_foe", "model_joint_foe.pth")
    FOE_A2_THETA     = os.path.join(ROOT_DIR2, "results_mnist_foe", "theta_joint_foe.pth")

    # FoE inner optimization
    FOE_INNER_STEPS = 400
    FOE_INNER_LR = 0.05

    DATA_ROOT = "../data_mnist"
    OUTPUT_DIR = "/mnt/hdd/mallika/_aa/visualization"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
#  HELPER FUNCTIONS
# ==========================================
def load_ckpt(path, device):
    """Load checkpoint, return state dict."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def build_physics(device):
    """Create the Gaussian blur physics operator."""
    import deepinv as dinv
    from deepinv.physics.blur import gaussian_blur
    kernel = gaussian_blur(sigma=(Config.BLUR_SIGMA, Config.BLUR_SIGMA))
    return dinv.physics.Blur(
        filter=kernel,
        padding="circular",
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=Config.NOISE_SIGMA),
    )


def foe_inner_solve(w_init, theta, y, physics_op, steps, lr):
    """Solve the FoE variational inner problem."""
    # Import FoE physics from mnist_foe
    foe_physics_path = os.path.join(SCRIPT_DIR, "mnist_foe")
    if foe_physics_path not in sys.path:
        sys.path.insert(0, foe_physics_path)
    from physics import inner_loss_func

    w = w_init.detach().clone().clamp(0, 1).requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)

    for _ in range(steps):
        opt.zero_grad()
        loss = inner_loss_func(w, theta, y, physics_op)
        loss.backward()
        opt.step()
        scheduler.step()

    return w.detach()


def save_image(tensor, path):
    """Save a single-channel tensor as a grayscale PNG (no title)."""
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=150)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_comparison_strip(images_dict, path):
    """Save all reconstructions as a single horizontal strip (no titles)."""
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.5), dpi=150)
    for ax, (name, tensor) in zip(axes, images_dict.items()):
        img = tensor.squeeze().cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    fig.subplots_adjust(wspace=0.02, hspace=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ==========================================
#  MAIN
# ==========================================
def main():
    torch.manual_seed(Config.SEED)
    device = Config.DEVICE
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"=== MNIST Visualization ===")
    print(f"    Target label: {Config.TARGET_LABEL}")
    print(f"    Device: {device}")

    # --- Build physics ---
    physics = build_physics(device)

    # --- Load MNIST test data ---
    ds = torchvision.datasets.MNIST(
        root=Config.DATA_ROOT, train=False, download=True,
        transform=transforms.ToTensor()
    )

    # Find samples with the target label
    indices = [i for i, (_, lbl) in enumerate(ds) if lbl == Config.TARGET_LABEL]
    indices = indices[:Config.NUM_SAMPLES]
    print(f"    Found {len(indices)} samples for label {Config.TARGET_LABEL}")

    # --- Load models ---
    # 1. Sequential: ReconstructionNet
    from task_adapted_recon_mnist.model import ReconstructionNet, JointModel
    from task_adapted_recon_mnist.config import Config as TARConfig

    tar_config = TARConfig()

    recon_seq = ReconstructionNet(in_channels=1).to(device)
    if os.path.exists(Config.SEQ_RECON_CKPT):
        recon_seq.load_state_dict(load_ckpt(Config.SEQ_RECON_CKPT, device))
        print(f"    Loaded sequential recon from {Config.SEQ_RECON_CKPT}")
    else:
        print(f"    WARNING: Sequential checkpoint not found: {Config.SEQ_RECON_CKPT}")
    recon_seq.eval()

    # 2. End-to-end: ReconstructionNet
    recon_e2e = ReconstructionNet(in_channels=1).to(device)
    if os.path.exists(Config.E2E_RECON_CKPT):
        recon_e2e.load_state_dict(load_ckpt(Config.E2E_RECON_CKPT, device))
        print(f"    Loaded e2e recon from {Config.E2E_RECON_CKPT}")
    else:
        print(f"    WARNING: E2E checkpoint not found: {Config.E2E_RECON_CKPT}")
    recon_e2e.eval()

    # 3. Joint: JointModel (recon + task)
    joint_model = JointModel(physics=physics, config=tar_config).to(device)
    if os.path.exists(Config.JOINT_CKPT):
        joint_model.load_state_dict(load_ckpt(Config.JOINT_CKPT, device))
        print(f"    Loaded joint model from {Config.JOINT_CKPT}")
    else:
        print(f"    WARNING: Joint checkpoint not found: {Config.JOINT_CKPT}")
    joint_model.eval()

    # 4. FoE Approach 1: theta only (uses upper bound classifier, we just need theta)
    theta_a1 = None
    if os.path.exists(Config.FOE_A1_THETA):
        a1_data = torch.load(Config.FOE_A1_THETA, map_location=device)
        theta_a1 = a1_data['theta'].to(device)
        print(f"    Loaded FoE A1 theta from {Config.FOE_A1_THETA}")
    else:
        print(f"    WARNING: FoE A1 theta not found: {Config.FOE_A1_THETA}")

    # 5. FoE Approach 2: joint theta + classifier
    theta_a2 = None
    if os.path.exists(Config.FOE_A2_THETA):
        a2_data = torch.load(Config.FOE_A2_THETA, map_location=device)
        theta_a2 = a2_data['theta'].to(device)
        print(f"    Loaded FoE A2 theta from {Config.FOE_A2_THETA}")
    else:
        print(f"    WARNING: FoE A2 theta not found: {Config.FOE_A2_THETA}")

    # --- Generate visualizations ---
    print(f"\nGenerating visualizations...")

    for sample_idx, ds_idx in enumerate(indices):
        image, label = ds[ds_idx]  # image: (1, 28, 28)
        x = image.unsqueeze(0).to(device)  # (1, 1, 28, 28)

        # Generate measurement
        with torch.no_grad():
            y = physics(x)
            a_dag = physics.A_dagger(y).clamp(0, 1)

        images_dict = {}

        # x - original
        images_dict["x (clean)"] = x

        # y - measurement (may not be image-shaped, visualize as best we can)
        images_dict["y (blurred)"] = y.clamp(0, 1)

        # A†(y) - pseudo-inverse
        images_dict["A†(y)"] = a_dag

        # Sequential reconstruction
        with torch.no_grad():
            x_seq = recon_seq(a_dag)
        images_dict["Sequential"] = x_seq.clamp(0, 1)

        # End-to-end reconstruction
        with torch.no_grad():
            x_e2e = recon_e2e(a_dag)
        images_dict["End-to-End"] = x_e2e.clamp(0, 1)

        # Joint reconstruction
        with torch.no_grad():
            x_joint, _ = joint_model(y)
        images_dict["Joint"] = x_joint.clamp(0, 1)

        # FoE Approach 1 reconstruction
        if theta_a1 is not None:
            x_foe_a1 = foe_inner_solve(a_dag, theta_a1, y, physics,
                                        Config.FOE_INNER_STEPS, Config.FOE_INNER_LR)
            images_dict["A1 (FoE)"] = x_foe_a1.clamp(0, 1)

        # FoE Approach 2 reconstruction
        if theta_a2 is not None:
            x_foe_a2 = foe_inner_solve(a_dag, theta_a2, y, physics,
                                        Config.FOE_INNER_STEPS, Config.FOE_INNER_LR)
            images_dict["A2 (FoE)"] = x_foe_a2.clamp(0, 1)

        # Save individual images
        prefix = f"label{label}_sample{sample_idx}"
        for name, tensor in images_dict.items():
            # Clean up the name so it's a valid filename
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("†", "dag").replace("/", "_")
            path = os.path.join(Config.OUTPUT_DIR, f"{prefix}_{safe_name}.png")
            
            # FIXED: Removed the title parameter
            save_image(tensor, path)

        # Save comparison strip
        strip_path = os.path.join(Config.OUTPUT_DIR, f"{prefix}_comparison.png")
        save_comparison_strip(images_dict, strip_path)

        print(f"  Sample {sample_idx}: saved {len(images_dict)} images + comparison strip")

    print(f"\nAll visualizations saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()