import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = "/mnt/hdd/mallika/_aa/results_medical"
ROOT_DIR2 = "/mnt/hdd/mallika/_aa"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "task_adapted_recon_medical"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "exp_tv"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "exp_foe"))

# ============================================================
#  SWITCH MODALITY HERE: "CT" or "MRI"
# ============================================================
MODALITY = "MRI"   # <-- Change to "CT" for spleen

class Config:
    NUM_SAMPLES = 5
    IMG_SIZE = 128
    MODALITY = MODALITY
    SEED = 42
    NOISE_SIGMA = 0.5
    CENTER_FRAC = 0.08
    INNER_STEPS = 500
    INNER_LR = 0.1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODALITY == "CT":
        ACCEL = 6
        DATA_ROOT = os.path.join(SCRIPT_DIR, "ct_data")
        TASK = "Task09_Spleen"
        SAMPLE_PREFIX = "spleen"
    else:  # MRI
        ACCEL = 4
        DATA_ROOT = os.path.join(SCRIPT_DIR, "mri_data")
        TASK = "Task02_Heart"
        SAMPLE_PREFIX = "heart"

    # --- Baseline checkpoints (from task_adapted_recon_medical) ---
    SEQ_RECON_CKPT = os.path.join(ROOT_DIR, "sequential", "recon_best.pt")
    SEQ_SEG_CKPT = os.path.join(ROOT_DIR, "sequential", "seg_best.pt")
    E2E_RECON_CKPT = os.path.join(ROOT_DIR, "end_to_end", "recon_best.pt")
    E2E_SEG_CKPT = os.path.join(ROOT_DIR, "end_to_end", "seg_best.pt")
    JOINT_CKPT = os.path.join(ROOT_DIR, "joint", "joint_best.pt")
    UPPER_SEG_CKPT = os.path.join(ROOT_DIR, "upper_bound", "upper_best.pt")

    # --- HOAG TV checkpoints ---
    TV_DIR = os.path.join(ROOT_DIR2, f"results_hoag_tv_{MODALITY}_{TASK}")
    TV_A1_THETA = os.path.join(TV_DIR, "hoag_theta.pth")
    TV_A1_MODEL = os.path.join(TV_DIR, "model_upper_clean.pth")
    TV_A2_MODEL = os.path.join(TV_DIR, "model_joint.pth")
    TV_A2_THETA = os.path.join(TV_DIR, "theta_joint.pth")

    # --- HOAG FoE checkpoints ---
    FOE_DIR = os.path.join(ROOT_DIR2, f"results_hoag_foe_{MODALITY}_{TASK}")
    FOE_A1_THETA = os.path.join(FOE_DIR, "hoag_theta_foe.pth")
    FOE_A1_MODEL = os.path.join(FOE_DIR, "model_upper_clean.pth")
    FOE_A2_MODEL = os.path.join(FOE_DIR, "model_joint_foe.pth")
    FOE_A2_THETA = os.path.join(FOE_DIR, "theta_joint_foe.pth")

    OUTPUT_DIR = os.path.join(ROOT_DIR2, f"visualization_medical_{MODALITY}")

# ============================================================
#  Normalization helpers
# ============================================================
def norm_ct(img):
    """CT HU windowing normalization."""
    img = torch.clamp(img, min=-150, max=250)
    img = (img + 150) / 400.0
    return img

def norm_z_score(img):
    """Z-score normalization (for MRI)."""
    mean = img.mean()
    std = img.std()
    if std > 0:
        img = (img - mean) / std
    else:
        img = torch.zeros_like(img)
    return img

def to_magnitude(x):
    """Extract 1-ch magnitude from 2-ch complex tensor (B,2,H,W) -> (B,1,H,W)."""
    return torch.sqrt(x[:, 0:1, :, :]**2 + x[:, 1:2, :, :]**2)

def apply_norm(img, modality):
    if modality == "CT":
        return norm_ct(img)
    else:
        return norm_z_score(img)

def to_display(tensor):
    """Min-max normalize a tensor to [0,1] for saving as image."""
    t = tensor.clone()
    tmin, tmax = t.min(), t.max()
    if tmax - tmin > 1e-7:
        t = (t - tmin) / (tmax - tmin)
    else:
        t = torch.zeros_like(t)
    return t

# ============================================================
#  Utilities
# ============================================================
def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt

def build_physics(img_size, accel, center_frac, device, modality):
    import deepinv as dinv
    if modality == "CT":
        num_views = int(180 / accel)
        angles = torch.linspace(0, 180, num_views).to(device)
        return dinv.physics.Tomography(
            angles=angles, img_width=img_size,
            circle=False, device=device, normalize=True
        )
    else:  # MRI
        mask = torch.zeros((1, img_size, img_size))
        pad = (img_size - int(img_size * center_frac) + 1) // 2
        width = max(1, int(img_size * center_frac))
        mask[:, :, pad:pad + width] = 1.0
        num_keep = int(img_size / accel)
        all_cols = np.arange(img_size)
        kept_cols = np.where(mask[0, 0, :].cpu().numpy() == 1)[0]
        zero_cols = np.setdiff1d(all_cols, kept_cols)
        if len(zero_cols) > 0 and (num_keep - len(kept_cols) > 0):
            chosen = np.random.choice(zero_cols, num_keep - len(kept_cols), replace=False)
            mask[:, :, chosen] = 1.0
        mask = mask.to(device)
        return dinv.physics.MRI(
            mask=mask, img_size=(1, img_size, img_size),
            device=device, normalize=True
        )

def load_test_slices(data_root, task, img_size, num_samples, seed=42):
    task_dir = os.path.join(data_root, task)
    if os.path.isdir(os.path.join(task_dir, task)):
        task_dir = os.path.join(task_dir, task)
    img_dir = os.path.join(task_dir, "imagesTr")
    mask_dir = os.path.join(task_dir, "labelsTr")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    if not img_files:
        raise FileNotFoundError(f"No NIfTI files in {img_dir}")
    random.seed(seed)
    slices = []
    for f in img_files:
        basename = os.path.basename(f)
        mask_path = os.path.join(mask_dir, basename)
        if not os.path.exists(mask_path):
            continue
        img_vol = nib.load(f).get_fdata().astype(np.float32)
        mask_vol = nib.load(mask_path).get_fdata().astype(np.float32)
        for s in range(img_vol.shape[2]):
            if mask_vol[:, :, s].sum() > 100:
                img_slice = img_vol[:, :, s]
                mask_slice = mask_vol[:, :, s]
                img_t = torch.from_numpy(img_slice).unsqueeze(0).unsqueeze(0).float()
                mask_t = torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0).float()
                img_t = F.interpolate(img_t, size=(img_size, img_size), mode='bilinear', align_corners=False)
                mask_t = F.interpolate(mask_t, size=(img_size, img_size), mode='nearest')
                slices.append((img_t.squeeze(0), mask_t.squeeze(0)))
        if len(slices) >= num_samples * 3:
            break
    random.shuffle(slices)
    return slices[:num_samples]

def tv_inner_solve(w_init, theta, y, physics_op, steps, lr):
    from exp_tv.physics import inner_loss_func as tv_inner_loss
    w = w_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = tv_inner_loss(w, theta, y, physics_op)
        loss.backward()
        opt.step()
        scheduler.step()
    return w.detach()

def foe_inner_solve(w_init, theta, y, physics_op, steps, lr):
    from exp_foe.physics import inner_loss_func as foe_inner_loss
    w = w_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = foe_inner_loss(w, theta, y, physics_op)
        loss.backward()
        opt.step()
        scheduler.step()
    return w.detach()

def save_image(tensor, path, cmap='gray'):
    img = tensor.squeeze().cpu().numpy()
    img = np.clip(img, 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_comparison_strip(recon_dict, mask_dict, gt_mask, path):
    names = list(recon_dict.keys())
    n = len(names) + 2
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5), dpi=150)
    col = 0
    for name in ["x (clean)"] + [k for k in names if k != "x (clean)"]:
        if name not in recon_dict:
            continue
        img = recon_dict[name].squeeze().cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, col].axis('off')
        if name == "x (clean)":
            m = gt_mask.squeeze().cpu().numpy()
        elif name in mask_dict:
            m = mask_dict[name].squeeze().cpu().numpy()
        else:
            m = np.zeros_like(img)
        m = np.clip(m, 0, 1)
        axes[1, col].imshow(m, cmap='gray', vmin=0, vmax=1)
        axes[1, col].axis('off')
        col += 1
    for c in range(col, n):
        axes[0, c].axis('off')
        axes[1, c].axis('off')
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ============================================================
#  MAIN
# ============================================================
def main():
    torch.manual_seed(Config.SEED)
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    device = Config.DEVICE
    modality = Config.MODALITY
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"=== Medical {modality} Visualization ({Config.TASK}) ===")
    print(f"    Device: {device}")
    print(f"    Output: {Config.OUTPUT_DIR}")

    physics = build_physics(Config.IMG_SIZE, Config.ACCEL, Config.CENTER_FRAC, device, modality)
    slices = load_test_slices(Config.DATA_ROOT, Config.TASK, Config.IMG_SIZE, Config.NUM_SAMPLES)
    print(f"    Loaded {len(slices)} test slices")

    # ---- Load baseline models (from task_adapted_recon_medical) ----
    from task_adapted_recon_medical.model import ReconstructionNet, SegmentationNet, JointModel
    from task_adapted_recon_medical.config import Config as MedConfig
    med_config = MedConfig(modality=modality.lower())
    recon_ch = med_config.recon_channels
    task_ch = med_config.task_channels

    # Sequential
    recon_seq = ReconstructionNet(in_channels=1, channels=recon_ch).to(device)
    seg_seq = SegmentationNet(in_channels=1, channels=task_ch).to(device)
    if os.path.exists(Config.SEQ_RECON_CKPT):
        recon_seq.load_state_dict(load_ckpt(Config.SEQ_RECON_CKPT, device))
    if os.path.exists(Config.SEQ_SEG_CKPT):
        seg_seq.load_state_dict(load_ckpt(Config.SEQ_SEG_CKPT, device))
    recon_seq.eval(); seg_seq.eval()

    # End-to-End
    recon_e2e = ReconstructionNet(in_channels=1, channels=recon_ch).to(device)
    seg_e2e = SegmentationNet(in_channels=1, channels=task_ch).to(device)
    if os.path.exists(Config.E2E_RECON_CKPT):
        recon_e2e.load_state_dict(load_ckpt(Config.E2E_RECON_CKPT, device))
    if os.path.exists(Config.E2E_SEG_CKPT):
        seg_e2e.load_state_dict(load_ckpt(Config.E2E_SEG_CKPT, device))
    recon_e2e.eval(); seg_e2e.eval()

    # Joint
    joint_model = JointModel(physics=physics, config=med_config).to(device)
    if os.path.exists(Config.JOINT_CKPT):
        joint_model.load_state_dict(load_ckpt(Config.JOINT_CKPT, device), strict=False)
    joint_model.eval()

    # ---- Load HOAG TV models ----
    from exp_tv.models import UNet as TVUNet
    tv_a1_seg = TVUNet().to(device)
    if os.path.exists(Config.TV_A1_MODEL):
        tv_a1_seg.load_state_dict(load_ckpt(Config.TV_A1_MODEL, device))
    tv_a1_seg.eval()
    theta_tv_a1 = None
    if os.path.exists(Config.TV_A1_THETA):
        d = torch.load(Config.TV_A1_THETA, map_location=device)
        theta_tv_a1 = d['theta'].to(device)

    tv_a2_seg = TVUNet().to(device)
    if os.path.exists(Config.TV_A2_MODEL):
        tv_a2_seg.load_state_dict(load_ckpt(Config.TV_A2_MODEL, device))
    tv_a2_seg.eval()
    theta_tv_a2 = None
    if os.path.exists(Config.TV_A2_THETA):
        d = torch.load(Config.TV_A2_THETA, map_location=device)
        theta_tv_a2 = d['theta'].to(device)

    # ---- Load HOAG FoE models ----
    from exp_foe.models import UNet as FoEUNet
    foe_a1_seg = FoEUNet().to(device)
    if os.path.exists(Config.FOE_A1_MODEL):
        foe_a1_seg.load_state_dict(load_ckpt(Config.FOE_A1_MODEL, device))
    foe_a1_seg.eval()
    theta_foe_a1 = None
    if os.path.exists(Config.FOE_A1_THETA):
        d = torch.load(Config.FOE_A1_THETA, map_location=device)
        theta_foe_a1 = d['theta'].to(device)

    foe_a2_seg = FoEUNet().to(device)
    if os.path.exists(Config.FOE_A2_MODEL):
        foe_a2_seg.load_state_dict(load_ckpt(Config.FOE_A2_MODEL, device))
    foe_a2_seg.eval()
    theta_foe_a2 = None
    if os.path.exists(Config.FOE_A2_THETA):
        d = torch.load(Config.FOE_A2_THETA, map_location=device)
        theta_foe_a2 = d['theta'].to(device)

    # ============================================================
    #  Generate visualizations
    # ============================================================
    print(f"\nGenerating visualizations...")
    for idx, (img, gt_mask) in enumerate(slices):
        img = img.to(device).unsqueeze(0)       # (1, 1, H, W)
        gt_mask = gt_mask.to(device).unsqueeze(0)

        with torch.no_grad():
            if modality == "MRI":
                # MRI: create 2-ch complex input for physics
                imaginary_part = torch.zeros_like(img)
                complex_input = torch.cat([img, imaginary_part], dim=1)  # (1,2,H,W)
                y_clean = physics(complex_input)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                a_dag = physics.A_dagger(y)                              # (1,2,H,W)
                a_dag_mag = to_magnitude(a_dag)                          # (1,1,H,W)
                a_dag_normed = norm_z_score(a_dag_mag)
                # For visualization: normalize clean image & a_dag to [0,1]
                img_vis = norm_z_score(img).clamp(0, 1)
            else:
                # CT: straightforward
                y_clean = physics(img)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                a_dag = physics.A_dagger(y)
                a_dag_normed = norm_ct(a_dag)
                img_vis = norm_ct(img)

        recon_dict = {}
        mask_dict = {}
        prefix = f"{Config.SAMPLE_PREFIX}_sample{idx}"
        if modality == "CT":
            recon_dict["x (clean)"] = img_vis  # already [0,1] from norm_ct
        else:
            recon_dict["x (clean)"] = to_display(img)  # min-max for display

        # Save sinogram (y) — resize for display
        y_vis = y.clone()
        if modality == "MRI":
            # MRI k-space: take magnitude and apply log scale for better visualization
            y_vis = to_magnitude(y_vis)
            y_vis = torch.log1p(y_vis)
        y_vis = F.interpolate(y_vis, size=(Config.IMG_SIZE, Config.IMG_SIZE), mode='bilinear', align_corners=False)
        y_vis = to_display(y_vis)
        save_image(y_vis, os.path.join(Config.OUTPUT_DIR, f"{prefix}_y_sinogram.png"))

        if modality == "CT":
            recon_dict["A_dag(y)"] = a_dag_normed.clamp(0, 1)
        else:
            recon_dict["A_dag(y)"] = to_display(a_dag_normed)

        # --- Sequential ---
        with torch.no_grad():
            x_seq = recon_seq(a_dag_normed)  # NO clamp — model expects z-score range for MRI
            m_seq = seg_seq(x_seq)
        recon_dict["Sequential"] = to_display(x_seq) if modality == "MRI" else x_seq.clamp(0, 1)
        mask_dict["Sequential"] = (m_seq > 0.5).float()

        # --- End-to-End ---
        with torch.no_grad():
            x_e2e = recon_e2e(a_dag_normed)  # NO clamp
            m_e2e = seg_e2e(x_e2e)
        recon_dict["End-to-End"] = to_display(x_e2e) if modality == "MRI" else x_e2e.clamp(0, 1)
        mask_dict["End-to-End"] = (m_e2e > 0.5).float()

        # --- Joint ---
        with torch.no_grad():
            x_joint, m_joint = joint_model(y)
        recon_dict["Joint"] = to_display(x_joint) if modality == "MRI" else x_joint.clamp(0, 1)
        mask_dict["Joint"] = (m_joint > 0.5).float()

        # --- Approach 1 (TV) ---
        if theta_tv_a1 is not None:
            x_tv_a1 = tv_inner_solve(a_dag, theta_tv_a1, y, physics, Config.INNER_STEPS, Config.INNER_LR)
            if modality == "MRI":
                x_tv_a1_n = norm_z_score(to_magnitude(x_tv_a1))  # NO clamp for model
            else:
                x_tv_a1_n = norm_ct(x_tv_a1)
            with torch.no_grad():
                m_tv_a1 = tv_a1_seg(x_tv_a1_n)
            recon_dict["A1 (TV)"] = to_display(x_tv_a1_n) if modality == "MRI" else x_tv_a1_n.clamp(0, 1)
            mask_dict["A1 (TV)"] = (m_tv_a1 > 0.5).float()

        # --- Approach 2 (TV) ---
        if theta_tv_a2 is not None:
            x_tv_a2 = tv_inner_solve(a_dag, theta_tv_a2, y, physics, Config.INNER_STEPS, Config.INNER_LR)
            if modality == "MRI":
                x_tv_a2_n = norm_z_score(to_magnitude(x_tv_a2))
            else:
                x_tv_a2_n = norm_ct(x_tv_a2)
            with torch.no_grad():
                m_tv_a2 = tv_a2_seg(x_tv_a2_n)
            recon_dict["A2 (TV)"] = to_display(x_tv_a2_n) if modality == "MRI" else x_tv_a2_n.clamp(0, 1)
            mask_dict["A2 (TV)"] = (m_tv_a2 > 0.5).float()

        # --- Approach 1 (FoE) ---
        if theta_foe_a1 is not None:
            x_foe_a1 = foe_inner_solve(a_dag, theta_foe_a1, y, physics, Config.INNER_STEPS, Config.INNER_LR)
            if modality == "MRI":
                x_foe_a1_n = norm_z_score(to_magnitude(x_foe_a1))
            else:
                x_foe_a1_n = norm_ct(x_foe_a1)
            with torch.no_grad():
                m_foe_a1 = foe_a1_seg(x_foe_a1_n)
            recon_dict["A1 (FoE)"] = to_display(x_foe_a1_n) if modality == "MRI" else x_foe_a1_n.clamp(0, 1)
            mask_dict["A1 (FoE)"] = (m_foe_a1 > 0.5).float()

        # --- Approach 2 (FoE) ---
        if theta_foe_a2 is not None:
            x_foe_a2 = foe_inner_solve(a_dag, theta_foe_a2, y, physics, Config.INNER_STEPS, Config.INNER_LR)
            if modality == "MRI":
                x_foe_a2_n = norm_z_score(to_magnitude(x_foe_a2))
            else:
                x_foe_a2_n = norm_ct(x_foe_a2)
            with torch.no_grad():
                m_foe_a2 = foe_a2_seg(x_foe_a2_n)
            recon_dict["A2 (FoE)"] = to_display(x_foe_a2_n) if modality == "MRI" else x_foe_a2_n.clamp(0, 1)
            mask_dict["A2 (FoE)"] = (m_foe_a2 > 0.5).float()

        # --- Save individual images ---
        for name, tensor in recon_dict.items():
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            save_image(tensor, os.path.join(Config.OUTPUT_DIR, f"{prefix}_{safe}_recon.png"))
        save_image(gt_mask, os.path.join(Config.OUTPUT_DIR, f"{prefix}_gt_mask.png"))
        for name, tensor in mask_dict.items():
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            save_image(tensor, os.path.join(Config.OUTPUT_DIR, f"{prefix}_{safe}_mask.png"))

        # --- Save comparison strip ---
        strip_path = os.path.join(Config.OUTPUT_DIR, f"{prefix}_comparison.png")
        save_comparison_strip(recon_dict, mask_dict, gt_mask, strip_path)

        # --- Compute Dice scores ---
        gt_flat = gt_mask.flatten()
        dice_scores = {}
        for name, pred_m in mask_dict.items():
            pred_flat = pred_m.flatten()
            inter = (pred_flat * gt_flat).sum()
            union = pred_flat.sum() + gt_flat.sum()
            dice = (2. * inter + 1e-6) / (union + 1e-6)
            dice_scores[name] = dice.item()
        dice_str = " | ".join([f"{k}: {v:.3f}" for k, v in dice_scores.items()])
        print(f"  Sample {idx}: Dice = {dice_str}")

    print(f"\nAll visualizations saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()