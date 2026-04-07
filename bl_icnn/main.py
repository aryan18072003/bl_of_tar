import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split

from models import UNet
from dataset import MSDDataset
from physics import get_physics_operator, inner_loss_func, regularizer_only
import convex_models
from convex_models import n_layers, n_filters, kernel_size

from hoag import HOAGState, hoag_step, solve_inner_problem


# ==========================================
#        1. CONFIGURATION
# ==========================================
class Config:

    DATA_ROOT = "../data_medical/ct_data"
    TASK = "Task09_Spleen"
    MODALITY = "CT"
    OUTPUT_DIR = f"./results_hoag_icnn_{MODALITY}_{TASK}"
    # Pre-trained ICNN regularizer checkpoints (from train_icnn.py)
    ICNN_MODEL_PATH = "./trained_models/"
    
    # Dataset Splits
    SUBSET_SIZE = None
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    IMG_SIZE = 128
    BATCH_SIZE = 16
    
    # --- SINGLE PHYSICS SETTING (SPARSE) ---
    ACCEL = 6
    NOISE_SIGMA = 0.5  
    CENTER_FRAC = 0.08
    
    # --- INNER OPTIMIZATION SETTINGS ---
    INNER_STEPS = 300   
    INNER_LR = 0.01     
    
    # --- OUTER OPTIMIZATION SETTINGS ---
    EPOCH_CLEAN = 50
    EPOCHS = 40               
    LR_UNET = 5e-3
    LR_THETA = 0.01           
    
    # --- HOAG-SPECIFIC SETTINGS ---
    HOAG_EPSILON_TOL_INIT = 1e-3
    HOAG_TOLERANCE_DECREASE = 'exponential'
    HOAG_DECREASE_FACTOR = 0.9
    HOAG_CG_MAX_ITER = 50  
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ==========================================
#        2. HELPER FUNCTIONS
# ==========================================
class DiceBCELoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        dice_loss = 1 - dice
        
        return 0.9 * bce_loss + 0.1 * dice_loss


def load_icnn_models(device):
    """Load pre-trained ICNN, SFB, L2net from checkpoints."""
    model_path = Config.ICNN_MODEL_PATH
    
    icnn = convex_models.ICNN(n_in_channels=1, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers).to(device)
    sfb = convex_models.SFB(n_in_channels=1, n_kernels=10, n_filters=32).to(device)
    l2_net = convex_models.L2net().to(device)
    
    icnn.load_state_dict(torch.load(os.path.join(model_path, "icnn.pt"), map_location=device))
    sfb.load_state_dict(torch.load(os.path.join(model_path, "sfb.pt"), map_location=device))
    l2_net.load_state_dict(torch.load(os.path.join(model_path, "l2_net.pt"), map_location=device))
    
    # Freeze all regularizer weights
    icnn.eval()
    sfb.eval()
    l2_net.eval()
    for p in icnn.parameters():
        p.requires_grad = False
    for p in sfb.parameters():
        p.requires_grad = False
    for p in l2_net.parameters():
        p.requires_grad = False
    
    icnn.zero_clip_weights()
    
    num_params = sum(p.numel() for p in icnn.parameters()) + \
                 sum(p.numel() for p in sfb.parameters()) + \
                 sum(p.numel() for p in l2_net.parameters())
    print(f"  Loaded ICNN regularizer ({num_params} frozen params) from {model_path}")
    
    x_test = torch.randn(1, 1, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    convex_models.test_convexity(icnn, x_test, device=device)
    
    return icnn, sfb, l2_net


def print_progress(epoch, batch, total_batches, loss, theta, info=""):
    vals = [f"θ{i}: {torch.exp(theta[i]).item():.5f}" for i in range(len(theta))]
    theta_str = " | ".join(vals)
    sys.stdout.write(f"\r[{info}] Ep {epoch+1} | Batch {batch+1}/{total_batches} | "
                     f"Loss: {loss:.4f} | {theta_str}")
    sys.stdout.flush()



def validate(model, val_loader, physics_op, icnn, sfb, l2_net,
             theta=None, steps=0, mode="clean", modality="CT"):
    model.eval()
    dice_score = 0.0
    
    for i, (img, mask) in enumerate(val_loader):
        img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
        
        # --- MODE 1: CLEAN (Upper Bound) ---
        if mode == "clean":
            if(modality=="CT"): x_in = norm(img)
            elif(modality=="MRI"): x_in = norm_z_score(img)
        
        # --- MODE 2: NOISY (Lower Bound) ---
        elif mode == "noisy":
            if(modality=="CT"):
                y_clean = physics_op(img)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                with torch.no_grad():
                    x_recon = physics_op.A_dagger(y)
                    x_in = norm(x_recon)
            elif(modality=="MRI"):
                imaginary_part = torch.zeros_like(img)
                complex_input = torch.cat([img, imaginary_part], dim=1)
                y_clean = physics_op(complex_input)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                with torch.no_grad():
                    x_recon = physics_op.A_dagger(y)
                    magnitude = torch.sqrt(x_recon[:, 0:1, :, :]**2 + x_recon[:, 1:2, :, :]**2)
                    x_in = norm_z_score(magnitude)    
            

        # --- MODE 3: HOAG (Optimized Reconstruction with ICNN regularizer) ---
        elif mode == "hoag":
            if(modality=="CT"):
                y_clean = physics_op(img)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                w = physics_op.A_dagger(y).detach().clone()
                w.requires_grad_(True)
                optimizer_inner = torch.optim.Adam([w], lr=Config.INNER_LR)
                scheduler_inner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_inner, T_max=steps, eta_min=Config.INNER_LR * 0.01)
                with torch.enable_grad():
                    for _ in range(steps):
                        optimizer_inner.zero_grad()
                        loss = inner_loss_func(w, theta, y, physics_op, icnn, sfb, l2_net)
                        loss.backward()
                        optimizer_inner.step()
                        scheduler_inner.step()     
                x_recon = w.detach()
                x_in = norm(x_recon)
            elif(modality=="MRI"):
                imaginary_part = torch.zeros_like(img)
                complex_input = torch.cat([img, imaginary_part], dim=1)
                y_clean = physics_op(complex_input)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                w = physics_op.A_dagger(y).detach().clone()
                w.requires_grad_(True)
                optimizer_inner = torch.optim.Adam([w], lr=Config.INNER_LR)
                scheduler_inner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_inner, T_max=steps, eta_min=Config.INNER_LR * 0.01)
                with torch.enable_grad():
                    for _ in range(steps):
                        optimizer_inner.zero_grad()
                        loss = inner_loss_func(w, theta, y, physics_op, icnn, sfb, l2_net)
                        loss.backward()
                        optimizer_inner.step()
                        scheduler_inner.step()     
                x_recon = w.detach()
                magnitude = torch.sqrt(x_recon[:, 0:1, :, :]**2 + x_recon[:, 1:2, :, :]**2)
                x_in = norm_z_score(magnitude)    

        # Predict segmentation mask
        with torch.no_grad():
            pred = (model(x_in) > 0.5).float()
            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum()
            dice_score += (2. * intersection + 1e-6) / (union + 1e-6)
            
    return dice_score.item() / len(val_loader)

# ==========================================
#        3. MAIN EXPERIMENT
# ==========================================
def run_experiment():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"--- Starting Experiment: {Config.TASK} (HOAG Bilevel with ICNN Regularizer) ---")
    print(f"    HOAG Settings: tol_init={Config.HOAG_EPSILON_TOL_INIT}, "
          f"schedule={Config.HOAG_TOLERANCE_DECREASE}, "
          f"decay={Config.HOAG_DECREASE_FACTOR}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # ====================================================================
    # DATA SETUP
    # ====================================================================
    full_ds = MSDDataset(Config.DATA_ROOT, Config.TASK, Config.IMG_SIZE, Config.MODALITY, Config.SUBSET_SIZE)
    train_len = int(Config.TRAIN_SPLIT * len(full_ds))
    val_len   = int(Config.VAL_SPLIT * len(full_ds))
    test_len  = len(full_ds) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)
    
    # ====================================================================
    # PHYSICS OPERATOR
    # ====================================================================
    physics = get_physics_operator(Config.IMG_SIZE, Config.ACCEL, Config.CENTER_FRAC, Config.DEVICE, modality=Config.MODALITY)
    
    # ====================================================================
    # LOAD PRE-TRAINED ICNN REGULARIZER (frozen weights)
    # ====================================================================
    print("\n--- Loading Pre-trained ICNN Regularizer ---")
    icnn, sfb, l2_net = load_icnn_models(Config.DEVICE)
    
    loss_fn = torch.nn.BCELoss()
    results = {}
    dummy_theta = torch.tensor([-10.0, -10.0, -10.0])
    
    # ====================================================================
    # PHASE 1: UPPER BOUND — Train UNet from scratch on clean images
    # ====================================================================
    print("\n--- PHASE 1: Upper Bound (Training UNet on Clean Images) ---")
    model_upper = UNet().to(Config.DEVICE)
    opt_upper = torch.optim.Adam(model_upper.parameters(), lr=Config.LR_UNET)
    scheduler_upper = torch.optim.lr_scheduler.CosineAnnealingLR(opt_upper, T_max=Config.EPOCH_CLEAN, eta_min=Config.LR_UNET * 0.01)
    
    local_ckpt_path = os.path.join(Config.OUTPUT_DIR, "model_upper_clean.pth")
    best_val_dice = 0.0
    
    for ep in range(Config.EPOCH_CLEAN):
        model_upper.train()
        epoch_loss = 0.0
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            # Clean input (upper bound = no corruption)
            if Config.MODALITY == "CT":
                x_in = norm(img)
            elif Config.MODALITY == "MRI":
                x_in = norm_z_score(img)
            
            opt_upper.zero_grad()
            pred = model_upper(x_in)
            loss = loss_fn(pred, mask)
            loss.backward()
            opt_upper.step()
            epoch_loss += loss.item()
            
            sys.stdout.write(f"\r[Upper Bound] Ep {ep+1}/{Config.EPOCH_CLEAN} | "
                             f"Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            sys.stdout.flush()
        
        scheduler_upper.step()
        avg_loss = epoch_loss / len(train_loader)
        
        # Validate and save best
        val_dice = validate(model_upper, val_loader, physics, icnn, sfb, l2_net,
                            theta=dummy_theta, mode="clean", modality=Config.MODALITY)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model_upper.state_dict(), local_ckpt_path)
        
        sys.stdout.write(f"\r[Upper Bound] Ep {ep+1}/{Config.EPOCH_CLEAN} | "
                         f"Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} | "
                         f"Best: {best_val_dice:.4f}")
        sys.stdout.flush()
    
    print()  # newline after progress
    
    # Load best model for evaluation
    model_upper.load_state_dict(torch.load(local_ckpt_path, map_location=Config.DEVICE))
    model_upper.eval()
    
    results['Upper Bound'] = validate(model_upper, test_loader, physics, icnn, sfb, l2_net,
                                       theta=dummy_theta, mode="clean", modality=Config.MODALITY)
    print(f" -> Final Upper Bound (Clean): {results['Upper Bound']:.4f}")

    # ====================================================================
    # PHASE 2: LOWER BOUND — Test Clean Model on Noisy Physics
    # ====================================================================
    print("\n--- PHASE 2: Lower Bound (Testing Clean Model on Noisy Physics) ---")
    results['Lower Bound'] = validate(model_upper, test_loader, physics, icnn, sfb, l2_net,
                                       theta=dummy_theta, mode="noisy", modality=Config.MODALITY)
    print(f" -> Final Lower Bound (Noisy): {results['Lower Bound']:.4f}")

    # ====================================================================
    # PHASE 3: APPROACH 1 — HOAG: Optimize theta Only (Fixed U-Net)
    # theta = [log λ_icnn, log λ_sfb, log λ_l2]
    # ====================================================================
    print("\n--- PHASE 3: Approach 1 (HOAG — Optimizing Theta Only) ---")
    
    model_fixed = UNet().to(Config.DEVICE)
    model_fixed.load_state_dict(torch.load(local_ckpt_path, map_location=Config.DEVICE)) 
    model_fixed.eval()
    for p in model_fixed.parameters():
        p.requires_grad = False
    
    theta = torch.tensor([-1.0, -2.0, -2.0], device=Config.DEVICE).requires_grad_(True)
    
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
    hoag_state = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )
    
    path_hoag = os.path.join(Config.OUTPUT_DIR, "hoag_theta.pth")

    for ep in range(Config.EPOCHS): 
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            if(Config.MODALITY=="CT"):
                y_clean = physics(img)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            elif(Config.MODALITY=="MRI"):
                imaginary_part = torch.zeros_like(img)
                complex_input = torch.cat([img, imaginary_part], dim=1)
                y_clean = physics(complex_input)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
    
            # HOAG STEP
            hyper_grad, val_loss_value, w_star = hoag_step(
                theta=theta,
                y=y,
                physics_op=physics,
                model=model_fixed,
                loss_fn=loss_fn,
                mask=mask,
                inner_loss_fn=inner_loss_func,
                state=hoag_state,
                icnn=icnn,
                sfb=sfb,
                l2_net=l2_net,
                inner_lr=Config.INNER_LR,
                inner_steps=Config.INNER_STEPS,
                cg_max_iter=Config.HOAG_CG_MAX_ITER,
                verbose=0,
                modality=Config.MODALITY
            )
            
            opt_theta.zero_grad()
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            
            with torch.no_grad():
                theta[0].clamp_(-9.0, 4.0)
                theta[1].clamp_(-9.0, 4.0)
                theta[2].clamp_(-9.0, 4.0)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 1 (HOAG)")
        
        hoag_state.decrease_tolerance()
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state.epsilon_tol, 'epoch': ep}, path_hoag)
        print(f"  [epsilon_tol: {hoag_state.epsilon_tol:.2e}]  saved theta")

    results['Approach 1'] = validate(model_fixed, test_loader, physics, icnn, sfb, l2_net,
                                      theta, Config.INNER_STEPS, mode="hoag", modality=Config.MODALITY)
    print(f" -> Final Approach 1 Score: {results['Approach 1']:.4f}")

    # ====================================================================
    # PHASE 4: APPROACH 2 — HOAG Joint Learning (theta + U-Net)
    # ====================================================================
    print("\n--- PHASE 4: Approach 2 (Joint Learning — Theta + U-Net) ---")
    
    model_joint = UNet().to(Config.DEVICE)
    model_joint.load_state_dict(torch.load(local_ckpt_path, map_location=Config.DEVICE))
    opt_model = torch.optim.Adam(model_joint.parameters(), lr=Config.LR_UNET)
    
    theta = torch.tensor([-1.0, -2.0, -2.0], device=Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_THETA)
    
    hoag_state_joint = HOAGState(
        epsilon_tol_init=Config.HOAG_EPSILON_TOL_INIT,
        tolerance_decrease=Config.HOAG_TOLERANCE_DECREASE,
        exponential_decrease_factor=Config.HOAG_DECREASE_FACTOR
    )
    
    path_joint = os.path.join(Config.OUTPUT_DIR, "model_joint.pth")
    path_theta_joint = os.path.join(Config.OUTPUT_DIR, "theta_joint.pth")

    for ep in range(Config.EPOCHS):
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            if(Config.MODALITY=="CT"):
                y_clean = physics(img)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            elif(Config.MODALITY=="MRI"):
                imaginary_part = torch.zeros_like(img)
                complex_input = torch.cat([img, imaginary_part], dim=1)
                y_clean = physics(complex_input)
                y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # STEP A: Solve Inner Problem ONCE
            w_star, _ = solve_inner_problem(
                w_init=physics.A_dagger(y).detach().clone(),
                theta=theta,
                y=y,
                physics_op=physics,
                inner_loss_fn=inner_loss_func,
                state=hoag_state_joint,
                icnn=icnn,
                sfb=sfb,
                l2_net=l2_net,
                lr=Config.INNER_LR,
                max_steps=Config.INNER_STEPS,
                verbose=0
            )
            
            # STEP B: Update U-Net Weights
            w_fixed = w_star.detach().clone().requires_grad_(False)
            if(Config.MODALITY=="CT"): x_in = norm(w_fixed)
            elif(Config.MODALITY=="MRI"):
                magnitude = torch.sqrt(w_fixed[:, 0:1, :, :]**2 + w_fixed[:, 1:2, :, :]**2)
                x_in = norm_z_score(magnitude)
            
            model_joint.train()
            opt_model.zero_grad()
            loss_unet = loss_fn(model_joint(x_in), mask)
            loss_unet.backward()
            opt_model.step()
            
            # STEP C: Update theta via HOAG Hypergradient
            model_joint.eval()
            
            hyper_grad, val_loss_value, w_star = hoag_step(
                theta=theta,
                y=y,
                physics_op=physics,
                model=model_joint,
                loss_fn=loss_fn,
                mask=mask,
                inner_loss_fn=inner_loss_func,
                state=hoag_state_joint,
                icnn=icnn,
                sfb=sfb,
                l2_net=l2_net,
                inner_lr=Config.INNER_LR,
                inner_steps=Config.INNER_STEPS,
                cg_max_iter=Config.HOAG_CG_MAX_ITER,
                verbose=0,
                modality=Config.MODALITY
            )
            
            opt_theta.zero_grad()
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            
            with torch.no_grad():
                theta[0].clamp_(-9.0, 4.0)
                theta[1].clamp_(-9.0, 4.0)
                theta[2].clamp_(-9.0, 4.0)
            
            print_progress(ep, i, len(train_loader), val_loss_value, theta, "Appr 2 (Joint)")
        
        hoag_state_joint.decrease_tolerance()
        torch.save(model_joint.state_dict(), path_joint)
        torch.save({'theta': theta, 'hoag_state_epsilon': hoag_state_joint.epsilon_tol, 'epoch': ep}, path_theta_joint)
        print(f"  [epsilon_tol: {hoag_state_joint.epsilon_tol:.2e}]  saved model + theta")

    results['Approach 2'] = validate(model_joint, test_loader, physics, icnn, sfb, l2_net,
                                      theta, Config.INNER_STEPS, mode="hoag", modality=Config.MODALITY)
    print(f" -> Final Approach 2 Score: {results['Approach 2']:.4f}")
    
    # ====================================================================
    # FINAL RESULTS
    # ====================================================================
    print("\n=== FINAL RESULTS ===")
    print(f"1. Upper Bound: {results['Upper Bound']:.4f}")
    print(f"2. Lower Bound: {results['Lower Bound']:.4f}")
    print(f"3. Approach 1:  {results['Approach 1']:.4f}")
    print(f"4. Approach 2:  {results['Approach 2']:.4f}")

    results_path = os.path.join(Config.OUTPUT_DIR, "final_results.txt")
    with open(results_path, "w") as f:
        f.write("=== FINAL RESULTS (ICNN Regularizer) ===\n")
        f.write(f"Task: {Config.TASK}\n")
        f.write(f"Accel: {Config.ACCEL}x | Noise: {Config.NOISE_SIGMA}\n")
        f.write(f"Epochs: {Config.EPOCHS} | Inner Steps: {Config.INNER_STEPS}\n")
        f.write(f"Regularizer: ICNN + SFB + L2net (pre-trained, frozen)\n")
        f.write(f"Theta (mixing weights): [log λ_icnn, log λ_sfb, log λ_l2]\n\n")
        f.write(f"1. Upper Bound (Clean):      {results['Upper Bound']:.4f}\n")
        f.write(f"2. Lower Bound (Noisy):      {results['Lower Bound']:.4f}\n")
        f.write(f"3. Approach 1 (HOAG theta):  {results['Approach 1']:.4f}\n")
        f.write(f"4. Approach 2 (HOAG joint):  {results['Approach 2']:.4f}\n")
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    run_experiment()
