"""
Pre-training script for the ICNN-based convex regularizer on medical data.
Uses adversarial convex regularizer (ACR) training:
    loss = R(clean) - R(A†(y)) + λ_gp · gradient_penalty(ICNN)

Must be run BEFORE main.py to produce checkpoints in ./trained_models/.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split

from dataset import MSDDataset
from physics import get_physics_operator
import convex_models
from convex_models import n_layers, n_filters, kernel_size


# ==========================================
#        CONFIGURATION
# ==========================================
class Config:
    DATA_ROOT = "../data_medical/ct_data"
    TASK = "Task09_Spleen"
    MODALITY = "CT"
    
    IMG_SIZE = 128
    BATCH_SIZE = 8
    
    # Physics settings (must match main.py)
    ACCEL = 6
    NOISE_SIGMA = 0.5
    CENTER_FRAC = 0.08
    
    # ACR Training settings
    N_EPOCHS = 20
    NUM_MINIBATCHES_LOG = 20
    LAMBDA_GP = 5.0
    LR_ACR = 2e-5
    LR_SFB = 2e-5
    
    SUBSET_SIZE = None
    TRAIN_SPLIT = 0.8
    
    MODEL_PATH = "./trained_models/"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm(img):
    """CT normalization: clamp HU values and scale to [0,1]."""
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
#  GRADIENT PENALTY (for ICNN Lipschitz)
# ==========================================
def compute_gradient_penalty(network, real_samples, fake_samples, device):
    """Gradient penalty loss (WGAN-GP style) to enforce Lipschitz on ICNN."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    validity = network(interpolates)
    fake = torch.ones(validity.shape).requires_grad_(False).to(device)
    gradients = autograd.grad(outputs=validity, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ==========================================
#        MAIN TRAINING
# ==========================================
def train():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"--- Pre-training ICNN Regularizer on {Config.TASK} ({Config.MODALITY}) ---")
    print(f"    Device: {Config.DEVICE}")
    
    # ====================================================================
    # DATA SETUP
    # ====================================================================
    full_ds = MSDDataset(Config.DATA_ROOT, Config.TASK, Config.IMG_SIZE, Config.MODALITY, Config.SUBSET_SIZE)
    train_len = int(Config.TRAIN_SPLIT * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, _ = random_split(full_ds, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # ====================================================================
    # PHYSICS OPERATOR
    # ====================================================================
    physics = get_physics_operator(Config.IMG_SIZE, Config.ACCEL, Config.CENTER_FRAC, Config.DEVICE, modality=Config.MODALITY)
    
    # ====================================================================
    # CREATE MODELS
    # ====================================================================
    acr = convex_models.ICNN(n_in_channels=1, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers).to(Config.DEVICE)
    acr.initialize_weights(device=Config.DEVICE)
    
    sfb = convex_models.SFB(n_in_channels=1, n_kernels=10, n_filters=32).to(Config.DEVICE)
    l2_net = convex_models.L2net().to(Config.DEVICE)
    
    num_params_acr = sum(p.numel() for p in acr.parameters())
    num_params_sfb = sum(p.numel() for p in sfb.parameters())
    num_params_l2 = sum(p.numel() for p in l2_net.parameters())
    print(f"# params: ACR(ICNN): {num_params_acr}, SFB: {num_params_sfb}, L2net: {num_params_l2}")
    
    # Convexity check
    x_test = torch.randn(1, 1, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    convex_models.test_convexity(acr, x_test, device=Config.DEVICE)
    
    # ====================================================================
    # OPTIMIZERS
    # ====================================================================
    import itertools
    optimizer_acr = torch.optim.Adam(acr.parameters(), lr=Config.LR_ACR, betas=(0.5, 0.99))
    optimizer_sfb = torch.optim.Adam(
        itertools.chain(sfb.parameters(), l2_net.parameters()), 
        lr=Config.LR_SFB, betas=(0.5, 0.99), weight_decay=1.0
    )
    
    # ====================================================================
    # TRAINING LOOP
    # ====================================================================
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    
    log_file_name = os.path.join(Config.MODEL_PATH, "training_log.txt")
    log_file = open(log_file_name, "w")
    log_file.write("################ training log for ICNN convex regularizer ################\n")
    
    acr.train()
    sfb.train()
    l2_net.train()
    
    for epoch in range(Config.N_EPOCHS):
        total_loss, total_gp_loss, total_diff = 0.0, 0.0, 0.0
        
        for idx, (img, mask) in enumerate(train_loader):
            img = img.to(Config.DEVICE)
            
            # Generate noisy reconstruction A†(y)
            with torch.no_grad():
                if Config.MODALITY == "CT":
                    y_clean = physics(img)
                    y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                    fbp = physics.A_dagger(y)
                    phantom = norm(img)
                    fbp_norm = norm(fbp)
                elif Config.MODALITY == "MRI":
                    imaginary_part = torch.zeros_like(img)
                    complex_input = torch.cat([img, imaginary_part], dim=1)
                    y_clean = physics(complex_input)
                    y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
                    x_recon = physics.A_dagger(y)
                    magnitude_recon = torch.sqrt(x_recon[:, 0:1, :, :]**2 + x_recon[:, 1:2, :, :]**2)
                    phantom = norm_z_score(img)
                    fbp_norm = norm_z_score(magnitude_recon)
            
            # ACR training loss: R(clean) - R(noisy)
            diff_loss = (acr(phantom).mean() + sfb(phantom).mean() + l2_net(phantom).mean()) \
                      - (acr(fbp_norm).mean() + sfb(fbp_norm).mean() + l2_net(fbp_norm).mean())
            
            gp_loss = compute_gradient_penalty(acr, phantom.data, fbp_norm.data, Config.DEVICE)
            loss = diff_loss + Config.LAMBDA_GP * gp_loss
            
            optimizer_acr.zero_grad()
            optimizer_sfb.zero_grad()
            loss.backward()
            optimizer_acr.step()
            optimizer_sfb.step()
            
            total_loss += loss.item()
            total_gp_loss += gp_loss.item()
            total_diff += diff_loss.item()
            
            # Preserve convexity
            acr.zero_clip_weights()
            
            if (idx % Config.NUM_MINIBATCHES_LOG == Config.NUM_MINIBATCHES_LOG - 1):
                avg_loss = total_loss / Config.NUM_MINIBATCHES_LOG
                avg_gp_loss = total_gp_loss / Config.NUM_MINIBATCHES_LOG
                avg_diff = total_diff / Config.NUM_MINIBATCHES_LOG
                total_loss, total_gp_loss, total_diff = 0.0, 0.0, 0.0
                
                train_log = (f"epoch: [{epoch+1}/{Config.N_EPOCHS}], "
                           f"batch: [{idx+1}/{len(train_loader)}], "
                           f"avg_loss: {avg_loss:.8f}, "
                           f"avg_gp: {avg_gp_loss:.8f}, "
                           f"avg_diff: {avg_diff:.8f}")
                print(train_log)
                log_file.write(train_log + "\n")
                
                with torch.no_grad():
                    train_log2 = (f'  phantom -> ICNN: {acr(phantom).mean():.6f}, '
                                f'SFB: {sfb(phantom).mean():.6f}, '
                                f'L2: {l2_net(phantom).mean():.6f}')
                    print(train_log2)
                    log_file.write(train_log2 + "\n")
                    
                    train_log3 = (f'  FBP     -> ICNN: {acr(fbp_norm).mean():.6f}, '
                                f'SFB: {sfb(fbp_norm).mean():.6f}, '
                                f'L2: {l2_net(fbp_norm).mean():.6f}')
                    print(train_log3)
                    log_file.write(train_log3 + "\n")
    
    # ====================================================================
    # SAVE MODELS
    # ====================================================================
    log_file.close()
    
    torch.save(acr.state_dict(), os.path.join(Config.MODEL_PATH, "icnn.pt"))
    torch.save(sfb.state_dict(), os.path.join(Config.MODEL_PATH, "sfb.pt"))
    torch.save(l2_net.state_dict(), os.path.join(Config.MODEL_PATH, "l2_net.pt"))
    print(f"\nModels saved to {Config.MODEL_PATH}")
    
    # Final convexity check
    acr.eval()
    convex_models.test_convexity(acr, x_test, device=Config.DEVICE)
    print("Done!")


if __name__ == "__main__":
    train()
