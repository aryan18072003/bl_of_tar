import torch
import torch.nn as nn
import numpy as np
import deepinv as dinv


# ==========================================
#  1. PHYSICS OPERATOR FACTORY
# ==========================================
def get_physics_operator(img_size, acceleration, center_frac, device, modality="CT"):

    if modality == "CT":
        if acceleration == 1:
            num_views = 180 
        else:
            num_views = int(180 / acceleration) 
            
        angles = torch.linspace(0, 180, num_views).to(device)
        
        physics = dinv.physics.Tomography(
            angles=angles,
            img_width=img_size,
            circle=False,
            device=device,
            normalize=True   
        )

        return physics

    elif modality == "MRI":
        mask = torch.zeros((1, img_size, img_size))
        
        pad = (img_size - int(img_size * center_frac) + 1) // 2
        width = max(1, int(img_size * center_frac))
        mask[:, :, pad:pad + width] = 1.0
        
        num_keep = int(img_size / acceleration)
        all_cols = np.arange(img_size)
        kept_cols = np.where(mask[0, 0, :].cpu().numpy() == 1)[0]
        zero_cols = np.setdiff1d(all_cols, kept_cols)
        
        if len(zero_cols) > 0 and (num_keep - len(kept_cols) > 0):
            chosen = np.random.choice(zero_cols, num_keep - len(kept_cols), replace=False)
            mask[:, :, chosen] = 1.0
            
        mask = mask.to(device)
        physics = dinv.physics.MRI(mask=mask, img_size=(1, img_size, img_size), device=device, normalize=True)
        return physics

    else:
        raise ValueError(f"Unsupported modality: {modality}")


# ==========================================
#  2a. REGULARIZER ONLY (for HVP computation)
# ==========================================
def regularizer_only(w, theta, icnn, sfb, l2_net):
    """
    Compute the ICNN-based regularizer R_θ(w):
    
        R_θ(w) = exp(θ[0]) · ICNN(w̃).mean() + exp(θ[1]) · SFB(w̃).mean() + exp(θ[2]) · L2net(w̃).mean()
    
    where w̃ = norm(w) is the CT-normalized version of w (to [0,1] range),
    matching the normalization used during ACR pre-training.
    θ = [log λ_icnn, log λ_sfb, log λ_l2] are learnable scalar mixing weights.
    The ICNN/SFB/L2net network weights are FROZEN (pre-trained).
    """
    # Normalize w to [0,1] to match ICNN training distribution
    w_norm = torch.clamp(w, min=-150, max=250)
    w_norm = (w_norm + 150.0) / 400.0
    
    lambda_icnn = torch.exp(theta[0].clamp(max=4.0))
    lambda_sfb  = torch.exp(theta[1].clamp(max=4.0))
    lambda_l2   = torch.exp(theta[2].clamp(max=4.0))
    
    reg = lambda_icnn * icnn(w_norm).mean() + lambda_sfb * sfb(w_norm).mean() + lambda_l2 * l2_net(w_norm).mean()
    
    return reg


# ==========================================
#  2b. INNER LOSS FUNCTION (h in HOAG)
# ==========================================
def inner_loss_func(w, theta, y, physics_op, icnn, sfb, l2_net):
    """
    Inner objective for HOAG:
    
        h(w, θ) = ||y - A(w)||² + R_θ(w)
    
    Data fidelity + ICNN-based convex regularizer.
    """
    residual = y - physics_op(w)
    fid = torch.mean(residual ** 2)
    
    reg = regularizer_only(w, theta, icnn, sfb, l2_net)
    
    return fid + reg


# ==========================================
#  3. NORMALIZATION UTILITY
# ==========================================
def robust_normalize(x):

    b = x.shape[0]
    x_flat = x.view(b, -1)
    
    val_min = torch.quantile(x_flat, 0.01, dim=1).view(b, 1, 1, 1)
    val_max = torch.quantile(x_flat, 0.99, dim=1).view(b, 1, 1, 1)
    
    x = torch.clamp(x, val_min, val_max)
    
    denom = val_max - val_min
    denom = torch.where(denom > 1e-7, denom, torch.ones_like(denom))
    
    return (x - val_min) / denom
