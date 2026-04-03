import torch
import torch.nn as nn
import numpy as np
import deepinv as dinv
from deepinv.physics.blur import gaussian_blur


# ==========================================
#  1. PHYSICS OPERATOR (Gaussian Blur)
# ==========================================
def build_physics(img_size, blur_sigma, noise_sigma, device):
    """Create the Gaussian blur forward operator."""
    kernel = gaussian_blur(
        sigma=(blur_sigma, blur_sigma),
    )
    return dinv.physics.Blur(
        filter=kernel,
        padding="circular",
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=noise_sigma),
    )


# ==========================================
#  2a. REGULARIZER ONLY (for exact HVP)
# ==========================================
def regularizer_only(w, theta):
    """TV regularization term only (no data fidelity)."""
    reg_weight = torch.exp(theta[0].clamp(max=1.0))
    eps = torch.exp(theta[1].clamp(min=-12.0))
    
    dx = torch.roll(w, 1, 2) - w
    dy = torch.roll(w, 1, 3) - w
    tv_penalty = torch.mean(torch.sqrt(dx**2 + dy**2 + eps))
    
    return reg_weight * tv_penalty


# ==========================================
#  2b. INNER LOSS FUNCTION (h in HOAG)
# ==========================================
def inner_loss_func(w, theta, y, physics_op):
    """Combined data fidelity + TV regularization."""
    # Data fidelity
    residual = y - physics_op(w)
    fid = torch.mean(residual ** 2)
    
    # TV regularization
    reg_weight = torch.exp(theta[0].clamp(max=1.0))
    eps = torch.exp(theta[1].clamp(min=-12.0))
    
    dx = torch.roll(w, 1, 2) - w
    dy = torch.roll(w, 1, 3) - w
    tv_penalty = torch.mean(torch.sqrt(dx**2 + dy**2 + eps))
    
    return fid + reg_weight * tv_penalty
