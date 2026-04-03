import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deepinv as dinv
from deepinv.physics.blur import gaussian_blur


# ==========================================
#  FoE CONFIGURATION
# ==========================================
NUM_EXPERTS = 5       # J: number of expert filters
FILTER_SIZE = 5       # K: spatial filter size
IN_CHANNELS = 1       # C: input channels (grayscale)

N_SCALAR_PARAMS = 1 + NUM_EXPERTS + NUM_EXPERTS   # 1 + J + J = 11
N_FILTER_PARAMS = NUM_EXPERTS * IN_CHANNELS * FILTER_SIZE * FILTER_SIZE  # 5*1*25 = 125
THETA_SIZE = N_SCALAR_PARAMS + N_FILTER_PARAMS     # 11 + 125 = 136


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
#  2. theta PARSING
# ==========================================
def parse_theta(theta):
    idx = 0
    global_weight = theta[idx]
    idx += 1

    filter_weights = theta[idx : idx + NUM_EXPERTS]
    idx += NUM_EXPERTS

    smoothing_params = theta[idx : idx + NUM_EXPERTS]
    idx += NUM_EXPERTS

    filters = theta[idx:].view(NUM_EXPERTS, IN_CHANNELS, FILTER_SIZE, FILTER_SIZE)
    return global_weight, filter_weights, smoothing_params, filters


# ==========================================
#  3. theta INITIALIZATION
# ==========================================
def initialize_theta(device):
    """Create initial theta for FoE regularizer with derivative-like filters."""
    global_weight = torch.tensor([-3.0])
    filter_weights = torch.full((NUM_EXPERTS,), -5.0)
    smoothing_params = torch.full((NUM_EXPERTS,), -4.6)

    filters = torch.zeros(NUM_EXPERTS, IN_CHANNELS, FILTER_SIZE, FILTER_SIZE)
    scale = 1.0 / np.sqrt(IN_CHANNELS)

    # Filter 1: Horizontal gradient
    h_pattern = torch.tensor([
        [-1, -1,  0,  1,  1],
        [-2, -2,  0,  2,  2],
        [-3, -3,  0,  3,  3],
        [-2, -2,  0,  2,  2],
        [-1, -1,  0,  1,  1]
    ], dtype=torch.float32) / 12.0
    for c in range(IN_CHANNELS):
        filters[0, c] = h_pattern * scale

    # Filter 2: Vertical gradient
    for c in range(IN_CHANNELS):
        filters[1, c] = h_pattern.T * scale

    # Filter 3: Diagonal (45 deg)
    d_pattern = torch.tensor([
        [-2, -1,  0,  0,  0],
        [-1, -2,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  2,  1],
        [ 0,  0,  0,  1,  2]
    ], dtype=torch.float32) / 6.0
    for c in range(IN_CHANNELS):
        filters[2, c] = d_pattern * scale

    # Filter 4: Anti-diagonal (135 deg)
    for c in range(IN_CHANNELS):
        filters[3, c] = d_pattern.flip(1) * scale

    # Filter 5: Laplacian of Gaussian
    log_pattern = torch.tensor([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -1,  0],
        [ 0,  0, -1,  0,  0]
    ], dtype=torch.float32) / 16.0
    for c in range(IN_CHANNELS):
        filters[4, c] = log_pattern * scale

    theta = torch.cat([global_weight, filter_weights, smoothing_params, filters.flatten()])
    return theta.to(device)


# ==========================================
#  4a. REGULARIZER ONLY (for exact HVP)
# ==========================================
def regularizer_only(w, theta):
    """Compute only the FoE regularization term R_theta(w)."""
    global_weight, filter_weights, smoothing_params, filters = parse_theta(theta)
    
    foe_sum = torch.tensor(0.0, device=w.device)
    for j in range(NUM_EXPERTS):
        c_j = filters[j:j+1]
        response = F.conv2d(w, c_j, padding=FILTER_SIZE // 2)
        nu_j = torch.exp(smoothing_params[j].clamp(max=2.0))
        smoothed_norm = torch.mean(torch.sqrt(response ** 2 + nu_j ** 2))
        foe_sum = foe_sum + torch.exp(filter_weights[j].clamp(max=2.0)) * smoothed_norm
    
    return torch.exp(global_weight.clamp(max=2.0)) * foe_sum


# ==========================================
#  4b. INNER LOSS (h in HOAG)
# ==========================================
def inner_loss_func(w, theta, y, physics_op):
    """Combined data fidelity + FoE regularization."""
    residual = y - physics_op(w)
    fid = torch.mean(residual ** 2)

    global_weight, filter_weights, smoothing_params, filters = parse_theta(theta)

    foe_sum = torch.tensor(0.0, device=w.device)

    for j in range(NUM_EXPERTS):
        c_j = filters[j:j+1]   # (1, C, K, K)
        response = F.conv2d(w, c_j, padding=FILTER_SIZE // 2)  # (B, 1, H, W)
        nu_j = torch.exp(smoothing_params[j].clamp(max=2.0))
        smoothed_norm = torch.mean(torch.sqrt(response ** 2 + nu_j ** 2))
        foe_sum = foe_sum + torch.exp(filter_weights[j].clamp(max=2.0)) * smoothed_norm

    foe_reg = torch.exp(global_weight.clamp(max=2.0)) * foe_sum

    return fid + foe_reg
