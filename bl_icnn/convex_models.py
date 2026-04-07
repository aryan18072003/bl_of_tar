#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICNN-based convex regularizer models.
Adapted from data_driven_convex_regularization-main/convex_models.py
for 128x128 medical images (reduced architecture).
"""

import torch
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Architecture hyperparameters (scaled down for 128x128 medical images)
n_layers, n_filters, kernel_size = 5, 32, 3


class ICNN(nn.Module):
    """Input-Convex Neural Network.
    
    Convexity is ensured by:
    - Non-negative weights on the z-path (wz layers + final_conv2d)
    - Leaky ReLU activations (which are convex and non-decreasing)
    - Skip connections from input x at every layer (wx layers)
    
    Output: scalar energy per image (via global average pooling).
    """
    def __init__(self, n_in_channels=1, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers):
        super(ICNN, self).__init__()
        
        self.n_layers = n_layers
        # These layers should have non-negative weights (z-path)
        self.wz = nn.ModuleList([nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
                                 for i in range(self.n_layers)])
        
        # These layers can have arbitrary weights (x skip-connections)
        self.wx = nn.ModuleList([nn.Conv2d(n_in_channels, n_filters, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
                                 for i in range(self.n_layers+1)])
        
        # One final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(n_filters, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        
        # Slope of leaky-relu
        self.negative_slope = 0.2 
        
    def forward(self, x):
        z = torch.nn.functional.leaky_relu(self.wx[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx[layer+1](x), negative_slope=self.negative_slope)
        z = self.final_conv2d(z)
        z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)
        
        return z_avg
    
    def initialize_weights(self, min_val=0.0, max_val=0.001, device=device):
        """Initialize non-negative layers with small positive random weights."""
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val) \
            * torch.rand(n_filters, n_filters, kernel_size, kernel_size).to(device)
        
        self.final_conv2d.weight.data = min_val + (max_val - min_val) \
        * torch.rand(1, n_filters, kernel_size, kernel_size).to(device)
        return self
    
    def zero_clip_weights(self): 
        """Clamp non-negative weights to >= 0 to preserve input-convexity."""
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        
        self.final_conv2d.weight.data.clamp_(0)
        return self    
    

def test_convexity(net, x, device=device):
    """Check convexity of the network numerically."""
    print('running a numerical convexity test...')
    n_trials = 100
    convexity = 0
    for trial in np.arange(n_trials):
        x1 = torch.rand(x.size()).to(device)
        x2 = torch.rand(x.size()).to(device)
        alpha = torch.rand(1).to(device)
    
        cvx_combo_of_input = net(alpha * x1 + (1-alpha)*x2)
        cvx_combo_of_output = alpha * net(x1) + (1-alpha)*net(x2)
    
        convexity += (cvx_combo_of_input.mean() <= cvx_combo_of_output.mean())
    if(convexity == n_trials):
        flag = True
        print('Passed convexity test!')
    else:
        flag = False
        print('Failed convexity test!')
    return flag
    
    
class SFB(nn.Module):
    """Sparsifying Filter Bank (SFB) module.
    
    Convex penalty with learned convolutional filters.
    Uses absolute value (convex) and a learnable penalty weight.
    """
    def __init__(self, n_in_channels=1, n_kernels=10, n_filters=32):
        super(SFB, self).__init__()
        self.penalty = nn.Parameter((-12.0) * torch.ones(1))
        self.n_kernels = n_kernels
        self.conv = nn.ModuleList([nn.Conv2d(n_in_channels, n_filters, kernel_size=7, stride=1, padding=3, bias=False)
                                   for i in range(self.n_kernels)])
        
    def forward(self, x):
        total_out = 0.0
        for kernel_idx in range(self.n_kernels):
            x_out = torch.abs(self.conv[kernel_idx](x))
            x_out_flat = x_out.view(x.size(0), -1)
            total_out += torch.sum(x_out_flat, dim=1)
        
        total_out = total_out.view(x.size(0), -1)
        return (torch.nn.functional.softplus(self.penalty)) * total_out
    
    
class L2net(nn.Module):
    """Learnable L2 penalty term."""
    def __init__(self):
        super(L2net, self).__init__()
        self.l2_penalty = nn.Parameter((-9.0) * torch.ones(1))
   
    def forward(self, x):
        l2_term = torch.sum(x.view(x.size(0), -1)**2, dim=1)
        out = ((torch.nn.functional.softplus(self.l2_penalty)) * l2_term).view(x.size(0), -1)
        return out
