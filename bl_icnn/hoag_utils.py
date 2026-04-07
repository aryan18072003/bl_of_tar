import torch
import torch.autograd as autograd


# ==========================================
#  1. HESSIAN-VECTOR PRODUCT (HVP)
#     Split into physics (analytical) + regularizer (autograd)
#
#     H = ∇²_ww h(w*, θ) = ∇²_ww [||y-A(w)||² + R_θ(w)]
#       = 2·A^T·A  +  ∇²_ww R_θ(w)
#
#     H·v = 2·A^T(A(v))  +  ∇²_ww R_θ(w)·v
#
#     The first term uses only forward+adjoint (no second derivatives).
#     The second term uses double-backprop through ICNN only (no grid_sample).
#     This avoids the "cudnn_grid_sampler_backward not implemented" error.
# ==========================================
def hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, v,
                           icnn, sfb, l2_net):
    """
    Compute the Hessian-vector product  H·v  where H = ∇²_ww h(w*, θ)
    
    Split into:
      1. Data fidelity HVP:  2 * A^T(A(v))  (analytical, no autograd needed)
      2. Regularizer HVP:    ∇²_ww R_θ(w) · v  (autograd through ICNN only)
    """
    v_detached = v.detach()
    
    # --------------------------------------------------
    # PART 1: Data fidelity HVP (analytical)
    #   ∇²_ww ||y - A(w)||² = 2 * A^T * A
    #   So HVP = 2 * A^T(A(v))
    # --------------------------------------------------
    with torch.no_grad():
        Av = physics_op(v_detached)       # A(v)
        AtAv = physics_op.A_adjoint(Av)   # A^T(A(v))
        hvp_data = 2.0 * AtAv
    
    # --------------------------------------------------
    # PART 2: Regularizer HVP (autograd double-backprop)
    #   Only differentiates through ICNN/SFB/L2net — no grid_sample
    # --------------------------------------------------
    from physics import regularizer_only
    
    w_for_hvp = w_star.detach().requires_grad_(True)
    
    with torch.enable_grad():
        reg = regularizer_only(w_for_hvp, theta.detach(), icnn, sfb, l2_net)
        grad_w_reg = autograd.grad(reg, w_for_hvp, create_graph=True)[0]
        grad_dot_v = torch.sum(grad_w_reg * v_detached)
        hvp_reg = autograd.grad(grad_dot_v, w_for_hvp, retain_graph=False)[0]
    
    return (hvp_data + hvp_reg).detach()


# ==========================================
#  2. CONJUGATE GRADIENT (CG) SOLVER
# ==========================================
def conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, b,
                       icnn, sfb, l2_net,
                       max_iter=10, tol=1e-4, warm_start=None):
    """
    Solve  H·q = b  for q using Conjugate Gradient,
    where H = ∇²_ww h(w*, θ).
    """
    b_norm = b.detach().norm().item()
    scaled_tol = tol * b_norm

    if warm_start is not None and warm_start.shape == b.shape:
        x = warm_start.detach().clone()
    else:
        x = torch.zeros_like(b)
        warm_start = None
    
    if warm_start is not None:
        Ax = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, x,
                                     icnn, sfb, l2_net)
        Ax = Ax + 1e-3 * x
        r = b.detach() - Ax
    else:
        r = b.detach().clone()
    
    p = r.clone()
    rsold = torch.sum(r * r)
    
    for i in range(max_iter):
        if torch.sqrt(rsold) < scaled_tol:
            break

        Ap = hessian_vector_product(inner_loss_fn, w_star, theta, y, physics_op, p,
                                     icnn, sfb, l2_net)
        
        Ap = Ap + 1e-3 * p
        
        pAp = torch.sum(p * Ap)
        if pAp.abs() < 1e-12:
            break
        
        alpha = rsold / (pAp + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
    
    return x.detach()
