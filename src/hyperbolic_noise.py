from __future__ import annotations
import torch.nn.functional as F
import torch
import geoopt

def is_hyperbolic(embedding_type: str | None) -> bool:
    return embedding_type is not None and embedding_type.lower() == "hyperbolic"


def _mobius_scalar_mul_batch(manifold, coeffs: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    B = points.shape[0]
    out = []
    if coeffs.dim() == 0:
        coeffs = coeffs.view(1)
    elif coeffs.dim() == 1:
        coeffs = coeffs
    else:
        coeffs = coeffs.view(B, -1)[:, 0]
    if coeffs.numel() == 1 and B > 1:
        coeffs = coeffs.repeat(B)
    coeffs = coeffs.to(points.device, points.dtype)
    for i in range(B):
        c = coeffs[i]
        out.append(manifold.mobius_scalar_mul(c, points[i]))
    return torch.stack(out, dim=0)


def _mobius_add_batch(manifold, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([manifold.mobius_add(x[i], y[i]) for i in range(x.shape[0])], dim=0)


def hyperbolic_forward_noise(manifold, x0, eps, sqrt_a, sqrt_one_minus_a):
    x0_man = manifold.expmap0(x0)
    eps_man = manifold.expmap0(eps)
    x0_scaled = _mobius_scalar_mul_batch(manifold, sqrt_a.view(-1), x0_man)
    noise_scaled = _mobius_scalar_mul_batch(manifold, sqrt_one_minus_a.view(-1), eps_man)
    x_t_man = _mobius_add_batch(manifold, x0_scaled, noise_scaled)
    return manifold.logmap0(x_t_man)


def hyperbolic_remove_noise(manifold, x_t, eps_hat, sqrt_a, sqrt_one_minus_a):
    x_t_man = manifold.expmap0(x_t)
    eps_man = manifold.expmap0(eps_hat)
    noise_scaled = _mobius_scalar_mul_batch(manifold, sqrt_one_minus_a.view(-1), eps_man)
    diff = _mobius_add_batch(manifold, x_t_man, -noise_scaled)
    inv_scale = (1.0 / (sqrt_a + 1e-8)).view(-1)
    x0_man = _mobius_scalar_mul_batch(manifold, inv_scale, diff)
    return manifold.logmap0(x0_man)


def hyperbolic_rectified_flow_loss(
    model,
    visit_encoder,
    flat_visits,
    visit_mask,
    manifold: geoopt.PoincareBall
):
    """
    Pure hyperbolic rectified flow in tangent space.
    This is the 2025 SOTA.
    """
    device = next(model.parameters()).device
    num_visits = visit_mask.shape[0] * visit_mask.shape[1]

    # Encode clean visits → tangent space
    x0_tangent = visit_encoder(flat_visits)  # [N, dim]
    x0_tangent = x0_tangent.view(-1, x0_tangent.size(-1))

    # Sample noise in tangent space
    noise = torch.randn_like(x0_tangent)

    # Time: uniform [0,1]
    t = torch.rand(x0_tangent.shape[0], device=device)

    # Straight line in tangent space = geodesic on manifold
    xt_tangent = (1 - t[:, None]) * noise + t[:, None] * x0_tangent

    # Target velocity
    target_velocity = x0_tangent - noise

    # Predict velocity
    pred_velocity = model(
        xt_tangent,
        t,
        visit_mask.view(-1, visit_mask.shape[1]) if visit_mask is not None else None
    )

    # MSE on velocity
    loss = F.mse_loss(pred_velocity, target_velocity, reduction='none').mean(dim=-1)
    if visit_mask is not None:
        mask_flat = visit_mask.view(-1).float()
        loss = (loss * mask_flat).sum() / mask_flat.sum()
    else:
        loss = loss.mean()

    return loss