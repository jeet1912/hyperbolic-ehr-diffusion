from __future__ import annotations

import torch


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
