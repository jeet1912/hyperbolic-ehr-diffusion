from __future__ import annotations

import torch
import torch.nn as nn


def code_pair_loss(code_emb, hier, device, num_pairs: int = 512) -> torch.Tensor:
    """Encourage embedding distances to correlate with tree distances."""
    n_real = len(hier.codes)
    idx_i = torch.randint(0, n_real, (num_pairs,), device=device)
    idx_j = torch.randint(0, n_real, (num_pairs,), device=device)

    base_emb = code_emb.emb
    emb_tensor = base_emb.weight if isinstance(base_emb, nn.Embedding) else base_emb
    emb = emb_tensor[:n_real]

    d_tree_list = []
    d_hyper_list = []

    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        if i == j:
            continue
        c1 = hier.idx2code[i]
        c2 = hier.idx2code[j]
        d_tree = hier.tree_distance(c1, c2)
        if d_tree is None:
            continue
        d_tree_list.append(d_tree)

        if hasattr(code_emb, "manifold"):
            v1 = emb[i]
            v2 = emb[j]
            d_h = code_emb.manifold.dist(v1.unsqueeze(0), v2.unsqueeze(0)).squeeze(0)
        else:
            d_h = torch.norm(emb[i] - emb[j])
        d_hyper_list.append(d_h)

    if not d_tree_list:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    d_tree_t = torch.tensor(d_tree_list, dtype=torch.float32, device=device)
    d_hyper_t = torch.stack(d_hyper_list).to(device)

    d_tree_t = (d_tree_t - d_tree_t.mean()) / (d_tree_t.std() + 1e-6)
    d_hyper_t = (d_hyper_t - d_hyper_t.mean()) / (d_hyper_t.std() + 1e-6)

    return torch.mean((d_hyper_t - d_tree_t) ** 2)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean() if reduction == 'mean' else focal.sum()