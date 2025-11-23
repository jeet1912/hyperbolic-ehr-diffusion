from __future__ import annotations

from typing import List, Sequence

import torch


def decode_visit_vectors(
    sampled_visits: torch.Tensor,
    code_emb,
    visit_enc,
    embedding_type: str,
    codes_per_visit: int,
) -> torch.Tensor:
    """Decode latent visit vectors by nearest neighbours in embedding space."""
    num_samples, max_len, dim = sampled_visits.shape
    visit_vecs = sampled_visits.view(num_samples * max_len, dim)

    if embedding_type == "hyperbolic":
        code_tangent = visit_enc.manifold.logmap0(code_emb.emb)
        pad_idx = getattr(visit_enc, "pad_idx", None)
        if pad_idx is not None:
            code_tangent = code_tangent[:pad_idx]
        sims = visit_vecs @ code_tangent.t()
    else:
        pad_idx = getattr(visit_enc, "pad_idx", None)
        if pad_idx is not None:
            code_matrix = code_emb.emb.weight[:pad_idx]
        else:
            code_matrix = code_emb.emb.weight
        diff = visit_vecs.unsqueeze(1) - code_matrix.unsqueeze(0)
        sims = -(diff**2).sum(-1)

    topk_idx = sims.topk(k=codes_per_visit, dim=-1).indices
    return topk_idx.view(num_samples, max_len, codes_per_visit).cpu()


def visits_from_indices(indices_tensor: torch.Tensor) -> List[List[int]]:
    visits: List[List[int]] = []
    flattened = indices_tensor.view(-1, indices_tensor.shape[-1])
    for visit_codes in flattened:
        unique_codes = sorted(set(int(c) for c in visit_codes.tolist()))
        visits.append(unique_codes)
    return visits


def mean_tree_distance_from_visits(visit_lists: Sequence[Sequence[int]], hier) -> float | None:
    dists: List[float] = []
    max_code_idx = len(hier.codes) - 1

    for visit in visit_lists:
        filtered = [c for c in visit if 0 <= c <= max_code_idx]
        if len(filtered) < 2:
            continue
        codes = [hier.idx2code[i] for i in filtered]
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                dist = hier.tree_distance(codes[i], codes[j])
                if dist is not None:
                    dists.append(dist)
    if not dists:
        return None
    return float(sum(dists) / len(dists))
