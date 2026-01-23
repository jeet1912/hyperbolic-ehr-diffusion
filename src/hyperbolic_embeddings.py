import torch
import torch.nn as nn
import geoopt
from torch.nn.utils.rnn import pad_sequence

class HyperbolicCodeEmbedding(nn.Module):
    def __init__(self, num_codes: int, dim: int = 16, c: float = 1.0):
        super().__init__()
        self.manifold = geoopt.manifolds.PoincareBall(c=c)
        init = torch.randn(num_codes, dim) * 1e-3
        init = self.manifold.projx(init)
        self.emb = geoopt.ManifoldParameter(init, manifold=self.manifold)

    def forward(self, code_ids: torch.Tensor) -> torch.Tensor:
        """
        code_ids: (...,) long
        returns: (..., dim) hyperbolic points
        """
        x = self.emb[code_ids]
        x = self.manifold.expmap0(self.manifold.logmap0(x))
        return self.manifold.projx(x)


class VisitEncoder(nn.Module):
    def __init__(self, code_embedding: HyperbolicCodeEmbedding, pad_idx: int):
        super().__init__()
        self.code_embedding = code_embedding
        self.manifold = code_embedding.manifold
        self.pad_idx = pad_idx

    def forward(self, code_ids_batch):
        """
        code_ids_batch: list of 1D LongTensors (variable length), len = B*T
        Returns: tensor [B*T, dim] in tangent space at origin.
        """
        visit_vecs = []
        d = self.code_embedding.emb.size(1)
        device = self.code_embedding.emb.device

        for ids in code_ids_batch:
            # remove pads
            ids = ids[ids != self.pad_idx]
            if ids.numel() == 0:
                # empty visit -> zero vector in tangent space
                visit_vecs.append(torch.zeros(d, device=device))
                continue

            x = self.code_embedding(ids)        # [k, d] on manifold
            tangents = self.manifold.logmap0(x) # [k, d] in R^d
            visit_vec = tangents.mean(dim=0)    # [d]
            visit_vecs.append(visit_vec)

        return torch.stack(visit_vecs, dim=0)


class HyperbolicVisitEncoder(nn.Module):
    def __init__(self, code_embedding, pad_idx: int):
        super().__init__()
        self.code_embedding = code_embedding
        self.manifold = code_embedding.manifold
        self.pad_idx = pad_idx
        # Get dimension from the embedding tensor directly
        self.dim = code_embedding.emb.size(-1)

    def forward(self, flat_visits: list):
        """
        flat_visits: list of LongTensor (variable length)
        """
        device = self.code_embedding.emb.device
        
        # 1. Pad sequence: [Batch, Max_Len]
        # We assume flat_visits contains LongTensors
        padded_visits = pad_sequence(flat_visits, batch_first=True, padding_value=self.pad_idx).to(device)
        
        # 2. Create Mask: [Batch, Max_Len] (1 for Real, 0 for Pad)
        mask = (padded_visits != self.pad_idx).float()
        
        # 3. Handle Empty Visits (Prevent NaN division later)
        # Add epsilon to sum to avoid division by zero for empty sets
        mask_sum = mask.sum(dim=1, keepdim=True)
        mask_sum[mask_sum == 0] = 1.0 
        
        # Normalize weights: sum(w) = 1 per row
        weights = mask / mask_sum
        
        # 4. Embed everything: [Batch, Max_Len, Dim]
        z = self.code_embedding(padded_visits)
        
        # 5. Batched Einstein Midpoint
        # geoopt can handle weighted average in one massive matrix op
        # reducedim=1 collapses the 'sequence length' dimension
        midpoint = self.manifold.weighted_midpoint(z, weights=weights, reducedim=[1])
        
        # 6. Map to Tangent Space
        tangent = self.manifold.logmap0(midpoint)
        
        # 7. Explicitly zero out empty visits
        # (Midpoint of padded zeros is technically origin, but just to be safe)
        is_empty = (mask.sum(dim=1, keepdim=True) == 0)
        tangent = torch.where(is_empty, torch.zeros_like(tangent), tangent)
        
        return tangent

class HyperbolicGraphVisitEncoder(nn.Module):
    """
    Hyperbolic visit encoder that uses a hyperbolic code embedding and
    aggregates per-visit codes via a weighted Einstein midpoint.

    Graph structure is injected at the *embedding* level via HDD + code_pair
    pretraining; this encoder then treats those embeddings as graph-aware.

    Args:
        code_embedding: HyperbolicCodeEmbedding (with .manifold and .emb)
        pad_idx: index of the padding code
    """
    def __init__(self, code_embedding, pad_idx: int, scales=None):
        super().__init__()
        self.code_embedding = code_embedding
        self.manifold = code_embedding.manifold
        self.pad_idx = pad_idx
        self.dim = code_embedding.emb.size(-1)
        self.scales = tuple(scales) if scales is not None else (0.5, 1.0, 2.0, 4.0)
        self.output_dim = self.dim * len(self.scales)

    def forward(self, flat_visits: list):
        """
        flat_visits: list of 1D LongTensor of variable length (codes for each visit)

        Returns:
            tangent: [B, output_dim] tangent-space visit representations
                     (B = len(flat_visits); output_dim = dim * len(scales))
        """
        device = self.code_embedding.emb.device

        # [B, V_max]
        padded = pad_sequence(
            flat_visits, batch_first=True, padding_value=self.pad_idx
        ).to(device)

        # mask: [B, V_max], 1 for real codes, 0 for pad
        mask = (padded != self.pad_idx).float()
        mask_sum = mask.sum(dim=1, keepdim=True)
        mask_sum[mask_sum == 0] = 1.0  # avoid division by zero

        weights = mask / mask_sum

        # Embed all codes: [B, V_max, dim] (in hyperbolic space)
        z = self.code_embedding(padded)

        # Multi-scale weighted Einstein midpoints
        tangents_multi = []
        for scale in self.scales:
            w_scaled = weights.pow(scale)
            denom = w_scaled.sum(dim=1, keepdim=True)
            denom[denom == 0] = 1.0
            w_scaled = w_scaled / denom
            midpoint = self.manifold.weighted_midpoint(z, weights=w_scaled, reducedim=[1])
            tangents_multi.append(self.manifold.logmap0(midpoint))

        # Concatenate tangent representations from all scales
        tangent = torch.cat(tangents_multi, dim=-1)

        # Explicitly zero-out empty visits
        is_empty = (mask.sum(dim=1, keepdim=True) == 0)
        tangent = torch.where(is_empty, torch.zeros_like(tangent), tangent)

        return tangent
