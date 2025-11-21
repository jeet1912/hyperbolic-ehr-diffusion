import torch
import torch.nn as nn
import geoopt


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
        return self.emb[code_ids]


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
