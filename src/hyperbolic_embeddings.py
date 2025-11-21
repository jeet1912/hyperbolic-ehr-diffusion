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
    """
    Map a set of code ids -> tangent-space visit vector.
    """
    def __init__(self, code_embedding: HyperbolicCodeEmbedding):
        super().__init__()
        self.code_embedding = code_embedding
        self.manifold = code_embedding.manifold

    def forward(self, code_ids_batch):
        """
        code_ids_batch: list of 1D LongTensors (variable length), len = B*T
        Returns: tensor [B*T, dim] in tangent space at origin.
        """
        dim = self.code_embedding.emb.shape[-1]
        visit_vecs = []
        for ids in code_ids_batch:
            valid = ids[ids >= 0]
            if valid.numel() == 0:
                visit_vec = torch.zeros(dim, device=ids.device)
            else:
                x = self.code_embedding(valid)             # [k, d] on manifold
                tangents = self.manifold.logmap0(x)        # [k, d] in R^d
                visit_vec = tangents.mean(dim=0)           # [d]
            visit_vecs.append(visit_vec)
        return torch.stack(visit_vecs, dim=0)
