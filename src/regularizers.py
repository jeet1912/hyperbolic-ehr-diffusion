import torch
import torch.nn as nn
import numpy as np

class HyperbolicDiffusionDistance:
    """
    Graph-based diffusion profile (heat kernel embedding) on ToyICDHierarchy.
    HDD loss tries to match manifold distances to diffusion distances.
    """
    def __init__(self, hier, scales=(0.1, 0.5, 1.0, 2.0), device=None):
        import networkx as nx  # local import to avoid hard dependency at top
        self.hier = hier
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.device = device or torch.device("cpu")
        self._build_graph_embeddings(nx)

    def _build_graph_embeddings(self, nx):
        codes = self.hier.codes
        n = len(codes)
        idx_map = self.hier.code2idx

        A = torch.zeros(n, n, dtype=torch.float32)
        undirected = self.hier.G.to_undirected()
        for u, v in undirected.edges:
            i = idx_map[u]
            j = idx_map[v]
            A[i, j] = 1.0
            A[j, i] = 1.0

        deg = A.sum(dim=-1)
        deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
        I = torch.eye(n)
        L = I - deg_inv_sqrt @ A @ deg_inv_sqrt  # normalized Laplacian

        evals, evecs = torch.linalg.eigh(L)
        self.evals = evals.to(self.device)
        self.evecs = evecs.to(self.device)

        # multi-scale heat-kernel features
        features = []
        for s in self.scales:
            coeff = torch.exp(-s * self.evals)
            phi = self.evecs * coeff.unsqueeze(0)
            features.append(phi)
        self.profile = torch.cat(features, dim=-1).to(self.device)
        self.num_codes = len(codes)

    def distance_tensor(self, idx_i, idx_j):
        prof_i = self.profile[idx_i]
        prof_j = self.profile[idx_j]
        return torch.norm(prof_i - prof_j, dim=-1)

    def embedding_loss(self, code_emb, device, num_pairs=512):
        idx_i = torch.randint(0, self.num_codes, (num_pairs,), device=device)
        idx_j = torch.randint(0, self.num_codes, (num_pairs,), device=device)
        target = self.distance_tensor(idx_i, idx_j)

        base_emb = code_emb.emb
        if isinstance(base_emb, nn.Embedding):
            emb_tensor = base_emb.weight[: self.num_codes]
        else:
            emb_tensor = base_emb[: self.num_codes]

        if hasattr(code_emb, "manifold"):
            v_i = emb_tensor[idx_i]
            v_j = emb_tensor[idx_j]
            dist = code_emb.manifold.dist(v_i, v_j).squeeze(-1)
        else:
            dist = torch.norm(emb_tensor[idx_i] - emb_tensor[idx_j], dim=-1)
        return torch.mean((dist - target) ** 2)


def radius_regularizer(code_emb, target_radius: float = 1.0):
    """
    Encourage all code embeddings to lie on a narrow shell of radius ~target_radius
    (measured from the origin in the manifold).
    """
    base = code_emb.emb
    if isinstance(base, nn.Embedding):
        emb = base.weight
    else:
        emb = base

    if hasattr(code_emb.manifold, "dist0"):
        r = code_emb.manifold.dist0(emb)
    else:
        r = torch.linalg.norm(emb, dim=-1)

    return ((r - target_radius) ** 2).mean()
