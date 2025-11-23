import torch
import torch.nn as nn

class EuclideanCodeEmbedding(nn.Module):
    def __init__(self, num_codes: int, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(num_codes, dim)

    def forward(self, code_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(code_ids)


class EuclideanVisitEncoder(nn.Module):
    def __init__(self, code_embedding: EuclideanCodeEmbedding, pad_idx: int):
        super().__init__()
        self.code_embedding = code_embedding
        self.pad_idx = pad_idx

    def forward(self, code_ids_batch):
        visit_vecs = []
        d = self.code_embedding.emb.embedding_dim
        device = self.code_embedding.emb.weight.device

        for ids in code_ids_batch:
            ids = ids[ids != self.pad_idx]
            if ids.numel() == 0:
                visit_vecs.append(torch.zeros(d, device=device))
                continue

            x = self.code_embedding(ids)    # [k, d]
            visit_vec = x.mean(dim=0)       # [d]
            visit_vecs.append(visit_vec)

        return torch.stack(visit_vecs, dim=0)

class LearnableVisitEncoder(nn.Module):
    """
    Learnable visit encoder for Euclidean code embeddings.
    Input: list of 1D LongTensors, one per visit (code indices, incl. pad_idx)
    Output: [num_visits, dim] latent visit vectors.
    """
    def __init__(self, code_emb, dim, pad_idx, hidden_dim=128, use_attention=True):
        super().__init__()
        self.code_emb = code_emb
        self.pad_idx = pad_idx
        self.dim = dim
        self.use_attention = use_attention

        in_dim = dim  # after embedding / logmap0 we are in R^dim

        # per-code MLP φ
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # pooling MLP ρ
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

        if use_attention:
            # optional attention-style pooling weights over codes in a visit
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.attn = None

    def _embed_codes(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: [num_codes] LongTensor
        returns: [num_codes, dim] Euclidean vectors
        """
        # drop pads
        mask = ids != self.pad_idx
        ids = ids[mask]
        if ids.numel() == 0:
            # empty visit: return a single zero vector
            return ids.new_zeros(1, self.dim).float()

        if hasattr(self.code_emb, "manifold"):  # hyperbolic
            z = self.code_emb.emb[ids]  # [N, dim] on Poincaré ball
            return self.code_emb.manifold.logmap0(z)  # [N, dim] in tangent space
        else:
            # Euclidean embedding (nn.Embedding)
            return self.code_emb.emb.weight[ids]  # [N, dim]

    def encode_single_visit(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: 1D LongTensor of code indices for a single visit.
        returns: [dim] latent visit vector.
        """
        x = self._embed_codes(ids)               # [N, dim]
        h = self.phi(x)                          # [N, hidden_dim]

        if self.attn is not None:
            # attention weights over codes
            a = self.attn(h)                     # [N, 1]
            a = torch.softmax(a, dim=0)          # [N, 1]
            h_pool = (a * h).sum(dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            h_pool = h.mean(dim=0, keepdim=True)       # [1, hidden_dim]

        v = self.rho(h_pool)                     # [1, dim]
        return v.squeeze(0)                      # [dim]

    def forward(self, flat_visits):
        """
        flat_visits: list of 1D LongTensors (one per visit).
        returns: [num_visits, dim]
        """
        device = self.code_emb.emb.device if hasattr(self.code_emb, "manifold") \
                 else self.code_emb.emb.weight.device
        latents = []
        for ids in flat_visits:
            ids = ids.to(device)
            v = self.encode_single_visit(ids)    # [dim]
            latents.append(v)
        return torch.stack(latents, dim=0)       # [num_visits, dim]
