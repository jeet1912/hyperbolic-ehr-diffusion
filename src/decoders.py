import torch
import torch.nn as nn
import geoopt
import geoopt.layers

class VisitDecoder(nn.Module):
    """
    Shared Euclidean decoder for both hyperbolic and Euclidean setups.

    Input: visit latent(s) in R^d
      - shape [B, L, d] or [B*L, d]
    Output:
      - logits over real codes (no pad) of shape [B, L, num_codes] or [B*L, num_codes]
    """
    def __init__(self, dim: int, num_codes: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),          
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codes)  # raw logits
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: [B, L, dim] or [B*L, dim]
        returns: logits over codes with matching leading dims
        """
        if v.dim() == 3:
            B, L, D = v.shape
            v_flat = v.view(B * L, D)
            logits_flat = self.net(v_flat)        # [B*L, num_codes]
            return logits_flat.view(B, L, self.num_codes)
        elif v.dim() == 2:
            return self.net(v)                    # [B*L, num_codes]
        else:
            raise ValueError(f"Expected v of shape [B,L,d] or [N,d], got {v.shape}")


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [..., dim]
        residual = x
        out = self.lin1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out

class StrongVisitDecoder(nn.Module):
    """
    More expressive decoder from visit latent(s) to code logits.
    Input: [B, L, dim] or [N, dim]
    Output: logits over real codes [B, L, num_codes] or [N, num_codes]
    """
    def __init__(self, dim: int, num_codes: int,
                 hidden_dim: int = 256, num_res_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes

        self.input_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)
        ])
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden_dim, num_codes)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: [B, L, dim] or [N, dim]
        returns: logits over codes with matching leading dims
        """
        orig_shape = v.shape[:-1]          # e.g. (B, L) or (N,)
        D = v.shape[-1]
        v_flat = v.reshape(-1, D)          # [N, dim]

        h = self.input_proj(v_flat)        # [N, hidden_dim]
        h = self.act(h)
        for blk in self.blocks:
            h = blk(h)                     # [N, hidden_dim]

        logits_flat = self.out_proj(h)     # [N, num_codes]
        return logits_flat.view(*orig_shape, self.num_codes)
    

class HyperbolicDistanceDecoder(nn.Module):
    """
    hyperbolic decoder:
    - Optional tiny hyperbolic linear layer
    - Distance-based logits
    - Temperature scheduling
    - Code-frequency-aware negative sampling
    """
    def __init__(self, code_embedding: geoopt.ManifoldParameter, 
                 manifold: geoopt.PoincareBall,
                 init_temperature: float = 1.0,
                 min_temperature: float = 0.07,
                 use_hyper_lin: bool = True,
                 code_freq: torch.Tensor = None):  # [num_codes], optional
        super().__init__()
        self.manifold = manifold
        self.code_emb = code_embedding
        self.dim = code_embedding.size(-1)
        self.use_hyper_lin = use_hyper_lin
        self.code_freq = code_freq  # for frequency-aware bias

        # Tiny hyperbolic linear (Möbius matrix multiplication)
        if use_hyper_lin:
            self.hyper_lin = geoopt.layers.PoincareMLP(
                in_features=self.dim,
                out_features=self.dim,
                c=manifold.c,
                num_layers=1  # just one layer is enough
            )

        # Temperature (will be updated externally)
        self.register_buffer("temperature", torch.tensor(init_temperature))
        self.min_temperature = min_temperature

        # Frequency bias (log-frequency boost for rare codes)
        if code_freq is not None:
            freq_bias = torch.log(code_freq + 1.0)  # avoid log(0)
            freq_bias = freq_bias - freq_bias.mean()
            self.register_buffer("freq_bias", freq_bias * 0.5)  # scale gently
        else:
            self.register_buffer("freq_bias", None)

    def set_temperature(self, temp: float):
        self.temperature = torch.tensor(max(temp, self.min_temperature), device=self.temperature.device)

    def forward(self, v_tangent: torch.Tensor) -> torch.Tensor:
        """
        v_tangent: [..., dim] in tangent space at origin
        returns: logits [..., num_codes]
        """
        batch_shape = v_tangent.shape[:-1]
        v_flat = v_tangent.view(-1, self.dim)

        # Optional tiny hyperbolic transformation
        if self.use_hyper_lin:
            v_manifold = self.manifold.expmap0(v_flat)
            v_manifold = self.hyper_lin(v_manifold)
            v_flat = self.manifold.logmap0(v_manifold)

        # Back to manifold for distance computation
        v_manifold = self.manifold.expmap0(v_flat)  # [N, dim]
        if hasattr(self.code_emb, "weight"):
            code_points = self.code_emb.weight
        else:
            code_points = self.code_emb

        # Hyperbolic distance squared
        dist_sq = self.manifold.dist2(v_manifold.unsqueeze(-2), code_points.unsqueeze(0))  # [N, 1, C] -> [N, C]
        logits = -dist_sq.squeeze(-2) / (self.temperature ** 2)  # [N, C]

        # Frequency-aware negative sampling boost
        if self.freq_bias is not None:
            logits = logits + self.freq_bias.unsqueeze(0)  # broadcast

        return logits.view(*batch_shape, -1)
