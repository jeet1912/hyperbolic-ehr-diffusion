import torch
import torch.nn as nn

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
