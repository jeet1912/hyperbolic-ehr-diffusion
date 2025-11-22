import torch
import torch.nn as nn
import torch.nn.functional as F

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
