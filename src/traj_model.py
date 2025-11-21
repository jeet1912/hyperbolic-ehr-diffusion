import torch
import torch.nn as nn
from diffusion import TimeEmbedding


class TrajectoryEpsModel(nn.Module):
    def __init__(self, dim, T_max, n_layers=2, n_heads=4, ff_dim=128):
        super().__init__()
        self.dim = dim
        self.time_mlp = TimeEmbedding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_t, t):
        """
        x_t: [B, L, dim] (L = max trajectory length)
        t:   [B] timesteps
        """
        B, L, D = x_t.shape
        t_emb = self.time_mlp(t)             # [B, dim]
        t_expanded = t_emb.unsqueeze(1).expand(-1, L, -1)
        h = x_t + t_expanded
        h = self.encoder(h)
        eps_hat = self.proj(h)
        return eps_hat
