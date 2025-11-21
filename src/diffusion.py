import torch
import torch.nn as nn
import math


def cosine_beta_schedule(T, s=0.008):
    steps = T
    x = torch.linspace(0, T, steps + 1)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t):
        """
        t: [B] int64 timesteps
        """
        half_dim = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        emb = self.lin1(emb)
        emb = self.act(emb)
        emb = self.lin2(emb)
        return emb  # [B, dim]
