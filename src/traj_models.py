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
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_t, t, visit_mask=None):
        """
        x_t: [B, L, dim]  (L = max trajectory length)
        t:   [B] timesteps
        visit_mask: [B, L] bool
            True  = real visit
            False = pad visit
        """
        B, L, D = x_t.shape
        t_emb = self.time_mlp(t)             # [B, dim]
        t_expanded = t_emb.unsqueeze(1).expand(-1, L, -1)
        h = x_t + t_expanded

        # nn.TransformerEncoder expects True where it should IGNORE (pad positions)
        src_key_padding_mask = None
        if visit_mask is not None:
            src_key_padding_mask = ~visit_mask  # [B, L]

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        eps_hat = self.proj(h)
        return eps_hat

class TrajectoryVelocityModel(nn.Module):
    """
    model for hyperbolic rectified flow.
    Input: points in tangent space at origin
    Output: velocity in tangent space
    """
    def __init__(self, dim: int, n_layers: int = 4, n_heads: int = 8, ff_dim: int = 512):
        super().__init__()
        self.dim = dim

        # Time embedding: t in [0,1] → scale to [0,1000] for stability
        self.time_mlp = TimeEmbedding(dim)

        # Bigger, stronger transformer — this is what wins
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Final projection to velocity
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.history_proj = nn.Linear(dim, dim)
        self.att_gate = nn.Linear(dim * 2, 2)

        # ------------- MedDiffusion step-wise attention (Eq. 3.11–3.12) ------------- #
        self.Wa = nn.Linear(dim, 2)
        self.Wh = nn.Linear(dim, dim)

    def forward(
        self,
        x_tangent: torch.Tensor,
        t: torch.Tensor,
        visit_mask: torch.Tensor = None,
        history: torch.Tensor | None = None,
    ):
        """
        x_tangent:   [B, L, dim] or [N, dim]  — tangent vectors at origin
        t:           [B] or [N]              — float time in [0,1]
        visit_mask:  [B, L] bool             — True = real visit
        """
        if x_tangent.dim() == 2:
            # [N, dim] → treat as [N, 1, dim]
            x_tangent = x_tangent.unsqueeze(1)
            if visit_mask is not None:
                visit_mask = visit_mask.view(-1, 1)

        B, L, D = x_tangent.shape

        # Time embedding
        t_emb = self.time_mlp(t.float() * 999 + 1)  # map [0,1] → [1,1000]
        t_emb = t_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, dim]

        fused = self._fuse_with_history(x_tangent, history)
        h = fused + t_emb

        # Padding mask: Transformer ignores positions where mask is True
        src_key_padding_mask = None if visit_mask is None else ~visit_mask

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        velocity = self.velocity_head(h)  # [B, L, dim]

        return velocity

    def _fuse_with_history(self, latent: torch.Tensor, history: torch.Tensor | None) -> torch.Tensor:
        if history is None:
            return latent
        if history.dim() == 2:
            history = history.unsqueeze(1)
        h_mapped = self.Wh(history.to(latent.dtype))
        fusion_input = latent + h_mapped
        gamma = torch.softmax(self.Wa(fusion_input), dim=-1)
        gamma_e = gamma[..., :1]
        gamma_h = gamma[..., 1:]
        return gamma_e * latent + gamma_h * h_mapped

    def fuse_latent_step(self, z_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Implements MedDiffusion 3.3.3 step-wise aggregation:
            ê = γ_e · z_t + γ_h · W_h(h_prev)
        """
        h_mapped = self.Wh(h_prev)
        fusion_input = z_t + h_mapped
        gamma = torch.softmax(self.Wa(fusion_input), dim=-1)
        gamma_e = gamma[..., :1]
        gamma_h = gamma[..., 1:2]
        return gamma_e * z_t + gamma_h * h_mapped
