import argparse
import copy
import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import MimicDataset, make_pad_collate
from hyperbolic_embeddings import HyperbolicCodeEmbedding
from traj_models import TrajectoryVelocityModel
from regularizers import radius_regularizer

# ----------------------------- Hyperparams ----------------------------- #

BATCH_SIZE = 32
TRAIN_LR = 1e-4
TRAIN_EPOCHS = 100
EARLY_STOP_PATIENCE = 5

EMBED_DIM = 128
DROPOUT_RATE = 0.2

LAMBDA_RADIUS = 0.003    # radius reg (pretrain only)
LAMBDA_HDD = 0.02        # HDD-style alignment (pretrain only)
LAMBDA_S = 1.0           # weight for synthetic risk loss (MedDiffusion λ_S)
LAMBDA_D = 1.0           # weight for rectified-flow loss (MedDiffusion λ_D)
LAMBDA_CONSISTENCY = 0.1 # synthetic/real feature consistency

# ----------------------------- 1. THE "BEST BET" CONFIGURATION ----------------------------- #
# Based on your ablation, this combination theoretically yields the highest AUPRC
# while maintaining a small amount of structural alignment.

BEST_BALANCED_CONFIG = {
    # GEOMETRY: Global context won slightly in ablations. Keep it.
    "diffusion_steps": [1, 2, 4, 8, 16],

    # REGULARIZATION: High dropout is crucial for the small MIMIC cohort.
    "dropout": 0.5,
    "train_lr": 1e-4,
    "train_epochs": 100,

    # TASK: Turn off Generative Loss (lambda_s=0).
    # It distracted the model in previous runs.
    # We rely on lambda_hdd alone for structure/correlation.
    "lambda_s": 0.0,
    "lambda_d": 0.0,
    "lambda_consistency": 0.0,

    # STRUCTURE: The "Goldilocks" zone.
    # 0.0 gave -0.002 correlation. 0.02 gave 0.83 correlation.
    # We try 0.01 to relax the constraint while keeping positive correlation.
    "lambda_hdd": 0.01,
    "lambda_radius": 0.003,

    # CAPACITY: 128 dim seems stable.
    "embed_dim": 128,
}


# ----------------------------- 2. THE HDD SWEEP GENERATOR ----------------------------- #
# Use this to find the exact trade-off point between AUPRC and Correlation.

def get_hdd_sweep_configs() -> List[Tuple[str, dict]]:
    experiments: List[Tuple[str, dict]] = []

    # We test magnitudes of HDD constraint while keeping the high-performance architecture fixed.
    # Range: From "Pure Risk" (0.0) to "High Structure" (0.05)

    hdd_values = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05]

    for hdd in hdd_values:
        cfg = copy.deepcopy(BEST_BALANCED_CONFIG)
        cfg["lambda_hdd"] = hdd

        # Naming convention for easy tracking
        tag = f"HDD_Sweep_{hdd}"
        experiments.append((tag, cfg))

    return experiments


def print_summary_table(records: List[dict]):
    if not records:
        print("[HyperMedDiff-Risk] No runs to summarize.")
        return

    headers = ["Run", "Experiment", "ValLoss", "AUROC", "AUPRC", "Accuracy", "F1"]
    rows = []
    for rec in records:
        metrics = rec.get("risk_metrics") or {}
        rows.append(
            [
                str(rec.get("run_index", "")),
                rec.get("experiment_name", ""),
                f"{rec.get('best_val_loss', 0.0):.4f}",
                f"{metrics.get('auroc', 0.0):.4f}",
                f"{metrics.get('auprc', 0.0):.4f}",
                f"{metrics.get('accuracy', 0.0):.4f}",
                f"{metrics.get('f1', 0.0):.4f}",
            ]
        )

    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def format_row(row_vals):
        return " | ".join(
            cell.ljust(col_widths[idx]) for idx, cell in enumerate(row_vals)
        )

    print("[HyperMedDiff-Risk] ==== Ablation Summary ====")
    print(format_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(format_row(row))


def plot_corr_vs_auprc(records: List[dict], output_path: str):
    """Scatter plot with diffusion correlation on X and AUPRC on Y."""
    if not records:
        return
    xs = []
    ys = []
    labels = []
    for rec in records:
        corr = rec.get("post_diff_corr")
        metrics = rec.get("risk_metrics") or {}
        auprc = metrics.get("auprc")
        if corr is None or auprc is None:
            continue
        xs.append(corr)
        ys.append(auprc)
        labels.append(rec.get("experiment_name", ""))
    if not xs:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c="tab:blue")
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha="left", fontsize=8)
    plt.xlabel("Diffusion/Embedding Correlation")
    plt.ylabel("AUPRC")
    plt.title("Correlation vs. AUPRC across HDD Sweep")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ----------------------------- Utils ----------------------------- #

def collect_unique_params(*modules):
    params = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            if not p.requires_grad:
                continue
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def build_diffusion_kernels_from_sequences(
    sequences, vocab_size: int, steps: List[int], device: torch.device
):
    """
    Build random-walk diffusion kernels over the code graph, like HDD.
    """
    adj = torch.zeros(vocab_size, vocab_size, dtype=torch.float32, device=device)
    for patient in sequences:
        for visit in patient:
            u = sorted({code for code in visit if 0 < code < vocab_size})
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    c1, c2 = u[i], u[j]
                    adj[c1, c2] += 1.0
                    adj[c2, c1] += 1.0
    adj = adj + torch.eye(vocab_size, device=device)
    deg = adj.sum(dim=1).clamp_min(1.0)
    row_norm = adj / deg.unsqueeze(1)

    kernels = []
    steps_sorted = sorted(set(steps))
    current = row_norm.clone()
    max_step = steps_sorted[-1]
    for step in range(1, max_step + 1):
        if step == 1:
            current = row_norm.clone()
        else:
            current = torch.matmul(current, row_norm)
        if step in steps_sorted:
            kernels.append(current.clone())
    return torch.stack(kernels, dim=0)   # [K, V, V]


# ----------------------------- Graph Diffusion Visit Encoder ----------------------------- #

class HyperbolicGraphDiffusionLayer(nn.Module):
    """
    Applies multi-scale diffusion over the code graph in tangent space,
    then projects back to a single embedding.
    """
    def __init__(self, manifold, dim, diffusion_kernels: torch.Tensor):
        super().__init__()
        self.manifold = manifold
        self.register_buffer("kernels", diffusion_kernels)
        self.proj = nn.Linear(dim * diffusion_kernels.size(0), dim)

    def forward(self, X_hyp: torch.Tensor) -> torch.Tensor:
        # X_hyp: [V, D] hyperbolic code embeddings
        Z0 = self.manifold.logmap0(X_hyp)      # [V, D] in tangent
        Z_scales = []
        for k in range(self.kernels.size(0)):
            Z_scales.append(torch.matmul(self.kernels[k], Z0))  # [V, D]
        Z_cat = torch.cat(Z_scales, dim=-1)    # [V, K*D]
        return self.proj(Z_cat)                # [V, D]


class GlobalSelfAttentionBlock(nn.Module):
    """
    Global attention block used to refine graph-diffused code embeddings.
    """
    def __init__(self, dim, num_heads=4, ff_dim=128, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D] or [L, D]; returns same shape.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x.squeeze(0)


class GraphHyperbolicVisitEncoderGlobal(nn.Module):
    """
    Your HGDM-style visit encoder:

    codes -> hyperbolic code embedding -> graph diffusion over co-occurrence
          -> global attention -> per-visit tangent latent.
    """
    def __init__(
        self,
        code_emb: HyperbolicCodeEmbedding,
        pad_idx: int,
        diffusion_kernels: torch.Tensor,
        num_attn_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.code_emb = code_emb
        self.manifold = code_emb.manifold
        self.pad_idx = pad_idx
        base = code_emb.emb
        if isinstance(base, nn.Embedding):
            self.dim = base.weight.size(-1)
        else:
            self.dim = base.size(-1)

        self.diff_layer = HyperbolicGraphDiffusionLayer(
            manifold=self.manifold,
            dim=self.dim,
            diffusion_kernels=diffusion_kernels,
        )

        self.attn_layers = nn.ModuleList(
            [
                GlobalSelfAttentionBlock(
                    dim=self.dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_attn_layers)
            ]
        )

        self.output_dim = self.dim
        self.time_freq = nn.Linear(1, self.dim)
        self.time_proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        flat_visits: Sequence[torch.Tensor],
        flat_deltas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.code_emb.emb
        if isinstance(base, nn.Embedding):
            device = base.weight.device
            X_hyp = base.weight
        else:
            device = base.device
            X_hyp = base

        # Graph diffusion in tangent space + global attention on codes
        Z_tan = self.diff_layer(X_hyp)   # [V, D]
        H = Z_tan.unsqueeze(0)           # [1, V, D]
        for layer in self.attn_layers:
            H = layer(H)
        H = H.squeeze(0)                 # [V, D]

        time_embeds = None
        if flat_deltas is not None:
            delta = flat_deltas.to(device).unsqueeze(-1)  # [N, 1]
            freq_term = self.time_freq(delta / 180.0)
            time_term = 1.0 - torch.tanh(freq_term) ** 2
            time_embeds = self.time_proj(time_term)

        visit_latents = []
        for idx, v in enumerate(flat_visits):
            codes = v.to(device)
            codes = codes[codes != self.pad_idx]
            if codes.numel() == 0:
                visit_vec = torch.zeros(self.dim, device=device)
            else:
                h_codes = H[codes]           # [n_codes, D]
                visit_vec = h_codes.mean(dim=0)
            if time_embeds is not None:
                visit_vec = visit_vec + time_embeds[idx]
            visit_latents.append(visit_vec)

        return torch.stack(visit_latents, dim=0)   # [B*L, D] in tangent coords


# ----------------------------- HDD-style code metric ----------------------------- #

class MimicDiffusionMetric:
    """
    HDD-like diffusion profile over the co-occurrence graph.
    """
    def __init__(self, profile: torch.Tensor):
        self.profile = profile
        self.num_codes = profile.shape[0]

    @classmethod
    def from_sequences(
        cls,
        sequences,
        vocab_size: int,
        steps=(1, 2, 4, 8),
        device: torch.device | None = None,
    ):
        device = device or torch.device("cpu")
        adj = torch.zeros(vocab_size, vocab_size, dtype=torch.float32, device=device)
        for patient in sequences:
            for visit in patient:
                visit_codes = sorted({code for code in visit if 0 < code < vocab_size})
                if len(visit_codes) <= 1:
                    continue
                for i in range(len(visit_codes)):
                    for j in range(i + 1, len(visit_codes)):
                        c1, c2 = visit_codes[i], visit_codes[j]
                        adj[c1, c2] += 1.0
                        adj[c2, c1] += 1.0
        deg = adj.sum(dim=1).clamp_min(1.0)
        A_norm = adj / deg.unsqueeze(1)
        eye = torch.eye(vocab_size, device=device)
        features = []
        current = eye
        for _ in steps:
            current = torch.matmul(A_norm, current)
            features.append(current)
        profile = torch.cat(features, dim=-1)   # [V, len(steps)*V]
        return cls(profile)

    def embedding_loss(self, code_emb, device, num_pairs=1024):
        idx_i = torch.randint(0, self.num_codes, (num_pairs,), device=device)
        idx_j = torch.randint(0, self.num_codes, (num_pairs,), device=device)
        diff = self.profile[idx_i] - self.profile[idx_j]
        target = torch.norm(diff, dim=-1)

        base = code_emb.emb
        emb_full = base.weight if isinstance(base, nn.Embedding) else base
        emb = emb_full[: self.num_codes]
        dist = code_emb.manifold.dist(emb[idx_i], emb[idx_j]).squeeze(-1)
        return torch.mean((dist - target) ** 2)


def pretrain_code_embedding(
    sequences,
    vocab_size: int,
    dim: int,
    device: torch.device,
    diffusion_metric: MimicDiffusionMetric,
    lambda_radius: float = LAMBDA_RADIUS,
    lambda_hdd: float = LAMBDA_HDD,
):
    """
    Pretrain HyperbolicCodeEmbedding with radius + HDD-style loss (no labels).
    """
    code_emb = HyperbolicCodeEmbedding(num_codes=vocab_size, dim=dim).to(device)
    params = collect_unique_params(code_emb)
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, 1 + 30):
        code_emb.train()
        loss_rad = radius_regularizer(code_emb)
        loss_hdd = diffusion_metric.embedding_loss(code_emb, device=device, num_pairs=2048)
        loss = lambda_radius * loss_rad + lambda_hdd * loss_hdd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            code_emb.eval()
            val_rad = radius_regularizer(code_emb)
            val_hdd = diffusion_metric.embedding_loss(code_emb, device=device, num_pairs=2048)
            val_loss = lambda_radius * val_rad + lambda_hdd * val_hdd

        print(f"[Pretrain] Epoch {epoch:02d} | train={loss.item():.4f} | val={val_loss.item():.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(code_emb.state_dict())

    if best_state is not None:
        code_emb.load_state_dict(best_state)

    return code_emb


# ----------------------------- Rectified Flow in Hyperbolic Latent Space ----------------------------- #

def compute_visit_deltas(visit_mask: torch.Tensor) -> torch.Tensor:
    """
    Approximate visit-to-visit gaps (Δt) from mask ordering.
    Without explicit timestamps we treat consecutive visits as one-day apart.
    """
    B, L = visit_mask.shape
    deltas = torch.zeros(B, L, dtype=torch.float32, device=visit_mask.device)
    for b in range(B):
        prev_idx = -1
        for l in range(L):
            if visit_mask[b, l] > 0:
                if prev_idx >= 0:
                    deltas[b, l] = float(l - prev_idx)
                else:
                    deltas[b, l] = 0.0
                prev_idx = l
    return deltas


def flatten_visits_from_multihot(
    padded_x: torch.Tensor,
    visit_mask: torch.Tensor,
    pad_idx: int = 0,
    visit_deltas: torch.Tensor | None = None,
) -> Tuple[List[torch.Tensor], torch.Tensor | None, int, int]:
    B, L, V = padded_x.shape
    flat_visits: List[torch.Tensor] = []
    delta_list: List[float] | None = [] if visit_deltas is not None else None
    for b in range(B):
        for l in range(L):
            if visit_mask[b, l] <= 0:
                flat_visits.append(torch.tensor([pad_idx], dtype=torch.long, device=padded_x.device))
                if delta_list is not None:
                    delta_list.append(float(0.0))
                continue
            codes = torch.nonzero(padded_x[b, l], as_tuple=False).squeeze(-1)
            if codes.numel() == 0:
                flat_visits.append(torch.tensor([pad_idx], dtype=torch.long, device=padded_x.device))
            else:
                flat_visits.append(codes)
            if delta_list is not None:
                delta_list.append(float(visit_deltas[b, l].item()))
    flat_delta_tensor = None
    if delta_list is not None:
        flat_delta_tensor = torch.tensor(delta_list, dtype=torch.float32, device=padded_x.device)
    return flat_visits, flat_delta_tensor, B, L


def rectified_flow_loss_hyperbolic(
    velocity_model,
    latents: torch.Tensor,
    visit_mask: torch.Tensor,
    manifold,
    history: torch.Tensor | None = None,
):
    """
    Hyperbolic rectified flow in tangent coordinates (no decoder needed).
    latents: [B, L, D] tangent visit latents (data).
    """
    device = latents.device
    B, L, _ = latents.shape

    x_data = manifold.expmap0(latents)
    eps = torch.randn_like(latents)
    x_noise = manifold.expmap0(eps)
    z_data = manifold.logmap0(x_data)
    z_noise = manifold.logmap0(x_noise)

    t = torch.rand(B, device=device)
    t_view = t.view(B, 1, 1)
    z_t = (1 - t_view) * z_noise + t_view * z_data

    target_velocity = z_data - z_noise
    pred_velocity = velocity_model(z_t, t, visit_mask, history=history)
    loss = (pred_velocity - target_velocity) ** 2
    mask = visit_mask.unsqueeze(-1).float()
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def build_history_context(h_seq: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    """
    Align h_{k-1} with visit k (first visit uses zeros).
    """
    history = torch.zeros_like(h_seq)
    history[:, 1:, :] = h_seq[:, :-1, :]
    return history * visit_mask.unsqueeze(-1).float()


def mimic_traj_stats(trajs: Sequence[Sequence[Sequence[int]]]):
    num_patients = len(trajs)
    total_visits = sum(len(p) for p in trajs)
    total_codes = sum(len(v) for p in trajs for v in p)
    avg_visits = total_visits / num_patients if num_patients else 0.0
    avg_codes = total_codes / total_visits if total_visits else 0.0
    max_visits = max((len(p) for p in trajs), default=0)
    max_codes = max((len(v) for p in trajs for v in p), default=0)
    return {
        "patients": num_patients,
        "avg_visits_per_patient": round(avg_visits, 2),
        "avg_codes_per_visit": round(avg_codes, 2),
        "max_visits": int(max_visits),
        "max_codes": int(max_codes),
    }


def sample_latents_from_flow(
    velocity_model,
    risk_lstm,
    visit_mask: torch.Tensor,
    latent_dim: int,
    steps: int,
    device: torch.device,
):
    """
    Sequential, conditional sampling: each visit latent depends on the
    previous hidden state h_{k-1} with step-wise attention fusion
    (MedDiffusion Eq. 3.9-3.12).
    """
    B, L = visit_mask.shape
    latents = torch.zeros(B, L, latent_dim, device=device)
    h_prev = torch.zeros(B, latent_dim, device=device)
    state = risk_lstm.init_state(B, device)
    dt = 1.0 / steps

    with torch.no_grad():
        for l in range(L):
            active = visit_mask[:, l].bool()
            if not active.any():
                continue

            z = torch.randn(B, latent_dim, device=device)
            mask_vec = active.unsqueeze(-1).float()
            z = z * mask_vec
            history = h_prev.unsqueeze(1)
            visit_mask_step = active.view(B, 1)

            for step_idx in range(steps):
                t = torch.full((B,), step_idx * dt, device=device)
                v = velocity_model(
                    z.unsqueeze(1),
                    t,
                    visit_mask=visit_mask_step,
                    history=history,
                ).squeeze(1)
                fused_z = velocity_model.fuse_latent_step(z, h_prev)
                z = fused_z * mask_vec
                z = z + v * dt
                z = z * mask_vec

            latents[:, l] = z
            new_h, state = risk_lstm.step(z, state, mask=active)
            mask_expand = active.unsqueeze(-1)
            h_prev = torch.where(mask_expand, new_h, h_prev)

    return latents


def diffusion_embedding_correlation(code_emb, diffusion_metric, device, num_pairs=5000):
    """
    Correlation between diffusion-profile distances and hyperbolic distances
    (HDD-like alignment score).
    """
    idx_i = torch.randint(0, diffusion_metric.num_codes, (num_pairs,), device=device)
    idx_j = torch.randint(0, diffusion_metric.num_codes, (num_pairs,), device=device)
    diff = diffusion_metric.profile[idx_i] - diffusion_metric.profile[idx_j]
    target = torch.norm(diff, dim=-1)

    base = code_emb.emb
    emb_full = base.weight if isinstance(base, nn.Embedding) else base
    emb = emb_full[: diffusion_metric.num_codes]
    dist = code_emb.manifold.dist(emb[idx_i], emb[idx_j]).squeeze(-1)
    diff_np = target.detach().cpu().numpy()
    dist_np = dist.detach().cpu().numpy()
    if diff_np.std() == 0 or dist_np.std() == 0:
        return 0.0
    return float(np.corrcoef(diff_np, dist_np)[0, 1])


def plot_training_curves(train_losses, val_losses, output_path, title):
    if not train_losses:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 15))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ----------------------------- Temporal LSTM + Risk Head (MedDiffusion backbone) ----------------------------- #

class TemporalLSTMEncoder(nn.Module):
    """
    MedDiffusion-style hidden state learning:
      [e1,...,eK] -> LSTM -> [h1,...,hK], use hK for risk prediction.
    Here we approximate that by taking the last *real* visit per patient.
    """
    def __init__(self, dim, hidden_dim=None, num_layers=1, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        latents: torch.Tensor,
        visit_mask: torch.Tensor,
        return_sequence: bool = False,
    ):
        """
        latents: [B, L, D]
        visit_mask: [B, L] (1=real, 0=pad)
        returns:
            reps: [B, hidden_dim]
            h_seq (optional): [B, L, hidden_dim]
        """
        B, L, D = latents.shape
        lengths = visit_mask.sum(dim=1).long()  # [B]
        h_seq, _ = self.lstm(latents)           # [B, L, H]

        reps = torch.zeros(B, self.hidden_dim, device=latents.device)
        for b in range(B):
            if lengths[b] > 0:
                reps[b] = h_seq[b, lengths[b] - 1]
            else:
                reps[b] = torch.zeros(self.hidden_dim, device=latents.device)
        if return_sequence:
            return reps, h_seq
        return reps

    def init_state(self, batch_size: int, device: torch.device):
        num_layers = self.lstm.num_layers
        h0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def step(self, latent_step: torch.Tensor, state, mask: torch.Tensor | None = None):
        """
        latent_step: [B, D]
        state: tuple(h, c)
        mask: [B] bool indicating which sequences are active.
        """
        inp = latent_step.unsqueeze(1)
        output, new_state = self.lstm(inp, state)
        output = output.squeeze(1)
        if mask is not None:
            mask_vec = mask.view(1, -1, 1).float()
            h = mask_vec * new_state[0] + (1 - mask_vec) * state[0]
            c = mask_vec * new_state[1] + (1 - mask_vec) * state[1]
            new_state = (h, c)
            output = torch.where(mask.unsqueeze(-1), output, torch.zeros_like(output))
        return output, new_state


class RiskHead(nn.Module):
    """
    Binary risk prediction head (e.g., HF vs non-HF).
    """
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # returns logits [B]
        return self.fc(h).squeeze(-1)


# ----------------------------- MedDiffusion-style metrics ----------------------------- #

def _binary_confusion(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def _safe_div(num, den):
    return num / den if den > 0 else 0.0


def auroc_score(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    tpr = tps / pos
    fpr = fps / neg
    return float(np.trapezoid(tpr, fpr))


def auprc_score(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = np.sum(y_true == 1)
    if pos == 0:
        return 0.0
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    precision = tps / (tps + fps + 1e-8)
    recall = tps / pos
    idx = np.argsort(recall)
    recall_sorted = recall[idx]
    precision_sorted = precision[idx]
    return float(np.trapezoid(precision_sorted, recall_sorted))


def cohen_kappa(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp, tn, fp, fn = _binary_confusion(y_true, y_pred)
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    po = (tp + tn) / total
    p_yes_true = (tp + fn) / total
    p_yes_pred = (tp + fp) / total
    p_no_true = (tn + fp) / total
    p_no_pred = (tn + fn) / total
    pe = p_yes_true * p_yes_pred + p_no_true * p_no_pred
    if pe == 1.0:
        return 0.0
    return float((po - pe) / (1 - pe))


def binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp, tn, fp, fn = _binary_confusion(y_true, y_pred)
    total = tp + tn + fp + fn
    acc = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    kappa = cohen_kappa(y_true, y_pred)
    roc = auroc_score(y_true, y_prob)
    pr = auprc_score(y_true, y_prob)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "kappa": float(kappa),
        "auroc": float(roc),
        "auprc": float(pr),
    }


# ----------------------------- Training / Eval Loops ----------------------------- #

def run_epoch(
    loader,
    velocity_model,
    visit_enc,
    risk_lstm,
    risk_head,
    device,
    lambda_s,
    lambda_d,
    lambda_consistency,
    synthetic_steps: int = 10,
    optimizer=None,
):
    """
    Single epoch of MedDiffusion-style training:

      L = L_real + λ_S L_synth + λ_D L_flow

    No decoder, no reconstruction here.
    """
    is_training = optimizer is not None
    modules = [velocity_model, visit_enc, risk_lstm, risk_head]
    for module in modules:
        module.train() if is_training else module.eval()

    trainable_params = collect_unique_params(*modules)
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for padded_x, labels, visit_mask in loader:
            padded_x = padded_x.to(device)
            labels = labels.float().to(device)
            visit_mask = visit_mask.to(device)
            visit_mask_bool = visit_mask.bool()
            visit_deltas = compute_visit_deltas(visit_mask)

            # Visit latents via hyperbolic graph encoder
            flat_visits, flat_deltas, B, L = flatten_visits_from_multihot(
                padded_x, visit_mask, pad_idx=0, visit_deltas=visit_deltas
            )
            latents = visit_enc(flat_visits, flat_deltas).to(device).view(B, L, -1)  # [B, L, D]

            # Rectified flow loss in hyperbolic space
            reps_real, h_seq = risk_lstm(latents, visit_mask_bool, return_sequence=True)
            history_context = build_history_context(h_seq, visit_mask_bool)
            flow_loss = rectified_flow_loss_hyperbolic(
                velocity_model,
                latents,
                visit_mask_bool,
                visit_enc.manifold,
                history=history_context,
            )

            # Risk on REAL trajectories
            logits_real = risk_head(reps_real)                 # [B]
            loss_real = bce(logits_real, labels)

            # Risk on SYNTHETIC trajectories (flow-generated latents)
            with torch.no_grad():
                latents_synth = sample_latents_from_flow(
                    velocity_model,
                    risk_lstm,
                    visit_mask_bool,
                    latent_dim=latents.size(-1),
                    steps=synthetic_steps,
                    device=device,
                )
            h_synth = risk_lstm(latents_synth, visit_mask_bool)
            logits_synth = risk_head(h_synth)
            loss_synth = bce(logits_synth, labels)

            loss_consistency = F.mse_loss(reps_real, h_synth.detach()) * lambda_consistency
            loss = loss_real + lambda_s * loss_synth + lambda_d * flow_loss + loss_consistency

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    return total_loss / max(total_samples, 1)


def train_risk_model(
    train_loader,
    val_loader,
    velocity_model,
    visit_enc,
    risk_lstm,
    risk_head,
    device,
    lambda_s,
    lambda_d,
    lambda_consistency,
    train_lr,
    train_epochs,
    early_stop_patience,
):
    optimizer = torch.optim.AdamW(
        collect_unique_params(velocity_model, visit_enc, risk_lstm, risk_head),
        lr=train_lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(1, train_epochs + 1):
        train_loss = run_epoch(
            train_loader,
            velocity_model,
            visit_enc,
            risk_lstm,
            risk_head,
            device,
            lambda_s=lambda_s,
            lambda_d=lambda_d,
            lambda_consistency=lambda_consistency,
            synthetic_steps=10,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            val_loader,
            velocity_model,
            visit_enc,
            risk_lstm,
            risk_head,
            device,
            lambda_s=lambda_s,
            lambda_d=lambda_d,
            lambda_consistency=lambda_consistency,
            synthetic_steps=10,
            optimizer=None,
        )

        scheduler.step()
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"[HyperMedDiff-Risk] Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "velocity": copy.deepcopy(velocity_model.state_dict()),
                "visit_enc": copy.deepcopy(visit_enc.state_dict()),
                "risk_lstm": copy.deepcopy(risk_lstm.state_dict()),
                "risk_head": copy.deepcopy(risk_head.state_dict()),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("[HyperMedDiff-Risk] Early stopping.")
                break

    if best_state is not None:
        velocity_model.load_state_dict(best_state["velocity"])
        visit_enc.load_state_dict(best_state["visit_enc"])
        risk_lstm.load_state_dict(best_state["risk_lstm"])
        risk_head.load_state_dict(best_state["risk_head"])

    return best_val, train_history, val_history


def evaluate_risk(
    loader,
    visit_enc,
    risk_lstm,
    risk_head,
    device,
):
    """
    Evaluate risk metrics on REAL data only (MedDiffusion-style main result).
    """
    visit_enc.eval()
    risk_lstm.eval()
    risk_head.eval()

    all_labels = []
    all_probs = []
    with torch.no_grad():
        for padded_x, labels, visit_mask in loader:
            padded_x = padded_x.to(device)
            labels = labels.float().to(device)
            visit_mask = visit_mask.to(device)

            visit_deltas = compute_visit_deltas(visit_mask)
            flat_visits, flat_deltas, B, L = flatten_visits_from_multihot(
                padded_x, visit_mask, pad_idx=0, visit_deltas=visit_deltas
            )
            latents = visit_enc(flat_visits, flat_deltas).to(device).view(B, L, -1)
            h = risk_lstm(latents, visit_mask.bool())
            logits = risk_head(h)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if not all_labels:
        return {}

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    metrics = binary_classification_metrics(y_true, y_prob, threshold=0.5)
    return metrics


# ----------------------------- Main ----------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Hyperbolic Graph Diffusion + Rectified Flow + MedDiffusion-style Risk Modeling."
    )
    parser.add_argument("--pkl", type=str, default="data/mimic_hf_cohort.pkl",
                        help="Path to mimic_hf_cohort pickle.")
    parser.add_argument("--output", type=str, default="results/checkpoints",
                        help="Directory for checkpoints.")
    parser.add_argument("--plot-dir", type=str, default="results/plots",
                        help="Directory for training curves.")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset
    dataset = MimicDataset(args.pkl)
    collate_fn = make_pad_collate(dataset.vocab_size)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=g
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    real_stats = mimic_traj_stats(dataset.x)
    print(f"[HyperMedDiff-Risk] Real trajectory stats: {json.dumps(real_stats, indent=2)}")

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    experiments = get_hdd_sweep_configs()
    print(f"[HyperMedDiff-Risk] Running {len(experiments)} HDD sweep configurations.")
    summary_records = []

    for run_idx, (exp_name, config) in enumerate(experiments, start=1):
        print(
            f"[HyperMedDiff-Risk] ===== Experiment {run_idx}/{len(experiments)}: {exp_name} ====="
        )
        print(json.dumps(config, indent=2))
        diffusion_metric = MimicDiffusionMetric.from_sequences(
            dataset.x,
            vocab_size=dataset.vocab_size,
            steps=config["diffusion_steps"],
            device=device,
        )
        diffusion_kernels = build_diffusion_kernels_from_sequences(
            dataset.x,
            dataset.vocab_size,
            config["diffusion_steps"],
            device,
        )

        code_emb = pretrain_code_embedding(
            dataset.x,
            dataset.vocab_size,
            config["embed_dim"],
            device,
            diffusion_metric,
            lambda_radius=config["lambda_radius"],
            lambda_hdd=config["lambda_hdd"],
        )
        pre_corr = diffusion_embedding_correlation(code_emb, diffusion_metric, device)
        print(f"[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: {pre_corr:.4f}")

        for p in code_emb.parameters():
            p.requires_grad = False

        visit_enc = GraphHyperbolicVisitEncoderGlobal(
            code_emb,
            pad_idx=0,
            diffusion_kernels=diffusion_kernels,
            num_attn_layers=3,
            num_heads=4,
            ff_dim=256,
            dropout=config["dropout"],
        ).to(device)
        latent_dim = visit_enc.output_dim

        velocity_model = TrajectoryVelocityModel(
            dim=latent_dim, n_layers=6, n_heads=8, ff_dim=1024
        ).to(device)

        risk_lstm = TemporalLSTMEncoder(
            dim=latent_dim,
            hidden_dim=latent_dim,
            num_layers=1,
            dropout=config["dropout"],
        ).to(device)
        risk_head = RiskHead(dim=latent_dim).to(device)

        best_val, train_history, val_history = train_risk_model(
            train_loader,
            val_loader,
            velocity_model,
            visit_enc,
            risk_lstm,
            risk_head,
            device,
            lambda_s=config["lambda_s"],
            lambda_d=config["lambda_d"],
            lambda_consistency=config["lambda_consistency"],
            train_lr=config["train_lr"],
            train_epochs=int(config["train_epochs"]),
            early_stop_patience=EARLY_STOP_PATIENCE,
        )
        print(f"[HyperMedDiff-Risk] Best validation total loss (run {run_idx}): {best_val:.4f}")

        plot_path = os.path.join(args.plot_dir, f"{exp_name}.png")
        plot_title = (
            f"{exp_name} | Risk Prediction (MIMIC) | "
            f"lambda_s={config['lambda_s']} | lambda_d={config['lambda_d']} | "
            f"lambda_consistency={config['lambda_consistency']} | dropout={config['dropout']} | "
            f"lr={config['train_lr']} | epochs={config['train_epochs']} | "
            f"lambda_radius={config['lambda_radius']} | lambda_hdd={config['lambda_hdd']} | "
            f"diffusion_steps={'-'.join(str(d) for d in config['diffusion_steps'])} | "
            f"embed_dim={config['embed_dim']}"
        )
        plot_training_curves(train_history, val_history, plot_path, plot_title)
        print(f"[HyperMedDiff-Risk] Saved training curve plot to {plot_path}")

        risk_metrics = evaluate_risk(test_loader, visit_enc, risk_lstm, risk_head, device)
        print("[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):")
        print(json.dumps(risk_metrics, indent=2))

        diff_corr = diffusion_embedding_correlation(code_emb, diffusion_metric, device)
        print(f"[HyperMedDiff-Risk] Diffusion/embedding correlation after training: {diff_corr:.4f}")

        ckpt_path = os.path.join(args.output, f"{exp_name}.pt")
        torch.save(
            {
                "velocity_model": velocity_model.state_dict(),
                "visit_enc": visit_enc.state_dict(),
                "risk_lstm": risk_lstm.state_dict(),
                "risk_head": risk_head.state_dict(),
                "code_emb": code_emb.state_dict(),
                "lambda_s": config["lambda_s"],
                "lambda_d": config["lambda_d"],
                "lambda_consistency": config["lambda_consistency"],
                "lambda_radius": config["lambda_radius"],
                "lambda_hdd": config["lambda_hdd"],
                "dropout": config["dropout"],
                "train_lr": config["train_lr"],
                "train_epochs": config["train_epochs"],
                "best_val_loss": best_val,
                "pretrain_diff_corr": pre_corr,
                "post_diff_corr": diff_corr,
                "risk_metrics": risk_metrics,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        summary_records.append(
            {
                "run_index": run_idx,
                "experiment_name": exp_name,
                "hyperparameters": copy.deepcopy(config),
                "best_val_loss": best_val,
                "risk_metrics": risk_metrics,
                "plot_path": plot_path,
                "checkpoint_path": ckpt_path,
                "post_diff_corr": diff_corr,
            }
        )

    print_summary_table(summary_records)
    corr_plot_path = os.path.join(args.plot_dir, "corr_vs_auprc.png")
    plot_corr_vs_auprc(summary_records, corr_plot_path)
    print(f"[HyperMedDiff-Risk] Saved corr vs. AUPRC plot to {corr_plot_path}")


if __name__ == "__main__":
    main()
