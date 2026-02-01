import argparse
import csv
import copy
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
try:
    from umap import UMAP
except ImportError:
    UMAP = None

from dataset import MimicCsvDataset, make_pad_collate
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

BASELINE_CONFIG = {
    "diffusion_steps": [1, 2, 4, 8],
    "embed_dim": EMBED_DIM,
    "lambda_hdd": LAMBDA_HDD,
    "dropout": DROPOUT_RATE,
    "train_lr": TRAIN_LR,
    "lambda_radius": LAMBDA_RADIUS,
    "lambda_s": LAMBDA_S,
    "lambda_d": LAMBDA_D,
    "lambda_consistency": LAMBDA_CONSISTENCY,
    "train_epochs": TRAIN_EPOCHS,
    "use_attention": True,
    "pretrain_code_emb": True,
    "freeze_code_emb": True,
}


def get_ablation_configs() -> List[Tuple[str, dict]]:
    """Returns ablation configurations."""
    experiments: List[Tuple[str, dict]] = []

    def add_exp(name: str, modifications: dict):
        cfg = copy.deepcopy(BASELINE_CONFIG)
        cfg.update(modifications)
        cfg["diffusion_steps"] = copy.deepcopy(cfg["diffusion_steps"])
        experiments.append((name, cfg))

    # --- 1. BASELINE ---
    add_exp("01_Baseline", {})

    # --- 2. GEOMETRY (Graph Scope) ---
    add_exp("02_NoDiffusion", {"diffusion_steps": [1]})
    add_exp("03_LocalDiff", {"diffusion_steps": [1, 2]})
    add_exp("04_GlobalDiff_Stress", {"diffusion_steps": [1, 2, 4, 8, 16]})

    # --- 3. STRUCTURE (Hyperbolic Alignment) ---
    add_exp("05_NoHDD", {"lambda_hdd": 0.0})
    add_exp("06_StrongHDD", {"lambda_hdd": 0.1})

    # --- 4. REGULARIZATION (Capacity) ---
    add_exp("07_HighDropout", {"dropout": 0.5})
    add_exp("08_SmallDim", {"embed_dim": 64})

    # --- 5. MULTI-TASK (Generative Weight) ---
    add_exp("09_NoSynthRisk", {"lambda_s": 0.0})  # No synthetic BCE term
    add_exp("10_GenFocus", {"lambda_s": 2.0, "lambda_d": 2.0})  # Stronger generative regularization

    # --- 6. Attention stack ---
    # Requires your model to read cfg["use_attention"] and bypass the self-attention block if False.
    add_exp("11_NoAttention", {"use_attention": False})

    # --- 7. Consistency coupling ---
    add_exp("12_NoConsistency", {"lambda_consistency": 0.0})

    # --- 8. Flow / rectified-flow module ---
    # Flow matching off; synthetic branch is skipped when lambda_d == 0.
    add_exp("13_NoFlow", {"lambda_d": 0.0})

    # --- 9. Truly discriminative-only (real risk only) ---
    # Removes ALL synthetic/generative couplings.
    add_exp("14_RealOnlyRisk", {"lambda_s": 0.0, "lambda_d": 0.0, "lambda_consistency": 0.0})

    # --- 10. Pretraining protocol toggles ---
    #  - "pretrain_code_emb": if False, skip code pretraining and initialize randomly.
    add_exp("15_NoPretrain_RandomInit", {"pretrain_code_emb": False, "lambda_radius": 0.0, "lambda_hdd": 0.0})
    #  - "freeze_code_emb": if False, allow embedding fine-tuning during downstream training.
    add_exp("16_UnfreezeEmbeddings", {"freeze_code_emb": False})
    
    # --- 11. isolate HDD vs radius pretraining components ---
    add_exp("17_Pretrain_RadiusOnly", {"lambda_hdd": 0.0})      # keep radius reg
    add_exp("18_Pretrain_HDDOnly", {"lambda_radius": 0.0})     # keep HDD alignment

    return experiments


def aggregate_distortion_stats(distortion_by_depth):
    """
    Weighted aggregation over depths.
    Returns (mean_ratio, std_ratio) or (None, None) if missing.
    """
    if not distortion_by_depth:
        return None, None

    total = 0
    mean_acc = 0.0
    var_acc = 0.0

    for entry in distortion_by_depth:
        count = entry.get("count", 0)
        if count <= 0:
            continue
        mean = entry.get("mean_ratio", 0.0)
        std = entry.get("std_ratio", 0.0)

        total += count
        mean_acc += count * mean
        var_acc += count * (std ** 2 + mean ** 2)

    if total == 0:
        return None, None

    mean = mean_acc / total
    var = var_acc / total - mean ** 2
    std = np.sqrt(max(var, 0.0))
    return float(mean), float(std)


def print_summary_table(records: List[dict]):
    if not records:
        print("[HyperMedDiff-Risk] No runs to summarize.")
        return

    headers = [
        "Run", "Experiment", "ValLoss", "AUROC", "AUPRC", "Accuracy", "F1",
        "Diff–Lat ρ", "Tree–Lat ρ", "Dist μ", "Dist σ"
    ]

    def fmt(x, nd=4):
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "N/A"

    rows = []
    for rec in records:
        metrics = rec.get("risk_metrics") or {}
        auroc = metrics.get("auroc")
        auprc = metrics.get("auprc")
        if auroc is None and "auroc_macro" in metrics:
            auroc = metrics.get("auroc_macro")
        if auprc is None and "auprc_macro" in metrics:
            auprc = metrics.get("auprc_macro")
        rows.append(
            [
                str(rec.get("run_index", "")),
                rec.get("experiment_name", ""),
                fmt(rec.get("best_val_loss", 0.0)),
                fmt(auroc),
                fmt(auprc),
                fmt(metrics.get("accuracy", 0.0)),
                fmt(metrics.get("f1", metrics.get("f1_macro", 0.0))),
                fmt(rec.get("diffusion_latent_spearman")),
                fmt(rec.get("tree_latent_spearman")),
                fmt(rec.get("distortion_depth_mean")),
                fmt(rec.get("distortion_depth_std")),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def format_row(vals):
        return " | ".join(vals[i].ljust(col_widths[i]) for i in range(len(vals)))

    print("[HyperMedDiff-Risk] ==== Ablation Summary ====")
    print(format_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(format_row(row))


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


def get_corr_pairs(
    num_pairs: int,
    path: str,
    valid_indices: Sequence[int],
    seed: int = 42,
) -> torch.Tensor:
    if os.path.exists(path):
        pairs = np.load(path)
        if pairs.ndim == 2 and pairs.shape[1] == 2:
            return torch.from_numpy(pairs).long()
    rng = np.random.default_rng(seed)
    choices = np.array(valid_indices, dtype=np.int64)
    if choices.size == 0 or num_pairs <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    idx_i = rng.choice(choices, size=num_pairs, replace=True)
    idx_j = rng.choice(choices, size=num_pairs, replace=True)
    pairs = np.stack([idx_i, idx_j], axis=1)
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    np.save(path, pairs)
    return torch.from_numpy(pairs).long()


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
    if not steps_sorted:
        raise ValueError("diffusion_steps must be non-empty (e.g., [1])")
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
        orig_2d = x.dim() == 2
        if orig_2d:
            x = x.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x.squeeze(0) if orig_2d else x


class GraphHyperbolicVisitEncoderGlobal(nn.Module):
    """
    Your HGDM-style visit encoder:

    codes -> hyperbolic code embedding -> graph diffusion over co-occurrence
          -> global attention -> per-visit hyperbolic latent.
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
        use_attention: bool = True,
        output_hyperbolic: bool = True,
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

        if use_attention and num_attn_layers > 0:
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
        else:
            self.attn_layers = nn.ModuleList([])

        self.output_dim = self.dim
        self.time_freq = nn.Linear(1, self.dim)
        self.time_proj = nn.Linear(self.dim, self.dim)
        self.output_hyperbolic = output_hyperbolic
        self.scale_factor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

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

        X_hyp = X_hyp * torch.tanh(self.scale_factor)
        # X_hyp = self.manifold.projx(X_hyp)
        if torch.isnan(X_hyp).any():
            print("[WARNING] NaN in code embeddings after projx")

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
            if self.output_hyperbolic:
                visit_vec = self.manifold.expmap0(visit_vec)
                visit_vec = self.manifold.projx(visit_vec)
            visit_latents.append(visit_vec)

        return torch.stack(visit_latents, dim=0)   # [B*L, D] (hyperbolic if enabled)


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
        adj = adj + torch.eye(vocab_size, device=device)
        deg = adj.sum(dim=1).clamp_min(1.0)
        A_norm = adj / deg.unsqueeze(1)
        features = []
        steps_sorted = sorted(set(steps))
        if not steps_sorted:
            return cls(torch.zeros(vocab_size, 0, device=device))
        current = A_norm.clone()
        max_step = steps_sorted[-1]
        for step in range(1, max_step + 1):
            if step == 1:
                current = A_norm.clone()
            else:
                current = torch.matmul(A_norm, current)
            if step in steps_sorted:
                features.append(current)
        profile = torch.cat(features, dim=-1)   # [V, len(steps)*V]
        return cls(profile)

    def embedding_loss(self, code_emb, device, num_pairs=1024, valid_indices: Sequence[int] | None = None):
        if valid_indices is None:
            idx_i = torch.randint(0, self.num_codes, (num_pairs,), device=device)
            idx_j = torch.randint(0, self.num_codes, (num_pairs,), device=device)
        else:
            idx_choices = torch.tensor(valid_indices, device=device, dtype=torch.long)
            if idx_choices.numel() == 0:
                return torch.tensor(0.0, device=device)
            choice_count = idx_choices.numel()
            idx_i = idx_choices[torch.randint(0, choice_count, (num_pairs,), device=device)]
            idx_j = idx_choices[torch.randint(0, choice_count, (num_pairs,), device=device)]
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
    valid_indices: Sequence[int] | None = None,
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
        loss_hdd = diffusion_metric.embedding_loss(
            code_emb, device=device, num_pairs=2048, valid_indices=valid_indices
        )
        loss = lambda_radius * loss_rad + lambda_hdd * loss_hdd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            code_emb.eval()
            val_rad = radius_regularizer(code_emb)
            val_hdd = diffusion_metric.embedding_loss(
                code_emb, device=device, num_pairs=2048, valid_indices=valid_indices
            )
            val_loss = lambda_radius * val_rad + lambda_hdd * val_hdd

        print(f"[Pretrain] Epoch {epoch:02d} | train={loss.item():.4f} | val={val_loss.item():.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(code_emb.state_dict())

    if best_state is not None:
        code_emb.load_state_dict(best_state)

    return code_emb


# ----------------------------- Rectified Flow in Tangent Space ----------------------------- #

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


def rectified_flow_loss(
    velocity_model,
    latents: torch.Tensor,
    visit_mask: torch.Tensor,
    history: torch.Tensor | None = None,
):
    """
    Rectified flow in tangent coordinates (no decoder needed).
    latents: [B, L, D] tangent visit latents (data).
    """
    device = latents.device
    B, _, _ = latents.shape

    z_data = latents
    z_noise = torch.randn_like(latents)

    t = torch.rand(B, device=device)
    t_view = t.view(B, 1, 1)
    z_t = (1 - t_view) * z_noise + t_view * z_data

    target_velocity = z_data - z_noise
    pred_velocity = velocity_model(z_t, t, visit_mask, history=history)
    loss = (pred_velocity - target_velocity) ** 2
    # Average over latent dimensions so flow loss is per-visit, not per-dim sum.
    loss = loss.mean(dim=-1)
    mask = visit_mask.float()
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
    manifold=None,
    output_hyperbolic: bool = False,
):
    """
    Sequential, conditional sampling: each visit latent depends on the
    previous hidden state h_{k-1} with step-wise attention fusion
    (MedDiffusion Eq. 3.9-3.12).
    If output_hyperbolic is True, returns manifold points via expmap0.
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

    if output_hyperbolic and manifold is not None:
        latents = manifold.expmap0(latents)
        return manifold.projx(latents)
    return latents


def diffusion_embedding_faithfulness(
    code_emb,
    diffusion_metric,
    pairs: torch.Tensor,
) -> float:
    """
    Correlation between diffusion-profile distances and hyperbolic distances
    over a fixed set of code pairs.
    """
    if pairs.numel() == 0:
        return 0.0
    code_emb.eval()
    base = code_emb.emb
    emb_full = base.weight if isinstance(base, nn.Embedding) else base
    device = emb_full.device
    with torch.no_grad():
        idx_i = pairs[:, 0].to(device=device)
        idx_j = pairs[:, 1].to(device=device)
        profile = diffusion_metric.profile.to(device=device, dtype=torch.float32)
        emb = emb_full.to(device=device, dtype=torch.float32)[: diffusion_metric.num_codes]
        diff = profile[idx_i] - profile[idx_j]
        target = torch.norm(diff, dim=-1)
        dist = code_emb.manifold.dist(emb[idx_i], emb[idx_j]).squeeze(-1)
        diff_np = target.detach().cpu().numpy()
        dist_np = dist.detach().cpu().numpy()
    if diff_np.std() == 0 or dist_np.std() == 0:
        return 0.0
    return float(np.corrcoef(diff_np, dist_np)[0, 1])


def diffusion_embedding_faithfulness_stats(
    code_emb,
    diffusion_metric,
    pair_sets: Sequence[torch.Tensor],
):
    values = [
        diffusion_embedding_faithfulness(code_emb, diffusion_metric, pairs)
        for pairs in pair_sets
    ]
    if not values:
        return 0.0, 0.0, []
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), [float(v) for v in values]


ROOT_CODE = "__ROOT__"


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def get_label_names(task_name: str) -> List[str]:
    if task_name == "mortality":
        return ["label_mortality"]
    if task_name == "los":
        return ["label_los_gt_7d"]
    if task_name == "readmission":
        return ["label_readmit_14d"]
    if task_name == "diagnosis":
        return [
            "label_septicemia",
            "label_diabetes_without_complication",
            "label_diabetes_with_complications",
            "label_lipid_disorders",
            "label_fluid_electrolyte_disorders",
            "label_essential_hypertension",
            "label_hypertension_with_complications",
            "label_acute_myocardial_infarction",
            "label_coronary_atherosclerosis",
            "label_conduction_disorders",
            "label_cardiac_dysrhythmias",
            "label_congestive_heart_failure",
            "label_acute_cerebrovascular_disease",
            "label_pneumonia",
            "label_copd_bronchiectasis",
            "label_pleurisy_pneumothorax_collapse",
            "label_respiratory_failure",
            "label_other_lower_respiratory",
            "label_other_upper_respiratory",
            "label_other_liver_disease",
            "label_gi_hemorrhage",
            "label_acute_renal_failure",
            "label_chronic_kidney_disease",
            "label_surgical_medical_complications",
            "label_shock",
        ]
    return ["label"]


def log_label_summary(y, task_name: str) -> None:
    y_arr = np.array(y)
    n = y_arr.shape[0]
    label_names = get_label_names(task_name)
    if y_arr.ndim == 1:
        pos = int(y_arr.sum())
        neg = int(n - pos)
        rate = pos / n if n else 0.0
        print(
            f"[HyperMedDiff-Risk] Labels ({label_names[0]}): "
            f"pos={pos} neg={neg} rate={rate:.4f}"
        )
        return
    if y_arr.ndim == 2:
        print(f"[HyperMedDiff-Risk] Labels ({task_name}):")
        for i in range(y_arr.shape[1]):
            name = label_names[i] if i < len(label_names) else f"label_{i}"
            pos = int(y_arr[:, i].sum())
            rate = pos / n if n else 0.0
            print(f"  - {name}: pos={pos} rate={rate:.4f}")


def load_icd_parent_map(tree_path: str) -> Dict[str, str]:
    with open(tree_path, "r") as f:
        raw = f.read().strip()
    if not raw:
        return {}
    if raw.startswith("{") or raw.startswith("["):
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(child): str(parent) for child, parent in data.items()}
        if isinstance(data, list):
            parent_map: Dict[str, str] = {}
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    child, parent = item[0], item[1]
                    parent_map[str(child)] = str(parent)
            return parent_map
        return {}

    parent_map: Dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
        if len(parts) < 2:
            continue
        child, parent = parts[0], parts[1]
        parent_map[str(child)] = str(parent)
    return parent_map


def normalize_icd9(code: str) -> str:
    if not code:
        return code
    code = code.strip().upper()
    if "." in code:
        return code
    if code.startswith("E"):
        return code if len(code) <= 4 else f"{code[:4]}.{code[4:]}"
    if code.startswith("V"):
        return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def normalize_icd10(code: str) -> str:
    if not code:
        return code
    return code.strip().upper().replace(".", "")


def load_icd10_to_icd9_gem(gem_path: str) -> Dict[str, str]:
    if not gem_path or not os.path.exists(gem_path):
        return {}
    mapping: Dict[str, str] = {}
    approx_map: Dict[str, int] = {}
    with open(gem_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            icd9 = str(row.get("icd9cm", "")).strip()
            icd10 = str(row.get("icd10cm", "")).strip()
            if not icd9 or not icd10:
                continue
            no_map = int(row.get("no_map", 0) or 0)
            if no_map:
                continue
            approximate = int(row.get("approximate", 0) or 0)
            icd10 = normalize_icd10(icd10)
            if icd10 in mapping:
                if approximate == 0 and approx_map.get(icd10, 1) != 0:
                    mapping[icd10] = icd9
                    approx_map[icd10] = approximate
                continue
            mapping[icd10] = icd9
            approx_map[icd10] = approximate
    return mapping


def build_icd9_tree_map(
    codes: Sequence[str],
    icd10_to_icd9: Dict[str, str],
) -> Dict[str, str]:
    code_map: Dict[str, str] = {}
    for code in codes:
        if code.startswith("ICD9:"):
            icd9 = normalize_icd9(code.split(":", 1)[1])
            if icd9:
                code_map[code] = icd9
            continue
        if code.startswith("ICD10:"):
            icd10 = normalize_icd10(code.split(":", 1)[1])
            icd9_raw = icd10_to_icd9.get(icd10)
            if icd9_raw:
                code_map[code] = normalize_icd9(icd9_raw)
    return code_map


def build_ancestor_paths(
    codes: Sequence[str],
    parent_map: Dict[str, str],
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    paths: Dict[str, List[str]] = {}
    depths: Dict[str, int] = {}
    visiting = set()

    def get_path(code: str) -> List[str]:
        if code in paths:
            return paths[code]
        if code == ROOT_CODE:
            paths[code] = [ROOT_CODE]
            depths[code] = 0
            return paths[code]
        if code in visiting:
            return [ROOT_CODE, code]
        visiting.add(code)
        parent = parent_map.get(code, ROOT_CODE)
        if parent == code or parent == ROOT_CODE:
            path = [ROOT_CODE, code]
        else:
            path = get_path(parent) + [code]
        visiting.remove(code)
        paths[code] = path
        depths[code] = len(path) - 1
        return path

    all_nodes = set(codes) | set(parent_map.keys()) | set(parent_map.values())
    # Filter out empty / whitespace nodes that can appear due to malformed parent_map entries
    all_nodes = {str(n) for n in all_nodes if n is not None and str(n).strip() != ""}
    for node in all_nodes:
        get_path(node)
    return paths, depths


def lca_depth(path_a: List[str], path_b: List[str]) -> int:
    n = min(len(path_a), len(path_b))
    i = 0
    while i < n and path_a[i] == path_b[i]:
        i += 1
    return i - 1


def sample_tree_latent_pairs(
    code_emb,
    idx_to_code: Dict[int, str],
    ancestor_paths: Dict[str, List[str]],
    depths: Dict[str, int],
    device: torch.device,
    num_pairs: int,
    seed: int,
):
    if not idx_to_code:
        return None, None, None, None
    rng = np.random.default_rng(seed)
    valid_indices = np.array(sorted(idx_to_code.keys()), dtype=np.int64)
    if valid_indices.size == 0:
        return None, None, None, None
    idx_i = rng.choice(valid_indices, size=num_pairs, replace=True)
    idx_j = rng.choice(valid_indices, size=num_pairs, replace=True)

    debug = {
        "total_drawn": int(num_pairs),
        "skipped_same": 0,
        "skipped_missing_code": 0,
        "skipped_missing_path": 0,
        "skipped_bad_lca": 0,
        "skipped_tree_dist_le0": 0,
        "kept": 0,
        "lca_depth_histogram": Counter(),
    }
    tree_dists = []
    lca_depths = []
    valid_i = []
    valid_j = []
    for i, j in zip(idx_i, idx_j):
        if i == j:
            debug["skipped_same"] += 1
            continue
        code_i = idx_to_code.get(int(i))
        code_j = idx_to_code.get(int(j))
        if code_i is None or code_j is None:
            debug["skipped_missing_code"] += 1
            continue
        path_i = ancestor_paths.get(code_i)
        path_j = ancestor_paths.get(code_j)
        if not path_i or not path_j:
            debug["skipped_missing_path"] += 1
            continue
        lca = lca_depth(path_i, path_j)
        if lca < 0:
            debug["skipped_bad_lca"] += 1
            continue
        depth_i = depths.get(code_i, len(path_i) - 1)
        depth_j = depths.get(code_j, len(path_j) - 1)
        dist_tree = (depth_i - lca) + (depth_j - lca)
        if dist_tree <= 0:
            debug["skipped_tree_dist_le0"] += 1
            continue
        tree_dists.append(dist_tree)
        lca_depths.append(lca)
        valid_i.append(int(i))
        valid_j.append(int(j))
        debug["kept"] += 1
        debug["lca_depth_histogram"][int(lca)] += 1

    if not tree_dists:
        return None, None, None, debug

    idx_i_t = torch.tensor(valid_i, device=device, dtype=torch.long)
    idx_j_t = torch.tensor(valid_j, device=device, dtype=torch.long)
    base = code_emb.emb
    emb_full = base.weight if isinstance(base, nn.Embedding) else base
    dist_latent = code_emb.manifold.dist(emb_full[idx_i_t], emb_full[idx_j_t]).squeeze(-1)
    dist_latent_np = dist_latent.detach().cpu().numpy()
    return (
        np.array(tree_dists, dtype=np.float64),
        dist_latent_np,
        np.array(lca_depths, dtype=int),
        debug,
    )


def diffusion_embedding_spearman(
    code_emb,
    diffusion_metric,
    device: torch.device,
    num_pairs: int,
    seed: int,
    valid_indices: Sequence[int] | None = None,
):
    rng = np.random.default_rng(seed)
    if valid_indices is None:
        idx_i = rng.integers(0, diffusion_metric.num_codes, size=num_pairs)
        idx_j = rng.integers(0, diffusion_metric.num_codes, size=num_pairs)
    else:
        choices = np.array(valid_indices, dtype=np.int64)
        if choices.size == 0:
            return 0.0
        idx_i = rng.choice(choices, size=num_pairs, replace=True)
        idx_j = rng.choice(choices, size=num_pairs, replace=True)
    idx_i_t = torch.tensor(idx_i, device=device, dtype=torch.long)
    idx_j_t = torch.tensor(idx_j, device=device, dtype=torch.long)

    diff = diffusion_metric.profile[idx_i_t] - diffusion_metric.profile[idx_j_t]
    target = torch.norm(diff, dim=-1)

    base = code_emb.emb
    emb_full = base.weight if isinstance(base, nn.Embedding) else base
    emb = emb_full[: diffusion_metric.num_codes]
    dist = code_emb.manifold.dist(emb[idx_i_t], emb[idx_j_t]).squeeze(-1)

    diff_np = target.detach().cpu().numpy()
    dist_np = dist.detach().cpu().numpy()
    return spearman_corr(diff_np, dist_np)


def distortion_stats_by_depth(
    tree_dists: np.ndarray,
    latent_dists: np.ndarray,
    lca_depths: np.ndarray,
):
    ratios = latent_dists / tree_dists
    depth_buckets: Dict[int, List[float]] = defaultdict(list)
    for depth, ratio in zip(lca_depths, ratios):
        if np.isfinite(ratio):
            depth_buckets[int(depth)].append(float(ratio))

    stats = []
    for depth in sorted(depth_buckets.keys()):
        vals = np.array(depth_buckets[depth], dtype=np.float64)
        stats.append(
            {
                "depth": int(depth),
                "count": int(vals.size),
                "mean_ratio": float(vals.mean()),
                "std_ratio": float(vals.std()),
            }
        )
    return stats


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


def plot_umap_embeddings(
    embeddings: torch.Tensor,
    output_path: str,
    title: str,
    max_points: int | None = None,
    seed: int = 42,
):
    if UMAP is None:
        print("[HyperMedDiff-Risk] UMAP not installed; skipping diffusion embedding UMAP.")
        return False
    if embeddings.numel() == 0:
        print("[HyperMedDiff-Risk] Empty embeddings; skipping diffusion embedding UMAP.")
        return False

    emb_np = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    if max_points is not None and max_points > 0 and emb_np.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(emb_np.shape[0], size=max_points, replace=False)
        emb_np = emb_np[idx]

    n_neighbors = min(15, emb_np.shape[0] - 1)
    if n_neighbors < 2:
        print("[HyperMedDiff-Risk] Not enough points for UMAP; skipping.")
        return False

    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    coords = reducer.fit_transform(emb_np)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.6)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def plot_distortion_vs_depth(stats, output_path: str, title: str):
    if not stats:
        print("[HyperMedDiff-Risk] No distortion stats to plot; skipping.")
        return False
    depths = [entry["depth"] for entry in stats]
    means = [entry["mean_ratio"] for entry in stats]
    stds = [entry["std_ratio"] for entry in stats]
    counts = [entry["count"] for entry in stats]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.plot(depths, means, linewidth=2)
    plt.scatter(depths, means, s=60, zorder=3)
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    plt.fill_between(depths, lower, upper, alpha=0.2)
    for depth, mean, count in zip(depths, means, counts):
        plt.annotate(f"n={count}", (depth, mean), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9)
    plt.xlabel("LCA depth")
    plt.ylabel("Latent / Tree distance")
    plt.title(title)
    plt.xticks(sorted(set(depths)))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    if len(stats) == 1:
        print("[HyperMedDiff-Risk] Distortion plot has a single depth bucket.")
    return True


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


def multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 2:
        raise ValueError("multilabel_metrics expects 2D arrays")

    num_labels = y_true.shape[1]
    per_label_auroc = []
    per_label_auprc = []
    per_label_f1 = []
    for i in range(num_labels):
        metrics = binary_classification_metrics(
            y_true[:, i], y_prob[:, i], threshold=threshold
        )
        per_label_auroc.append(metrics["auroc"])
        per_label_auprc.append(metrics["auprc"])
        per_label_f1.append(metrics["f1"])

    micro_auroc = auroc_score(y_true.ravel(), y_prob.ravel())
    micro_auprc = auprc_score(y_true.ravel(), y_prob.ravel())

    return {
        "auroc_macro": float(np.mean(per_label_auroc)) if per_label_auroc else 0.0,
        "auprc_macro": float(np.mean(per_label_auprc)) if per_label_auprc else 0.0,
        "f1_macro": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "auroc_micro": float(micro_auroc),
        "auprc_micro": float(micro_auprc),
        "per_label_auprc": per_label_auprc,
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
    code_emb=None,
    check_code_emb_grad: bool = False,
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
    effective_lambda_s = lambda_s
    effective_lambda_consistency = lambda_consistency
    if lambda_d == 0:
        effective_lambda_s = 0.0
        effective_lambda_consistency = 0.0

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
            if getattr(visit_enc, "output_hyperbolic", False):
                latents_tan = visit_enc.manifold.logmap0(latents)
            else:
                latents_tan = latents

            if lambda_d > 0:
                reps_real, h_seq = risk_lstm(latents_tan, visit_mask_bool, return_sequence=True)
                history_context = build_history_context(h_seq, visit_mask_bool)
                flow_loss = rectified_flow_loss(
                    velocity_model,
                    latents_tan,
                    visit_mask_bool,
                    history=history_context,
                )
            else:
                reps_real = risk_lstm(latents_tan, visit_mask_bool, return_sequence=False)
                flow_loss = latents_tan.new_tensor(0.0)

            # Risk on REAL trajectories
            logits_real = risk_head(reps_real)                 # [B]
            loss_real = bce(logits_real, labels)

            loss_synth = latents_tan.new_tensor(0.0)
            loss_consistency = latents_tan.new_tensor(0.0)
            if effective_lambda_s > 0 or effective_lambda_consistency > 0:
                with torch.no_grad():
                    latents_synth = sample_latents_from_flow(
                        velocity_model,
                        risk_lstm,
                        visit_mask_bool,
                        latent_dim=latents_tan.size(-1),
                        steps=synthetic_steps,
                        device=device,
                        manifold=visit_enc.manifold,
                        output_hyperbolic=getattr(visit_enc, "output_hyperbolic", False),
                    )
                if getattr(visit_enc, "output_hyperbolic", False):
                    latents_synth_tan = visit_enc.manifold.logmap0(latents_synth)
                else:
                    latents_synth_tan = latents_synth
                h_synth = risk_lstm(latents_synth_tan, visit_mask_bool)
                if effective_lambda_s > 0:
                    logits_synth = risk_head(h_synth)
                    loss_synth = bce(logits_synth, labels)
                if effective_lambda_consistency > 0:
                    loss_consistency = (
                        F.mse_loss(reps_real, h_synth.detach()) * effective_lambda_consistency
                    )
            loss = (
                loss_real
                + effective_lambda_s * loss_synth
                + lambda_d * flow_loss
                + loss_consistency
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                if check_code_emb_grad and code_emb is not None:
                    if any(p.requires_grad and p.grad is not None for p in code_emb.parameters()):
                        raise RuntimeError("code_emb got gradients despite freezing.")
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
    code_emb=None,
    check_code_emb_grad: bool = False,
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
            code_emb=code_emb,
            check_code_emb_grad=check_code_emb_grad,
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
            code_emb=code_emb,
            check_code_emb_grad=check_code_emb_grad,
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
            if getattr(visit_enc, "output_hyperbolic", False):
                latents = visit_enc.manifold.logmap0(latents)
            h = risk_lstm(latents, visit_mask.bool())
            logits = risk_head(h)
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if not all_labels:
        return {}

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    if y_true.ndim == 2:
        metrics = multilabel_metrics(y_true, y_prob, threshold=0.5)
    else:
        metrics = binary_classification_metrics(y_true, y_prob, threshold=0.5)
    return metrics

def group_split_indices(subject_ids, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(subject_ids)), dtype=np.int64)
    rng.shuffle(unique)

    n = len(unique)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_subj = set(unique[:n_train])
    val_subj = set(unique[n_train:n_train+n_val])
    test_subj = set(unique[n_train+n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for i, s in enumerate(subject_ids):
        if s in train_subj:
            train_idx.append(i)
        elif s in val_subj:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


# ----------------------------- Main ----------------------------- #

# Usage (CSV):
# python3 src/task_prediction_mimic_iv.py --task-csv data/mimiciv/llemr_readmission_task.csv --cohort-csv data/mimiciv/llemr_cohort.csv --task-name readmission --icd-tree data/mimiciii/icd9_parent_map.csv --icd10-gem data/icd9toicd10cmgem.csv

def main():
    parser = argparse.ArgumentParser(
        description="Hyperbolic Graph Diffusion + Rectified Flow + MedDiffusion-style Risk Modeling."
    )
    parser.add_argument("--task-csv", type=str, default=None,
                        help="LLemr task CSV (e.g., llemr_mortality_task.csv).")
    parser.add_argument("--cohort-csv", type=str, default=None,
                        help="LLemr cohort CSV for global subject splits.")
    parser.add_argument("--task-name", type=str, default=None,
                        choices=["mortality", "los", "readmission", "diagnosis"],
                        help="Task name for CSV loader.")
    parser.add_argument("--bin-hours", type=int, default=6,
                        help="Bin size in hours for CSV loader.")
    parser.add_argument("--drop-negative", action="store_true",
                        help="Drop events with negative timestamps.")
    parser.add_argument("--truncate", type=str, default="latest",
                        choices=["latest", "earliest"],
                        help="Truncate long sequences when loading CSVs.")
    parser.add_argument("--t-max", type=int, default=256,
                        help="Max visits for readmission/diagnosis when loading CSVs.")
    parser.add_argument("--output", type=str, default="results/checkpoints",
                        help="Directory for checkpoints.")
    parser.add_argument("--plot-dir", type=str, default="results/plots",
                        help="Directory for training curves.")
    parser.add_argument("--umap", action="store_true",
                        help="Generate UMAP plots for diffusion embeddings.")
    parser.add_argument("--umap-max-points", type=int, default=0,
                        help="If >0, subsample at most this many codes for UMAP.")
    parser.add_argument("--icd-tree", type=str, default=None,
                        help="Path to ICD tree file (child,parent per line or JSON mapping).")
    parser.add_argument(
        "--icd10-gem",
        type=str,
        default="data/icd9toicd10cmgem.csv",
        help="GEM crosswalk CSV used to map ICD10 codes to ICD9 for tree metrics.",
    )
    parser.add_argument("--metric-pairs", type=int, default=5000,
                        help="Number of random code pairs for tree/diffusion metrics.")
    parser.add_argument("--metric-seed", type=int, default=42,
                        help="Random seed for tree/diffusion metric sampling.")
    parser.add_argument("--tree-metric-debug", action="store_true",
                        help="Print detailed tree metric debug stats.")
    args = parser.parse_args()

    if not args.icd_tree:
        parser.error(
            "Missing --icd-tree. Build it first with "
            "scripts/icd9/build_icd9_parent_map.py and pass the CSV path."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Dataset (CSV only)
    if not args.task_csv:
        parser.error("Missing --task-csv.")
    if not args.cohort_csv:
        parser.error("Missing --cohort-csv.")
    if not args.task_name:
        parser.error("Missing --task-name.")
    dataset = MimicCsvDataset(
        task_csv=args.task_csv,
        cohort_csv=args.cohort_csv,
        task_name=args.task_name,
        bin_hours=args.bin_hours,
        drop_negative=args.drop_negative,
        truncate=args.truncate,
        t_max=args.t_max,
    )
    collate_fn = make_pad_collate(dataset.vocab_size)

    train_idx, val_idx, test_idx = group_split_indices(dataset.subject_id, seed=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    test_ds  = torch.utils.data.Subset(dataset, test_idx)


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    real_stats = mimic_traj_stats(dataset.x)
    print(f"[HyperMedDiff-Risk] Real trajectory stats: {json.dumps(real_stats, indent=2)}")
    log_label_summary(dataset.y, args.task_name)

    idx_to_code = {idx: code for code, idx in dataset.code_map.items() if idx != 0}
    valid_code_indices = sorted(idx_to_code.keys())
    codes = list(idx_to_code.values())
    corr_cache_dir = os.path.join("results", "cache")
    corr_seeds = [41, 42, 43, 44, 45]
    corr_pair_sets = []
    corr_pair_paths = []
    for seed in corr_seeds:
        filename = (
            f"mimic_corr_pairs_{args.metric_pairs}_seed{seed}_n{len(valid_code_indices)}.npy"
        )
        path = os.path.join(corr_cache_dir, filename)
        corr_pair_sets.append(
            get_corr_pairs(args.metric_pairs, path, valid_code_indices, seed=seed)
        )
        corr_pair_paths.append(path)
    try:
        parent_map = load_icd_parent_map(args.icd_tree)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load ICD tree. Run scripts/icd9/build_icd9_parent_map.py "
            "and pass the generated CSV via --icd-tree."
        ) from exc

    icd10_to_icd9 = load_icd10_to_icd9_gem(args.icd10_gem)
    icd_code_map = build_icd9_tree_map(codes, icd10_to_icd9)
    idx_to_tree_code = {
        idx: icd_code_map[code]
        for idx, code in idx_to_code.items()
        if code in icd_code_map
    }
    tree_codes = list(idx_to_tree_code.values())
    tree_source = "icd9+gem"

    for code in tree_codes:
        parent_map.setdefault(code, ROOT_CODE)

    ancestor_paths, depth_map = build_ancestor_paths(tree_codes, parent_map)
    print(
        f"[HyperMedDiff-Risk] ICD tree source: {tree_source} | "
        f"tree_codes={len(depth_map)} | total_codes={len(codes)}"
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    experiments = get_ablation_configs()

    print(f"[HyperMedDiff-Risk] Running {len(experiments)} ablation configurations.")
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
        pretrain_code_emb = config.get("pretrain_code_emb", True)
        if pretrain_code_emb:
            code_emb = pretrain_code_embedding(
                dataset.x,
                dataset.vocab_size,
                config["embed_dim"],
                device,
                diffusion_metric,
                lambda_radius=config["lambda_radius"],
                lambda_hdd=config["lambda_hdd"],
                valid_indices=valid_code_indices,
            )
            print("[HyperMedDiff-Risk] Code embedding pretraining enabled.")
        else:
            code_emb = HyperbolicCodeEmbedding(
                num_codes=dataset.vocab_size, dim=config["embed_dim"]
            ).to(device)
            print("[HyperMedDiff-Risk] Code embedding pretraining disabled (random init).")

        freeze_code_emb = config.get("freeze_code_emb", True)
        faithfulness_prefix = "Frozen-code" if freeze_code_emb else "Code"
        pre_faith_mean, pre_faith_std, pre_faith_values = diffusion_embedding_faithfulness_stats(
            code_emb,
            diffusion_metric,
            corr_pair_sets,
        )
        print(
            f"[HyperMedDiff-Risk] {faithfulness_prefix} diffusion faithfulness "
            f"(post-pretrain eval): {pre_faith_mean:.4f} ± {pre_faith_std:.4f}"
        )
        if freeze_code_emb:
            for p in code_emb.parameters():
                p.requires_grad = False
            assert all(not p.requires_grad for p in code_emb.parameters())
        base = code_emb.emb
        weight = base.weight if isinstance(base, nn.Embedding) else base
        print(f"[Check] code_emb.requires_grad = {weight.requires_grad}")

        visit_enc = GraphHyperbolicVisitEncoderGlobal(
            code_emb,
            pad_idx=0,
            diffusion_kernels=diffusion_kernels,
            num_attn_layers=3,
            num_heads=4,
            ff_dim=256,
            dropout=config["dropout"],
            use_attention=config.get("use_attention", True),
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
            code_emb=code_emb,
            check_code_emb_grad=freeze_code_emb,
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

        post_faith_mean, post_faith_std, post_faith_values = diffusion_embedding_faithfulness_stats(
            code_emb,
            diffusion_metric,
            corr_pair_sets,
        )
        print(
            f"[HyperMedDiff-Risk] {faithfulness_prefix} diffusion faithfulness "
            f"(post-train eval): {post_faith_mean:.4f} ± {post_faith_std:.4f}"
        )

        diff_spearman = diffusion_embedding_spearman(
            code_emb,
            diffusion_metric,
            device,
            num_pairs=args.metric_pairs,
            seed=args.metric_seed,
            valid_indices=valid_code_indices,
        )
        print(f"[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: {diff_spearman:.4f}")

        tree_spearman = None
        distortion_stats = None
        distortion_path = None
        tree_dists, latent_dists, lca_depths, tree_debug = sample_tree_latent_pairs(
            code_emb,
            idx_to_tree_code,
            ancestor_paths,
            depth_map,
            device,
            num_pairs=args.metric_pairs,
            seed=args.metric_seed,
        )
        if tree_debug:
            kept = tree_debug.get("kept", 0)
            summary = (
                f"kept={kept} | "
                f"same={tree_debug.get('skipped_same', 0)} | "
                f"missing_code={tree_debug.get('skipped_missing_code', 0)} | "
                f"missing_path={tree_debug.get('skipped_missing_path', 0)} | "
                f"bad_lca={tree_debug.get('skipped_bad_lca', 0)} | "
                f"tree_dist_le0={tree_debug.get('skipped_tree_dist_le0', 0)}"
            )
            if args.tree_metric_debug:
                hist = dict(sorted(tree_debug.get("lca_depth_histogram", {}).items()))
                print(f"[HyperMedDiff-Risk] Tree metric debug: {summary}")
                print(f"[HyperMedDiff-Risk] LCA depth histogram: {hist}")
            else:
                print(f"[HyperMedDiff-Risk] Tree metric debug summary: {summary}")
            if kept:
                lca0 = tree_debug.get("lca_depth_histogram", {}).get(0, 0)
                if lca0 / kept > 0.8:
                    print("[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.")

        if tree_dists is not None:
            tree_label = tree_source
            tree_spearman = spearman_corr(tree_dists, latent_dists)
            print(
                f"[HyperMedDiff-Risk] Tree/embedding Spearman rho ({tree_label}): {tree_spearman:.4f}"
            )
            distortion_stats = distortion_stats_by_depth(tree_dists, latent_dists, lca_depths)
            if distortion_stats:
                print("[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):")
                for entry in distortion_stats:
                    print(
                        f"[HyperMedDiff-Risk] depth={entry['depth']} "
                        f"count={entry['count']} "
                        f"mean={entry['mean_ratio']:.4f} "
                        f"std={entry['std_ratio']:.4f}"
                    )
            distortion_path = os.path.join(args.plot_dir, f"{exp_name}_distortion_depth.png")
            distortion_title = (
                f"MIMICIII | {exp_name} | Distortion vs LCA Depth | tree={tree_label}"
            )
            if plot_distortion_vs_depth(distortion_stats, distortion_path, distortion_title):
                print(f"[HyperMedDiff-Risk] Saved distortion vs depth plot to {distortion_path}")
        else:
            print("[HyperMedDiff-Risk] Tree metrics skipped (insufficient pairs).")

        if args.umap:
            visit_enc.eval()
            with torch.no_grad():
                base = visit_enc.code_emb.emb
                X_hyp = base.weight if isinstance(base, nn.Embedding) else base
                diff_embeds = visit_enc.diff_layer(X_hyp)
                if diff_embeds.size(0) > 1:
                    diff_embeds = diff_embeds[1:]
                else:
                    diff_embeds = diff_embeds[:0]

            umap_path = os.path.join(args.plot_dir, f"{exp_name}_umap.png")
            umap_title = (
                f"MIMICIII | {exp_name} | Diffusion Embeddings UMAP | "
                f"diffusion_steps={'-'.join(str(d) for d in config['diffusion_steps'])} | "
                f"embed_dim={config['embed_dim']}"
            )
            max_points = args.umap_max_points if args.umap_max_points > 0 else None
            if plot_umap_embeddings(
                diff_embeds, umap_path, umap_title, max_points=max_points, seed=42
            ):
                print(f"[HyperMedDiff-Risk] Saved diffusion embedding UMAP to {umap_path}")

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
                "pretrain_diffusion_faithfulness_mean": pre_faith_mean,
                "pretrain_diffusion_faithfulness_std": pre_faith_std,
                "pretrain_diffusion_faithfulness_values": pre_faith_values,
                "posttrain_diffusion_faithfulness_mean": post_faith_mean,
                "posttrain_diffusion_faithfulness_std": post_faith_std,
                "posttrain_diffusion_faithfulness_values": post_faith_values,
                "diffusion_faithfulness_pair_seeds": corr_seeds,
                "diffusion_faithfulness_pair_paths": corr_pair_paths,
                "diffusion_latent_spearman": diff_spearman,
                "tree_latent_spearman": tree_spearman,
                "distortion_by_depth": distortion_stats,
                "distortion_plot_path": distortion_path,
                "icd_tree_source": tree_source,
                "metric_pairs": args.metric_pairs,
                "metric_seed": args.metric_seed,
                "risk_metrics": risk_metrics,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        dist_mean, dist_std = aggregate_distortion_stats(distortion_stats)
        summary_records.append(
            {
                "run_index": run_idx,
                "experiment_name": exp_name,
                "hyperparameters": copy.deepcopy(config),
                "best_val_loss": best_val,
                "risk_metrics": risk_metrics,
                # Faithfulness metrics
                "diffusion_latent_spearman": diff_spearman,
                "tree_latent_spearman": tree_spearman,
                "distortion_depth_mean": dist_mean,
                "distortion_depth_std": dist_std,
                "plot_path": plot_path,
                "distortion_plot_path": distortion_path,
                "checkpoint_path": ckpt_path,
            }
        )

    print_summary_table(summary_records)


if __name__ == "__main__":
    main()
