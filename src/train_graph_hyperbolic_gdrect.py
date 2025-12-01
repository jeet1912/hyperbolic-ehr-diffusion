import argparse
import copy
import itertools
import os
from typing import List, Sequence, Tuple

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import MimicDataset, make_pad_collate
from decoders import HyperbolicDistanceDecoder
from hyperbolic_embeddings import HyperbolicCodeEmbedding
from traj_models import TrajectoryVelocityModel
from regularizers import radius_regularizer
from losses import focal_loss

BATCH_SIZE = 32         
TRAIN_LR = 1e-4         
TRAIN_EPOCHS = 100      
EARLY_STOP_PATIENCE = 5
EMBED_DIM = 128         
DROPOUT_RATE = 0.2      
LAMBDA_RECON = 200.0      
LAMBDA_RADIUS = 0.003   
LAMBDA_PAIR = 0.01      
LAMBDA_HDD = 0.02       

# DIFFUSION
DIFFUSION_STEPS = [1, 2, 4, 8]


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
    return torch.stack(kernels, dim=0)


class HyperbolicGraphDiffusionLayer(nn.Module):
    def __init__(self, manifold, dim, diffusion_kernels: torch.Tensor):
        super().__init__()
        self.manifold = manifold
        self.register_buffer("kernels", diffusion_kernels)
        self.proj = nn.Linear(dim * diffusion_kernels.size(0), dim)

    def forward(self, X_hyp: torch.Tensor) -> torch.Tensor:
        Z0 = self.manifold.logmap0(X_hyp)
        Z_scales = []
        for k in range(self.kernels.size(0)):
            Z_scales.append(torch.matmul(self.kernels[k], Z0))
        Z_cat = torch.cat(Z_scales, dim=-1)
        return self.proj(Z_cat)


class GlobalSelfAttentionBlock(nn.Module):
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
        if x.dim() == 2:
            x = x.unsqueeze(0)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x.squeeze(0)


class GraphHyperbolicVisitEncoderGlobal(nn.Module):
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

    def forward(self, flat_visits: Sequence[torch.Tensor]) -> torch.Tensor:
        base = self.code_emb.emb
        if isinstance(base, nn.Embedding):
            device = base.weight.device
            X_hyp = base.weight
        else:
            device = base.device
            X_hyp = base
        Z_tan = self.diff_layer(X_hyp)
        H = Z_tan
        for layer in self.attn_layers:
            H = layer(H)

        visit_latents = []
        for v in flat_visits:
            codes = v.to(device)
            codes = codes[codes != self.pad_idx]
            if codes.numel() == 0:
                visit_latents.append(torch.zeros(self.dim, device=device))
                continue
            h_codes = H[codes]
            visit_latents.append(h_codes.mean(dim=0))

        return torch.stack(visit_latents, dim=0)


class MimicDiffusionMetric:
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
        profile = torch.cat(features, dim=-1)
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
    diffusion_metric,
):
    code_emb = HyperbolicCodeEmbedding(num_codes=vocab_size, dim=dim).to(device)
    params = collect_unique_params(code_emb)
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, 1 + 30):
        code_emb.train()
        loss_rad = radius_regularizer(code_emb)
        loss_hdd = diffusion_metric.embedding_loss(code_emb, device=device, num_pairs=2048)
        loss = LAMBDA_RADIUS * loss_rad + LAMBDA_HDD * loss_hdd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            code_emb.eval()
            val_rad = radius_regularizer(code_emb)
            val_hdd = diffusion_metric.embedding_loss(code_emb, device=device, num_pairs=2048)
            val_loss = LAMBDA_RADIUS * val_rad + LAMBDA_HDD * val_hdd

        print(f"[Pretrain] Epoch {epoch:02d} | train={loss.item():.4f} | val={val_loss.item():.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(code_emb.state_dict())

    if best_state is not None:
        code_emb.load_state_dict(best_state)

    return code_emb


def flatten_visits_from_multihot(
    padded_x: torch.Tensor,
    visit_mask: torch.Tensor,
    pad_idx: int = 0,
) -> Tuple[List[torch.Tensor], int, int]:
    B, L, V = padded_x.shape
    flat_visits: List[torch.Tensor] = []
    for b in range(B):
        for l in range(L):
            if visit_mask[b, l] <= 0:
                flat_visits.append(torch.tensor([pad_idx], dtype=torch.long, device=padded_x.device))
                continue
            codes = torch.nonzero(padded_x[b, l], as_tuple=False).squeeze(-1)
            if codes.numel() == 0:
                flat_visits.append(torch.tensor([pad_idx], dtype=torch.long, device=padded_x.device))
            else:
                flat_visits.append(codes)
    return flat_visits, B, L


def rectified_flow_loss_hyperbolic(
    velocity_model,
    latents: torch.Tensor,
    visit_mask: torch.Tensor,
    manifold,
):
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
    pred_velocity = velocity_model(z_t, t, visit_mask)
    loss = (pred_velocity - target_velocity) ** 2
    mask = visit_mask.unsqueeze(-1).float()
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def compute_code_frequency(sequences, vocab_size: int) -> torch.Tensor:
    counts = torch.ones(vocab_size, dtype=torch.float32)
    for patient in sequences:
        for visit in patient:
            for code in visit:
                if 0 < code < vocab_size:
                    counts[code] += 1
    return counts / counts.sum()


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


def convert_multihot_to_sequences(multihot: torch.Tensor, mask: torch.Tensor):
    sequences = []
    for i in range(multihot.size(0)):
        patient = []
        for j in range(multihot.size(1)):
            if mask[i, j] <= 0:
                continue
            codes = torch.nonzero(multihot[i, j], as_tuple=False).squeeze(-1).tolist()
            patient.append(codes)
        sequences.append(patient)
    return sequences


def logits_to_multihot(logits: torch.Tensor, k: int = 64) -> torch.Tensor:
    """
    Turn logits into a multi-hot with exactly k active codes per non-padded visit
    (except PAD=0, which we always zero out). Used only for sampling.
    """
    probs = logits.sigmoid()
    _, topk_idx = probs.topk(k=k, dim=-1)
    multihot = torch.zeros_like(probs)
    multihot.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=multihot.dtype))
    multihot[..., 0] = 0  # never predict PAD
    return multihot


def sample_from_flow(
    velocity_model,
    visit_dec,
    tangent_proj,
    mask_template: torch.Tensor,
    latent_dim: int,
    steps: int,
    device: torch.device,
):
    B, L = mask_template.shape
    latents = torch.randn(B, L, latent_dim, device=device)
    visit_mask = mask_template.bool().to(device)
    mask_expand = visit_mask.unsqueeze(-1)
    dt = 1.0 / steps
    for n in range(steps):
        t_n = torch.full((B,), n * dt, device=device)
        v1 = velocity_model(latents, t_n, visit_mask)
        latents_pred = latents + dt * v1
        latents_pred = torch.where(mask_expand, latents_pred, latents)
        t_np1 = t_n + dt
        v2 = velocity_model(latents_pred, t_np1, visit_mask)
        latents = latents + 0.5 * dt * (v1 + v2)
        latents = torch.where(mask_expand, latents, torch.zeros_like(latents))
    logits = visit_dec(tangent_proj(latents.view(B * L, -1))).view(B, L, -1)
    logits[..., 0] = -1e9
    samples = logits_to_multihot(logits, k=64).cpu()
    return convert_multihot_to_sequences(samples, mask_template.cpu())


def diffusion_embedding_correlation(code_emb, diffusion_metric, device, num_pairs=5000):
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


def run_epoch(
    loader,
    velocity_model,
    visit_enc,
    visit_dec,
    tangent_proj,
    device,
    lambda_recon,
    optimizer=None,
):
    is_training = optimizer is not None
    modules = [velocity_model, visit_enc, visit_dec, tangent_proj]
    for module in modules:
        module.train() if is_training else module.eval()

    trainable_params = collect_unique_params(*modules)

    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for padded_x, _, visit_mask in loader:
            padded_x = padded_x.to(device)
            visit_mask = visit_mask.to(device)

            flat_visits, B, L = flatten_visits_from_multihot(padded_x, visit_mask, pad_idx=0)
            latents = visit_enc(flat_visits).to(device).view(B, L, -1)

            flow_loss = rectified_flow_loss_hyperbolic(
                velocity_model, latents, visit_mask.bool(), visit_enc.manifold
            )
            clean_latents = latents.view(B * L, -1)
            decoder_inputs = tangent_proj(clean_latents)
            logits = visit_dec(decoder_inputs).view(B, L, -1)

            targets = padded_x
            mask = visit_mask.unsqueeze(-1)
            recon = focal_loss(logits, targets, reduction="none")
            recon = (recon * mask).sum() / (mask.sum() * targets.size(-1) + 1e-8)

            loss = flow_loss + lambda_recon * recon

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    return total_loss / max(total_samples, 1)


def train_rectified(
    train_loader,
    val_loader,
    velocity_model,
    visit_enc,
    visit_dec,
    tangent_proj,
    device,
    lambda_recon,
):
    optimizer = torch.optim.AdamW(
        collect_unique_params(velocity_model, visit_enc, visit_dec, tangent_proj),
        lr=TRAIN_LR,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(1, TRAIN_EPOCHS + 1):
        train_loss = run_epoch(
            train_loader,
            velocity_model,
            visit_enc,
            visit_dec,
            tangent_proj,
            device,
            lambda_recon,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            val_loader,
            velocity_model,
            visit_enc,
            visit_dec,
            tangent_proj,
            device,
            lambda_recon,
            optimizer=None,
        )

        scheduler.step()
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"[Rect-GD] Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "velocity": copy.deepcopy(velocity_model.state_dict()),
                "visit_enc": copy.deepcopy(visit_enc.state_dict()),
                "visit_dec": copy.deepcopy(visit_dec.state_dict()),
                "tangent_proj": copy.deepcopy(tangent_proj.state_dict()),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("[Rect-GD] Early stopping triggered.")
                break

    if best_state is not None:
        velocity_model.load_state_dict(best_state["velocity"])
        visit_enc.load_state_dict(best_state["visit_enc"])
        visit_dec.load_state_dict(best_state["visit_dec"])
        tangent_proj.load_state_dict(best_state["tangent_proj"])

    return best_val, train_history, val_history


def evaluate(loader, velocity_model, visit_enc, visit_dec, tangent_proj, device):
    velocity_model.eval()
    visit_enc.eval()
    visit_dec.eval()
    tangent_proj.eval()

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for padded_x, _, visit_mask in loader:
            padded_x = padded_x.to(device)
            visit_mask = visit_mask.to(device)
            flat_visits, B, L = flatten_visits_from_multihot(padded_x, visit_mask, pad_idx=0)
            latents = visit_enc(flat_visits).to(device).view(B, L, -1)
            logits = visit_dec(tangent_proj(latents.view(B * L, -1))).view(B, L, -1)

            mask = visit_mask.unsqueeze(-1)
            recon = focal_loss(logits, padded_x, reduction="none")
            recon = (recon * mask).sum() / (mask.sum() * padded_x.size(-1) + 1e-8)

            total_loss += recon.item() * B
            total_samples += B
    return total_loss / max(total_samples, 1)


def evaluate_recall(
    loader,
    visit_enc,
    visit_dec,
    tangent_proj,
    device,
    k: int = 4,
) -> float:
    visit_enc.eval()
    visit_dec.eval()
    tangent_proj.eval()

    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for padded_x, _, visit_mask in loader:
            padded_x = padded_x.to(device)
            visit_mask = visit_mask.to(device)
            flat_visits, B, L = flatten_visits_from_multihot(padded_x, visit_mask, pad_idx=0)
            latents = visit_enc(flat_visits).to(device).view(B, L, -1)
            logits = visit_dec(tangent_proj(latents.view(B * L, -1))).view(B, L, -1)
            preds = logits.topk(k=k, dim=-1).indices

            for b in range(B):
                for l in range(L):
                    if visit_mask[b, l] <= 0:
                        continue
                    true_codes = torch.nonzero(padded_x[b, l], as_tuple=False).squeeze(-1).tolist()
                    true_codes = [c for c in true_codes if c > 0]
                    if not true_codes:
                        continue
                    pred_set = set(int(c) for c in preds[b, l].tolist())
                    for code in true_codes:
                        if int(code) in pred_set:
                            total_correct += 1
                    total_items += len(true_codes)

    return float(total_correct) / max(total_items, 1)


def main():
    parser = argparse.ArgumentParser(description="Rectified Flow on MIMIC HF cohort with hyperbolic embeddings.")
    parser.add_argument("--pkl", type=str, default="data/mimic_hf_cohort.pkl", help="Path to mimic_hf_cohort pickle.")
    parser.add_argument("--output", type=str, default="results/checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--plot-dir", type=str, default="results/plots", help="Directory for training plots.")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    dataset = MimicDataset(args.pkl)
    collate_fn = make_pad_collate(dataset.vocab_size)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=g)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    real_stats = mimic_traj_stats(dataset.x)
    print(f"[Rect-GD] Real trajectory stats: {json.dumps(real_stats, indent=2)}")

    diffusion_metric = MimicDiffusionMetric.from_sequences(
        dataset.x, vocab_size=dataset.vocab_size, steps=DIFFUSION_STEPS, device=device
    )
    freq = compute_code_frequency(dataset.x, dataset.vocab_size).to(device)
    diffusion_kernels = build_diffusion_kernels_from_sequences(
        dataset.x, dataset.vocab_size, DIFFUSION_STEPS, device
    )
    code_emb = pretrain_code_embedding(
        dataset.x, dataset.vocab_size, EMBED_DIM, device, diffusion_metric
    )
    for p in code_emb.parameters():
        p.requires_grad = False
    pre_corr = diffusion_embedding_correlation(code_emb, diffusion_metric, device)
    print(f"[Rect-GD] Diffusion/embedding correlation after pretraining: {pre_corr:.4f}")
    visit_dec = HyperbolicDistanceDecoder(
        code_embedding=code_emb.emb,
        manifold=code_emb.manifold,
        code_freq=freq,
    ).to(device)

    visit_enc = GraphHyperbolicVisitEncoderGlobal(
        code_emb,
        pad_idx=0,
        diffusion_kernels=diffusion_kernels,
        num_attn_layers=3,
        num_heads=4,
        ff_dim=256,
        dropout=0.0,
    ).to(device)
    latent_dim = visit_enc.output_dim
    tangent_proj = torch.nn.Linear(latent_dim, EMBED_DIM).to(device)
    velocity_model = TrajectoryVelocityModel(dim=latent_dim, n_layers=6, n_heads=8, ff_dim=1024).to(device)

    best_val, train_history, val_history = train_rectified(
        train_loader,
        val_loader,
        velocity_model,
        visit_enc,
        visit_dec,
        tangent_proj,
        device,
        lambda_recon=LAMBDA_RECON,
    )
    print(f"[Rect-GD] Best validation total loss: {best_val:.4f}")

    os.makedirs(args.plot_dir, exist_ok=True)
    plot_name = f"graph_gdrect_best_{best_val:.4f}_mimic.png"
    plot_path = os.path.join(args.plot_dir, plot_name)
    plot_title = (
        "Graph-GD Rectified (MIMIC) | "
        f"lambda_recon={LAMBDA_RECON} | lambda_radius={LAMBDA_RADIUS} | "
        f"lambda_hdd={LAMBDA_HDD} | lr={TRAIN_LR} | epochs={TRAIN_EPOCHS}"
    )
    plot_training_curves(train_history, val_history, plot_path, plot_title)
    print(f"[Rect-GD] Saved training curve plot to {plot_path}")

    test_loss = evaluate(test_loader, velocity_model, visit_enc, visit_dec, tangent_proj, device)
    print(f"[Rect-GD] Test recon focal: {test_loss:.4f}")
    test_recall = evaluate_recall(
        test_loader, visit_enc, visit_dec, tangent_proj, device, k=4
    )
    print(f"[Rect-GD] Test Recall@4: {test_recall:.4f}")
    test_recall64 = evaluate_recall(
        test_loader, visit_enc, visit_dec, tangent_proj, device, k=64
    )
    print(f"[Rect-GD] Test Recall@64: {test_recall64:.4f}")

    diff_corr = diffusion_embedding_correlation(code_emb, diffusion_metric, device)
    print(f"[Rect-GD] Diffusion/embedding correlation after training: {diff_corr:.4f}")

    sample_batch = next(iter(test_loader))
    mask_template = sample_batch[2].to(device)
    random_mask = (torch.rand_like(mask_template) > 0.25).float()
    ensure_visit = (random_mask.sum(dim=1, keepdim=True) == 0)
    random_mask[:, 0:1] = torch.where(
        ensure_visit, torch.ones_like(random_mask[:, 0:1]), random_mask[:, 0:1]
    )
    synthetic = sample_from_flow(
        velocity_model,
        visit_dec,
        tangent_proj,
        random_mask,
        latent_dim,
        steps=64,
        device=device,
    )
    synthetic_stats = mimic_traj_stats(synthetic)
    print(f"[Rect-GD] Synthetic stats: {json.dumps(synthetic_stats, indent=2)}")

    os.makedirs(args.output, exist_ok=True)
    ckpt_path = os.path.join(args.output, f"graph_gdrect_best_{best_val:.4f}.pt")
    torch.save(
        {
            "velocity_model": velocity_model.state_dict(),
            "visit_enc": visit_enc.state_dict(),
            "visit_dec": visit_dec.state_dict(),
            "tangent_proj": tangent_proj.state_dict(),
            "code_emb": code_emb.state_dict(),
            "lambda_recon": LAMBDA_RECON,
            "test_recon_focal": test_loss,
            "test_recall@4": test_recall,
            "test_recall@64": test_recall64,
            "pretrain_diff_corr": pre_corr,
            "post_diff_corr": diff_corr,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
