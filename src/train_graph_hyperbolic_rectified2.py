import os
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data_icd_toy import ToyICDHierarchy, sample_toy_trajectories
from data_utils import TrajDataset, make_collate_fn
from decoders import HyperbolicDistanceDecoder
from traj_models import TrajectoryVelocityModel
from losses import code_pair_loss, focal_loss
from metrics_toy import traj_stats
from hyperbolic_embeddings import HyperbolicCodeEmbedding, HyperbolicGraphVisitEncoder
from regularizers import radius_regularizer


BATCH_SIZE = 128
EMBED_DIM = 32
MAX_LEN = 6
TRAIN_EPOCHS = 30
PRETRAIN_EPOCHS = 15
EARLY_STOP_PATIENCE = 5
TRAIN_LR = 3e-4
NUM_SAMPLES_FOR_SYNTHETIC = 1000
EXTRA_DEPTHS = [0, 5]

LAMBDA_RECON_VALUES = [1, 10, 100, 1000, 2000]
DEFAULT_LAMBDA_RECON = LAMBDA_RECON_VALUES[-1]
LAMBDA_RADIUS = 1.0
LAMBDA_PAIR = 0.01

def collect_unique_params(*modules):
    unique = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for p in module.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                unique.append(p)
    return unique

def save_loss_curves(train_losses, val_losses, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train")
    if val_losses:
        plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(tag)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_loss.png"))
    plt.close()

def split_trajectories(trajs, train_ratio=0.8, val_ratio=0.1, seed=42):
    indices = list(range(len(trajs)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def gather(idxs):
        return [trajs[i] for i in idxs]

    return gather(train_idx), gather(val_idx), gather(test_idx)

def rectified_flow_loss_hyperbolic(model, latents: torch.Tensor,
                                   visit_mask: torch.Tensor,
                                   manifold) -> torch.Tensor:
    """
    Hyperbolic rectified flow:
    - sample 'noise' in tangent space
    - map both noise and data to the hyperbolic manifold via expmap0
    - work in logmap0 coordinates to get a geodesic-like interpolation
      and target velocity.
    """
    device = latents.device
    B, L, D = latents.shape

    # 1) Map data latents to manifold
    x_data = manifold.expmap0(latents)         # [B, L, D] manifold points

    # 2) Sample hyperbolic "noise": Gaussian in tangent, then expmap0
    eps = torch.randn_like(latents)
    x_noise = manifold.expmap0(eps)            # [B, L, D]

    # 3) Work in log-coordinates at origin
    z_data = manifold.logmap0(x_data)          # [B, L, D]
    z_noise = manifold.logmap0(x_noise)        # [B, L, D]

    # 4) Draw times and interpolate z_noise -> z_data
    t = torch.rand(B, device=device)           # [B]
    t_view = t.view(B, 1, 1)                   # [B,1,1]

    z_t = (1.0 - t_view) * z_noise + t_view * z_data   # path in log space
    xt = z_t                                           # model sees tangent coords

    # 5) Target velocity in log space
    target_velocity = z_data - z_noise                 # [B, L, D]
    pred_velocity = model(xt, t, visit_mask)           # same shape

    loss = (pred_velocity - target_velocity) ** 2
    mask = visit_mask.unsqueeze(-1).float()
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def compute_code_frequency(trajs, num_codes, device):
    counts = torch.zeros(num_codes, dtype=torch.float32)
    for traj in trajs:
        for visit in traj:
            for code in visit:
                if 0 <= code < num_codes:
                    counts[code] += 1
    if counts.sum() == 0:
        counts += 1.0
    freq = counts / counts.sum()
    return freq.to(device)

def pretrain_embeddings(
    hier,
    device,
    dim,
    pad_idx,
    lambda_radius=LAMBDA_RADIUS,
    lambda_pair=LAMBDA_PAIR,
    n_epochs=PRETRAIN_EPOCHS,
    plot_dir="results/plots",
):
    print("\n=== Pretraining hyperbolic graph embeddings (Rectified) ===")
    code_emb = HyperbolicCodeEmbedding(num_codes=len(hier.codes) + 1, dim=dim, c=1.0).to(device)
    visit_enc = HyperbolicGraphVisitEncoder(code_emb, pad_idx=pad_idx).to(device)

    params = collect_unique_params(visit_enc)
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        code_emb.train()
        visit_enc.train()

        loss_rad = radius_regularizer(code_emb)
        loss_pair = code_pair_loss(code_emb, hier, device=device, num_pairs=1024)
        loss = lambda_radius * loss_rad + lambda_pair * loss_pair
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            code_emb.eval()
            visit_enc.eval()
            val_rad = radius_regularizer(code_emb)
            val_pair = code_pair_loss(code_emb, hier, device=device, num_pairs=1024)
            val_loss = lambda_radius * val_rad + lambda_pair * val_pair

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        print(
            f"[Pretrain] Epoch {epoch:3d} | "
            f"train={loss.item():.6f} | val={val_loss.item():.6f} | "
            f"rad={lambda_radius} pair={lambda_pair}"
        )

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {
                "code_emb": copy.deepcopy(code_emb.state_dict()),
                "visit_enc": copy.deepcopy(visit_enc.state_dict()),
            }

    tag = f"pretrain_rectified2_rad{lambda_radius}_pair{lambda_pair}"
    save_loss_curves(train_losses, val_losses, plot_dir, tag)

    if best_state is not None:
        code_emb.load_state_dict(best_state["code_emb"])
        visit_enc.load_state_dict(best_state["visit_enc"])

    ckpt_dir = "results/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"hyperbolic_rectified_pretrain_rad{lambda_radius}_pair{lambda_pair}_val{best_val:.4f}.pt",
    )
    torch.save(
        {
            "code_emb": code_emb.state_dict(),
            "visit_enc": visit_enc.state_dict(),
            "extra_depth": hier.extra_depth,
            "lambda_radius": lambda_radius,
            "lambda_pair": lambda_pair,
            "best_val": best_val,
        },
        ckpt_path,
    )
    print(f"Saved pretraining checkpoint to {ckpt_path}")
    return code_emb, visit_enc, ckpt_path, best_val

def compute_batch_loss(
    flat_visits,
    B,
    L,
    visit_mask,
    velocity_model,
    visit_enc,
    visit_dec,
    tangent_proj,
    code_emb,
    hier,
    lambda_recon=DEFAULT_LAMBDA_RECON,
):
    latents = visit_enc(flat_visits).contiguous()
    dim = latents.shape[-1]
    latents = latents.view(B, L, dim)

    flow_loss = rectified_flow_loss_hyperbolic(velocity_model, latents, visit_mask, visit_enc.manifold)


    clean_latents = latents.view(B * L, dim)
    decoder_inputs = tangent_proj(clean_latents)
    logits = visit_dec(decoder_inputs)

    pad_idx = len(hier.codes)
    targets = torch.zeros_like(logits)
    visit_mask_flat = visit_mask.view(-1)
    for idx, (visit, is_real) in enumerate(zip(flat_visits, visit_mask_flat)):
        if not bool(is_real.item()):
            continue
        valid_codes = visit[visit != pad_idx].long()
        if valid_codes.numel() > 0:
            targets[idx, valid_codes] = 1.0

    recon_loss = focal_loss(logits, targets, gamma=2.0, alpha=0.25)
    total_loss = flow_loss + lambda_recon * recon_loss
    return total_loss

def run_epoch(
    loader,
    velocity_model,
    visit_enc,
    visit_dec,
    tangent_proj,
    code_emb,
    hier,
    device,
    lambda_recon,
    optimizer=None,
):
    is_training = optimizer is not None
    modules = [velocity_model, visit_enc, visit_dec]
    for m in modules:
        m.train() if is_training else m.eval()

    trainable_params = collect_unique_params(*modules)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)

            loss = compute_batch_loss(
                flat_visits,
                B,
                L,
                visit_mask,
                velocity_model,
                visit_enc,
                visit_dec,
                tangent_proj,
                code_emb,
                hier,
                lambda_recon=lambda_recon,
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    return total_loss / max(total_samples, 1)

def train_rectified(
    velocity_model,
    code_emb,
    visit_enc,
    visit_dec,
    tangent_proj,
    train_loader,
    val_loader,
    hier,
    device,
    lambda_recon=DEFAULT_LAMBDA_RECON,
    n_epochs=TRAIN_EPOCHS,
    plot_dir="results/plots",
    tag="rectified_graph",
):
    optimizer = torch.optim.AdamW(
        collect_unique_params(velocity_model, visit_enc, visit_dec, tangent_proj),
        lr=TRAIN_LR,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    patience = EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(
            train_loader,
            velocity_model,
            visit_enc,
            visit_dec,
            tangent_proj,
            code_emb,
            hier,
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
            code_emb,
            hier,
            device,
            lambda_recon,
            optimizer=None,
        )

        scheduler.step()
        print(
            f"[Rectified] Epoch {epoch:3d} | Train {train_loss:.6f} "
            f"| Val {val_loss:.6f} | lambda_recon={lambda_recon}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "velocity": copy.deepcopy(velocity_model.state_dict()),
                "decoder": copy.deepcopy(visit_dec.state_dict()),
                "code_emb": copy.deepcopy(code_emb.state_dict()),
                "visit_enc": copy.deepcopy(visit_enc.state_dict()),
                "tangent_proj": copy.deepcopy(tangent_proj.state_dict()),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[Rectified] Early stopping.")
                break

    tag_full = f"{tag}_lrecon{int(lambda_recon)}"
    save_loss_curves(train_losses, val_losses, plot_dir, tag_full)

    if best_state is not None:
        velocity_model.load_state_dict(best_state["velocity"])
        visit_dec.load_state_dict(best_state["decoder"])
        code_emb.load_state_dict(best_state["code_emb"])
        visit_enc.load_state_dict(best_state["visit_enc"])
        tangent_proj.load_state_dict(best_state["tangent_proj"])

    return best_val

def evaluate_recall(
    loader,
    visit_enc,
    visit_dec,
    tangent_proj,
    hier,
    device,
    codes_per_visit=4,
):
    visit_enc.eval()
    visit_dec.eval()
    pad_idx = len(hier.codes)

    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)
            latents = visit_enc(flat_visits).contiguous().view(B, L, -1)
            decoder_inputs = tangent_proj(latents.view(B * L, -1))
            logits = visit_dec(decoder_inputs)
            preds = logits.topk(k=codes_per_visit, dim=-1).indices.view(B * L, codes_per_visit)
            mask_flat = visit_mask.view(-1)
            for visit_tensor, mask_value, pred_codes in zip(flat_visits, mask_flat, preds):
                if not bool(mask_value.item()):
                    continue
                true_codes = visit_tensor[visit_tensor != pad_idx].tolist()
                if not true_codes:
                    continue
                pred_set = set(int(c) for c in pred_codes.tolist())
                for code in true_codes:
                    if int(code) in pred_set:
                        total_correct += 1
                total_items += len(true_codes)
    return float(total_correct) / max(total_items, 1)

def correlation_tree_vs_embedding(code_emb, hier, device, num_pairs=5000):
    n_real = len(hier.codes)
    base_emb = code_emb.emb
    base_tensor = base_emb.weight if isinstance(base_emb, nn.Embedding) else base_emb
    emb = base_tensor[:n_real].detach().to(device)

    tree_dists = []
    embed_dists = []
    idx_i = torch.randint(0, n_real, (num_pairs,), device=device)
    idx_j = torch.randint(0, n_real, (num_pairs,), device=device)

    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        if i == j:
            continue
        c1 = hier.idx2code[i]
        c2 = hier.idx2code[j]
        d_tree = hier.tree_distance(c1, c2)
        if d_tree is None:
            continue
        tree_dists.append(d_tree)
        if hasattr(code_emb, "manifold"):
            d = code_emb.manifold.dist(emb[i].unsqueeze(0), emb[j].unsqueeze(0)).item()
        else:
            d = torch.norm(emb[i] - emb[j]).item()
        embed_dists.append(d)

    if not tree_dists:
        return 0.0
    tree_arr = np.array(tree_dists)
    emb_arr = np.array(embed_dists)
    return float(np.corrcoef(tree_arr, emb_arr)[0, 1])

def sample_trajectories_hyperbolic(
    velocity_model,
    visit_dec,
    hier,
    manifold,
    max_len,
    dim,
    num_samples,
    device,
    codes_per_visit=4,
    num_steps=15,
    print_examples=3,
):
    velocity_model.eval()
    visit_dec.eval()
    with torch.no_grad():
        # Start from hyperbolic "noise" in tangent space
        z_t = torch.randn(num_samples, max_len, dim, device=device)
        visit_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=device)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((num_samples,), step * dt, device=device)
            v = velocity_model(z_t, t, visit_mask)
            z_t = z_t + v * dt

        # Final manifold points (if decoder expects manifold coords)
        x_final = manifold.expmap0(z_t)             # [B, L, D]
        logits = visit_dec(x_final)                 # [B, L, num_codes]
        decoded_idx = logits.topk(k=codes_per_visit, dim=-1).indices

    decoded_idx = decoded_idx.cpu()
    pad_idx = len(hier.codes)
    synthetic = []
    for i in range(num_samples):
        traj = []
        for j in range(max_len):
            codes = sorted(
                int(c) for c in decoded_idx[i, j].tolist() if int(c) < pad_idx
            )
            traj.append(codes)
        synthetic.append(traj)

    for idx in range(min(print_examples, num_samples)):
        print(f"\n[Rectified-Hyp] Sample trajectory {idx + 1}:")
        for v_idx, codes in enumerate(synthetic[idx]):
            names = [hier.idx2code[c] for c in codes]
            print(f"  Visit {v_idx + 1}: {names}")

    return synthetic


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    for extra_depth in EXTRA_DEPTHS:
        hier = ToyICDHierarchy(extra_depth=extra_depth)
        name = f"rectified_depth{hier.max_depth}"

        trajs = sample_toy_trajectories(hier, num_patients=20000)
        train_trajs, val_trajs, test_trajs = split_trajectories(trajs)
        real_stats = traj_stats(trajs, hier)
        print(f"\n{name} | max_depth = {hier.max_depth} | Real stats: {real_stats}")

        pad_idx = len(hier.codes)
        # ---- Stage 0: pretrain embeddings ----
        code_emb, visit_enc, ckpt_path, best_pre_val = pretrain_embeddings(
            hier,
            device,
            EMBED_DIM,
            pad_idx,
            lambda_radius=LAMBDA_RADIUS,
            lambda_pair=LAMBDA_PAIR,
            n_epochs=PRETRAIN_EPOCHS,
        )

        # freeze code embeddings for downstream / generative training
        for p in code_emb.parameters():
            p.requires_grad = False

        # ---- Build loaders ----
        collate = make_collate_fn(pad_idx)
        train_dl = DataLoader(
            TrajDataset(train_trajs, MAX_LEN, pad_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate,
        )
        val_dl = DataLoader(
            TrajDataset(val_trajs, MAX_LEN, pad_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
        )
        test_dl = DataLoader(
            TrajDataset(test_trajs, MAX_LEN, pad_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
        )

        latent_dim = visit_enc.output_dim
        pretrained_code = copy.deepcopy(code_emb.state_dict())
        pretrained_enc = copy.deepcopy(visit_enc.state_dict())
        freq = compute_code_frequency(train_trajs, len(hier.codes), device)
        depth_results = []

        for lambda_recon in LAMBDA_RECON_VALUES:
            # reset modules to pretrained states
            code_emb.load_state_dict(pretrained_code)
            visit_enc.load_state_dict(pretrained_enc)
            for p in code_emb.parameters():
                p.requires_grad = False

            tangent_proj = nn.Linear(latent_dim, EMBED_DIM).to(device)
            visit_dec = HyperbolicDistanceDecoder(
                code_embedding=code_emb.emb,
                manifold=code_emb.manifold,
                code_freq=freq,
            ).to(device)
            velocity_model = TrajectoryVelocityModel(
                dim=latent_dim, n_layers=6, n_heads=8, ff_dim=1024
            ).to(device)

            best_val = train_rectified(
                velocity_model,
                code_emb,
                visit_enc,
                visit_dec,
                tangent_proj,
                train_dl,
                val_dl,
                hier,
                device,
                lambda_recon=lambda_recon,
                n_epochs=TRAIN_EPOCHS,
                tag=f"{name}_graph",
            )
            print(
                f"[Summary Rectified2] depth={hier.max_depth} | "
                f"lambda_recon={lambda_recon} | pretrain_val={best_pre_val:.6f} | best_val={best_val:.6f}"
            )

            test_recall = evaluate_recall(test_dl, visit_enc, visit_dec, tangent_proj, hier, device)
            corr = correlation_tree_vs_embedding(code_emb, hier, device=device)
            print(f"Test Recall@4: {test_recall:.4f}")
            print(f"Tree-Embedding Correlation: {corr:.4f}")

            synthetic = sample_trajectories_hyperbolic(
                velocity_model,
                visit_dec,
                hier,
                visit_enc.manifold,
                MAX_LEN,
                latent_dim,
                num_samples=NUM_SAMPLES_FOR_SYNTHETIC,
                device=device,
            )
            stats = traj_stats(synthetic, hier)
            print(f"Synthetic stats (N={len(synthetic)}): {stats}")

            depth_results.append((lambda_recon, best_val, test_recall, corr))

            ckpt_dir = "results/checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            model_ckpt = os.path.join(
                ckpt_dir,
                f"hyperbolic_rectified2_lrecon{int(lambda_recon)}_depth{hier.max_depth}_best{best_val:.4f}.pt",
            )
            torch.save(
                {
                    "velocity_model": velocity_model.state_dict(),
                    "visit_dec": visit_dec.state_dict(),
                    "visit_enc": visit_enc.state_dict(),
                    "tangent_proj": tangent_proj.state_dict(),
                    "code_emb": code_emb.state_dict(),
                    "lambda_recon": lambda_recon,
                    "depth": hier.max_depth,
                    "test_recall": test_recall,
                    "tree_corr": corr,
                },
                model_ckpt,
            )
            print(f"Saved rectified model checkpoint to {model_ckpt}")

        for lambda_recon, best_val, test_recall, corr in depth_results:
            print(
                f"[Depth {hier.max_depth}] lambda_recon={lambda_recon} | "
                f"best_val={best_val:.6f} | test_recall={test_recall:.4f} | corr={corr:.4f}"
            )


if __name__ == "__main__":
    main()
