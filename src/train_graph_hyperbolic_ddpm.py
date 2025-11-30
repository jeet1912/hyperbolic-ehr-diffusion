import os
import random
import copy
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_icd_toy import ToyICDHierarchy, sample_toy_trajectories
from data_utils import TrajDataset, make_collate_fn
from decoders import HyperbolicDistanceDecoder
from diffusion import cosine_beta_schedule
from hyperbolic_embeddings import HyperbolicCodeEmbedding, HyperbolicGraphVisitEncoder
from hyperbolic_noise import hyperbolic_forward_noise, hyperbolic_remove_noise
from losses import code_pair_loss
from metrics_toy import traj_stats
from traj_models import TrajectoryEpsModel
from regularizers import radius_regularizer, HyperbolicDiffusionDistance


BATCH_SIZE = 128
EMBED_DIM = 32
MAX_LEN = 6
CODES_PER_VISIT = 4
EPS_LAYERS = 6
EPS_HEADS = 8
EPS_FF_DIM = 1024
TRAIN_LR = 3e-4
TRAIN_EPOCHS = 50
PRETRAIN_EPOCHS = 30
NUM_SAMPLES_FOR_SYNTHETIC = 1000
EARLY_STOP_PATIENCE = 3
EXTRA_DEPTHS = [5]

LAMBDA_RADIUS = 0.003
LAMBDA_PAIR = 0.01
LAMBDA_HDD = 0.02
LAMBDA_RECON_VALUES = [1, 10, 100, 1000, 2000]

# ---- utils copied/adapted from your DDPM script ----

def collect_unique_params(*modules):
    unique = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                unique.append(param)
    return unique

def split_trajectories(
    trajs: Sequence[Sequence[Sequence[int]]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[list, list, list]:
    indices = list(range(len(trajs)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def gather(idxs):
        return [trajs[i] for i in idxs]
    return gather(train_idx), gather(val_idx), gather(test_idx)

def encode_visits(
    visit_enc: HyperbolicGraphVisitEncoder,
    flat_visits: Sequence[torch.Tensor],
    B: int,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    visit_tensors = [v.to(device) for v in flat_visits]
    latents = visit_enc(visit_tensors)  # [B, dim]
    dim = latents.shape[-1]
    return latents.view(B, L, dim)

def build_targets(
    flat_visits: Sequence[torch.Tensor],
    visit_mask: torch.Tensor,
    num_codes: int,
    pad_idx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    visit_mask_flat = visit_mask.view(-1)
    targets = torch.zeros(visit_mask_flat.shape[0], num_codes, device=device)
    for idx, (visit_tensor, is_real) in enumerate(zip(flat_visits, visit_mask_flat)):
        if not bool(is_real.item()):
            continue
        valid_codes = visit_tensor[visit_tensor != pad_idx].long()
        if valid_codes.numel() > 0:
            targets[idx, valid_codes] = 1.0
    return targets, visit_mask_flat

def compute_batch_loss(
    flat_visits: Sequence[torch.Tensor],
    B: int,
    L: int,
    visit_mask: torch.Tensor,
    eps_model: TrajectoryEpsModel,
    visit_enc: HyperbolicGraphVisitEncoder,
    visit_dec: HyperbolicDistanceDecoder,
    tangent_proj: nn.Module,
    code_emb: HyperbolicCodeEmbedding,
    hier: ToyICDHierarchy,
    alphas_cumprod: torch.Tensor,
    T: int,
    device: torch.device,
    lambda_recon: float,
) -> torch.Tensor:
    visit_mask = visit_mask.to(device)
    flat_visits = [v.to(device) for v in flat_visits]
    x0 = encode_visits(visit_enc, flat_visits, B, L, device)  # [B, L, dim]

    t = torch.randint(0, T, (B,), device=device).long()
    a_bar_t = alphas_cumprod[t].view(B, 1, 1)
    sqrt_a = torch.sqrt(a_bar_t)
    sqrt_one_minus = torch.sqrt(1 - a_bar_t)
    eps = torch.randn_like(x0)

    manifold = visit_enc.manifold
    x_t = hyperbolic_forward_noise(manifold, x0, eps, sqrt_a, sqrt_one_minus)
    eps_hat = eps_model(x_t, t, visit_mask=visit_mask)

    mask = visit_mask.unsqueeze(-1).float()
    ddpm_loss = (mask * (eps - eps_hat) ** 2).sum() / mask.sum().clamp_min(1.0)

    decoder_inputs = tangent_proj(x0.reshape(B * L, -1)).view(B, L, -1)
    logits = visit_dec(decoder_inputs)  # [B, L, num_codes]
    num_codes = logits.shape[-1]
    logits_flat = logits.view(B * L, num_codes)
    targets, visit_mask_flat = build_targets(
        flat_visits, visit_mask, num_codes, len(hier.codes), device
    )

    bce = nn.BCEWithLogitsLoss(reduction="none")
    recon_all = bce(logits_flat, targets)
    mask_flat = visit_mask_flat.float().unsqueeze(-1)
    recon = (recon_all * mask_flat).sum() / (mask_flat.sum() * num_codes + 1e-8)

    loss = ddpm_loss + lambda_recon * recon
    return loss

def run_epoch(
    loader,
    eps_model,
    visit_enc,
    visit_dec,
    tangent_proj,
    code_emb,
    hier,
    alphas_cumprod,
    T,
    device,
    lambda_recon,
    optimizer=None,
):
    is_training = optimizer is not None
    modules = [eps_model, visit_enc, visit_dec]
    for m in modules:
        m.train() if is_training else m.eval()
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for flat_visits, B, L, visit_mask in loader:
            visit_mask = visit_mask.to(device)
            loss = compute_batch_loss(
                flat_visits,
                B,
                L,
                visit_mask,
                eps_model,
                visit_enc,
                visit_dec,
                tangent_proj,
                code_emb,
                hier,
                alphas_cumprod,
                T,
                device,
                lambda_recon,
            )
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * B
            total_samples += B
    return total_loss / max(total_samples, 1)

def train_model(
    eps_model,
    visit_enc,
    visit_dec,
    tangent_proj,
    code_emb,
    train_loader,
    val_loader,
    hier,
    alphas_cumprod,
    T,
    device,
    lambda_recon,
    n_epochs,
    lr,
    plot_dir,
    tag,
):
    params = collect_unique_params(eps_model, visit_enc, visit_dec, tangent_proj)
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val = float("inf")
    best_state = {
        "eps": copy.deepcopy(eps_model.state_dict()),
        "enc": copy.deepcopy(visit_enc.state_dict()),
        "code": copy.deepcopy(code_emb.state_dict()),
        "dec": copy.deepcopy(visit_dec.state_dict()),
        "proj": copy.deepcopy(tangent_proj.state_dict()),
    }

    train_losses, val_losses = [], []
    epochs_without_improvement = 0
    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(
            train_loader,
            eps_model,
            visit_enc,
            visit_dec,
            tangent_proj,
            code_emb,
            hier,
            alphas_cumprod,
            T,
            device,
            lambda_recon,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            val_loader,
            eps_model,
            visit_enc,
            visit_dec,
            tangent_proj,
            code_emb,
            hier,
            alphas_cumprod,
            T,
            device,
            lambda_recon,
            optimizer=None,
        )

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"[DDPM] Epoch {epoch:3d} | Train {train_loss:.6f} | Val {val_loss:.6f} "
            f"| lambda_recon={lambda_recon}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "eps": copy.deepcopy(eps_model.state_dict()),
                "enc": copy.deepcopy(visit_enc.state_dict()),
                "code": copy.deepcopy(code_emb.state_dict()),
                "dec": copy.deepcopy(visit_dec.state_dict()),
                "proj": copy.deepcopy(tangent_proj.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"[DDPM] Early stopping at epoch {epoch} after "
                    f"{EARLY_STOP_PATIENCE} epochs without improvement."
                )
                break

    eps_model.load_state_dict(best_state["eps"])
    visit_enc.load_state_dict(best_state["enc"])
    code_emb.load_state_dict(best_state["code"])
    visit_dec.load_state_dict(best_state["dec"])
    tangent_proj.load_state_dict(best_state["proj"])

    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"graph ddpm | depth={tag['depth']} | lambda_recon={tag['lambda_recon']} | "
        f"lambda_radius={tag['lambda_radius']} | lambda_pair={tag['lambda_pair']} | "
        f"lambda_hdd={tag['lambda_hdd']} | lr={lr} | epochs={n_epochs} | "
        f"patience={EARLY_STOP_PATIENCE} | batch={BATCH_SIZE}"
    )
    plt.legend()
    plt.tight_layout()
    fname = (
        f"graph_ddpm_depth{tag['depth']}_lrecon{int(tag['lambda_recon'])}_"
        f"lrad{tag['lambda_radius']}_lpair{tag['lambda_pair']}_lhdd{tag['lambda_hdd']}_"
        f"lr{str(lr).replace('.', 'p')}_ep{n_epochs}.png"
    )
    plt.savefig(os.path.join(plot_dir, fname))
    plt.close()

    return train_losses, val_losses, best_val

def evaluate_recall(
    loader,
    eps_model,
    visit_enc,
    visit_dec,
    tangent_proj,
    alphas_cumprod,
    T,
    device,
    codes_per_visit,
    pad_idx,
    hier,
):
    eps_model.eval()
    visit_enc.eval()
    visit_dec.eval()
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)
            x0 = encode_visits(visit_enc, flat_visits, B, L, device)
            t = torch.randint(0, T, (B,), device=device).long()
            a_bar_t = alphas_cumprod[t].view(B, 1, 1)
            sqrt_a = torch.sqrt(a_bar_t)
            sqrt_one_minus = torch.sqrt(1 - a_bar_t)
            eps = torch.randn_like(x0)
            x_t = hyperbolic_forward_noise(visit_enc.manifold, x0, eps, sqrt_a, sqrt_one_minus)
            eps_hat = eps_model(x_t, t, visit_mask=visit_mask)
            x0_pred = hyperbolic_remove_noise(
                visit_enc.manifold, x_t, eps_hat, sqrt_a, sqrt_one_minus
            )

            decoder_inputs = tangent_proj(x0_pred.view(B * L, -1)).view(B, L, -1)
            logits = visit_dec(decoder_inputs)
            preds = logits.topk(k=codes_per_visit, dim=-1).indices.view(
                B * L, codes_per_visit
            )
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

def sample_trajectories(
    eps_model,
    visit_dec,
    visit_enc,
    tangent_proj,
    hier,
    betas,
    alphas_cumprod,
    alphas_cumprod_prev,
    max_len,
    dim,
    num_samples,
    codes_per_visit,
    device,
    print_examples=3,
):
    eps_model.eval()
    visit_dec.eval()
    manifold = visit_enc.manifold
    with torch.no_grad():
        x_t = torch.randn(num_samples, max_len, dim, device=device)
        visit_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=device)
        T = betas.shape[0]
        for timestep in reversed(range(T)):
            t_batch = torch.full((num_samples,), timestep, dtype=torch.long, device=device)
            eps_hat = eps_model(x_t, t_batch, visit_mask=visit_mask)

            alpha_bar = alphas_cumprod[timestep]
            sqrt_alpha_bar = torch.sqrt(alpha_bar).view(1, 1, 1)
            sqrt_one_minus = torch.sqrt(1 - alpha_bar).view(1, 1, 1)
            x0_est = hyperbolic_remove_noise(
                manifold, x_t, eps_hat, sqrt_alpha_bar, sqrt_one_minus
            )

            if timestep > 0:
                noise = torch.randn_like(x_t)
                alpha_bar_prev = alphas_cumprod_prev[timestep]
                sqrt_alpha_prev = torch.sqrt(alpha_bar_prev).view(1, 1, 1)
                sqrt_one_minus_prev = torch.sqrt(1 - alpha_bar_prev).view(1, 1, 1)
                x_t = hyperbolic_forward_noise(
                    manifold, x0_est, noise, sqrt_alpha_prev, sqrt_one_minus_prev
                )
            else:
                x_t = x0_est

        decoder_inputs = tangent_proj(x_t.view(num_samples * max_len, dim)).view(
            num_samples, max_len, -1
        )
        logits = visit_dec(decoder_inputs)
        decoded_idx = logits.topk(k=codes_per_visit, dim=-1).indices

    decoded_idx = decoded_idx.cpu()
    pad_idx = len(hier.codes)
    synthetic = []
    for sample_idx in range(num_samples):
        traj_visits = []
        for visit_idx in range(max_len):
            visit_codes = [
                int(c)
                for c in decoded_idx[sample_idx, visit_idx].tolist()
                if int(c) < pad_idx
            ]
            traj_visits.append(sorted(set(visit_codes)))
        synthetic.append(traj_visits)

    for i in range(min(print_examples, num_samples)):
        print(f"\n[DDPM] Sample hyperbolic trajectory {i + 1}:")
        for visit_idx, codes in enumerate(synthetic[i]):
            names = [hier.idx2code[c] for c in codes]
            print(f"  Visit {visit_idx + 1}: {names}")

    return synthetic

def correlation_tree_vs_embedding(code_emb, hier, device, num_pairs=5000):
    n_real = len(hier.codes)
    base_tensor = code_emb.emb
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
        d = code_emb.manifold.dist(emb[i].unsqueeze(0), emb[j].unsqueeze(0)).item()
        embed_dists.append(d)

    if not tree_dists:
        return 0.0
    tree = np.array(tree_dists)
    embd = np.array(embed_dists)
    return float(np.corrcoef(tree, embd)[0, 1])

def compute_code_frequency(trajs, num_codes, device):
    counts = torch.zeros(num_codes, dtype=torch.float32, device=device)
    for traj in trajs:
        for visit in traj:
            for code in visit:
                if 0 <= code < num_codes:
                    counts[code] += 1
    if counts.sum() == 0:
        counts += 1.0
    return counts / counts.sum()

def pretrain_embeddings(
    hier,
    device,
    dim,
    pad_idx,
    lambda_radius=LAMBDA_RADIUS,
    lambda_pair=LAMBDA_PAIR,
    lambda_hdd=LAMBDA_HDD,
    n_epochs=PRETRAIN_EPOCHS,
):
    print("\n=== Pretraining hyperbolic graph embeddings (DDPM) ===")
    code_emb = HyperbolicCodeEmbedding(num_codes=len(hier.codes) + 1, dim=dim).to(device)
    visit_enc = HyperbolicGraphVisitEncoder(code_emb, pad_idx=pad_idx).to(device)

    params = collect_unique_params(visit_enc)
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)

    hdd_metric = HyperbolicDiffusionDistance(hier, device=device)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        code_emb.train()
        visit_enc.train()
        loss_rad = radius_regularizer(code_emb)
        loss_pair = code_pair_loss(code_emb, hier, device=device, num_pairs=1024)
        hdd_loss = hdd_metric.embedding_loss(code_emb, device=device, num_pairs=1024)
        loss = lambda_radius * loss_rad + lambda_pair * loss_pair + lambda_hdd * hdd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            code_emb.eval()
            visit_enc.eval()
            val_rad = radius_regularizer(code_emb)
            val_pair = code_pair_loss(code_emb, hier, device=device, num_pairs=1024)
            val_hdd = hdd_metric.embedding_loss(code_emb, device=device, num_pairs=1024)
            val_loss = lambda_radius * val_rad + lambda_pair * val_pair + lambda_hdd * val_hdd

        print(
            f"[Pretrain-DDPM] Epoch {epoch:3d} | "
            f"train={loss.item():.6f} | val={val_loss.item():.6f} | "
            f"rad={lambda_radius} pair={lambda_pair} hdd={lambda_hdd}"
        )

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {
                "code_emb": copy.deepcopy(code_emb.state_dict()),
                "visit_enc": copy.deepcopy(visit_enc.state_dict()),
            }

    if best_state is not None:
        code_emb.load_state_dict(best_state["code_emb"])
        visit_enc.load_state_dict(best_state["visit_enc"])

    ckpt_dir = "results/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"hyperbolic_ddpm_pretrain_rad{lambda_radius}_pair{lambda_pair}_hdd{lambda_hdd}_val{best_val:.4f}.pt",
    )
    torch.save(
        {
            "code_emb": code_emb.state_dict(),
            "visit_enc": visit_enc.state_dict(),
            "extra_depth": hier.extra_depth,
            "lambda_radius": lambda_radius,
            "lambda_pair": lambda_pair,
            "lambda_hdd": lambda_hdd,
            "best_val": best_val,
        },
        ckpt_path,
    )
    print(f"Saved pretraining checkpoint to {ckpt_path}")
    return code_emb, visit_enc, ckpt_path, best_val

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

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for extra_depth in EXTRA_DEPTHS:
        hier = ToyICDHierarchy(extra_depth=extra_depth)
        name = f"hyperbolic_graph_ddpm_depth{hier.max_depth}"

        trajs = sample_toy_trajectories(hier, num_patients=20000)
        train_trajs, val_trajs, test_trajs = split_trajectories(trajs, seed=seed)
        real_stats = traj_stats(trajs, hier)
        print(f"\n{name} | max_depth = {hier.max_depth} | Real stats: {real_stats}")

        pad_idx = len(hier.codes)
        collate = make_collate_fn(pad_idx)
        train_ds = TrajDataset(train_trajs, max_len=MAX_LEN, pad_idx=pad_idx)
        val_ds = TrajDataset(val_trajs, max_len=MAX_LEN, pad_idx=pad_idx)
        test_ds = TrajDataset(test_trajs, max_len=MAX_LEN, pad_idx=pad_idx)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

        code_emb, visit_enc, _, best_pre_val = pretrain_embeddings(
            hier,
            device,
            EMBED_DIM,
            pad_idx,
            lambda_radius=LAMBDA_RADIUS,
            lambda_pair=LAMBDA_PAIR,
            lambda_hdd=LAMBDA_HDD,
            n_epochs=PRETRAIN_EPOCHS,
        )
        pretrained_code = copy.deepcopy(code_emb.state_dict())
        pretrained_enc = copy.deepcopy(visit_enc.state_dict())

        # Diffusion schedule
        T = 1000
        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
        )

        freq = compute_code_frequency(train_trajs, len(hier.codes), device)

        for lambda_recon in LAMBDA_RECON_VALUES:
            print(f"\nTraining hyperbolic Graph DDPM | depth={hier.max_depth} | lambda_recon={lambda_recon}")

            code_emb.load_state_dict(pretrained_code)
            visit_enc.load_state_dict(pretrained_enc)
            for p in code_emb.parameters():
                p.requires_grad = False

            latent_dim = visit_enc.output_dim
            tangent_proj = nn.Linear(latent_dim, EMBED_DIM).to(device)

            visit_dec = HyperbolicDistanceDecoder(
                code_embedding=code_emb.emb,
                manifold=code_emb.manifold,
                code_freq=freq,
            ).to(device)

            eps_model = TrajectoryEpsModel(
                dim=latent_dim,
                n_layers=EPS_LAYERS,
                n_heads=EPS_HEADS,
                ff_dim=EPS_FF_DIM,
                T_max=T,
            ).to(device)

            plot_dir = os.path.join("results", "plots")
            tag = {
                "depth": hier.max_depth,
                "lambda_recon": lambda_recon,
                "lambda_radius": LAMBDA_RADIUS,
                "lambda_pair": LAMBDA_PAIR,
                "lambda_hdd": LAMBDA_HDD,
            }
            train_losses, val_losses, best_val = train_model(
                eps_model,
                visit_enc,
                visit_dec,
                tangent_proj,
                code_emb,
                train_dl,
                val_dl,
                hier,
                alphas_cumprod,
                T,
                device,
                lambda_recon=lambda_recon,
                n_epochs=TRAIN_EPOCHS,
                lr=TRAIN_LR,
                plot_dir=plot_dir,
                tag=tag,
            )
            print(f"[DDPM] depth={hier.max_depth} | lambda_recon={lambda_recon} | pretrain_val={best_pre_val:.6f} | best_val={best_val:.6f}")

            test_recall = evaluate_recall(
                test_dl,
                eps_model,
                visit_enc,
                visit_dec,
                tangent_proj,
                alphas_cumprod,
                T,
                device,
                CODES_PER_VISIT,
                pad_idx,
                hier,
            )
            corr = correlation_tree_vs_embedding(code_emb, hier, device=device)
            print(f"Test Recall@{CODES_PER_VISIT}: {test_recall:.4f}")
            print(f"Tree/embedding correlation: {corr:.4f}")

            synthetic = sample_trajectories(
                eps_model,
                visit_dec,
                visit_enc,
                tangent_proj,
                hier,
                betas,
                alphas_cumprod,
                alphas_cumprod_prev,
                MAX_LEN,
                latent_dim,
                NUM_SAMPLES_FOR_SYNTHETIC,
                CODES_PER_VISIT,
                device,
            )
            stats = traj_stats(synthetic, hier)
            print(f"Synthetic stats (N={len(synthetic)}): {stats}")

            ckpt_dir = "results/checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            model_ckpt = os.path.join(
                ckpt_dir,
                f"hyperbolic_ddpm_lrecon{int(lambda_recon)}_depth{hier.max_depth}_best{best_val:.4f}.pt",
            )
            torch.save(
                {
                    "eps_model": eps_model.state_dict(),
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
            print(f"Saved DDPM model checkpoint to {model_ckpt}")


if __name__ == "__main__":
    main()
