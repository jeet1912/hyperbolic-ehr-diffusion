import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import geoopt

from data_icd_toy import ToyICDHierarchy, sample_toy_trajectories
from hyperbolic_embeddings import HyperbolicCodeEmbedding, HyperbolicVisitEncoder
from euclidean_embeddings import EuclideanCodeEmbedding, LearnableVisitEncoder
from traj_models import TrajectoryVelocityModel
from metrics_toy import traj_stats
from data_utils import TrajDataset
from losses import code_pair_loss, focal_loss
from decoders import StrongVisitDecoder, HyperbolicDistanceDecoder


def collect_unique_params(*modules):
    """Return a list of unique trainable parameters from given modules."""
    unique = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                unique.append(param)
    return unique

class VisitCollator:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        B = len(batch)
        L = len(batch[0])

        flat_visits = []
        visit_mask = []

        for traj in batch:
            row_mask = []
            for visit_codes in traj:
                v = torch.tensor(visit_codes, dtype=torch.long)
                flat_visits.append(v)

                if len(visit_codes) == 1 and visit_codes[0] == self.pad_idx:
                    row_mask.append(False)
                else:
                    row_mask.append(True)

            visit_mask.append(row_mask)

        visit_mask = torch.tensor(visit_mask, dtype=torch.bool)
        return flat_visits, B, L, visit_mask


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
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    def gather(idxs):
        return [trajs[i] for i in idxs]

    return gather(train_idx), gather(val_idx), gather(test_idx)

def update_temperature(decoder, epoch, total_epochs=400, init_temp=1.0, final_temp=0.07):
    if not hasattr(decoder, 'set_temperature'):
        return
    progress = epoch / total_epochs
    temp = init_temp * (final_temp / init_temp) ** progress
    decoder.set_temperature(temp)

def rectified_flow_loss(model, latents: torch.Tensor, visit_mask: torch.Tensor):
    """
    latents: [B, L, dim] tangent visit representations
    visit_mask: [B, L] bool mask indicating real visits
    """
    device = latents.device
    B, L, _ = latents.shape
    noise = torch.randn_like(latents)
    t = torch.rand(B, device=device)

    xt = (1 - t[:, None, None]) * noise + t[:, None, None] * latents
    target_velocity = latents - noise
    pred_velocity = model(xt, t, visit_mask)

    loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
    mask = visit_mask.unsqueeze(-1).float()
    return (loss * mask).sum() / (mask.sum() + 1e-8)

def sample_trajectories(
    velocity_model,
    visit_enc,
    visit_dec,
    hier,
    max_len,
    dim,
    num_samples,
    embeddingType,
    device,
    codes_per_visit,
    num_steps=15,
    print_examples=3,
):
    velocity_model.eval()
    visit_enc.eval()
    visit_dec.eval()
    manifold = getattr(visit_enc, "manifold", None) if embeddingType == "hyperbolic" else None

    with torch.no_grad():
        x_t = torch.randn(num_samples, max_len, dim, device=device)
        visit_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=device)

        dt = 1.0 / num_steps
        # In sample_trajectories(), inside the step loop:
        for step in range(num_steps):
            t = torch.full((num_samples,), step * dt, device=device)
            v = velocity_model(x_t, t, visit_mask)
            x_t = x_t + v * dt

            # PROJECT BACK TO TANGENT SPACE EVERY FEW STEPS
            if embeddingType == "hyperbolic" and step % 3 == 0:
                with torch.no_grad():
                    x_manifold = manifold.expmap0(x_t)
                    x_t = manifold.logmap0(x_manifold)

        x_final = x_t  # remain in tangent space
        logits = visit_dec(x_final)                                     # [B, L, C]
        decoded_idx = logits.topk(k=codes_per_visit, dim=-1).indices

    # Convert to list of lists
    synthetic_trajs = []
    decoded_idx = decoded_idx.cpu()
    pad_idx = len(hier.codes)
    for i in range(num_samples):
        traj = []
        for j in range(max_len):
            codes = sorted(int(c) for c in decoded_idx[i, j] if int(c) < pad_idx)
            traj.append(codes)
        synthetic_trajs.append(traj)

    for idx in range(min(print_examples, num_samples)):
        print(f"\nSample trajectory ({embeddingType}) {idx + 1}:")
        for v_idx, codes in enumerate(synthetic_trajs[idx]):
            names = [hier.idx2code[c] for c in codes if c < len(hier.codes)]
            print(f"  Visit {v_idx + 1}: {names}")

    return synthetic_trajs

def compute_batch_loss(
    flat_visits, B, L, visit_mask,
    velocity_model, visit_enc, visit_dec,
    hier,
    lambda_recon=1000.0
):
    latents = visit_enc(flat_visits).contiguous()        # [B*L, dim]
    dim = latents.shape[-1]
    latents = latents.view(B, L, dim)

    flow_loss = rectified_flow_loss(velocity_model, latents, visit_mask)

    clean_latents = latents.view(B * L, dim)
    logits = visit_dec(clean_latents)                    # [B*L, num_codes]

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

def run_epoch(loader, velocity_model, visit_enc, visit_dec, hier,
              device, embedding_type, codes_per_visit, lambda_recon, optimizer=None):
    is_training = optimizer is not None
    modules = [velocity_model, visit_enc, visit_dec]
    for module in modules:
        module.train() if is_training else module.eval()
    clip_params = collect_unique_params(*modules) if is_training else None

    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)

            loss = compute_batch_loss(
                flat_visits, B, L, visit_mask,
                velocity_model, visit_enc, visit_dec,
                hier,
                lambda_recon=lambda_recon
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    return total_loss / max(total_samples, 1)

def train_model(
    velocity_model, code_emb, visit_enc, visit_dec,
    train_loader, val_loader, hier, device,
    embedding_type, codes_per_visit,
    lambda_recon=1000.0, n_epochs=50, plot_dir="results/plots", tag="archFix"
):
    params = collect_unique_params(velocity_model, visit_enc, visit_dec)
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    patience = 5
    epochs_without_improve = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(train_loader, velocity_model, visit_enc, visit_dec,
                               hier, device, embedding_type, codes_per_visit,
                               lambda_recon, optimizer=optimizer)

        val_loss = run_epoch(val_loader, velocity_model, visit_enc, visit_dec,
                             hier, device, embedding_type, codes_per_visit,
                             lambda_recon, optimizer=None)

        scheduler.step()
        if embedding_type == "hyperbolic":
            update_temperature(visit_dec, epoch, total_epochs=n_epochs)

        print(
            f"Epoch {epoch:3d} | Train {train_loss:.5f} | Val {val_loss:.5f} "
            f"| lambda_recon={lambda_recon}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            # when saving
            best_state = {
                "velocity": velocity_model.state_dict(),
                "decoder": visit_dec.state_dict(),
                "code_emb": code_emb.state_dict()
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break

    # when loading
    velocity_model.load_state_dict(best_state["velocity"])
    visit_dec.load_state_dict(best_state["decoder"])
    code_emb.load_state_dict(best_state["code_emb"])
    save_loss_curves(train_losses, val_losses, plot_dir, tag)
    return best_val


def evaluate_test_accuracy(
    loader,
    velocity_model,
    visit_enc,
    visit_dec,
    code_emb,
    max_len,
    dim,
    device,
    embedding_type,
    codes_per_visit,
    pad_idx,
):
    del velocity_model, code_emb, embedding_type, max_len, dim  # unused in this eval
    visit_enc.eval()
    visit_dec.eval()
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)
            latents = visit_enc(flat_visits).contiguous().view(B, L, -1)
            logits = visit_dec(latents)                   # [B, L, num_codes]
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

def run_experiment(embeddingType, hier, traj_splits, experiment_name, device, n_epochs=50):
    train_trajs, val_trajs, test_trajs = traj_splits
    max_len = 6
    batch_size = 128
    dim = 32
    pad_idx = len(hier.codes)
    collate = VisitCollator(pad_idx)
    pin_memory = device.type == "cuda"
    loader_kwargs = dict(collate_fn=collate, pin_memory=pin_memory)

    train_dl = DataLoader(TrajDataset(train_trajs, max_len, pad_idx), batch_size=batch_size, shuffle=True, num_workers=0, **loader_kwargs)
    val_dl = DataLoader(TrajDataset(val_trajs, max_len, pad_idx), batch_size=batch_size, shuffle=False, num_workers=0, **loader_kwargs)
    test_dl = DataLoader(TrajDataset(test_trajs, max_len, pad_idx), batch_size=batch_size, shuffle=False, num_workers=0, **loader_kwargs)

    if embeddingType == "euclidean":
        code_emb = EuclideanCodeEmbedding(num_codes=len(hier.codes) + 1, dim=dim).to(device)
        visit_enc = LearnableVisitEncoder(code_emb, dim=dim, pad_idx=pad_idx, hidden_dim=256, use_attention=True).to(device)
        visit_dec = StrongVisitDecoder(dim=dim, num_codes=len(hier.codes), hidden_dim=512, num_res_blocks=6).to(device)
    else:
        code_emb = HyperbolicCodeEmbedding(num_codes=len(hier.codes) + 1, dim=dim, c=1.0).to(device)
        visit_enc = HyperbolicVisitEncoder(code_emb, pad_idx=pad_idx).to(device)
        freq = compute_code_frequency(train_trajs, len(hier.codes), device)
        visit_dec = HyperbolicDistanceDecoder(code_embedding=code_emb.emb, manifold=code_emb.manifold, code_freq=freq).to(device)

    velocity_model = TrajectoryVelocityModel(dim=dim, n_layers=6, n_heads=8, ff_dim=1024).to(device)
    codes_per_visit = 4
    lambda_recon_values = [1.0, 10.0, 100.0, 1000.0]

    results = []
    for lambda_recon in lambda_recon_values:
        # Re-init for each run to be fair/clean, or keep weights? Usually re-init.
        # Simplified: Just re-instantiate everything to be safe as in original script loop
        if embeddingType == "euclidean":
            code_emb = EuclideanCodeEmbedding(num_codes=len(hier.codes) + 1, dim=dim).to(device)
            visit_enc = LearnableVisitEncoder(code_emb, dim=dim, pad_idx=pad_idx, hidden_dim=256, use_attention=True).to(device)
            visit_dec = StrongVisitDecoder(dim=dim, num_codes=len(hier.codes), hidden_dim=512, num_res_blocks=6).to(device)
        else:
            code_emb = HyperbolicCodeEmbedding(num_codes=len(hier.codes) + 1, dim=dim, c=1.0).to(device)
            visit_enc = HyperbolicVisitEncoder(code_emb, pad_idx=pad_idx).to(device)
            freq = compute_code_frequency(train_trajs, len(hier.codes), device)
            visit_dec = HyperbolicDistanceDecoder(code_embedding=code_emb.emb, manifold=code_emb.manifold, code_freq=freq).to(device)

        velocity_model = TrajectoryVelocityModel(dim=dim, n_layers=6, n_heads=8, ff_dim=1024).to(device)

        print(f"\nTraining {embeddingType.upper()} | Depth {hier.max_depth} | lambda_recon={lambda_recon}")
        best_val = train_model(
            velocity_model, code_emb, visit_enc, visit_dec,
            train_dl, val_dl, hier, device,
            embedding_type=embeddingType,
            codes_per_visit=codes_per_visit,
            lambda_recon=lambda_recon,
            n_epochs=n_epochs,
            plot_dir=os.path.join("results", "plots"),
            tag=f"{experiment_name}_{embeddingType}_lrecon{int(lambda_recon)}"
        )
        print(f"Best validation loss (lambda_recon={lambda_recon}): {best_val:.6f}")

        test_recall = evaluate_test_accuracy(
            test_dl, velocity_model, visit_enc, visit_dec, code_emb,
            max_len, dim, device, embeddingType, codes_per_visit, pad_idx
        )
        print(f"Test Recall@{codes_per_visit} (lambda_recon={lambda_recon}): {test_recall:.4f}")

        synthetic = sample_trajectories(
            velocity_model, visit_enc, visit_dec, hier,
            max_len, dim, num_samples=1000, embeddingType=embeddingType,
            device=device, codes_per_visit=codes_per_visit, num_steps=15
        )

        corr = correlation_tree_vs_embedding(code_emb, hier, device)
        print(f"Tree-Embedding Correlation (lambda_recon={lambda_recon}): {corr:.4f}")

        stats = traj_stats(synthetic, hier)
        print(f"Synthetic ({embeddingType}, lambda_recon={lambda_recon}) stats (N={len(synthetic)}): {stats}")
        results.append((lambda_recon, best_val, test_recall, corr))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra_depth", type=int, default=5, help="Extra depth for ToyICD hierarchy (0=depth2, 5=depth7)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    hier = ToyICDHierarchy(extra_depth=args.extra_depth)
    name = f"depth{hier.max_depth}_final"

    trajs = sample_toy_trajectories(hier, num_patients=20000)
    splits = split_trajectories(trajs)
    real_stats = traj_stats(trajs, hier)
    print(f"\n{name} | max_depth = {hier.max_depth} | Real stats: {real_stats}")

    for emb in ["hyperbolic"]:
        print(f"\n--- Running {emb} ---")
        results = run_experiment(
            emb, hier, splits, name, device,
            n_epochs=args.epochs,
        )
        for lambda_recon, best_val, test_recall, corr in results:
            print(
                f"[Summary] {name} | {emb} | lambda_recon={lambda_recon}: "
                f"best_val={best_val:.6f}, test_recall={test_recall:.4f}, corr={corr:.4f}"
            )

if __name__ == "__main__":
    main()
