import os
import torch
import torch.nn as nn
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_icd_toy import ToyICDHierarchy, sample_toy_trajectories
from hyperbolic_embeddings import HyperbolicCodeEmbedding, VisitEncoder
from euclidean_embeddings import EuclideanCodeEmbedding, EuclideanVisitEncoder
from diffusion import cosine_beta_schedule
from traj_models import TrajectoryEpsModel
from metrics_toy import traj_stats
from data_utils import TrajDataset, make_collate_fn, build_visit_tensor
from losses import code_pair_loss
from hyperbolic_noise import (
    is_hyperbolic,
    hyperbolic_forward_noise,
    hyperbolic_remove_noise,
)
from decoders import VisitDecoder


def sample_fake_visit_indices(
    eps_model,
    num_samples,
    max_len,
    dim,
    T,
    alphas_cumprod,
    code_emb,
    visit_enc,
    visit_dec,
    codes_per_visit,
    device,
    embedding_type,
):
    """
    Quick helper for synthetic visits from random latents.
    Now uses VisitDecoder instead of embedding nearest-neighbors.
    """
    was_training = eps_model.training
    eps_model.eval()
    with torch.no_grad():
        use_hyp = is_hyperbolic(embedding_type)
        manifold = getattr(visit_enc, "manifold", None) if use_hyp else None
        x_t = torch.randn(num_samples, max_len, dim, device=device)
        t = torch.randint(0, T, (num_samples,), device=device)
        a_bar_t = alphas_cumprod[t].view(num_samples, 1, 1)

        # all positions are "real" for these fake samples
        visit_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=device)

        eps_hat = eps_model(x_t, t, visit_mask=visit_mask)
        sqrt_a = torch.sqrt(a_bar_t)
        sqrt_one_minus = torch.sqrt(1 - a_bar_t)
        if use_hyp and manifold is not None:
            x0_pred = hyperbolic_remove_noise(manifold, x_t, eps_hat, sqrt_a, sqrt_one_minus)
        else:
            x0_pred = (x_t - sqrt_one_minus * eps_hat) / sqrt_a

        # decode via VisitDecoder
        logits = visit_dec(x0_pred)  # [num_samples, max_len, num_codes_real]
        topk_idx = logits.topk(k=codes_per_visit, dim=-1).indices  # [num_samples, max_len, K]

    if was_training:
        eps_model.train()
    return topk_idx.cpu()


def sample_trajectories(
    eps_model,
    code_emb,
    visit_enc,
    visit_dec,
    hier,
    alphas,
    betas,
    alphas_cumprod,
    alphas_cumprod_prev,
    max_len,
    dim,
    num_samples,
    embeddingType,
    device,
    codes_per_visit,
    print_examples=3,
):
    eps_model.eval()
    visit_enc.eval()
    visit_dec.eval()
    synthetic_trajs = []
    use_hyperbolic = is_hyperbolic(embeddingType)
    manifold = getattr(visit_enc, "manifold", None) if use_hyperbolic else None

    with torch.no_grad():
        x_t = torch.randn(num_samples, max_len, dim, device=device)
        T = betas.shape[0]
        visit_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=device)

        for timestep in reversed(range(T)):
            t_batch = torch.full((num_samples,), timestep, dtype=torch.long, device=device)
            eps_hat = eps_model(x_t, t_batch, visit_mask=visit_mask)

            alpha_bar = alphas_cumprod[timestep]
            sqrt_alpha_bar = torch.sqrt(alpha_bar).view(1, 1, 1).repeat(num_samples, 1, 1)
            sqrt_one_minus = torch.sqrt(1 - alpha_bar).view(1, 1, 1).repeat(num_samples, 1, 1)

            if use_hyperbolic and manifold is not None:
                x0_est = hyperbolic_remove_noise(
                    manifold, x_t, eps_hat, sqrt_alpha_bar, sqrt_one_minus
                )
            else:
                x0_est = (x_t - sqrt_one_minus * eps_hat) / sqrt_alpha_bar

            if timestep > 0:
                noise = torch.randn_like(x_t)
                alpha_bar_prev = alphas_cumprod_prev[timestep]
                sqrt_alpha_prev = torch.sqrt(alpha_bar_prev).view(1, 1, 1).repeat(num_samples, 1, 1)
                sqrt_one_minus_prev = torch.sqrt(1 - alpha_bar_prev).view(1, 1, 1).repeat(num_samples, 1, 1)

                if use_hyperbolic and manifold is not None:
                    x_t = hyperbolic_forward_noise(
                        manifold,
                        x0_est,
                        noise,
                        sqrt_alpha_prev,
                        sqrt_one_minus_prev,
                    )
                else:
                    x_t = sqrt_alpha_prev * x0_est + sqrt_one_minus_prev * noise
            else:
                x_t = x0_est

        sampled_visits = x_t  # approx x0: [num_samples, max_len, dim]

        # Decode with VisitDecoder
        logits = visit_dec(sampled_visits)  # [num_samples, max_len, num_codes_real]
        decoded_idx = logits.topk(k=codes_per_visit, dim=-1).indices  # [num_samples, max_len, K]

    decoded_idx = decoded_idx.view(num_samples, max_len, codes_per_visit)

    for sample_idx in range(num_samples):
        traj_visits = []
        for visit_idx in range(max_len):
            visit_codes = decoded_idx[sample_idx, visit_idx].tolist()
            visit_codes = sorted(set(int(c) for c in visit_codes))
            traj_visits.append(visit_codes)
        synthetic_trajs.append(traj_visits)

    for sample_idx in range(min(print_examples, num_samples)):
        print(f"\nSample trajectory ({embeddingType}) {sample_idx + 1}:")
        traj_visits = synthetic_trajs[sample_idx]
        for visit_idx, visit_codes in enumerate(traj_visits):
            code_names = [hier.idx2code[idx] for idx in visit_codes]
            print(f"  Visit {visit_idx + 1}: {code_names}")

    return synthetic_trajs


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


def compute_batch_loss(
    flat_visits,
    B,
    L,
    visit_mask,
    eps_model,
    visit_enc,
    visit_dec,
    code_emb,
    hier,
    alphas_cumprod,
    T,
    max_len,
    dim,
    device,
    embedding_type,
    codes_per_visit,
    lambda_tree,
    lambda_radius,
    depth_targets,
    lambda_recon=1.0,
):
    """
    Main training objective:
      - DDPM prediction loss in latent space
      - geometry-aware code_pair_loss
      - (hyperbolic) radius-depth regularizer
      - reconstruction loss via VisitDecoder (multi-label BCE on real visits)
    """
    # 1) Standard DDPM latent loss
    x0 = build_visit_tensor(visit_enc, flat_visits, B, L, dim, device)  # [B, L, D]
    t = torch.randint(0, T, (B,), device=device).long()
    a_bar_t = alphas_cumprod[t].view(B, 1, 1)
    eps = torch.randn_like(x0)
    use_hyperbolic = is_hyperbolic(embedding_type)
    manifold = getattr(visit_enc, "manifold", None) if use_hyperbolic else None
    if use_hyperbolic and manifold is not None:
        sqrt_a = torch.sqrt(a_bar_t)
        sqrt_one_minus = torch.sqrt(1 - a_bar_t)
        x_t = hyperbolic_forward_noise(manifold, x0, eps, sqrt_a, sqrt_one_minus)
    else:
        x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * eps
    eps_hat = eps_model(x_t, t, visit_mask=visit_mask)
    loss = torch.mean((eps - eps_hat) ** 2)

    # 2) Code-pair geometry regularizer
    if lambda_tree > 0.0:
        pair_loss = code_pair_loss(code_emb, hier, device=device, num_pairs=512)
        loss = loss + lambda_tree * pair_loss

    # 3) Radius-depth regularizer for hyperbolic embeddings
    if use_hyperbolic and depth_targets is not None and lambda_radius > 0.0:
        # radius over all codes, including pad
        base_emb = code_emb.emb
        if isinstance(base_emb, nn.Embedding):
            emb_tensor = base_emb.weight
        else:
            emb_tensor = base_emb
        radius = torch.norm(emb_tensor, dim=-1)             # [num_codes + 1]
        radius_mismatch = torch.mean((radius - depth_targets) ** 2)
        loss = loss + lambda_radius * radius_mismatch

    # 4) Reconstruction loss via VisitDecoder (multi-label BCE on real visits)
    if lambda_recon > 0.0:
        num_real_codes = len(hier.codes)
        # logits: [B, L, num_real_codes]
        logits = visit_dec(x0)
        B_, L_, C_ = logits.shape
        assert B_ == B and L_ == L and C_ == num_real_codes

        logits_flat = logits.view(B * L, num_real_codes)  # [B*L, C]
        visit_mask_flat = visit_mask.view(-1)             # [B*L]

        # build multi-hot targets per visit
        targets = torch.zeros_like(logits_flat)           # [B*L, C]
        for idx, visit_tensor in enumerate(flat_visits):
            if not bool(visit_mask_flat[idx].item()):
                continue
            visit_codes = visit_tensor.detach().cpu().tolist()
            for c in visit_codes:
                if 0 <= int(c) < num_real_codes:
                    targets[idx, int(c)] = 1.0

        bce = nn.BCEWithLogitsLoss(reduction="none")
        recon_loss_all = bce(logits_flat, targets)        # [B*L, C]
        # mask out pad visits
        if visit_mask_flat.any():
            mask_rows = visit_mask_flat.to(logits_flat.dtype).unsqueeze(-1)  # [B*L,1]
            recon_loss_all = recon_loss_all * mask_rows
            recon_loss = recon_loss_all.sum() / (mask_rows.sum() * num_real_codes + 1e-8)
            loss = loss + lambda_recon * recon_loss

    return loss


def compute_batch_accuracy(
    flat_visits,
    B,
    L,
    visit_mask,
    eps_model,
    visit_enc,
    visit_dec,
    code_emb,
    alphas_cumprod,
    T,
    max_len,
    dim,
    device,
    embedding_type,
    codes_per_visit,
    pad_idx,
):
    """
    Reconstruction recall@K on real visits, using VisitDecoder.
    """
    visit_enc.eval()
    visit_dec.eval()
    use_hyperbolic = is_hyperbolic(embedding_type)
    manifold = getattr(visit_enc, "manifold", None) if use_hyperbolic else None
    with torch.no_grad():
        x0 = build_visit_tensor(visit_enc, flat_visits, B, L, dim, device)
        t = torch.randint(0, T, (B,), device=device).long()
        a_bar_t = alphas_cumprod[t].view(B, 1, 1)
        eps = torch.randn_like(x0)
        if use_hyperbolic and manifold is not None:
            sqrt_a = torch.sqrt(a_bar_t)
            sqrt_one_minus = torch.sqrt(1 - a_bar_t)
            x_t = hyperbolic_forward_noise(manifold, x0, eps, sqrt_a, sqrt_one_minus)
        else:
            sqrt_a = torch.sqrt(a_bar_t)
            sqrt_one_minus = torch.sqrt(1 - a_bar_t)
            x_t = sqrt_a * x0 + sqrt_one_minus * eps
        eps_hat = eps_model(x_t, t, visit_mask=visit_mask)
        if use_hyperbolic and manifold is not None:
            x0_pred = hyperbolic_remove_noise(manifold, x_t, eps_hat, sqrt_a, sqrt_one_minus)
        else:
            x0_pred = (x_t - sqrt_one_minus * eps_hat) / sqrt_a

        # decode with VisitDecoder
        logits = visit_dec(x0_pred)  # [B, L, num_real_codes]
        decoded_idx = logits.topk(k=codes_per_visit, dim=-1).indices  # [B, L, K]

    decoded_idx = decoded_idx.view(B * L, codes_per_visit)

    visit_mask_flat = visit_mask.view(-1).cpu()
    total = 0
    correct = 0

    for i, (visit_tensor, mask_value, preds) in enumerate(
        zip(flat_visits, visit_mask_flat, decoded_idx)
    ):
        if not bool(mask_value.item()):
            continue
        visit_codes = visit_tensor.detach().cpu().tolist()
        true_codes = [int(c) for c in visit_codes if int(c) != pad_idx]
        if not true_codes:
            continue
        pred_set = set(int(c) for c in preds.tolist())
        for code in true_codes:
            if code in pred_set:
                correct += 1
        total += len(true_codes)

    return correct, total


def run_epoch(
    loader,
    eps_model,
    visit_enc,
    visit_dec,
    code_emb,
    hier,
    alphas_cumprod,
    T,
    max_len,
    dim,
    device,
    embedding_type,
    codes_per_visit,
    lambda_tree_eff,
    lambda_radius_eff,
    depth_targets,
    lambda_recon,
    optimizer=None,
):
    is_training = optimizer is not None
    if is_training:
        eps_model.train()
        visit_dec.train()
    else:
        eps_model.eval()
        visit_dec.eval()

    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)  # [B, L]

            loss = compute_batch_loss(
                flat_visits,
                B,
                L,
                visit_mask,
                eps_model,
                visit_enc,
                visit_dec,
                code_emb,
                hier,
                alphas_cumprod,
                T,
                max_len,
                dim,
                device,
                embedding_type,
                codes_per_visit,
                lambda_tree_eff,
                lambda_radius_eff,
                depth_targets,
                lambda_recon=lambda_recon,
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


def train_model(
    eps_model,
    code_emb,
    visit_enc,
    visit_dec,
    train_loader,
    val_loader,
    hier,
    alphas_cumprod,
    T,
    max_len,
    dim,
    device,
    embedding_type,
    codes_per_visit,
    lambda_tree,
    lambda_radius,
    depth_targets,
    lambda_recon=1.0,
    n_epochs=20,
):
    optimizer = torch.optim.Adam(
        list(eps_model.parameters()) +
        list(code_emb.parameters()) +
        list(visit_dec.parameters()),
        lr=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val = float("inf")
    best_weights = {
        "eps_model": copy.deepcopy(eps_model.state_dict()),
        "code_emb": copy.deepcopy(code_emb.state_dict()),
        "visit_dec": copy.deepcopy(visit_dec.state_dict()),
    }

    train_losses = []
    val_losses = []

    # warm-up config: first 30% of epochs for geometry regs
    warmup_epochs = max(1, int(0.3 * n_epochs))

    for epoch in range(1, n_epochs + 1):
        # linearly ramp up regularization for tree/radius terms
        scale = min(1.0, epoch / warmup_epochs)
        lambda_tree_eff = lambda_tree * scale
        lambda_radius_eff = lambda_radius * scale

        train_loss = run_epoch(
            train_loader,
            eps_model,
            visit_enc,
            visit_dec,
            code_emb,
            hier,
            alphas_cumprod,
            T,
            max_len,
            dim,
            device,
            embedding_type,
            codes_per_visit,
            lambda_tree_eff,
            lambda_radius_eff,
            depth_targets,
            lambda_recon=lambda_recon,
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            val_loader,
            eps_model,
            visit_enc,
            visit_dec,
            code_emb,
            hier,
            alphas_cumprod,
            T,
            max_len,
            dim,
            device,
            embedding_type,
            codes_per_visit,
            lambda_tree_eff,
            lambda_radius_eff,
            depth_targets,
            lambda_recon=lambda_recon,
            optimizer=None,
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{n_epochs}, "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"lambda_tree_eff={lambda_tree_eff:.4f}, lambda_radius_eff={lambda_radius_eff:.4f}, "
            f"lambda_recon={lambda_recon:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {
                "eps_model": copy.deepcopy(eps_model.state_dict()),
                "code_emb": copy.deepcopy(code_emb.state_dict()),
                "visit_dec": copy.deepcopy(visit_dec.state_dict()),
            }

    eps_model.load_state_dict(best_weights["eps_model"])
    code_emb.load_state_dict(best_weights["code_emb"])
    visit_dec.load_state_dict(best_weights["visit_dec"])
    return train_losses, val_losses, best_val


def evaluate_loader(
    loader,
    eps_model,
    visit_enc,
    visit_dec,
    code_emb,
    hier,
    alphas_cumprod,
    T,
    max_len,
    dim,
    device,
    embedding_type,
    codes_per_visit,
    lambda_tree,
    lambda_radius,
    depth_targets,
    lambda_recon,
):
    eps_model.eval()
    visit_dec.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_samples = 0
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)
            loss = compute_batch_loss(
                flat_visits,
                B,
                L,
                visit_mask,
                eps_model,
                visit_enc,
                visit_dec,
                code_emb,
                hier,
                alphas_cumprod,
                T,
                max_len,
                dim,
                device,
                embedding_type,
                codes_per_visit,
                lambda_tree,
                lambda_radius,
                depth_targets,
                lambda_recon=lambda_recon,
            )
            total_loss += loss.item() * B
            total_samples += B
    return total_loss / max(total_samples, 1)


def evaluate_test_accuracy(
    loader,
    eps_model,
    visit_enc,
    visit_dec,
    code_emb,
    alphas_cumprod,
    T,
    max_len,
    dim,
    device,
    embedding_type,
    codes_per_visit,
    pad_idx,
):
    """
    Reconstruction recall@K on real visits, averaged over the test loader.
    """
    eps_model.eval()
    visit_enc.eval()
    visit_dec.eval()
    with torch.no_grad():
        total_correct = 0
        total_items = 0
        for flat_visits, B, L, visit_mask in loader:
            flat_visits = [v.to(device) for v in flat_visits]
            visit_mask = visit_mask.to(device)
            correct, total = compute_batch_accuracy(
                flat_visits,
                B,
                L,
                visit_mask,
                eps_model,
                visit_enc,
                visit_dec,
                code_emb,
                alphas_cumprod,
                T,
                max_len,
                dim,
                device,
                embedding_type,
                codes_per_visit,
                pad_idx,
            )
            total_correct += correct
            total_items += total
    if total_items == 0:
        return 0.0
    return float(total_correct) / float(total_items)


def _format_float_for_name(val):
    if isinstance(val, float):
        s = f"{val:.4f}".rstrip("0").rstrip(".")
        return s.replace(".", "p") if s else "0"
    return str(val)


def _plot_single_curve(values, title, filename):
    epochs = range(1, len(values) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_loss_curves(train_losses, val_losses, meta):
    base_dir = os.path.join("results", "plots")
    os.makedirs(base_dir, exist_ok=True)
    tag = (
        f"{meta['experiment']}_depth{meta['hier_depth']}_"
        f"{meta['embedding']}_dim{meta['dim']}_layers{meta['layers']}_T{meta['T']}_"
        f"reg{'on' if meta['use_reg'] else 'off'}_"
        f"lt{_format_float_for_name(meta['lambda_tree'])}_"
        f"lr{_format_float_for_name(meta['lambda_radius'])}"
    )
    tag = tag.replace("/", "-")

    depth_info = f"depth={meta['hier_depth']}, exp={meta['experiment']}"
    train_title = (
        f"{meta['embedding']} Train Loss (dim={meta['dim']}, layers={meta['layers']}, {depth_info})"
    )
    train_path = os.path.join(base_dir, f"{tag}_train.png")
    _plot_single_curve(train_losses, train_title, train_path)

    if val_losses:
        val_title = (
            f"{meta['embedding']} Val Loss (dim={meta['dim']}, layers={meta['layers']}, {depth_info})"
        )
        val_path = os.path.join(base_dir, f"{tag}_val.png")
        _plot_single_curve(val_losses, val_title, val_path)


def correlation_tree_vs_embedding(code_emb, hier, device, num_pairs=5000):
    n_real = len(hier.codes)
    base_emb = code_emb.emb
    if isinstance(base_emb, nn.Embedding):
        base_tensor = base_emb.weight
    else:
        base_tensor = base_emb
    emb = base_tensor[:n_real].detach().to(device)  # [N_real, D]

    tree_dists = []
    embed_dists = []

    idx_i = torch.randint(0, n_real, (num_pairs,), device=device)
    idx_j = torch.randint(0, n_real, (num_pairs,), device=device)

    for i, j in zip(idx_i.tolist(), idx_j.tolist()):
        if i == j:
            continue
        c1 = hier.idx2code[i]
        c2 = hier.idx2code[j]
        dist = hier.tree_distance(c1, c2)
        if dist is None:
            continue
        tree_dists.append(dist)

        if hasattr(code_emb, "manifold"):
            d = code_emb.manifold.dist(emb[i].unsqueeze(0), emb[j].unsqueeze(0)).item()
        else:
            d = torch.norm(emb[i] - emb[j]).item()
        embed_dists.append(d)

    tree = np.array(tree_dists)
    embd = np.array(embed_dists)
    if len(tree) == 0:
        return 0.0
    return float(np.corrcoef(tree, embd)[0, 1])


def main(
    embeddingType: str,
    use_regularization: bool = True,
    traj_splits=None,
    hier=None,
    experiment_name: str = "base",
):
    device = torch.device("cpu")

    # 1) data
    if hier is None:
        hier = ToyICDHierarchy()
    if traj_splits is None:
        trajs = sample_toy_trajectories(hier, num_patients=20000)
        train_trajs, val_trajs, test_trajs = split_trajectories(trajs, seed=42)
    else:
        train_trajs, val_trajs, test_trajs = traj_splits
        trajs = train_trajs + val_trajs + test_trajs

    max_len = 6

    batch_size = 64
    dim = 16
    pad_idx = len(hier.codes)
    collate = make_collate_fn(pad_idx)

    train_ds = TrajDataset(train_trajs, max_len=max_len, pad_idx=pad_idx)
    val_ds = TrajDataset(val_trajs, max_len=max_len, pad_idx=pad_idx)
    test_ds = TrajDataset(test_trajs, max_len=max_len, pad_idx=pad_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # 2) embeddings + visit encoder
    if embeddingType == "euclidean":
        code_emb = EuclideanCodeEmbedding(
            num_codes=len(hier.codes) + 1,  # +1 for pad
            dim=dim,
        ).to(device)
        visit_enc = EuclideanVisitEncoder(code_emb, pad_idx=pad_idx).to(device)
    else:
        code_emb = HyperbolicCodeEmbedding(
            num_codes=len(hier.codes) + 1,  # +1 for pad
            dim=dim,
        ).to(device)
        visit_enc = VisitEncoder(code_emb, pad_idx=pad_idx).to(device)

    # 2b) visit decoder (shared across geometries, only real codes)
    num_real_codes = len(hier.codes)
    visit_dec = VisitDecoder(dim=dim, num_codes=num_real_codes).to(device)

    # 3) diffusion params
    T = 1000
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
    )

    # 4) model
    n_layers = 4
    eps_model = TrajectoryEpsModel(dim=dim, n_layers=n_layers, T_max=T).to(device)
    codes_per_visit = 4

    # base lambdas (max strength after warm-up for geometry)
    lambda_tree = 0.01 if use_regularization else 0.0
    lambda_radius = 0.003 if (embeddingType == "hyperbolic" and use_regularization) else 0.0
    lambda_recon = 1.0  # decoder reconstruction weight

    depth_targets = None
    if embeddingType == "hyperbolic" and lambda_radius > 0.0:
        max_depth = max(hier.depth(code) for code in hier.codes)
        depth_vals = [hier.depth(code) / max_depth for code in hier.codes]  # real codes
        depth_targets = torch.tensor(depth_vals, dtype=torch.float32, device=device)
        # add one entry for pad index (e.g. target radius = 0)
        depth_targets = torch.cat([depth_targets, torch.zeros(1, device=device)], dim=0)

    train_losses, val_losses, best_val = train_model(
        eps_model,
        code_emb,
        visit_enc,
        visit_dec,
        train_dl,
        val_dl,
        hier,
        alphas_cumprod,
        T,
        max_len,
        dim,
        device,
        embeddingType,
        codes_per_visit,
        lambda_tree,
        lambda_radius,
        depth_targets,
        lambda_recon=lambda_recon,
        n_epochs=20,
    )
    print(f"Best validation loss: {best_val:.6f}")
    meta = {
        "embedding": embeddingType,
        "dim": dim,
        "layers": n_layers,
        "T": T,
        "use_reg": use_regularization,
        "lambda_tree": lambda_tree,
        "lambda_radius": lambda_radius,
        "hier_depth": getattr(hier, "max_depth", None),
        "experiment": experiment_name,
    }
    save_loss_curves(train_losses, val_losses, meta)
    print("Saved loss curves to results/plots")

    test_accuracy = evaluate_test_accuracy(
        test_dl,
        eps_model,
        visit_enc,
        visit_dec,
        code_emb,
        alphas_cumprod,
        T,
        max_len,
        dim,
        device,
        embeddingType,
        codes_per_visit,
        pad_idx,
    )
    print(f"Test recall@{codes_per_visit}: {test_accuracy:.4f}")

    num_samples_for_eval = 1000
    synthetic_trajs = sample_trajectories(
        eps_model,
        code_emb,
        visit_enc,
        visit_dec,
        hier,
        alphas,
        betas,
        alphas_cumprod,
        alphas_cumprod_prev,
        max_len,
        dim,
        num_samples_for_eval,
        embeddingType,
        device,
        codes_per_visit,
        print_examples=3,
    )

    corr = correlation_tree_vs_embedding(code_emb, hier, device=device, num_pairs=5000)
    print(f"Correlation(tree_dist, {embeddingType}_embedding_dist) = {corr:.4f}")

    syn_stats = traj_stats(synthetic_trajs, hier)
    print(f"\nSynthetic ({embeddingType}) stats (N={num_samples_for_eval}):", syn_stats)
    return test_accuracy


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    experiments = [
        ("depth2_base_w-DecHypNoise", ToyICDHierarchy(extra_depth=0)),
        ("depth7_extended_wDecHypNoise", ToyICDHierarchy(extra_depth=5)),
    ]

    configs = [
        ("hyperbolic", False),
        ("euclidean", False),
        ("hyperbolic", True),
        ("euclidean", True),
    ]

    for exp_name, hier in experiments:
        all_trajs = sample_toy_trajectories(hier, num_patients=20000)
        splits = split_trajectories(all_trajs, seed=42)
        real_stats = traj_stats(all_trajs, hier)
        print(f"\nReal stats ({exp_name}, max_depth={hier.max_depth}): {real_stats}")

        for embedding, use_reg in configs:
            print(
                f"\n=== Experiment {exp_name} | depth {hier.max_depth} | "
                f"{embedding} | regularization={'on' if use_reg else 'off'} ==="
            )
            main(
                embeddingType=embedding,
                use_regularization=use_reg,
                traj_splits=splits,
                hier=hier,
                experiment_name=exp_name,
            )
