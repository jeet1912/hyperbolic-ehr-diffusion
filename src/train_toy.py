import os
import torch
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from data_icd_toy import ToyICDHierarchy, sample_toy_trajectories
from hyperbolic_embeddings import HyperbolicCodeEmbedding, VisitEncoder
from euclidean_embeddings import EuclideanCodeEmbedding, EuclideanVisitEncoder
from diffusion import cosine_beta_schedule
from traj_model import TrajectoryEpsModel
from metrics_toy import traj_stats


class TrajDataset(Dataset):
    def __init__(self, trajs, max_len, pad_idx):
        self.trajs = trajs
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj = self.trajs[idx]
        # pad / truncate to max_len
        if len(traj) >= self.max_len:
            traj = traj[: self.max_len]
        else:
            pad_visits = [[self.pad_idx]] * (self.max_len - len(traj))
            traj = traj + pad_visits
        return traj  # list of visits (each visit is list of code indices)



def make_collate_fn(pad_idx):
    def collate_fn(batch):
        B = len(batch)
        L = len(batch[0])

        flat_visits = []
        visit_mask = []

        for traj in batch:
            row_mask = []
            for visit_codes in traj:
                v = torch.tensor(visit_codes, dtype=torch.long)
                flat_visits.append(v)

                if len(visit_codes) == 1 and visit_codes[0] == pad_idx:
                    row_mask.append(False)
                else:
                    row_mask.append(True)

            visit_mask.append(row_mask)

        visit_mask = torch.tensor(visit_mask, dtype=torch.bool)
        return flat_visits, B, L, visit_mask
    return collate_fn



def build_visit_tensor(visit_enc, flat_visits, B, L, dim, device):
    visit_enc.eval()
    with torch.no_grad():
        visit_vecs = visit_enc(flat_visits).to(device)  # [B*L, dim]
    x0 = visit_vecs.view(B, L, dim)
    return x0


def decode_visit_vectors(sampled_visits, code_emb, visit_enc, embedding_type, codes_per_visit):
    num_samples, max_len, dim = sampled_visits.shape
    visit_vecs = sampled_visits.view(num_samples * max_len, dim)  # [B*L, D]

    if embedding_type == "hyperbolic":
        # code_emb.emb: [N_all, D] on the Poincaré ball
        code_tangent = visit_enc.manifold.logmap0(code_emb.emb)  # [N_all, D]
        pad_idx = getattr(visit_enc, "pad_idx", None)
        if pad_idx is not None:
            # keep only real codes (0..pad_idx-1), drop pad row
            code_tangent = code_tangent[:pad_idx]  # [N_real, D]
        sims = visit_vecs @ code_tangent.t()       # [B*L, N_real]

    else:
        pad_idx = getattr(visit_enc, "pad_idx", None)
        if pad_idx is not None:
            code_matrix = code_emb.emb.weight[:pad_idx]  # [N_real, D]
        else:
            code_matrix = code_emb.emb.weight            # [N_all, D]

        # negative squared Euclidean distance as similarity
        diff = visit_vecs.unsqueeze(1) - code_matrix.unsqueeze(0)  # [B*L, N_real, D]
        sims = -(diff ** 2).sum(-1)                               # [B*L, N_real]

    topk_idx = sims.topk(k=codes_per_visit, dim=-1).indices       # [B*L, K]
    return topk_idx.view(num_samples, max_len, codes_per_visit).cpu()


def visits_from_indices(indices_tensor):
    visits = []
    flattened = indices_tensor.view(-1, indices_tensor.shape[-1])
    for visit_codes in flattened:
        unique_codes = sorted(set(int(c) for c in visit_codes.tolist()))
        visits.append(unique_codes)
    return visits


def mean_tree_distance_from_visits(visit_lists, hier):
    dists = []
    max_code_idx = len(hier.codes) - 1

    for visit in visit_lists:
        # keep only real codes: 0 .. len(hier.codes)-1
        filtered = [c for c in visit if 0 <= c <= max_code_idx]
        if len(filtered) < 2:
            continue
        codes = [hier.idx2code[i] for i in filtered]
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                dist = hier.tree_distance(codes[i], codes[j])
                if dist is not None:
                    dists.append(dist)
    if not dists:
        return None
    return float(sum(dists) / len(dists))



def sample_fake_visit_indices(
    eps_model,
    num_samples,
    max_len,
    dim,
    T,
    alphas_cumprod,
    code_emb,
    visit_enc,
    codes_per_visit,
    device,
    embedding_type,
):
    was_training = eps_model.training
    eps_model.eval()
    with torch.no_grad():
        x_t = torch.randn(num_samples, max_len, dim, device=device)
        t = torch.randint(0, T, (num_samples,), device=device)
        a_bar_t = alphas_cumprod[t].view(num_samples, 1, 1)

        # all positions are "real" for these fake samples
        visit_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=device)

        eps_hat = eps_model(x_t, t, visit_mask=visit_mask)
        x0_pred = (x_t - torch.sqrt(1 - a_bar_t) * eps_hat) / torch.sqrt(a_bar_t)
    if was_training:
        eps_model.train()
    return decode_visit_vectors(x0_pred, code_emb, visit_enc, embedding_type, codes_per_visit)


def sample_trajectories(
    eps_model,
    code_emb,
    visit_enc,
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
    synthetic_trajs = []

    with torch.no_grad():
        x_t = torch.randn(num_samples, max_len, dim, device=device)
        T = betas.shape[0]
        visit_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=device)

        for timestep in reversed(range(T)):
            t_batch = torch.full((num_samples,), timestep, dtype=torch.long, device=device)
            eps_hat = eps_model(x_t, t_batch, visit_mask=visit_mask)

            alpha = alphas[timestep]
            alpha_bar = alphas_cumprod[timestep]
            alpha_bar_prev = alphas_cumprod_prev[timestep]
            beta = betas[timestep]

            coeff = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            mean = (x_t - coeff * eps_hat) / torch.sqrt(alpha)

            if timestep > 0:
                posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_var) * noise
            else:
                x_t = mean

        sampled_visits = x_t  # approx x0: [num_samples, max_len, dim]
        decoded_idx = decode_visit_vectors(
            sampled_visits, code_emb, visit_enc, embeddingType, codes_per_visit
        )

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
    fake_sample_size=2,
):
    x0 = build_visit_tensor(visit_enc, flat_visits, B, L, dim, device)
    t = torch.randint(0, T, (B,), device=device).long()
    a_bar_t = alphas_cumprod[t].view(B, 1, 1)
    eps = torch.randn_like(x0)
    x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * eps
    eps_hat = eps_model(x_t, t, visit_mask=visit_mask)
    loss = torch.mean((eps - eps_hat) ** 2)

    if lambda_tree > 0.0:
        max_code_idx = len(hier.codes) - 1
        real_visit_lists = []
        for visit in flat_visits:
            visit_codes = [int(x) for x in visit.tolist()
                        if 0 <= int(x) <= max_code_idx]
            real_visit_lists.append(visit_codes)
        D_tree_real = mean_tree_distance_from_visits(real_visit_lists, hier)
        fake_indices = sample_fake_visit_indices(
            eps_model,
            num_samples=fake_sample_size,
            max_len=max_len,
            dim=dim,
            T=T,
            alphas_cumprod=alphas_cumprod,
            code_emb=code_emb,
            visit_enc=visit_enc,
            codes_per_visit=codes_per_visit,
            device=device,
            embedding_type=embedding_type,
        )
        fake_visit_lists = visits_from_indices(fake_indices)
        D_tree_fake = mean_tree_distance_from_visits(fake_visit_lists, hier)
        D_tree_real = 0.0 if D_tree_real is None else D_tree_real
        D_tree_fake = 0.0 if D_tree_fake is None else D_tree_fake
        tree_penalty = lambda_tree * abs(D_tree_real - D_tree_fake)
        loss = loss + torch.tensor(tree_penalty, dtype=torch.float32, device=device)
       
        if embedding_type == "hyperbolic" and lambda_radius > 0.0:
            max_depth = max(hier.depth(code) for code in hier.codes)
            depth_vals = [hier.depth(code) / max_depth for code in hier.codes]
            depth_targets = torch.tensor(depth_vals, dtype=torch.float32, device=device)
            # add target for pad index (e.g. 0)
            depth_targets = torch.cat([
                depth_targets,
                torch.zeros(1, device=device)
            ], dim=0)
            radius = torch.norm(code_emb.emb, dim=-1)
            radius_mismatch = torch.mean((radius - depth_targets) ** 2)
            loss = loss + lambda_radius * radius_mismatch

    return loss


def run_epoch(
    loader,
    eps_model,
    visit_enc,
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
    optimizer=None,
    fake_sample_size=2,
):
    is_training = optimizer is not None
    if is_training:
        eps_model.train()
    else:
        eps_model.eval()

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
                fake_sample_size=fake_sample_size,
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
    n_epochs=20,
):
    optimizer = torch.optim.Adam(
        list(eps_model.parameters()) + list(code_emb.parameters()), lr=2e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val = float("inf")
    best_weights = {
        "eps_model": copy.deepcopy(eps_model.state_dict()),
        "code_emb": copy.deepcopy(code_emb.state_dict()),
    }

    train_losses = []
    val_losses = []

    # warm-up config: first 30% of epochs
    warmup_epochs = max(1, int(0.3 * n_epochs))

    for epoch in range(1, n_epochs + 1):
        # linearly ramp up regularization
        scale = min(1.0, epoch / warmup_epochs)
        lambda_tree_eff = lambda_tree * scale
        lambda_radius_eff = lambda_radius * scale

        train_loss = run_epoch(
            train_loader,
            eps_model,
            visit_enc,
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
            optimizer=optimizer,
        )
        val_loss = run_epoch(
            val_loader,
            eps_model,
            visit_enc,
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
            optimizer=None,
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{n_epochs}, "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"lambda_tree_eff={lambda_tree_eff:.4f}, lambda_radius_eff={lambda_radius_eff:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_weights = {
                "eps_model": copy.deepcopy(eps_model.state_dict()),
                "code_emb": copy.deepcopy(code_emb.state_dict()),
            }

    eps_model.load_state_dict(best_weights["eps_model"])
    code_emb.load_state_dict(best_weights["code_emb"])
    return train_losses, val_losses, best_val


def evaluate_loader(
    loader,
    eps_model,
    visit_enc,
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
):
    eps_model.eval()
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
            )
            total_loss += loss.item() * B
            total_samples += B
    return total_loss / max(total_samples, 1)


def _format_float_for_name(val):
    if isinstance(val, float):
        s = f"{val:.4f}".rstrip("0").rstrip(".")
        return s.replace(".", "p") if s else "0"
    return str(val)


def _plot_single_curve(values, title, filename):
    epochs = range(1, len(values) + 1)
    plt.figure(figsize=(6, 4))
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
        f"{meta['embedding']}_dim{meta['dim']}_layers{meta['layers']}_T{meta['T']}_"
        f"reg{'on' if meta['use_reg'] else 'off'}_"
        f"lt{_format_float_for_name(meta['lambda_tree'])}_"
        f"lr{_format_float_for_name(meta['lambda_radius'])}"
    )

    train_title = f"{meta['embedding']} Train Loss (dim={meta['dim']}, layers={meta['layers']})"
    train_path = os.path.join(base_dir, f"{tag}_train.png")
    _plot_single_curve(train_losses, train_title, train_path)

    if val_losses:
        val_title = f"{meta['embedding']} Val Loss (dim={meta['dim']}, layers={meta['layers']})"
        val_path = os.path.join(base_dir, f"{tag}_val.png")
        _plot_single_curve(val_losses, val_title, val_path)


def main(
    embeddingType: str,
    use_regularization: bool = True,
    traj_splits=None,
    hier=None,
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
    val_ds   = TrajDataset(val_trajs,   max_len=max_len, pad_idx=pad_idx)
    test_ds  = TrajDataset(test_trajs,  max_len=max_len, pad_idx=pad_idx)


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    # 2) embeddings + visit encoder
    if embeddingType == "euclidean":
        code_emb = EuclideanCodeEmbedding(
            num_codes=len(hier.codes) + 1,  # +1 for pad
            dim=dim
        ).to(device)
        visit_enc = EuclideanVisitEncoder(code_emb, pad_idx=pad_idx).to(device)
    else:
        code_emb = HyperbolicCodeEmbedding(
            num_codes=len(hier.codes) + 1,  # +1 for pad
            dim=dim
        ).to(device)
        visit_enc = VisitEncoder(code_emb, pad_idx=pad_idx).to(device)


    # 3) diffusion params
    T = 1000
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
    )

    # 4) model
    n_layers = 2
    eps_model = TrajectoryEpsModel(dim=dim, n_layers=n_layers, T_max=T).to(device)
    codes_per_visit = 4

    # base lambdas (max strength after warm-up)
    lambda_tree = 0.01 if use_regularization else 0.0
    lambda_radius = 0.003 if (embeddingType == "hyperbolic" and use_regularization) else 0.0

    depth_targets = None
    if embeddingType == "hyperbolic" and lambda_radius > 0.0:
        max_depth = max(hier.depth(code) for code in hier.codes)
        depth_targets = torch.tensor(
            [hier.depth(code) / max_depth for code in hier.codes],
            dtype=torch.float32,
            device=device,
        )

    train_losses, val_losses, best_val = train_model(
        eps_model,
        code_emb,
        visit_enc,
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
        n_epochs=5,
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
    }
    save_loss_curves(train_losses, val_losses, meta)
    print("Saved loss curves to results/plots")

    test_loss = evaluate_loader(
        test_dl,
        eps_model,
        visit_enc,
        code_emb,
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
    )
    print(f"Test loss: {test_loss:.6f}")

    num_samples_for_eval = 1000
    synthetic_trajs = sample_trajectories(
        eps_model,
        code_emb,
        visit_enc,
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

    syn_stats = traj_stats(synthetic_trajs, hier)
    print(f"\nSynthetic ({embeddingType}) stats (N={num_samples_for_eval}):", syn_stats)
    return syn_stats


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hier = ToyICDHierarchy()
    all_trajs = sample_toy_trajectories(hier, num_patients=20000)
    splits = split_trajectories(all_trajs, seed=42)
    real_stats = traj_stats(all_trajs, hier)
    print("\nReal stats:", real_stats)

    main(
        embeddingType="hyperbolic",
        use_regularization=False,
        traj_splits=splits,
        hier=hier,
    )
    main(
        embeddingType="euclidean",
        use_regularization=False,
        traj_splits=splits,
        hier=hier,
    )
