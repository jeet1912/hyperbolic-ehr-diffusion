import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_icd_toy import ToyICDHierarchy, sample_toy_trajectories
from hyperbolic_embeddings import HyperbolicCodeEmbedding, VisitEncoder
from euclidean_embeddings import EuclideanCodeEmbedding, EuclideanVisitEncoder
from diffusion import cosine_beta_schedule
from traj_model import TrajectoryEpsModel
from metrics_toy import traj_stats


class TrajDataset(Dataset):
    def __init__(self, trajs, max_len, pad_idx=-1):
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
        return traj  # list of [code_indices]


def collate_fn(batch):
    # batch: list of trajectories (list[list[int]])
    # output: list of visits (length B*L), plus shape info
    B = len(batch)
    L = len(batch[0])
    flat_visits = []
    for traj in batch:
        for visit_codes in traj:
            flat_visits.append(torch.tensor(visit_codes, dtype=torch.long))
    return flat_visits, B, L


def build_visit_tensor(visit_enc, flat_visits, B, L, dim, device):
    visit_enc.eval()
    with torch.no_grad():
        visit_vecs = visit_enc(flat_visits).to(device)  # [B*L, dim]
    x0 = visit_vecs.view(B, L, dim)
    return x0


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
    print_examples=3,
):
    eps_model.eval()
    visit_enc.eval()
    synthetic_trajs = []

    with torch.no_grad():
        x_t = torch.randn(num_samples, max_len, dim, device=device)
        T = betas.shape[0]

        for timestep in reversed(range(T)):
            t_batch = torch.full((num_samples,), timestep, dtype=torch.long, device=device)
            eps_hat = eps_model(x_t, t_batch)

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
        visit_vecs = sampled_visits.view(num_samples * max_len, dim)

        codes_per_visit = 4

        if embeddingType == "hyperbolic":
            code_tangent = visit_enc.manifold.logmap0(code_emb.emb)  # [N, dim]
            sims = visit_vecs @ code_tangent.t()
            topk_idx = sims.topk(k=codes_per_visit, dim=-1).indices.cpu().tolist()
        else:
            code_matrix = code_emb.emb.weight                        # [N, dim]
            diff = visit_vecs.unsqueeze(1) - code_matrix.unsqueeze(0)  # [B*L, N, dim]
            dists_sq = (diff ** 2).sum(-1)                             # [B*L, N]
            sims = -dists_sq
            topk_idx = sims.topk(k=codes_per_visit, dim=-1).indices.cpu().tolist()

    # turn flat visits into trajectories
    for sample_idx in range(num_samples):
        traj_visits = []
        for visit_idx in range(max_len):
            flat_idx = sample_idx * max_len + visit_idx
            visit_codes = sorted(set(topk_idx[flat_idx]))
            traj_visits.append(visit_codes)
        synthetic_trajs.append(traj_visits)

    # print a few examples for sanity
    for sample_idx in range(min(print_examples, num_samples)):
        print(f"\nSample trajectory ({embeddingType}) {sample_idx + 1}:")
        traj_visits = synthetic_trajs[sample_idx]
        for visit_idx, visit_codes in enumerate(traj_visits):
            code_names = [hier.idx2code[idx] for idx in visit_codes]
            print(f"  Visit {visit_idx + 1}: {code_names}")

    return synthetic_trajs



def main(embeddingType: str):
    device = torch.device("cpu")

    # 1) data
    hier = ToyICDHierarchy()
    trajs = sample_toy_trajectories(hier, num_patients=20000)
    max_len = 6
    ds = TrajDataset(trajs, max_len=max_len)
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # 2) embeddings + visit encoder
    dim = 16
    if embeddingType == "euclidean":
        code_emb = EuclideanCodeEmbedding(num_codes=len(hier.codes), dim=dim).to(device)
        visit_enc = EuclideanVisitEncoder(code_emb).to(device)
    else:
        code_emb = HyperbolicCodeEmbedding(num_codes=len(hier.codes), dim=dim).to(device)
        visit_enc = VisitEncoder(code_emb).to(device)

    # 3) diffusion params
    T = 1000
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
    )

    # 4) model
    eps_model = TrajectoryEpsModel(dim=dim, n_layers=4, T_max=T).to(device)
    optimizer = torch.optim.Adam(
        list(eps_model.parameters()) + list(code_emb.parameters()), lr=2e-4
    )

    n_epochs = 20

    for epoch in range(n_epochs):
        eps_model.train()
        for flat_visits, B, L in tqdm(dl, desc=f"Epoch {epoch+1}"):
            flat_visits = [v.to(device) for v in flat_visits]

            # x0 in (hyperbolic tangent or Euclidean) space: [B, L, dim]
            x0 = build_visit_tensor(visit_enc, flat_visits, B, L, dim, device)

            # sample timesteps
            t = torch.randint(0, T, (B,), device=device).long()
            # extract alphas
            a_bar_t = alphas_cumprod[t].view(B, 1, 1)

            eps = torch.randn_like(x0)
            x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1 - a_bar_t) * eps

            eps_hat = eps_model(x_t, t)

            loss = torch.mean((eps - eps_hat) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} loss: {loss.item():.4f}")

    # 5) metrics
    real_stats = traj_stats(trajs, hier)
    print("\nReal stats:", real_stats)

    # sample MANY for evaluation
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
        print_examples=3,
    )

    syn_stats = traj_stats(synthetic_trajs, hier)
    print(f"\nSynthetic ({embeddingType}) stats (N={num_samples_for_eval}):", syn_stats)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # hyperbolic
    main(embeddingType="hyperbolic")
    # euclidean
    main(embeddingType="euclidean")
