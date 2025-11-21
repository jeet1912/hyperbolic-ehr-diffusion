import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_icd_toy import ToyICDHierarchy, sample_toy_trajectories
from hyperbolic_embeddings import HyperbolicCodeEmbedding, VisitEncoder
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


def build_visit_tensor(visit_enc: VisitEncoder, flat_visits, B, L, dim, device):
    visit_enc.eval()
    with torch.no_grad():
        visit_vecs = visit_enc(flat_visits).to(device)  # [B*L, dim]
    x0 = visit_vecs.view(B, L, dim)
    return x0


def main():
    device = torch.device("cpu")

    # 1) data
    hier = ToyICDHierarchy()
    trajs = sample_toy_trajectories(hier, num_patients=20000)
    max_len = 6
    ds = TrajDataset(trajs, max_len=max_len)
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

    # 2) hyperbolic embeddings + visit encoder
    dim = 16
    code_emb = HyperbolicCodeEmbedding(num_codes=len(hier.codes), dim=dim).to(device)
    visit_enc = VisitEncoder(code_emb).to(device)

    # 3) diffusion params
    T = 1000
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

    # 4) model
    eps_model = TrajectoryEpsModel(dim=dim, T_max=T).to(device)
    optimizer = torch.optim.Adam(list(eps_model.parameters()) + list(code_emb.parameters()), lr=2e-4)

    n_epochs = 5

    for epoch in range(n_epochs):
        eps_model.train()
        for flat_visits, B, L in tqdm(dl, desc=f"Epoch {epoch+1}"):
            flat_visits = [v.to(device) for v in flat_visits]

            # x0 in tangent space: [B, L, dim]
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

    synthetic_trajs = []
    # diffusion sampling
    eps_model.eval()
    visit_enc.eval()
    num_samples = 3
    with torch.no_grad():
        x_t = torch.randn(num_samples, max_len, dim, device=device)
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

        sampled_visits = x_t  # approximated x_0

        # decode visits by nearest hyperbolic code embeddings in tangent space
        code_tangent = visit_enc.manifold.logmap0(code_emb.emb)
        visit_vecs = sampled_visits.view(num_samples * max_len, dim)
        sims = visit_vecs @ code_tangent.t()
        codes_per_visit = 4
        topk_idx = sims.topk(k=codes_per_visit, dim=-1).indices.cpu().tolist()

    for sample_idx in range(num_samples):
        print(f"\nSample trajectory {sample_idx + 1}:")
        traj_visits = []
        for visit_idx in range(max_len):
            flat_idx = sample_idx * max_len + visit_idx
            visit_codes = sorted(set(topk_idx[flat_idx]))
            traj_visits.append(visit_codes)
            code_names = [hier.idx2code[idx] for idx in visit_codes]
            print(f"  Visit {visit_idx + 1}: {code_names}")
        synthetic_trajs.append(traj_visits)

    real_stats = traj_stats(trajs, hier)
    syn_stats = traj_stats(synthetic_trajs, hier)
    print("\nReal stats:", real_stats)
    print("Synthetic (hyperbolic) stats:", syn_stats)

if __name__ == "__main__":
    main()
