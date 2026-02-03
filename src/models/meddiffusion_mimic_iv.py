import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_auc_score = None
    average_precision_score = None

BATCH_SIZE = 32
TRAIN_LR = 1e-3
TRAIN_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
DROPOUT_RATE = 0.2


def _build_global_split_map(subject_ids, seed, train_frac, val_frac):
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(subject_ids)), dtype=np.int64)
    rng.shuffle(unique)

    n = len(unique)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_subj = set(unique[:n_train])
    val_subj = set(unique[n_train:n_train + n_val])
    test_subj = set(unique[n_train + n_val:])

    split_map = {}
    split_map.update({int(s): "train" for s in train_subj})
    split_map.update({int(s): "val" for s in val_subj})
    split_map.update({int(s): "test" for s in test_subj})
    return split_map


def _infer_label_cols(task_name: str):
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
    raise ValueError(f"Unsupported task_name: {task_name}")


def _prepare_events(df, bin_hours, drop_negative):
    df = df.copy()
    df["event_time_hours"] = pd.to_numeric(df["event_time_hours"], errors="coerce")
    df["event_time_avail_hours"] = pd.to_numeric(df["event_time_avail_hours"], errors="coerce")
    df = df.dropna(subset=["event_time_hours"])
    if drop_negative:
        df = df[df["event_time_hours"] >= 0]

    df["bin"] = (df["event_time_hours"] // bin_hours).astype(int)
    bin_end = (df["bin"] + 1) * bin_hours
    bin_end = np.minimum(bin_end, df["t_pred_hours"])
    df = df[
        df["event_time_avail_hours"].isna()
        | (df["event_time_avail_hours"] <= bin_end)
    ]
    df = df[df["event_time_hours"] <= df["t_pred_hours"]]

    df = df[df["code"].notna()]
    df["code"] = df["code"].astype(str).str.strip()
    df = df[~df["code"].isin(["", "nan", "None", "NULL", "null"])]
    return df


def _build_vocab(df_train):
    tokens = pd.unique(df_train["code"])
    return {t: int(i + 1) for i, t in enumerate(tokens)}


def _tokenize_events(df, code_map):
    df = df.copy()
    df["token_id"] = df["code"].map(code_map).fillna(0).astype(int)
    return df


def _dedupe_visit_tokens(df):
    df = df.sort_values(["hadm_id", "bin", "token_id"])
    df = df.drop_duplicates(subset=["hadm_id", "bin", "token_id"])
    return df


def _build_ragged_sequences(df):
    visits = {}
    bins = {}
    grouped = df.groupby(["hadm_id", "bin"], sort=True)["token_id"].apply(list).reset_index()
    for hadm_id, group in grouped.groupby("hadm_id", sort=True):
        group_sorted = group.sort_values("bin")
        visits[int(hadm_id)] = group_sorted["token_id"].tolist()
        bins[int(hadm_id)] = group_sorted["bin"].tolist()
    return visits, bins


class MedDiffusionCsvDataset(Dataset):
    """
    Build sequences on-the-fly from LLemr task CSVs (MedDiffusion only).
    Stores ragged visits + labels + bin indices.
    """

    def __init__(
        self,
        task_csv: str,
        cohort_csv: str,
        task_name: str,
        bin_hours: int = 6,
        drop_negative: bool = False,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        seed: int = 42,
        truncate: str = "latest",
        t_max: int = 256,
    ):
        df = pd.read_csv(task_csv)

        label_cols = _infer_label_cols(task_name)
        required = {
            "subject_id",
            "hadm_id",
            "task_name",
            "t_pred_hours",
            "event_type",
            "code",
            "code_system",
            "event_time_hours",
            "event_time_avail_hours",
        } | set(label_cols)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if task_name in ("mortality", "los"):
            t_max = int(48 // bin_hours) + 1

        cohort_df = pd.read_csv(cohort_csv, usecols=["subject_id"])
        split_map = _build_global_split_map(
            cohort_df["subject_id"].tolist(),
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
        )

        df = _prepare_events(df, bin_hours, drop_negative)
        df["split"] = df["subject_id"].map(split_map)
        df_train = df[df["split"] == "train"]

        self.code_map = _build_vocab(df_train)
        self.vocab_size = len(self.code_map) + 1

        df = _tokenize_events(df, self.code_map)
        df = _dedupe_visit_tokens(df)
        visits, bins = _build_ragged_sequences(df)

        hadm_to_subject = df.groupby("hadm_id")["subject_id"].first().to_dict()
        if len(label_cols) == 1:
            labels = df.groupby("hadm_id")[label_cols[0]].max().to_dict()
        else:
            labels = df.groupby("hadm_id")[label_cols].max().to_dict(orient="index")

        hadm_ids = sorted(visits.keys())
        x = [visits[h] for h in hadm_ids]
        bin_seqs = [bins[h] for h in hadm_ids]
        subject_ids = [int(hadm_to_subject[h]) for h in hadm_ids]
        if len(label_cols) == 1:
            y = [int(labels[h]) for h in hadm_ids]
        else:
            y = [[int(labels[h][c]) for c in label_cols] for h in hadm_ids]

        if truncate not in ("latest", "earliest"):
            raise ValueError("truncate must be 'latest' or 'earliest'")
        if t_max is not None:
            for i in range(len(x)):
                if len(x[i]) > t_max:
                    if truncate == "latest":
                        x[i] = x[i][-t_max:]
                        bin_seqs[i] = bin_seqs[i][-t_max:]
                    else:
                        x[i] = x[i][:t_max]
                        bin_seqs[i] = bin_seqs[i][:t_max]

        self.x = x
        self.bins = bin_seqs
        self.y = y
        self.subject_id = subject_ids
        self.hadm_id = hadm_ids
        self.split = [split_map[int(hadm_to_subject[h])] for h in hadm_ids]
        self.split_indices = {
            "train": [i for i, s in enumerate(self.split) if s == "train"],
            "val": [i for i, s in enumerate(self.split) if s == "val"],
            "test": [i for i, s in enumerate(self.split) if s == "test"],
        }

        print(
            f"[MIMIC] Admissions: {len(self.x)} | Vocab size: {self.vocab_size}"
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.bins[idx]


def make_pad_collate_with_bins(vocab_size: int):
    """
    Collate with padded bin indices for MedDiffusion only.
    """

    def pad_collate(batch):
        batch_x, batch_y, batch_bins = zip(*batch)

        max_visits = max(len(p) for p in batch_x)
        max_codes = max((len(v) for p in batch_x for v in p), default=0)
        if max_codes == 0:
            max_codes = 1

        padded_x = torch.zeros(len(batch_x), max_visits, vocab_size, dtype=torch.float32)
        mask = torch.zeros(len(batch_x), max_visits, dtype=torch.float32)
        padded_bins = torch.zeros(len(batch_x), max_visits, dtype=torch.long)

        for i, (patient, bins) in enumerate(zip(batch_x, batch_bins)):
            for j, visit in enumerate(patient):
                mask[i, j] = 1.0
                if j < len(bins):
                    padded_bins[i, j] = int(bins[j]) + 1
                for code in visit:
                    if 0 <= code < vocab_size:
                        padded_x[i, j, code] = 1.0

        tensor_y = torch.tensor(batch_y, dtype=torch.float32)
        return padded_x, tensor_y, mask, padded_bins

    return pad_collate


def get_last_hidden(h: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    lengths = visit_mask.sum(dim=1).long().clamp_min(1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
    return h.gather(1, idx).squeeze(1)


class MedDiffusionModel(nn.Module):
    """
    MedDiffusion-style model with diffusion-based augmentation on visit embeddings.
    Inputs:
      padded_x: [B, L, V] multi-hot visits
      visit_mask: [B, L] (1=real, 0=pad)
    Outputs:
      logits_real, logits_syn_or_None, diffusion_loss
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.0,
        diffusion_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        max_bin: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.diffusion_steps = diffusion_steps

        self.visit_embed = nn.Linear(vocab_size, embed_dim, bias=False)
        self.bin_embed = nn.Embedding(max_bin + 2, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, out_dim)

        self.proj_h = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.time_embed = nn.Embedding(diffusion_steps + 1, embed_dim)
        self.fuse = nn.Linear(embed_dim * 2, 2)
        self.eps_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(
        self,
        padded_x: torch.Tensor,
        visit_mask: torch.Tensor,
        padded_bins: torch.Tensor,
        train: bool = True,
    ):
        padded_x = padded_x.float()
        visit_mask = visit_mask.float()

        v = self.visit_embed(padded_x)
        padded_bins = padded_bins.long()
        delta_bins = torch.zeros_like(padded_bins)
        if padded_bins.size(1) > 1:
            delta_bins[:, 1:] = (padded_bins[:, 1:] - padded_bins[:, :-1]).clamp_min(0)
        delta_bins = delta_bins.clamp_max(self.bin_embed.num_embeddings - 1)
        delta_bins = delta_bins.masked_fill(visit_mask <= 0, 0)

        v = v + self.bin_embed(delta_bins)
        v = v * visit_mask.unsqueeze(-1)
        v = self.dropout(v)
        lengths = visit_mask.sum(dim=1).long().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            v, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=v.size(1)
        )

        h_last = get_last_hidden(h, visit_mask)
        logits_real = self.classifier(h_last)
        if self.out_dim == 1:
            logits_real = logits_real.squeeze(-1)

        if not train:
            return logits_real, None, v.new_tensor(0.0)

        diffusion_loss = self._diffusion_loss(v, h, visit_mask)
        v_syn = self._sample_synthetic(h, visit_mask)
        packed_syn = nn.utils.rnn.pack_padded_sequence(
            v_syn, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_syn_out, _ = self.lstm(packed_syn)
        h_syn, _ = nn.utils.rnn.pad_packed_sequence(
            packed_syn_out, batch_first=True, total_length=v_syn.size(1)
        )
        h_syn_last = get_last_hidden(h_syn, visit_mask)
        logits_syn = self.classifier(h_syn_last)
        if self.out_dim == 1:
            logits_syn = logits_syn.squeeze(-1)

        return logits_real, logits_syn, diffusion_loss

    def _diffusion_loss(self, v: torch.Tensor, h: torch.Tensor, visit_mask: torch.Tensor):
        B, L, _ = v.shape
        device = v.device

        t = torch.randint(1, self.diffusion_steps + 1, (B,), device=device)
        t = t.view(B, 1).expand(B, L)
        t_safe = torch.where(visit_mask > 0, t, torch.ones_like(t))
        alpha_bar = self.alpha_bars[t_safe - 1].view(B, L, 1)

        eps = torch.randn_like(v)
        z_t = torch.sqrt(alpha_bar) * v + torch.sqrt(1.0 - alpha_bar) * eps
        z_t = z_t * visit_mask.unsqueeze(-1)

        h_prev = torch.zeros_like(h)
        h_prev[:, 1:, :] = h[:, :-1, :]
        h_prev_proj = self.proj_h(h_prev) * visit_mask.unsqueeze(-1)

        t_emb = self.time_embed(t_safe)
        fuse_in = torch.cat([z_t, h_prev_proj], dim=-1)
        alpha = torch.softmax(self.fuse(fuse_in), dim=-1)
        g = alpha[..., :1] * z_t + alpha[..., 1:] * h_prev_proj

        eps_pred = self.eps_net(torch.cat([g, t_emb], dim=-1))

        mask = visit_mask.unsqueeze(-1)
        diff = (eps_pred - eps) ** 2
        diff = diff * mask
        denom = mask.sum().clamp_min(1.0)
        return diff.sum() / denom

    def _sample_synthetic(self, h: torch.Tensor, visit_mask: torch.Tensor):
        B, L, _ = h.shape
        device = h.device
        z = torch.randn(B, L, self.embed_dim, device=device)

        h_prev = torch.zeros_like(h)
        h_prev[:, 1:, :] = h[:, :-1, :]
        h_prev_proj = self.proj_h(h_prev) * visit_mask.unsqueeze(-1)

        for k in range(self.diffusion_steps, 0, -1):
            t = torch.full((B, L), k, device=device, dtype=torch.long)
            t_safe = torch.where(visit_mask > 0, t, torch.ones_like(t))
            t_emb = self.time_embed(t_safe)

            fuse_in = torch.cat([z, h_prev_proj], dim=-1)
            alpha = torch.softmax(self.fuse(fuse_in), dim=-1)
            g = alpha[..., :1] * z + alpha[..., 1:] * h_prev_proj

            eps_pred = self.eps_net(torch.cat([g, t_emb], dim=-1))

            alpha = self.alphas[k - 1].view(1, 1, 1)
            alpha_bar = self.alpha_bars[k - 1].view(1, 1, 1)
            beta = self.betas[k - 1].view(1, 1, 1)

            mean = (z - (beta / torch.sqrt(1.0 - alpha_bar)) * eps_pred) / torch.sqrt(alpha)
            if k > 1:
                noise = torch.randn_like(z)
                z = mean + torch.sqrt(beta) * noise
            else:
                z = mean
            z = z * visit_mask.unsqueeze(-1)

        return z


def binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {"accuracy": float(acc), "f1": float(f1)}
    if roc_auc_score is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auroc"] = float("nan")
    if average_precision_score is not None:
        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["auprc"] = float("nan")
    return metrics


def select_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        metrics = binary_classification_metrics(y_true, y_prob, threshold=thr)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thr = float(thr)
    return best_thr


def multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 2:
        raise ValueError("multilabel_metrics expects 2D arrays")

    per_label_f1 = []
    per_label_auroc = []
    per_label_auprc = []
    for i in range(y_true.shape[1]):
        metrics = binary_classification_metrics(
            y_true[:, i], y_prob[:, i], threshold=threshold
        )
        per_label_f1.append(metrics["f1"])
        label = y_true[:, i]
        if not (np.all(label == 0) or np.all(label == 1)):
            if "auroc" in metrics:
                per_label_auroc.append(metrics["auroc"])
            if "auprc" in metrics:
                per_label_auprc.append(metrics["auprc"])

    return {
        "f1_macro": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "auroc_macro": float(np.nanmean(per_label_auroc)) if per_label_auroc else 0.0,
        "auprc_macro": float(np.nanmean(per_label_auprc)) if per_label_auprc else 0.0,
        "auroc_per_label": per_label_auroc,
        "auprc_per_label": per_label_auprc,
    }


def run_epoch(loader, model, device, optimizer=None, lambda_gen=1.0, lambda_diff=1.0):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for padded_x, labels, visit_mask, padded_bins in loader:
            padded_x = padded_x.to(device)
            labels = labels.float().to(device)
            visit_mask = visit_mask.to(device)
            padded_bins = padded_bins.to(device)

            logits_real, logits_syn, diff_loss = model(
                padded_x, visit_mask, padded_bins, train=is_training
            )
            if labels.ndim == 1:
                if logits_real.shape != labels.shape:
                    raise ValueError(f"Binary logits_real shape {logits_real.shape} != labels {labels.shape}")
                if logits_syn is not None and logits_syn.shape != labels.shape:
                    raise ValueError(f"Binary logits_syn shape {logits_syn.shape} != labels {labels.shape}")
            else:
                if logits_real.shape != labels.shape:
                    raise ValueError(f"Multilabel logits_real shape {logits_real.shape} != labels {labels.shape}")
                if logits_syn is not None and logits_syn.shape != labels.shape:
                    raise ValueError(f"Multilabel logits_syn shape {logits_syn.shape} != labels {labels.shape}")
            loss_real = bce(logits_real, labels)
            loss_syn = bce(logits_syn, labels) if logits_syn is not None else 0.0
            loss = loss_real + lambda_gen * loss_syn + lambda_diff * diff_loss

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * padded_x.size(0)
            total_samples += padded_x.size(0)

    return total_loss / max(total_samples, 1)


def train_model(train_loader, val_loader, model, device, lambda_gen=1.0, lambda_diff=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.2
    )

    best_val = float("inf")
    best_state = None
    patience = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, TRAIN_EPOCHS + 1):
        train_loss = run_epoch(
            train_loader,
            model,
            device,
            optimizer=optimizer,
            lambda_gen=lambda_gen,
            lambda_diff=lambda_diff,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            device,
            optimizer=None,
            lambda_gen=lambda_gen,
            lambda_diff=lambda_diff,
        )
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[MedDiffusion] Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("[MedDiffusion] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, train_losses, val_losses


def collect_probs(loader, model, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for padded_x, labels, visit_mask, padded_bins in loader:
            padded_x = padded_x.to(device)
            labels = labels.float().to(device)
            visit_mask = visit_mask.to(device)
            padded_bins = padded_bins.to(device)

            logits_real, _, _ = model(padded_x, visit_mask, padded_bins, train=False)
            probs = torch.sigmoid(logits_real)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if not all_labels:
        return None, None
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    return y_true, y_prob


def evaluate(loader, model, device, threshold=0.5):
    y_true, y_prob = collect_probs(loader, model, device)
    if y_true is None:
        return {}
    if y_true.ndim == 2:
        return multilabel_metrics(y_true, y_prob, threshold=threshold)
    return binary_classification_metrics(y_true, y_prob, threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description="MedDiffusion for LLemr task CSVs.")
    parser.add_argument("--task-csv", type=str, required=True)
    parser.add_argument("--cohort-csv", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True,
                        choices=["mortality", "los", "readmission", "diagnosis"])
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--lambda-gen", type=float, default=1.0)
    parser.add_argument("--lambda-diff", type=float, default=1.0)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    dataset = MedDiffusionCsvDataset(
        task_csv=args.task_csv,
        cohort_csv=args.cohort_csv,
        task_name=args.task_name,
    )
    collate_fn = make_pad_collate_with_bins(dataset.vocab_size)

    train_idx = dataset.split_indices["train"]
    val_idx = dataset.split_indices["val"]
    test_idx = dataset.split_indices["test"]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    out_dim = 1
    if isinstance(dataset.y[0], (list, tuple, np.ndarray)):
        out_dim = len(dataset.y[0])

    def _safe_max_bin(bin_seq):
        if bin_seq is None:
            return 0
        if isinstance(bin_seq, torch.Tensor):
            return int(bin_seq.max().item()) if bin_seq.numel() else 0
        arr = np.asarray(bin_seq)
        return int(arr.max()) if arr.size else 0

    max_bin = max((_safe_max_bin(b) for b in dataset.bins), default=0)
    max_bin = max_bin + 1

    model = MedDiffusionModel(
        vocab_size=dataset.vocab_size,
        embed_dim=128,
        hidden_dim=128,
        out_dim=out_dim,
        dropout=DROPOUT_RATE,
        diffusion_steps=1000,
        max_bin=max_bin,
    ).to(device)

    best_val, _, _ = train_model(
        train_loader,
        val_loader,
        model,
        device,
        lambda_gen=args.lambda_gen,
        lambda_diff=args.lambda_diff,
    )
    print(f"[MedDiffusion] Best validation loss: {best_val:.4f}")

    val_y, val_p = collect_probs(val_loader, model, device)
    threshold = 0.5
    if val_y is not None and val_y.ndim == 1:
        threshold = select_best_threshold(val_y, val_p)
    metrics = evaluate(test_loader, model, device, threshold=threshold)
    if val_y is not None and val_y.ndim == 1:
        metrics["threshold"] = float(threshold)
    print("[MedDiffusion] Test metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
