import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

from dataset import MimicCsvDataset, make_pad_collate

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover - fallback handled in code
    roc_auc_score = None
    average_precision_score = None


BATCH_SIZE = 32
TRAIN_LR = 1e-4
TRAIN_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
DROPOUT_RATE = 0.2


class RETAIN(nn.Module):
    """
    RETAIN-style model for visit sequences (Choi et al., 2016).
    Expects multihot inputs: [B, L, V] and visit mask: [B, L] (1=real).
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.code_embed = nn.Linear(vocab_size, embed_dim, bias=False)
        self.gru_alpha = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.gru_beta = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.alpha_proj = nn.Linear(hidden_dim, 1)
        self.beta_proj = nn.Linear(hidden_dim, embed_dim)
        self.out = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor, visit_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        if visit_mask is None:
            visit_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.float32)

        lengths = visit_mask.sum(dim=1).long()
        v = self.code_embed(x)  # [B, L, D]
        v = self.dropout(v)

        v_rev = reverse_by_lengths(v, lengths)
        mask_rev = reverse_by_lengths(visit_mask, lengths)

        h_alpha, _ = self.gru_alpha(v_rev)
        h_beta, _ = self.gru_beta(v_rev)

        alpha = self.alpha_proj(h_alpha).squeeze(-1)  # [B, L]
        alpha = alpha.masked_fill(mask_rev == 0, float("-inf"))
        alpha = torch.softmax(alpha, dim=1).unsqueeze(-1)

        beta = torch.tanh(self.beta_proj(h_beta))
        context_rev = alpha * beta * v_rev
        context = context_rev.sum(dim=1)
        if (lengths == 0).any():
            zero_mask = (lengths == 0).unsqueeze(-1)
            context = torch.where(zero_mask, torch.zeros_like(context), context)

        logits = self.out(context)
        if self.out_dim == 1:
            return logits.squeeze(-1)
        return logits


def reverse_by_lengths(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Reverse the first lengths[b] timesteps for each batch element.
    Supports x shaped [B, L, D] or [B, L].
    """
    if x.dim() == 2:
        x = x.unsqueeze(-1)
        squeeze_last = True
    else:
        squeeze_last = False
    B, L, D = x.shape
    out = x.clone()
    for b in range(B):
        l = int(lengths[b].item())
        if l <= 1:
            continue
        out[b, :l] = torch.flip(x[b, :l], dims=[0])
    if squeeze_last:
        out = out.squeeze(-1)
    return out


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
        label = y_true[:, i]
        metrics = binary_classification_metrics(
            y_true[:, i], y_prob[:, i], threshold=threshold
        )
        per_label_f1.append(metrics["f1"])
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


def run_epoch(loader, model, device, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for padded_x, labels, visit_mask in loader:
            padded_x = padded_x.to(device)
            labels = labels.float().to(device)
            visit_mask = visit_mask.to(device)

            logits = model(padded_x, visit_mask)
            loss = bce(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * padded_x.size(0)
            total_samples += padded_x.size(0)

    return total_loss / max(total_samples, 1)


def train_model(train_loader, val_loader, model, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

    best_val = float("inf")
    best_state = None
    patience = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, TRAIN_EPOCHS + 1):
        train_loss = run_epoch(train_loader, model, device, optimizer=optimizer)
        val_loss = run_epoch(val_loader, model, device, optimizer=None)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[RETAIN] Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("[RETAIN] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if train_losses and val_losses:
        print(f"[RETAIN] Final Train/Val Loss: {train_losses[-1]:.4f} / {val_losses[-1]:.4f}")
    return best_val, train_losses, val_losses


def collect_probs(loader, model, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for padded_x, labels, visit_mask in loader:
            padded_x = padded_x.to(device)
            labels = labels.float().to(device)
            visit_mask = visit_mask.to(device)

            logits = model(padded_x, visit_mask)
            probs = torch.sigmoid(logits)

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
    parser = argparse.ArgumentParser(description="RETAIN baseline for LLemr task CSVs.")
    parser.add_argument("--task-csv", type=str, required=True)
    parser.add_argument("--cohort-csv", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True,
                        choices=["mortality", "los", "readmission", "diagnosis"])
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
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

    dataset = MimicCsvDataset(
        task_csv=args.task_csv,
        cohort_csv=args.cohort_csv,
        task_name=args.task_name,
    )
    collate_fn = make_pad_collate(dataset.vocab_size)

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
    model = RETAIN(
        vocab_size=dataset.vocab_size,
        out_dim=out_dim,
        dropout=DROPOUT_RATE,
    ).to(device)

    best_val, _, _ = train_model(train_loader, val_loader, model, device)
    print(f"[RETAIN] Best validation loss: {best_val:.4f}")

    val_y, val_p = collect_probs(val_loader, model, device)
    threshold = 0.5
    if val_y is not None and val_y.ndim == 1:
        threshold = select_best_threshold(val_y, val_p)
    metrics = evaluate(test_loader, model, device, threshold=threshold)
    if val_y is not None and val_y.ndim == 1:
        metrics["threshold"] = float(threshold)
    print("[RETAIN] Test metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
