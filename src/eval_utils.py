import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # pragma: no cover
    roc_auc_score = None
    average_precision_score = None


def _binary_confusion(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def _safe_div(num, den):
    return num / den if den > 0 else 0.0


def _cohen_kappa(y_true, y_pred):
    tp, tn, fp, fn = _binary_confusion(y_true, y_pred)
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    po = (tp + tn) / total
    p_yes_true = (tp + fn) / total
    p_yes_pred = (tp + fp) / total
    p_no_true = (tn + fp) / total
    p_no_pred = (tn + fn) / total
    pe = p_yes_true * p_yes_pred + p_no_true * p_no_pred
    if pe == 1.0:
        return 0.0
    return float((po - pe) / (1 - pe))


def _auroc_fallback(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    tpr = tps / pos
    fpr = fps / neg
    return float(np.trapezoid(tpr, fpr))


def _auprc_fallback(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = np.sum(y_true == 1)
    if pos == 0:
        return float("nan")
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    precision = tps / (tps + fps + 1e-8)
    recall = tps / pos
    idx = np.argsort(recall)
    recall_sorted = recall[idx]
    precision_sorted = precision[idx]
    return float(np.trapezoid(precision_sorted, recall_sorted))


def _auroc_score(y_true, y_prob):
    if roc_auc_score is None:
        return _auroc_fallback(y_true, y_prob)
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _auprc_score(y_true, y_prob):
    if average_precision_score is None:
        return _auprc_fallback(y_true, y_prob)
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def binary_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp, tn, fp, fn = _binary_confusion(y_true, y_pred)
    total = tp + tn + fp + fn
    acc = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    kappa = _cohen_kappa(y_true, y_pred)
    roc = _auroc_score(y_true, y_prob)
    pr = _auprc_score(y_true, y_prob)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "kappa": float(kappa),
        "auroc": float(roc),
        "auprc": float(pr),
    }


def multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 2:
        raise ValueError("multilabel_metrics expects 2D arrays")

    num_labels = y_true.shape[1]
    per_label_auroc = []
    per_label_auprc = []
    per_label_f1 = []
    for i in range(num_labels):
        metrics = binary_classification_metrics(
            y_true[:, i], y_prob[:, i], threshold=threshold
        )
        per_label_f1.append(metrics["f1"])
        label = y_true[:, i]
        if not (np.all(label == 0) or np.all(label == 1)):
            per_label_auroc.append(metrics["auroc"])
            per_label_auprc.append(metrics["auprc"])

    micro_auroc = _auroc_score(y_true.ravel(), y_prob.ravel())
    micro_auprc = _auprc_score(y_true.ravel(), y_prob.ravel())

    return {
        "auroc_macro": float(np.nanmean(per_label_auroc)) if per_label_auroc else float("nan"),
        "auprc_macro": float(np.nanmean(per_label_auprc)) if per_label_auprc else float("nan"),
        "f1_macro": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "auroc_micro": float(micro_auroc),
        "auprc_micro": float(micro_auprc),
        "auroc_per_label": per_label_auroc,
        "auprc_per_label": per_label_auprc,
    }


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


def epoch_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim == 1:
        metrics = binary_classification_metrics(y_true, y_prob, threshold=threshold)
        return {
            "accuracy": metrics["accuracy"],
            "auprc": metrics["auprc"],
        }
    if y_true.ndim == 2:
        # Macro accuracy across labels at fixed threshold.
        accs = []
        for i in range(y_true.shape[1]):
            label_metrics = binary_classification_metrics(
                y_true[:, i], y_prob[:, i], threshold=threshold
            )
            accs.append(label_metrics["accuracy"])
        macro = multilabel_metrics(y_true, y_prob, threshold=threshold)
        return {
            "accuracy": float(np.mean(accs)) if accs else 0.0,
            "auprc": macro["auprc_macro"],
        }
    raise ValueError("epoch_metrics expects 1D or 2D y_true")
