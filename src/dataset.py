import pickle
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MimicDataset(Dataset):
    """
    Lightweight wrapper around the mimic_hf_cohort.pkl file produced by data/create.py.
    Each sample is a list of visits, and each visit is a list of code indices.
    """

    def __init__(self, pkl_path: str):
        print(f"[MIMIC] Loading {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.x = data["x"]
        self.y = data["y"]
        self.code_map = data["code_map"]
        self.vocab_size = data["vocab_size"]
        self.subject_id = data["subject_id"]

        print(f"[MIMIC] Patients: {len(self.x)} | Vocab size: {self.vocab_size}")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[List[List[int]], int]:
        return self.x[idx], self.y[idx]


def make_pad_collate(vocab_size: int) -> Callable:
    """
    Returns a collate function that converts jagged patient sequences into
    multi-hot padded tensors.
    """

    def pad_collate(batch: Sequence[Tuple[List[List[int]], int]]):
        batch_x, batch_y = zip(*batch)

        max_visits = max(len(p) for p in batch_x)
        max_codes = max((len(v) for p in batch_x for v in p), default=0)
        if max_codes == 0:
            max_codes = 1

        padded_x = torch.zeros(len(batch_x), max_visits, vocab_size, dtype=torch.float32)
        mask = torch.zeros(len(batch_x), max_visits, dtype=torch.float32)

        for i, patient in enumerate(batch_x):
            for j, visit in enumerate(patient):
                mask[i, j] = 1.0
                for code in visit:
                    if 0 <= code < vocab_size:
                        padded_x[i, j, code] = 1.0

        tensor_y = torch.tensor(batch_y, dtype=torch.float32)
        return padded_x, tensor_y, mask

    return pad_collate


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
    grouped = (
        df.groupby(["hadm_id", "bin"], sort=True)["token_id"]
        .apply(lambda x: sorted(set(int(v) for v in x if v > 0)))
        .reset_index()
    )
    grouped = grouped[grouped["token_id"].map(len) > 0]
    return grouped


def _build_ragged_sequences(grouped):
    visits = {}
    bins = {}
    for hadm_id, group in grouped.groupby("hadm_id", sort=True):
        group_sorted = group.sort_values("bin")
        visits[int(hadm_id)] = group_sorted["token_id"].tolist()
        bins[int(hadm_id)] = group_sorted["bin"].tolist()
    return visits, bins


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


class MimicCsvDataset(Dataset):
    """
    Build sequences on-the-fly from LLemr task CSVs (no PKL).
    Stores ragged visits + labels, and builds vocab from train split only.
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
        print(f"[MIMIC] Loading task CSV {task_csv} ...")
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
        grouped = _dedupe_visit_tokens(df)
        visits, bins = _build_ragged_sequences(grouped)

        hadm_to_subject = df.groupby("hadm_id")["subject_id"].first().to_dict()
        if len(label_cols) == 1:
            labels = df.groupby("hadm_id")[label_cols[0]].max().to_dict()
        else:
            labels = df.groupby("hadm_id")[label_cols].max().to_dict(orient="index")

        hadm_ids = sorted(visits.keys())
        x = [visits[h] for h in hadm_ids]
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
                    x[i] = x[i][-t_max:] if truncate == "latest" else x[i][:t_max]

        self.x = x
        self.y = y
        self.subject_id = subject_ids
        self.hadm_id = hadm_ids

        print(
            f"[MIMIC] Admissions: {len(self.x)} | Vocab size: {self.vocab_size}"
        )
