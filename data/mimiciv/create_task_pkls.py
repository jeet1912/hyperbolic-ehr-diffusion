#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


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
    return split_map, train_subj, val_subj, test_subj


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
    code_map = {t: int(i + 1) for i, t in enumerate(tokens)}
    return code_map


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


def _pad_multihot(visits, bins, hadm_ids, vocab_size, bin_hours, t_max, truncate):
    T = t_max
    V = vocab_size
    x_multi = np.zeros((len(hadm_ids), T, V), dtype=np.float32)
    mask = np.zeros((len(hadm_ids), T), dtype=np.float32)
    visit_time = np.zeros((len(hadm_ids), T), dtype=np.float32)

    for i, hadm_id in enumerate(hadm_ids):
        v = visits.get(hadm_id, [])
        b = bins.get(hadm_id, [])
        if not v:
            continue
        if len(v) > T:
            if truncate == "latest":
                v = v[-T:]
                b = b[-T:]
            else:
                v = v[:T]
                b = b[:T]

        for t, codes in enumerate(v):
            mask[i, t] = 1.0
            visit_time[i, t] = b[t] * bin_hours
            for code in codes:
                if 0 < code < V:
                    x_multi[i, t, code] = 1.0
    return x_multi, mask, visit_time


def _assemble_pkl(df, label_cols, bin_hours, t_max, truncate, drop_negative, seed, train_frac, val_frac, split_map):
    df = _prepare_events(df, bin_hours, drop_negative)

    df["split"] = df["subject_id"].map(split_map)
    df_train = df[df["split"] == "train"]

    code_map = _build_vocab(df_train)
    vocab_size = len(code_map) + 1

    df = _tokenize_events(df, code_map)
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
    splits = [split_map[int(hadm_to_subject[h])] for h in hadm_ids]
    if len(label_cols) == 1:
        y = [int(labels[h]) for h in hadm_ids]
    else:
        y = [[int(labels[h][c]) for c in label_cols] for h in hadm_ids]

    x_multihot, mask, visit_time = _pad_multihot(
        visits,
        bins,
        hadm_ids,
        vocab_size,
        bin_hours,
        t_max,
        truncate,
    )

    return {
        "x": x,
        "y": y,
        "x_multihot": x_multihot,
        "mask": mask,
        "visit_time_hours": visit_time,
        "code_map": code_map,
        "vocab_size": vocab_size,
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "split": splits,
    }


def main():
    parser = argparse.ArgumentParser(description="Create task PKLs with structured codes.")
    parser.add_argument("--task", required=True, choices=["mortality", "los", "readmission", "diagnosis"])
    parser.add_argument("--input-dir", default="data/mimiciv")
    parser.add_argument("--output-dir", default="data/mimiciv/pkl")
    parser.add_argument("--cohort-path", default="data/mimiciv/llemr_cohort.csv")
    parser.add_argument("--bin-hours", type=int, default=6)
    parser.add_argument("--drop-negative", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--truncate", choices=["latest", "earliest"], default="latest")
    parser.add_argument("--t-max", type=int, default=256)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_map = {
        "mortality": ("llemr_mortality_task.csv", ["label_mortality"]),
        "los": ("llemr_los_task.csv", ["label_los_gt_7d"]),
        "readmission": ("llemr_readmission_task.csv", ["label_readmit_14d"]),
        "diagnosis": (
            "llemr_diagnosis_task.csv",
            [
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
            ],
        ),
    }

    input_file, label_cols = task_map[args.task]
    df = pd.read_csv(input_dir / input_file)

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
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    if args.task in ("mortality", "los"):
        t_max = int(48 // args.bin_hours) + 1
    else:
        t_max = args.t_max

    cohort_df = pd.read_csv(args.cohort_path, usecols=["subject_id"])
    split_map, train_subj, val_subj, test_subj = _build_global_split_map(
        cohort_df["subject_id"].tolist(),
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    data = _assemble_pkl(
        df,
        label_cols=label_cols,
        bin_hours=args.bin_hours,
        t_max=t_max,
        truncate=args.truncate,
        drop_negative=args.drop_negative,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        split_map=split_map,
    )

    out_path = output_dir / f"mimiciv_{args.task}_cohort.pkl"
    with out_path.open("wb") as f:
        pickle.dump(data, f)

    print(
        f"Wrote {out_path} | admissions={len(data['x'])} | "
        f"vocab={data['vocab_size']} | T={t_max} | "
        f"splits: train={len(train_subj)} val={len(val_subj)} test={len(test_subj)}"
    )


if __name__ == "__main__":
    main()
