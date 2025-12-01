import os
import json
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm


# =========================================================
# 1) Load CSV files produced from MySQL extraction
# =========================================================

matched = pd.read_csv("tmp_matched_cohort.csv")           # SUBJECT_ID, LABEL, INDEX_DATE
visits = pd.read_csv("tmp_windowed_data.csv")             # SUBJECT_ID, HADM_ID
codes  = pd.read_csv("tmp_windowed_codes.csv")            # HADM_ID, ICD9_CODE
vocab_df = pd.read_csv("tmp_global_vocabulary.csv")       # ICD9_CODE (unique)

valid_pool = None
if os.path.exists("tmp_valid_pool.csv"):
    valid_pool = pd.read_csv("tmp_valid_pool.csv")        # SUBJECT_ID, GENDER, AGE, RACE_BUCKET, TOTAL_VISITS, ...


# =========================================================
# 2) Build vocabulary (ICD9 → integer index)
# =========================================================

vocab_list = sorted(vocab_df["ICD9_CODE"].unique())
vocab = {code: i + 1 for i, code in enumerate(vocab_list)}   # 0 reserved for PAD

with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print(f"[Vocab] Size = {len(vocab)}")


# =========================================================
# 3) Join visits → codes into visit sequences
# =========================================================

hadm_to_codes = defaultdict(list)
for _, row in codes.iterrows():
    c = row["ICD9_CODE"]
    if c in vocab:
        hadm_to_codes[row["HADM_ID"]].append(vocab[c])

subject_to_visits = defaultdict(list)
for _, row in visits.iterrows():
    subject_to_visits[row["SUBJECT_ID"]].append(row["HADM_ID"])


# =========================================================
# 4) Convert each patient → sorted visits → list of code indices
# =========================================================

ehr_sequences = []
labels = []
subject_ids = []   # keep aligned SUBJECT_IDs
dropped_no_codes = 0

print("\n[Build] Constructing patient EHR sequences...\n")

for _, row in tqdm(matched.iterrows(), total=len(matched)):
    sid = row["SUBJECT_ID"]
    label = int(row["LABEL"])

    hadm_list = sorted(subject_to_visits.get(sid, []))  # time-ordered
    patient_visits = [hadm_to_codes[h] for h in hadm_list if h in hadm_to_codes and len(hadm_to_codes[h]) > 0]

    if len(patient_visits) == 0:
        # patient in matched cohort but no usable codes in window
        dropped_no_codes += 1
        continue

    ehr_sequences.append(patient_visits)
    labels.append(label)
    subject_ids.append(sid)

print(f"[Build] Patients in matched cohort: {len(matched)}")
print(f"[Build] Patients with at least one coded visit in window: {len(ehr_sequences)}")
print(f"[Build] Patients dropped due to no codes in window: {dropped_no_codes}")


# =========================================================
# 5) Padding for uniform visits × codes (for batching)
# =========================================================

max_visits = max(len(v) for v in ehr_sequences)
max_codes = max(len(codes) for v in ehr_sequences for codes in v)

def pad_patient(visits_one_patient):
    padded = []
    mask = []
    for v in visits_one_patient:
        v_trunc = v[:max_codes]
        v_pad = v_trunc + [0] * (max_codes - len(v_trunc))
        padded.append(v_pad)
        mask.append(1)
    for _ in range(max_visits - len(visits_one_patient)):
        padded.append([0] * max_codes)
        mask.append(0)
    return padded, mask

padded_seqs = []
padded_masks = []

for v in ehr_sequences:
    p, m = pad_patient(v)
    padded_seqs.append(p)
    padded_masks.append(m)


# =========================================================
# 6) Save tensors for MedDiffusion
# =========================================================

ehr_tensor = torch.tensor(padded_seqs, dtype=torch.long)
label_tensor = torch.tensor(labels, dtype=torch.long)
mask_tensor = torch.tensor(padded_masks, dtype=torch.float32)

torch.save(ehr_tensor, "ehr_sequences.pt")
torch.save(label_tensor, "labels.pt")
torch.save(mask_tensor, "visit_masks.pt")

# also save aligned SUBJECT_IDs for cross-checking later
with open("subject_ids.json", "w") as f:
    json.dump(subject_ids, f, indent=2)


# =========================================================
# 7) Statistics & Verification
# =========================================================

print("\n[Stats] Verifying dataset statistics...")

num_patients = len(ehr_sequences)
num_visits = sum(len(v) for v in ehr_sequences)
num_codes = sum(len(codes_in_visit) for v in ehr_sequences for codes_in_visit in v)

avg_visits = num_visits / num_patients if num_patients > 0 else 0.0
avg_codes_per_visit = num_codes / num_visits if num_visits > 0 else 0.0

# Positive / Negative counts (after possible drops)
pos_count = sum(1 for y in labels if y == 1)
neg_count = sum(1 for y in labels if y == 0)

# Unique codes actually used (might be smaller than global vocab)
used_code_ids = set()
for v in ehr_sequences:
    for visit_codes in v:
        used_code_ids.update([c for c in visit_codes if c != 0])
unique_codes_used = len(used_code_ids)

stats = {
    "patients_kept": num_patients,
    "patients_dropped_no_codes": dropped_no_codes,
    "positives": pos_count,
    "negatives": neg_count,
    "class_ratio_pos_neg": f"{pos_count}:{neg_count}" if neg_count > 0 else "inf",
    "vocab_size_declared": len(vocab),
    "unique_codes_used": unique_codes_used,
    "total_visits": num_visits,
    "total_codes": num_codes,
    "avg_visits_per_patient": round(avg_visits, 2),
    "avg_codes_per_visit": round(avg_codes_per_visit, 2),
    "max_visits": max_visits,
    "max_codes_per_visit": max_codes,
}

# Optional: Demographic sanity checks if tmp_valid_pool.csv is present
if valid_pool is not None:
    vp = valid_pool.set_index("SUBJECT_ID")
    aligned = vp.loc[vp.index.intersection(subject_ids)]

    if "AGE" in aligned.columns:
        stats["mean_age"] = float(aligned["AGE"].mean())
        stats["age_min"] = int(aligned["AGE"].min())
        stats["age_max"] = int(aligned["AGE"].max())

    if "GENDER" in aligned.columns:
        stats["gender_counts"] = aligned["GENDER"].value_counts().to_dict()

    if "RACE_BUCKET" in aligned.columns:
        stats["race_counts"] = aligned["RACE_BUCKET"].value_counts().to_dict()

    if "TOTAL_VISITS" in aligned.columns:
        stats["mean_total_visits_in_db"] = float(aligned["TOTAL_VISITS"].mean())

with open("stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))

print("\n==============================")
print(" MedDiffusion Preprocess Done ")
print("==============================")