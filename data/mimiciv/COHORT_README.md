# Cohort and Task Filtering Notes

I built the cohort in BigQuery with `data/mimiciv/llemr_cohort_bigquery.sql`. I start from ICU admissions via `icustays` and apply the filters described in the paper and confirmed in `llemr-main/src/preprocess/01_cohort_selection.ipynb`: require discharge summaries (`mimiciv_note.discharge`), keep admissions with exactly one ICU stay, remove patients younger than 18 at admission, remove negative ICU or hospital LOS, and apply the selected-event length cutoff (`len_selected <= 1256.65`).

I build `event_selected` from the paper-listed sources plus the static records used in llemr-main: hosp/diagnoses_icd, hosp/labevents, hosp/microbiologyevents, hosp/prescriptions, hosp/transfers, icu/inputevents, icu/outputevents, icu/procedureevents, and `patient_demographics` + `admission_info`. icu/chartevents are excluded.

Task cohorts are derived from the filtered cohort. Mortality uses the first 48 hours of hospital admission and labels death at discharge; LOS uses the first 48 hours and labels LOS > 7 days; both exclude admissions with LOS < 48 hours. Readmission uses all events from the admission and excludes in-hospital deaths, then labels readmission within 14 days. I do train/val/test splits after export in Python, and I split by subject_id to avoid patient leakage.

Diagnosis phenotyping follows the 25 CCS groups from Harutyunyan et al. (MIMIC-III benchmark). I keep ICD-9 definitions in `data/mimiciv/phenotype_icd9.csv` and map to ICD-10 using the GEMs crosswalk in `data/icd9toicd10cmgem.csv` (excluding no-map entries). The resulting labels are stored in `llemr_diagnosis`. For sequence models, I use structured event codes (`event_code`) as tokens (e.g., ICD9/10 codes, item IDs with prefixes), not free-text event values.

Latest local CSV counts (downloaded from BigQuery on 2026-02-01):
- Cohort admissions: 53,419.
- Mortality admissions: 47,660.
- LOS admissions: 47,660.
- Readmission admissions: 49,159.
- Diagnosis admissions: 53,419 (25 labels).

Sequence preprocessing for RETAIN/MedDiffusion/my model:
- Visits are 6-hour bins. Events are only included if `timestamp_avail` is not later than the end of the bin (availability gating).
- Codes are deduplicated within a visit (binary per code per visit).
- Subject-level splits are global (computed once from the base cohort and reused across tasks).
- Vocab is built on train split only; unseen tokens map to 0 in val/test.
- I keep ragged visit sequences (`x`) and also emit fixed-length multihot tensors (`x_multihot`) with `mask` and `visit_time_hours`.
- For mortality/LOS, T is fixed at `ceil(48 / bin_hours)`; for readmission/diagnosis I truncate to a max T (newest-first).

Canonical task export format (BigQuery `*_task` tables) includes: subject_id, hadm_id, task_name, label(s), t_pred_hours, event_time_hours, event_time_avail_hours, code, code_system, event_type, event_value. Codes are controlled identifiers (ICD9/10, LAB/INPUT/OUTPUT/PROC item IDs, NDC for drugs, MICRO test_itemid). Free-text is only kept in `event_value` for Llemr; sequence models use `code`.

## Code hierarchy used by the model (ICD only)

For LLemr cohorts, the `code` field mixes ICD9/ICD10 diagnosis codes with other structured codes
(LAB, INPUT/OUTPUT/PROC, NDC, MICRO). In `src/risk_prediction_mimic.py`, the hierarchy is an ICD
tree provided via `--icd-tree` (e.g., `data/mimiciii/icd9_parent_map.csv`). This hierarchy applies
only to ICD codes; non-ICD codes are treated as flat tokens.

Example ICD hierarchy (broad -> specific):

```
Root
└─ ICD9
   └─ 390–459 (Circulatory system)
      └─ 410 (Acute myocardial infarction)
         └─ 410.0 (AMI of anterolateral wall)
            └─ 410.01 (initial episode of care)
```

And ICD10:

```
Root
└─ ICD10
   └─ I00–I99 (Diseases of the circulatory system)
      └─ I21 (Acute myocardial infarction)
         └─ I21.0 (STEMI of anterior wall)
```

## End-to-end rebuild steps (BigQuery -> CSVs -> PKLs)

1) Run the BigQuery build script (creates/overwrites tables + exports task CSVs):

```bash
bq --location=US query --use_legacy_sql=false < data/mimiciv/llemr_cohort_bigquery.sql
```

2) Export the base cohort table (not exported by the script):

```bash
bq --location=US query --use_legacy_sql=false <<'SQL'
EXPORT DATA OPTIONS (
  uri = 'gs://mimiciv-exports-485411/llemr_cohort_*.csv',
  format = 'CSV',
  overwrite = true,
  header = true
) AS
SELECT * FROM `mimic-iv-485411.hyperbolic_ehr.llemr_cohort`;
SQL
```

3) Download all exported shards from GCS:

```bash
mkdir -p data/mimiciv/gcs_exports
gsutil -m cp "gs://mimiciv-exports-485411/llemr_*_*.csv" data/mimiciv/gcs_exports/
```

4) Merge shards into single local CSVs:

```bash
python3 - <<'PY'
import glob, pandas as pd

def combine(prefix, out_path):
    parts = sorted(glob.glob(f"data/mimiciv/gcs_exports/{prefix}_*.csv"))
    if not parts:
        raise SystemExit(f"missing shards for {prefix}")
    df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"[wrote] {out_path} rows={len(df)}")

combine("llemr_cohort", "data/mimiciv/llemr_cohort.csv")
combine("llemr_mortality_task", "data/mimiciv/llemr_mortality_task.csv")
combine("llemr_los_task", "data/mimiciv/llemr_los_task.csv")
combine("llemr_readmission_task", "data/mimiciv/llemr_readmission_task.csv")
combine("llemr_diagnosis_task", "data/mimiciv/llemr_diagnosis_task.csv")
PY
```

5) Build PKLs for RETAIN/MedDiffusion/our model:

```bash
python3 data/mimiciv/create_task_pkls.py --task mortality --input-dir data/mimiciv --output-dir data/mimiciv/pkl --cohort-path data/mimiciv/llemr_cohort.csv
python3 data/mimiciv/create_task_pkls.py --task los        --input-dir data/mimiciv --output-dir data/mimiciv/pkl --cohort-path data/mimiciv/llemr_cohort.csv
python3 data/mimiciv/create_task_pkls.py --task readmission --input-dir data/mimiciv --output-dir data/mimiciv/pkl --cohort-path data/mimiciv/llemr_cohort.csv
python3 data/mimiciv/create_task_pkls.py --task diagnosis   --input-dir data/mimiciv --output-dir data/mimiciv/pkl --cohort-path data/mimiciv/llemr_cohort.csv
```

Notes:
- The BigQuery script exports only the `llemr_*_task` tables; the base cohort export must be run separately (step 2).
- LLemr uses the merged `llemr_*_task.csv` files directly. RETAIN/MedDiffusion/our model use the PKLs from step 5.
- Readmission/diagnosis PKLs are large (multihot tensors). If you hit "No space left on device", free disk space or set `--output-dir` to a larger drive before rerunning those tasks.
