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
