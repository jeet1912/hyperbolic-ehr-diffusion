-- BigQuery cohort build (base + event_selected + len_selected)
-- Project: mimic-iv-485411
-- Dataset: hyperbolic_ehr

-- Discharge summaries
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.discharge_summary_hadm` AS
SELECT DISTINCT hadm_id
FROM `physionet-data.mimiciv_note.discharge`
WHERE hadm_id IS NOT NULL;

-- ICU stay counts + bad LOS flags
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.icu_stay_counts` AS
SELECT
  hadm_id,
  COUNT(*) AS icu_stay_count,
  SUM(CASE WHEN outtime < intime OR los < 0 THEN 1 ELSE 0 END) AS bad_icu_los
FROM `physionet-data.mimiciv_3_1_icu.icustays`
GROUP BY hadm_id;

-- Base cohort
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` AS
SELECT
  a.subject_id,
  a.hadm_id,
  s.icu_stay_count
FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
JOIN `mimic-iv-485411.hyperbolic_ehr.icu_stay_counts` s
  ON s.hadm_id = a.hadm_id
JOIN `mimic-iv-485411.hyperbolic_ehr.discharge_summary_hadm` ds
  ON ds.hadm_id = a.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
  ON p.subject_id = a.subject_id
WHERE a.dischtime >= a.admittime
  AND s.icu_stay_count = 1
  AND s.bad_icu_los = 0
  AND (EXTRACT(YEAR FROM a.admittime) - p.anchor_year + p.anchor_age) >= 18;

-- Selected events
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.event_selected` AS
SELECT
  c.hadm_id,
  'patient_demographics' AS event_type,
  0.0 AS timestamp,
  CONCAT(
    'gender: ', p.gender,
    ', age: ', CAST(EXTRACT(YEAR FROM a.admittime) - p.anchor_year + p.anchor_age AS STRING),
    ', race: ', a.race,
    IF(a.marital_status IS NOT NULL AND a.marital_status <> '',
      CONCAT(', marital status: ', a.marital_status), ''),
    ', insurance: ', a.insurance
  ) AS event_value,
  0.0 AS timestamp_avail
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = c.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
  ON p.subject_id = c.subject_id

UNION ALL
SELECT
  c.hadm_id,
  'admission_info' AS event_type,
  0.0 AS timestamp,
  CONCAT('type: ', a.admission_type, ', location: ', a.admission_location) AS event_value,
  0.0 AS timestamp_avail
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = c.hadm_id

UNION ALL
SELECT
  l.hadm_id,
  'labevents' AS event_type,
  DATETIME_DIFF(l.charttime, a.admittime, SECOND) / 3600.0 AS timestamp,
  CONCAT(
    CASE
      WHEN l.valueuom IS NULL OR l.valueuom = ''
        THEN CONCAT(dl.fluid, ' ', dl.label, ' ', dl.category, ': ', CAST(l.value AS STRING))
      ELSE CONCAT(dl.fluid, ' ', dl.label, ' ', dl.category, ': ', CAST(l.value AS STRING), ' ', l.valueuom)
    END,
    CASE WHEN l.flag IS NULL OR l.flag = '' THEN ' (normal)' ELSE ' (abnormal)' END
  ) AS event_value,
  DATETIME_DIFF(l.storetime, a.admittime, SECOND) / 3600.0 AS timestamp_avail
FROM `physionet-data.mimiciv_3_1_hosp.labevents` l
JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
  ON c.hadm_id = l.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = l.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` dl
  ON dl.itemid = l.itemid
WHERE l.value IS NOT NULL AND l.charttime IS NOT NULL AND l.storetime IS NOT NULL

UNION ALL
SELECT
  m.hadm_id,
  'microbiologyevents' AS event_type,
  DATETIME_DIFF(m.charttime, a.admittime, SECOND) / 3600.0 AS timestamp,
  CONCAT(
    m.test_name, ' on ', m.spec_type_desc,
    IF(m.org_name IS NOT NULL AND m.org_name <> '', CONCAT(', organism grew: ', m.org_name), ''),
    IF(m.ab_name IS NOT NULL AND m.ab_name <> '', CONCAT(', antibiotic tested: ', m.ab_name), ''),
    IF(m.interpretation IS NOT NULL AND m.interpretation <> '', CONCAT(', antibiotic sensitivity: ', m.interpretation), ''),
    IF(m.comments IS NOT NULL AND m.comments <> '', CONCAT(', comments: ', m.comments), '')
  ) AS event_value,
  DATETIME_DIFF(m.storetime, a.admittime, SECOND) / 3600.0 AS timestamp_avail
FROM `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
  ON c.hadm_id = m.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = m.hadm_id
WHERE m.charttime IS NOT NULL AND m.storetime IS NOT NULL

UNION ALL
SELECT
  p.hadm_id,
  'prescriptions' AS event_type,
  DATETIME_DIFF(p.starttime, a.admittime, SECOND) / 3600.0 AS timestamp,
  CONCAT(
    p.drug,
    IF(p.prod_strength IS NOT NULL AND p.prod_strength <> '', CONCAT(' (', p.prod_strength, ')'), ''),
    IF(p.dose_val_rx IS NOT NULL AND p.dose_val_rx <> '', CONCAT(', prescribed dose: ', p.dose_val_rx), ''),
    IF(p.dose_unit_rx IS NOT NULL AND p.dose_unit_rx <> '', CONCAT(' ', p.dose_unit_rx), ''),
    ', route: ', p.route,
    ', duration: ', IFNULL(CAST(ROUND(DATETIME_DIFF(p.stoptime, p.starttime, SECOND) / 3600.0, 2) AS STRING), 'nan'),
    ' hour'
  ) AS event_value,
  DATETIME_DIFF(p.starttime, a.admittime, SECOND) / 3600.0 AS timestamp_avail
FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` p
JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
  ON c.hadm_id = p.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = p.hadm_id
WHERE p.starttime IS NOT NULL

UNION ALL
SELECT
  t.hadm_id,
  'transfers' AS event_type,
  DATETIME_DIFF(t.intime, a.admittime, SECOND) / 3600.0 AS timestamp,
  CONCAT(t.eventtype, IF(t.careunit IS NOT NULL AND t.careunit <> '', CONCAT(' to ', t.careunit), '')) AS event_value,
  DATETIME_DIFF(t.intime, a.admittime, SECOND) / 3600.0 AS timestamp_avail
FROM `physionet-data.mimiciv_3_1_hosp.transfers` t
JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
  ON c.hadm_id = t.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = t.hadm_id
WHERE t.intime IS NOT NULL

UNION ALL
SELECT
  pe.hadm_id,
  'procedureevents' AS event_type,
  DATETIME_DIFF(pe.starttime, a.admittime, SECOND) / 3600.0 AS timestamp,
  CONCAT(
    di.label,
    ' for ',
    IFNULL(CAST(ROUND(DATETIME_DIFF(pe.endtime, pe.starttime, SECOND) / 3600.0, 2) AS STRING), 'nan'),
    ' hour'
  ) AS event_value,
  DATETIME_DIFF(pe.starttime, a.admittime, SECOND) / 3600.0 AS timestamp_avail
FROM `physionet-data.mimiciv_3_1_icu.procedureevents` pe
JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
  ON c.hadm_id = pe.hadm_id
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = pe.hadm_id
JOIN `physionet-data.mimiciv_3_1_icu.d_items` di
  ON di.itemid = pe.itemid
WHERE pe.starttime IS NOT NULL;

-- Per-admission selected event length
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.event_selected_lengths` AS
SELECT hadm_id, COUNT(*) AS len_selected
FROM `mimic-iv-485411.hyperbolic_ehr.event_selected`
GROUP BY hadm_id;

-- Final cohort with len_selected cutoff
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_cohort` AS
SELECT
  c.subject_id,
  c.hadm_id,
  c.icu_stay_count,
  COALESCE(es.len_selected, 0) AS len_selected
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_cohort_base` c
LEFT JOIN `mimic-iv-485411.hyperbolic_ehr.event_selected_lengths` es
  ON es.hadm_id = c.hadm_id
WHERE COALESCE(es.len_selected, 0) <= 1256.65;

-- ------------------------
-- Task cohorts (no splits)
-- ------------------------

CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_admission_windows` AS
SELECT
  c.subject_id,
  c.hadm_id,
  a.admittime,
  a.dischtime,
  DATETIME_DIFF(a.dischtime, a.admittime, HOUR) AS hosp_los_hours,
  a.hospital_expire_flag
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_cohort` c
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
  ON a.hadm_id = c.hadm_id;

-- Mortality: LOS >= 48 hours, 48h window.
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_mortality` AS
SELECT
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  hosp_los_hours,
  hospital_expire_flag AS label_mortality,
  admittime AS window_start,
  DATETIME_ADD(admittime, INTERVAL 48 HOUR) AS window_end
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_admission_windows`
WHERE hosp_los_hours >= 48;

-- LOS: LOS >= 48 hours, label > 7 days, 48h window.
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_los` AS
SELECT
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  hosp_los_hours,
  IF(hosp_los_hours > 168, 1, 0) AS label_los_gt_7d,
  admittime AS window_start,
  DATETIME_ADD(admittime, INTERVAL 48 HOUR) AS window_end
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_admission_windows`
WHERE hosp_los_hours >= 48;

-- Readmission: exclude deceased, all events.
CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_next_admit` AS
SELECT
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime
FROM `physionet-data.mimiciv_3_1_hosp.admissions`;

CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_readmission` AS
SELECT
  w.subject_id,
  w.hadm_id,
  w.admittime,
  w.dischtime,
  w.hosp_los_hours,
  IF(n.next_admittime IS NOT NULL
     AND n.next_admittime <= DATETIME_ADD(w.dischtime, INTERVAL 14 DAY), 1, 0) AS label_readmit_14d
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_admission_windows` w
JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_next_admit` n
  ON n.hadm_id = w.hadm_id
WHERE w.hospital_expire_flag = 0;

-- ------------------------
-- Diagnosis phenotyping (requires uploaded CSVs)
-- ------------------------

CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_phenotype_icd10` AS
SELECT DISTINCT
  p.phenotype,
  p.phenotype_type,
  g.icd10cm AS icd10_code
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_phenotype_icd9` p
JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_icd9_to_icd10_gem` g
  ON g.icd9cm = p.icd9_code
WHERE g.no_map = 0;

CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_diagnosis_labels` AS
SELECT DISTINCT
  c.subject_id,
  d.hadm_id,
  COALESCE(p9.phenotype, p10.phenotype) AS phenotype
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_cohort` c
JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
  ON d.hadm_id = c.hadm_id
LEFT JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_phenotype_icd9` p9
  ON d.icd_version = 9
  AND d.icd_code = p9.icd9_code
LEFT JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_phenotype_icd10` p10
  ON d.icd_version = 10
  AND d.icd_code = p10.icd10_code
WHERE COALESCE(p9.phenotype, p10.phenotype) IS NOT NULL;

CREATE OR REPLACE TABLE `mimic-iv-485411.hyperbolic_ehr.llemr_diagnosis` AS
SELECT
  c.subject_id,
  c.hadm_id,
  MAX(CASE WHEN l.phenotype = 'Septicemia (except in labor)' THEN 1 ELSE 0 END) AS label_septicemia,
  MAX(CASE WHEN l.phenotype = 'Diabetes mellitus without complication' THEN 1 ELSE 0 END)
    AS label_diabetes_without_complication,
  MAX(CASE WHEN l.phenotype = 'Diabetes mellitus with complications' THEN 1 ELSE 0 END)
    AS label_diabetes_with_complications,
  MAX(CASE WHEN l.phenotype = 'Disorders of lipid metabolism' THEN 1 ELSE 0 END)
    AS label_lipid_disorders,
  MAX(CASE WHEN l.phenotype = 'Fluid and electrolyte disorders' THEN 1 ELSE 0 END)
    AS label_fluid_electrolyte_disorders,
  MAX(CASE WHEN l.phenotype = 'Essential hypertension' THEN 1 ELSE 0 END)
    AS label_essential_hypertension,
  MAX(CASE WHEN l.phenotype = 'Hypertension with complications and secondary hypertension' THEN 1 ELSE 0 END)
    AS label_hypertension_with_complications,
  MAX(CASE WHEN l.phenotype = 'Acute myocardial infarction' THEN 1 ELSE 0 END)
    AS label_acute_myocardial_infarction,
  MAX(CASE WHEN l.phenotype = 'Coronary atherosclerosis and other heart disease' THEN 1 ELSE 0 END)
    AS label_coronary_atherosclerosis,
  MAX(CASE WHEN l.phenotype = 'Conduction disorders' THEN 1 ELSE 0 END)
    AS label_conduction_disorders,
  MAX(CASE WHEN l.phenotype = 'Cardiac dysrhythmias' THEN 1 ELSE 0 END)
    AS label_cardiac_dysrhythmias,
  MAX(CASE WHEN l.phenotype = 'Congestive heart failure; nonhypertensive' THEN 1 ELSE 0 END)
    AS label_congestive_heart_failure,
  MAX(CASE WHEN l.phenotype = 'Acute cerebrovascular disease' THEN 1 ELSE 0 END)
    AS label_acute_cerebrovascular_disease,
  MAX(CASE WHEN l.phenotype =
    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)' THEN 1 ELSE 0 END)
    AS label_pneumonia,
  MAX(CASE WHEN l.phenotype = 'Chronic obstructive pulmonary disease and bronchiectasis' THEN 1 ELSE 0 END)
    AS label_copd_bronchiectasis,
  MAX(CASE WHEN l.phenotype = 'Pleurisy; pneumothorax; pulmonary collapse' THEN 1 ELSE 0 END)
    AS label_pleurisy_pneumothorax_collapse,
  MAX(CASE WHEN l.phenotype = 'Respiratory failure; insufficiency; arrest (adult)' THEN 1 ELSE 0 END)
    AS label_respiratory_failure,
  MAX(CASE WHEN l.phenotype = 'Other lower respiratory disease' THEN 1 ELSE 0 END)
    AS label_other_lower_respiratory,
  MAX(CASE WHEN l.phenotype = 'Other upper respiratory disease' THEN 1 ELSE 0 END)
    AS label_other_upper_respiratory,
  MAX(CASE WHEN l.phenotype = 'Other liver diseases' THEN 1 ELSE 0 END)
    AS label_other_liver_disease,
  MAX(CASE WHEN l.phenotype = 'Gastrointestinal hemorrhage' THEN 1 ELSE 0 END)
    AS label_gi_hemorrhage,
  MAX(CASE WHEN l.phenotype = 'Acute and unspecified renal failure' THEN 1 ELSE 0 END)
    AS label_acute_renal_failure,
  MAX(CASE WHEN l.phenotype = 'Chronic kidney disease' THEN 1 ELSE 0 END)
    AS label_chronic_kidney_disease,
  MAX(CASE WHEN l.phenotype = 'Complications of surgical procedures or medical care' THEN 1 ELSE 0 END)
    AS label_surgical_medical_complications,
  MAX(CASE WHEN l.phenotype = 'Shock' THEN 1 ELSE 0 END)
    AS label_shock
FROM `mimic-iv-485411.hyperbolic_ehr.llemr_cohort` c
LEFT JOIN `mimic-iv-485411.hyperbolic_ehr.llemr_diagnosis_labels` l
  ON l.hadm_id = c.hadm_id
GROUP BY c.subject_id, c.hadm_id;
