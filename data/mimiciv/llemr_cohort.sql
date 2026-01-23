USE mimic4;

-- Require discharge summaries from MIMIC-IV-Note.
SET @discharge_exists := (
  SELECT COUNT(*)
  FROM information_schema.tables
  WHERE table_schema = DATABASE()
    AND table_name = 'discharge'
);

-- Fail fast if discharge summaries are unavailable.
SET @require_discharge := CASE WHEN @discharge_exists = 0 THEN (1/0) ELSE 1 END;

-- Require event_selected table for length cutoff.
SET @event_selected_exists := (
  SELECT COUNT(*)
  FROM information_schema.tables
  WHERE table_schema = DATABASE()
    AND table_name = 'event_selected'
);
SET @require_event_selected := CASE WHEN @event_selected_exists = 0 THEN (1/0) ELSE 1 END;

DROP TEMPORARY TABLE IF EXISTS discharge_summary_hadm;
CREATE TEMPORARY TABLE discharge_summary_hadm AS
SELECT DISTINCT hadm_id
FROM discharge
WHERE hadm_id IS NOT NULL
;

-- Per-admission selected event length.
DROP TEMPORARY TABLE IF EXISTS event_selected_lengths;
CREATE TEMPORARY TABLE event_selected_lengths AS
SELECT
  hadm_id,
  COUNT(*) AS len_selected
FROM event_selected
GROUP BY hadm_id;

-- ICU stay counts and LOS checks.
DROP TEMPORARY TABLE IF EXISTS icu_stay_counts;
CREATE TEMPORARY TABLE icu_stay_counts AS
SELECT
  hadm_id,
  COUNT(*) AS icu_stay_count,
  SUM(CASE WHEN outtime < intime OR los < 0 THEN 1 ELSE 0 END) AS bad_icu_los
FROM icustays
GROUP BY hadm_id;

-- -------------
-- Cohort table
-- -------------

DROP TABLE IF EXISTS llemr_cohort;
CREATE TABLE llemr_cohort AS
SELECT
  a.subject_id,
  a.hadm_id,
  s.icu_stay_count,
  COALESCE(es.len_selected, 0) AS len_selected
FROM admissions a
JOIN icu_stay_counts s
  ON s.hadm_id = a.hadm_id
JOIN discharge_summary_hadm ds
  ON ds.hadm_id = a.hadm_id
LEFT JOIN event_selected_lengths es
  ON es.hadm_id = a.hadm_id
WHERE a.dischtime >= a.admittime
  AND s.icu_stay_count = 1
  AND s.bad_icu_los = 0
  AND COALESCE(es.len_selected, 0) <= 1256.65;

-- ------------------------
-- Cohort-filtered views
-- ------------------------

DROP VIEW IF EXISTS llemr_patients;
CREATE VIEW llemr_patients AS
SELECT p.*
FROM patients p
JOIN llemr_cohort c
  ON c.subject_id = p.subject_id;

DROP VIEW IF EXISTS llemr_admissions;
CREATE VIEW llemr_admissions AS
SELECT a.*
FROM admissions a
JOIN llemr_cohort c
  ON c.hadm_id = a.hadm_id;

DROP VIEW IF EXISTS llemr_diagnoses_icd;
CREATE VIEW llemr_diagnoses_icd AS
SELECT d.*
FROM diagnoses_icd d
JOIN llemr_cohort c
  ON c.hadm_id = d.hadm_id;

DROP VIEW IF EXISTS llemr_labevents;
CREATE VIEW llemr_labevents AS
SELECT l.*
FROM labevents l
JOIN llemr_cohort c
  ON c.hadm_id = l.hadm_id;

DROP VIEW IF EXISTS llemr_microbiologyevents;
CREATE VIEW llemr_microbiologyevents AS
SELECT m.*
FROM microbiologyevents m
JOIN llemr_cohort c
  ON c.hadm_id = m.hadm_id;

DROP VIEW IF EXISTS llemr_prescriptions;
CREATE VIEW llemr_prescriptions AS
SELECT p.*
FROM prescriptions p
JOIN llemr_cohort c
  ON c.hadm_id = p.hadm_id;

DROP VIEW IF EXISTS llemr_transfers;
CREATE VIEW llemr_transfers AS
SELECT t.*
FROM transfers t
JOIN llemr_cohort c
  ON c.hadm_id = t.hadm_id;

DROP VIEW IF EXISTS llemr_icustays;
CREATE VIEW llemr_icustays AS
SELECT i.*
FROM icustays i
JOIN llemr_cohort c
  ON c.hadm_id = i.hadm_id;

DROP VIEW IF EXISTS llemr_inputevents;
CREATE VIEW llemr_inputevents AS
SELECT ie.*
FROM inputevents ie
JOIN llemr_icustays i
  ON i.stay_id = ie.stay_id;

DROP VIEW IF EXISTS llemr_outputevents;
CREATE VIEW llemr_outputevents AS
SELECT oe.*
FROM outputevents oe
JOIN llemr_icustays i
  ON i.stay_id = oe.stay_id;

DROP VIEW IF EXISTS llemr_procedureevents;
CREATE VIEW llemr_procedureevents AS
SELECT pe.*
FROM procedureevents pe
JOIN llemr_icustays i
  ON i.stay_id = pe.stay_id;

-- ------------------------
-- Task-specific cohorts
-- ------------------------
DROP TABLE IF EXISTS llemr_admission_windows;
CREATE TABLE llemr_admission_windows AS
SELECT
  c.subject_id,
  c.hadm_id,
  a.admittime,
  a.dischtime,
  TIMESTAMPDIFF(HOUR, a.admittime, a.dischtime) AS hosp_los_hours,
  a.hospital_expire_flag
FROM llemr_cohort c
JOIN admissions a
  ON a.hadm_id = c.hadm_id;

-- Mortality prediction: first 48 hours, LOS >= 48 hours.
DROP TABLE IF EXISTS llemr_mortality;
CREATE TABLE llemr_mortality AS
SELECT
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  hosp_los_hours,
  label_mortality,
  window_start,
  window_end,
  CASE
    WHEN rn <= cnt * 0.8 THEN 'train'
    WHEN rn <= cnt * 0.9 THEN 'val'
    ELSE 'test'
  END AS split
FROM (
  SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    hosp_los_hours,
    hospital_expire_flag AS label_mortality,
    admittime AS window_start,
    admittime + INTERVAL 48 HOUR AS window_end,
    ROW_NUMBER() OVER (PARTITION BY hospital_expire_flag ORDER BY RAND(42)) AS rn,
    COUNT(*) OVER (PARTITION BY hospital_expire_flag) AS cnt
  FROM llemr_admission_windows
  WHERE hosp_los_hours >= 48
) base;

DROP TABLE IF EXISTS llemr_mortality_train;
CREATE TABLE llemr_mortality_train AS
SELECT *
FROM llemr_mortality
WHERE split = 'train';

DROP TABLE IF EXISTS llemr_mortality_val;
CREATE TABLE llemr_mortality_val AS
SELECT *
FROM llemr_mortality
WHERE split = 'val';

DROP TABLE IF EXISTS llemr_mortality_test;
CREATE TABLE llemr_mortality_test AS
SELECT *
FROM llemr_mortality
WHERE split = 'test';

-- Length-of-stay prediction: first 48 hours, LOS >= 48 hours.
DROP TABLE IF EXISTS llemr_los;
CREATE TABLE llemr_los AS
SELECT
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  hosp_los_hours,
  label_los_gt_7d,
  window_start,
  window_end,
  CASE
    WHEN rn <= cnt * 0.8 THEN 'train'
    WHEN rn <= cnt * 0.9 THEN 'val'
    ELSE 'test'
  END AS split
FROM (
  SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    hosp_los_hours,
    CASE WHEN hosp_los_hours > 168 THEN 1 ELSE 0 END AS label_los_gt_7d,
    admittime AS window_start,
    admittime + INTERVAL 48 HOUR AS window_end,
    ROW_NUMBER() OVER (
      PARTITION BY CASE WHEN hosp_los_hours > 168 THEN 1 ELSE 0 END
      ORDER BY RAND(42)
    ) AS rn,
    COUNT(*) OVER (
      PARTITION BY CASE WHEN hosp_los_hours > 168 THEN 1 ELSE 0 END
    ) AS cnt
  FROM llemr_admission_windows
  WHERE hosp_los_hours >= 48
) base;

DROP TABLE IF EXISTS llemr_los_train;
CREATE TABLE llemr_los_train AS
SELECT *
FROM llemr_los
WHERE split = 'train';

DROP TABLE IF EXISTS llemr_los_val;
CREATE TABLE llemr_los_val AS
SELECT *
FROM llemr_los
WHERE split = 'val';

DROP TABLE IF EXISTS llemr_los_test;
CREATE TABLE llemr_los_test AS
SELECT *
FROM llemr_los
WHERE split = 'test';

-- Readmission prediction: all events, exclude deceased admissions.
DROP TEMPORARY TABLE IF EXISTS llemr_next_admit;
CREATE TEMPORARY TABLE llemr_next_admit AS
SELECT
  subject_id,
  hadm_id,
  admittime,
  dischtime,
  LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admittime
FROM admissions;

DROP TABLE IF EXISTS llemr_readmission;
CREATE TABLE llemr_readmission AS
SELECT
  w.subject_id,
  w.hadm_id,
  w.admittime,
  w.dischtime,
  w.hosp_los_hours,
  label_readmit_14d,
  CASE
    WHEN rn <= cnt * 0.8 THEN 'train'
    WHEN rn <= cnt * 0.9 THEN 'val'
    ELSE 'test'
  END AS split
FROM (
  SELECT
    w.subject_id,
    w.hadm_id,
    w.admittime,
    w.dischtime,
    w.hosp_los_hours,
    CASE
      WHEN n.next_admittime IS NOT NULL
        AND n.next_admittime <= w.dischtime + INTERVAL 14 DAY
        THEN 1 ELSE 0 END AS label_readmit_14d,
    ROW_NUMBER() OVER (
      PARTITION BY CASE
        WHEN n.next_admittime IS NOT NULL
          AND n.next_admittime <= w.dischtime + INTERVAL 14 DAY
          THEN 1 ELSE 0 END
      ORDER BY RAND(42)
    ) AS rn,
    COUNT(*) OVER (
      PARTITION BY CASE
        WHEN n.next_admittime IS NOT NULL
          AND n.next_admittime <= w.dischtime + INTERVAL 14 DAY
          THEN 1 ELSE 0 END
    ) AS cnt
  FROM llemr_admission_windows w
  JOIN llemr_next_admit n
    ON n.hadm_id = w.hadm_id
  WHERE w.hospital_expire_flag = 0
) base;

DROP TABLE IF EXISTS llemr_readmission_train;
CREATE TABLE llemr_readmission_train AS
SELECT *
FROM llemr_readmission
WHERE split = 'train';

DROP TABLE IF EXISTS llemr_readmission_val;
CREATE TABLE llemr_readmission_val AS
SELECT *
FROM llemr_readmission
WHERE split = 'val';

DROP TABLE IF EXISTS llemr_readmission_test;
CREATE TABLE llemr_readmission_test AS
SELECT *
FROM llemr_readmission
WHERE split = 'test';

-- ------------------------
-- Task-specific event views
-- ------------------------

-- Mortality / LOS: restrict to first 48 hours.
DROP VIEW IF EXISTS llemr_mortality_patients;
CREATE VIEW llemr_mortality_patients AS
SELECT p.*
FROM patients p
JOIN llemr_mortality m
  ON m.subject_id = p.subject_id;

DROP VIEW IF EXISTS llemr_mortality_admissions;
CREATE VIEW llemr_mortality_admissions AS
SELECT a.*
FROM admissions a
JOIN llemr_mortality m
  ON m.hadm_id = a.hadm_id;

DROP VIEW IF EXISTS llemr_mortality_diagnoses_icd;
CREATE VIEW llemr_mortality_diagnoses_icd AS
SELECT d.*
FROM diagnoses_icd d
JOIN llemr_mortality m
  ON m.hadm_id = d.hadm_id;

DROP VIEW IF EXISTS llemr_mortality_labevents;
CREATE VIEW llemr_mortality_labevents AS
SELECT l.*
FROM labevents l
JOIN llemr_mortality m
  ON m.hadm_id = l.hadm_id
WHERE l.charttime >= m.window_start
  AND l.charttime < m.window_end;

DROP VIEW IF EXISTS llemr_mortality_microbiologyevents;
CREATE VIEW llemr_mortality_microbiologyevents AS
SELECT mbe.*
FROM microbiologyevents mbe
JOIN llemr_mortality m
  ON m.hadm_id = mbe.hadm_id
WHERE COALESCE(mbe.charttime, CAST(mbe.chartdate AS DATETIME)) >= m.window_start
  AND COALESCE(mbe.charttime, CAST(mbe.chartdate AS DATETIME)) < m.window_end;

DROP VIEW IF EXISTS llemr_mortality_prescriptions;
CREATE VIEW llemr_mortality_prescriptions AS
SELECT pr.*
FROM prescriptions pr
JOIN llemr_mortality m
  ON m.hadm_id = pr.hadm_id
WHERE COALESCE(pr.starttime, pr.stoptime) >= m.window_start
  AND COALESCE(pr.starttime, pr.stoptime) < m.window_end;

DROP VIEW IF EXISTS llemr_mortality_transfers;
CREATE VIEW llemr_mortality_transfers AS
SELECT t.*
FROM transfers t
JOIN llemr_mortality m
  ON m.hadm_id = t.hadm_id
WHERE t.intime >= m.window_start
  AND t.intime < m.window_end;

DROP VIEW IF EXISTS llemr_mortality_icustays;
CREATE VIEW llemr_mortality_icustays AS
SELECT i.*
FROM icustays i
JOIN llemr_mortality m
  ON m.hadm_id = i.hadm_id
WHERE i.intime >= m.window_start
  AND i.intime < m.window_end;

DROP VIEW IF EXISTS llemr_mortality_inputevents;
CREATE VIEW llemr_mortality_inputevents AS
SELECT ie.*
FROM inputevents ie
JOIN llemr_mortality_icustays i
  ON i.stay_id = ie.stay_id
JOIN llemr_mortality m
  ON m.hadm_id = i.hadm_id
WHERE ie.starttime >= m.window_start
  AND ie.starttime < m.window_end;

DROP VIEW IF EXISTS llemr_mortality_outputevents;
CREATE VIEW llemr_mortality_outputevents AS
SELECT oe.*
FROM outputevents oe
JOIN llemr_mortality_icustays i
  ON i.stay_id = oe.stay_id
JOIN llemr_mortality m
  ON m.hadm_id = i.hadm_id
WHERE oe.charttime >= m.window_start
  AND oe.charttime < m.window_end;

DROP VIEW IF EXISTS llemr_mortality_procedureevents;
CREATE VIEW llemr_mortality_procedureevents AS
SELECT pe.*
FROM procedureevents pe
JOIN llemr_mortality_icustays i
  ON i.stay_id = pe.stay_id
JOIN llemr_mortality m
  ON m.hadm_id = i.hadm_id
WHERE pe.starttime >= m.window_start
  AND pe.starttime < m.window_end;

-- LOS prediction views mirror mortality windows.
DROP VIEW IF EXISTS llemr_los_patients;
CREATE VIEW llemr_los_patients AS
SELECT p.*
FROM patients p
JOIN llemr_los l
  ON l.subject_id = p.subject_id;

DROP VIEW IF EXISTS llemr_los_admissions;
CREATE VIEW llemr_los_admissions AS
SELECT a.*
FROM admissions a
JOIN llemr_los l
  ON l.hadm_id = a.hadm_id;

DROP VIEW IF EXISTS llemr_los_diagnoses_icd;
CREATE VIEW llemr_los_diagnoses_icd AS
SELECT d.*
FROM diagnoses_icd d
JOIN llemr_los l
  ON l.hadm_id = d.hadm_id;

DROP VIEW IF EXISTS llemr_los_labevents;
CREATE VIEW llemr_los_labevents AS
SELECT l.*
FROM labevents l
JOIN llemr_los c
  ON c.hadm_id = l.hadm_id
WHERE l.charttime >= c.window_start
  AND l.charttime < c.window_end;

DROP VIEW IF EXISTS llemr_los_microbiologyevents;
CREATE VIEW llemr_los_microbiologyevents AS
SELECT mbe.*
FROM microbiologyevents mbe
JOIN llemr_los c
  ON c.hadm_id = mbe.hadm_id
WHERE COALESCE(mbe.charttime, CAST(mbe.chartdate AS DATETIME)) >= c.window_start
  AND COALESCE(mbe.charttime, CAST(mbe.chartdate AS DATETIME)) < c.window_end;

DROP VIEW IF EXISTS llemr_los_prescriptions;
CREATE VIEW llemr_los_prescriptions AS
SELECT pr.*
FROM prescriptions pr
JOIN llemr_los c
  ON c.hadm_id = pr.hadm_id
WHERE COALESCE(pr.starttime, pr.stoptime) >= c.window_start
  AND COALESCE(pr.starttime, pr.stoptime) < c.window_end;

DROP VIEW IF EXISTS llemr_los_transfers;
CREATE VIEW llemr_los_transfers AS
SELECT t.*
FROM transfers t
JOIN llemr_los c
  ON c.hadm_id = t.hadm_id
WHERE t.intime >= c.window_start
  AND t.intime < c.window_end;

DROP VIEW IF EXISTS llemr_los_icustays;
CREATE VIEW llemr_los_icustays AS
SELECT i.*
FROM icustays i
JOIN llemr_los c
  ON c.hadm_id = i.hadm_id
WHERE i.intime >= c.window_start
  AND i.intime < c.window_end;

DROP VIEW IF EXISTS llemr_los_inputevents;
CREATE VIEW llemr_los_inputevents AS
SELECT ie.*
FROM inputevents ie
JOIN llemr_los_icustays i
  ON i.stay_id = ie.stay_id
JOIN llemr_los c
  ON c.hadm_id = i.hadm_id
WHERE ie.starttime >= c.window_start
  AND ie.starttime < c.window_end;

DROP VIEW IF EXISTS llemr_los_outputevents;
CREATE VIEW llemr_los_outputevents AS
SELECT oe.*
FROM outputevents oe
JOIN llemr_los_icustays i
  ON i.stay_id = oe.stay_id
JOIN llemr_los c
  ON c.hadm_id = i.hadm_id
WHERE oe.charttime >= c.window_start
  AND oe.charttime < c.window_end;

DROP VIEW IF EXISTS llemr_los_procedureevents;
CREATE VIEW llemr_los_procedureevents AS
SELECT pe.*
FROM procedureevents pe
JOIN llemr_los_icustays i
  ON i.stay_id = pe.stay_id
JOIN llemr_los c
  ON c.hadm_id = i.hadm_id
WHERE pe.starttime >= c.window_start
  AND pe.starttime < c.window_end;

-- Readmission: use all events in the admission (no 48h limit).
DROP VIEW IF EXISTS llemr_readmission_patients;
CREATE VIEW llemr_readmission_patients AS
SELECT p.*
FROM patients p
JOIN llemr_readmission r
  ON r.subject_id = p.subject_id;

DROP VIEW IF EXISTS llemr_readmission_admissions;
CREATE VIEW llemr_readmission_admissions AS
SELECT a.*
FROM admissions a
JOIN llemr_readmission r
  ON r.hadm_id = a.hadm_id;

DROP VIEW IF EXISTS llemr_readmission_diagnoses_icd;
CREATE VIEW llemr_readmission_diagnoses_icd AS
SELECT d.*
FROM diagnoses_icd d
JOIN llemr_readmission r
  ON r.hadm_id = d.hadm_id;

DROP VIEW IF EXISTS llemr_readmission_labevents;
CREATE VIEW llemr_readmission_labevents AS
SELECT l.*
FROM labevents l
JOIN llemr_readmission r
  ON r.hadm_id = l.hadm_id;

DROP VIEW IF EXISTS llemr_readmission_microbiologyevents;
CREATE VIEW llemr_readmission_microbiologyevents AS
SELECT mbe.*
FROM microbiologyevents mbe
JOIN llemr_readmission r
  ON r.hadm_id = mbe.hadm_id;

DROP VIEW IF EXISTS llemr_readmission_prescriptions;
CREATE VIEW llemr_readmission_prescriptions AS
SELECT pr.*
FROM prescriptions pr
JOIN llemr_readmission r
  ON r.hadm_id = pr.hadm_id;

DROP VIEW IF EXISTS llemr_readmission_transfers;
CREATE VIEW llemr_readmission_transfers AS
SELECT t.*
FROM transfers t
JOIN llemr_readmission r
  ON r.hadm_id = t.hadm_id;

DROP VIEW IF EXISTS llemr_readmission_icustays;
CREATE VIEW llemr_readmission_icustays AS
SELECT i.*
FROM icustays i
JOIN llemr_readmission r
  ON r.hadm_id = i.hadm_id;

DROP VIEW IF EXISTS llemr_readmission_inputevents;
CREATE VIEW llemr_readmission_inputevents AS
SELECT ie.*
FROM inputevents ie
JOIN llemr_readmission_icustays i
  ON i.stay_id = ie.stay_id;

DROP VIEW IF EXISTS llemr_readmission_outputevents;
CREATE VIEW llemr_readmission_outputevents AS
SELECT oe.*
FROM outputevents oe
JOIN llemr_readmission_icustays i
  ON i.stay_id = oe.stay_id;

DROP VIEW IF EXISTS llemr_readmission_procedureevents;
CREATE VIEW llemr_readmission_procedureevents AS
SELECT pe.*
FROM procedureevents pe
JOIN llemr_readmission_icustays i
  ON i.stay_id = pe.stay_id;

-- ------------------------
-- Final patient counts
-- ------------------------

SELECT COUNT(DISTINCT subject_id) AS icu_patients
FROM icustays;

SELECT COUNT(DISTINCT hadm_id) AS icu_admissions
FROM icustays;

SELECT COUNT(DISTINCT subject_id) AS cohort_patients
FROM llemr_cohort;

SELECT COUNT(*) AS cohort_admissions
FROM llemr_cohort;

SELECT COUNT(DISTINCT subject_id) AS mortality_patients
FROM llemr_mortality;

SELECT COUNT(*) AS mortality_admissions
FROM llemr_mortality;

SELECT label_mortality, COUNT(*) AS admissions
FROM llemr_mortality
GROUP BY label_mortality;

SELECT split, COUNT(*) AS admissions
FROM llemr_mortality
GROUP BY split;

SELECT COUNT(*) AS mortality_train_admissions
FROM llemr_mortality_train;

SELECT COUNT(*) AS mortality_val_admissions
FROM llemr_mortality_val;

SELECT COUNT(*) AS mortality_test_admissions
FROM llemr_mortality_test;

SELECT COUNT(DISTINCT subject_id) AS los_patients
FROM llemr_los;

SELECT COUNT(*) AS los_admissions
FROM llemr_los;

SELECT label_los_gt_7d, COUNT(*) AS admissions
FROM llemr_los
GROUP BY label_los_gt_7d;

SELECT split, COUNT(*) AS admissions
FROM llemr_los
GROUP BY split;

SELECT COUNT(*) AS los_train_admissions
FROM llemr_los_train;

SELECT COUNT(*) AS los_val_admissions
FROM llemr_los_val;

SELECT COUNT(*) AS los_test_admissions
FROM llemr_los_test;

SELECT COUNT(DISTINCT subject_id) AS readmission_patients
FROM llemr_readmission;

SELECT COUNT(*) AS readmission_admissions
FROM llemr_readmission;

SELECT label_readmit_14d, COUNT(*) AS admissions
FROM llemr_readmission
GROUP BY label_readmit_14d;

SELECT split, COUNT(*) AS admissions
FROM llemr_readmission
GROUP BY split;

SELECT COUNT(*) AS readmission_train_admissions
FROM llemr_readmission_train;

SELECT COUNT(*) AS readmission_val_admissions
FROM llemr_readmission_val;

SELECT COUNT(*) AS readmission_test_admissions
FROM llemr_readmission_test;

-- 10-row previews for task cohorts (admission-level tables).
SELECT *
FROM llemr_mortality
ORDER BY subject_id, hadm_id
LIMIT 10;

SELECT *
FROM llemr_los
ORDER BY subject_id, hadm_id
LIMIT 10;

SELECT *
FROM llemr_readmission
ORDER BY subject_id, hadm_id
LIMIT 10;

-- 10 distinct patients per task (subject_id only).
SELECT DISTINCT subject_id
FROM llemr_mortality
ORDER BY subject_id
LIMIT 10;

SELECT DISTINCT subject_id
FROM llemr_los
ORDER BY subject_id
LIMIT 10;

SELECT DISTINCT subject_id
FROM llemr_readmission
ORDER BY subject_id
LIMIT 10;
