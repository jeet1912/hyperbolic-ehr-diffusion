-- Active: 1753949249665@@127.0.0.1@3306@mimiciii4
DROP TABLE IF EXISTS meddiff_hf_cohort;
DROP TABLE IF EXISTS meddiff_hf_stats;

DROP TEMPORARY TABLE IF EXISTS hf_codes;
CREATE TEMPORARY TABLE hf_codes AS
SELECT '39891' AS code UNION ALL
SELECT '40201' UNION ALL SELECT '40211' UNION ALL SELECT '40291' UNION ALL
SELECT '40401' UNION ALL SELECT '40403' UNION ALL SELECT '40411' UNION ALL SELECT '40413' UNION ALL SELECT '40491' UNION ALL SELECT '40493' UNION ALL
SELECT '4280' UNION ALL SELECT '4281' UNION ALL SELECT '42820' UNION ALL SELECT '42821' UNION ALL SELECT '42822' UNION ALL SELECT '42823' UNION ALL
SELECT '42830' UNION ALL SELECT '42831' UNION ALL SELECT '42832' UNION ALL SELECT '42833' UNION ALL
SELECT '42840' UNION ALL SELECT '42841' UNION ALL SELECT '42842' UNION ALL SELECT '42843' UNION ALL SELECT '4289';
ALTER TABLE hf_codes ADD INDEX idx_hf_codes_code (code);

DROP TEMPORARY TABLE IF EXISTS eth_map;
CREATE TEMPORARY TABLE eth_map AS
SELECT
    subject_id,
    hadm_id,
    admittime,
    CASE
        WHEN ethnicity LIKE 'WHITE%' THEN 'WHITE'
        WHEN ethnicity LIKE 'BLACK%' OR ethnicity LIKE '%AFRICAN%' THEN 'BLACK'
        WHEN ethnicity LIKE 'HISPANIC%' OR ethnicity LIKE '%LATINO%' THEN 'HISPANIC'
        WHEN ethnicity LIKE 'ASIAN%' THEN 'ASIAN'
        WHEN ethnicity IS NULL OR ethnicity = '' THEN 'UNKNOWN'
        ELSE 'OTHER'
    END AS ethnicity_group
FROM ADMISSIONS;
ALTER TABLE eth_map
    ADD INDEX idx_eth_map_hadm (hadm_id),
    ADD INDEX idx_eth_map_subject_time (subject_id, admittime);

DROP TEMPORARY TABLE IF EXISTS hf_diag;
CREATE TEMPORARY TABLE hf_diag AS
SELECT d.subject_id, d.hadm_id, a.admittime, REPLACE(d.icd9_code, '.', '') AS icd9_nodot
FROM DIAGNOSES_ICD d
JOIN ADMISSIONS a ON a.hadm_id = d.hadm_id
WHERE REPLACE(d.icd9_code, '.', '') LIKE '428%'
   OR REPLACE(d.icd9_code, '.', '') IN (SELECT code FROM hf_codes);
ALTER TABLE hf_diag
    ADD INDEX idx_hf_diag_subject_time (subject_id, admittime, hadm_id),
    ADD INDEX idx_hf_diag_hadm (hadm_id);

DROP TEMPORARY TABLE IF EXISTS case_index;
CREATE TEMPORARY TABLE case_index AS
SELECT subject_id, hadm_id AS index_hadm_id, admittime AS index_time
FROM (
    SELECT subject_id, hadm_id, admittime,
           ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS rn
    FROM hf_diag
) t
WHERE rn = 1;
ALTER TABLE case_index
    ADD INDEX idx_case_index_subject_time (subject_id, index_time),
    ADD INDEX idx_case_index_hadm (index_hadm_id);

DROP TEMPORARY TABLE IF EXISTS case_keys;
CREATE TEMPORARY TABLE case_keys AS
SELECT
    ci.subject_id,
    ci.index_time,
    p.gender,
    FLOOR(TIMESTAMPDIFF(YEAR, p.dob, ci.index_time)/5)*5 AS age_bucket,
    em.ethnicity_group
FROM case_index ci
JOIN PATIENTS p ON p.subject_id = ci.subject_id
JOIN eth_map em ON em.hadm_id = ci.index_hadm_id;
ALTER TABLE case_keys
    ADD INDEX idx_case_keys_subject_time (subject_id, index_time),
    ADD INDEX idx_case_keys_match (gender, age_bucket, ethnicity_group);

DROP TEMPORARY TABLE IF EXISTS eligible_cases;
CREATE TEMPORARY TABLE eligible_cases AS
SELECT ck.*
FROM case_keys ck
WHERE EXISTS (
    SELECT 1
    FROM ADMISSIONS a
    WHERE a.subject_id = ck.subject_id
      AND a.admittime >= DATE_SUB(ck.index_time, INTERVAL 180 DAY)
      AND a.admittime <= ck.index_time
);
ALTER TABLE eligible_cases
    ADD INDEX idx_eligible_cases_subject_time (subject_id, index_time),
    ADD INDEX idx_eligible_cases_match (gender, age_bucket, ethnicity_group, subject_id);

DROP TEMPORARY TABLE IF EXISTS control_pool;
CREATE TEMPORARY TABLE control_pool AS
SELECT DISTINCT p.subject_id, p.gender, p.dob
FROM PATIENTS p
WHERE p.subject_id NOT IN (SELECT DISTINCT subject_id FROM hf_diag);
ALTER TABLE control_pool
    ADD INDEX idx_control_pool_gender (gender, subject_id),
    ADD INDEX idx_control_pool_gender_dob (gender, dob, subject_id);

DROP TEMPORARY TABLE IF EXISTS control_latest_adm;
DROP TEMPORARY TABLE IF EXISTS control_admissions;
CREATE TEMPORARY TABLE control_admissions AS
SELECT a.subject_id, a.hadm_id, a.admittime
FROM ADMISSIONS a
JOIN control_pool cp ON cp.subject_id = a.subject_id;
ALTER TABLE control_admissions
    ADD INDEX idx_control_admissions_subject_time (subject_id, admittime, hadm_id);

CREATE TEMPORARY TABLE control_latest_adm AS
SELECT
    ck.subject_id AS case_id,
    ck.index_time,
    cp.subject_id AS ctrl_id,
    cp.gender AS ctrl_gender,
    cp.dob AS ctrl_dob,
    a.hadm_id,
    ROW_NUMBER() OVER (
        PARTITION BY ck.subject_id, cp.subject_id, ck.index_time
        ORDER BY a.admittime DESC, a.hadm_id DESC
    ) AS rn
FROM eligible_cases ck
JOIN control_pool cp
  ON cp.gender = ck.gender
LEFT JOIN control_admissions a
  ON a.subject_id = cp.subject_id
 AND a.admittime <= ck.index_time;
ALTER TABLE control_latest_adm
    ADD INDEX idx_control_latest_key (case_id, index_time, ctrl_id, rn),
    ADD INDEX idx_control_latest_hadm (ctrl_id, hadm_id),
    ADD INDEX idx_control_latest_case_time (case_id, index_time);

DROP TEMPORARY TABLE IF EXISTS matched_controls;
CREATE TEMPORARY TABLE matched_controls AS
SELECT
    ck.subject_id AS case_id,
    ck.index_time,
    cla.ctrl_id,
    ck.gender,
    ck.ethnicity_group,
    ck.age_bucket AS case_age_bucket,
    FLOOR(TIMESTAMPDIFF(YEAR, cla.ctrl_dob, ck.index_time)/5)*5 AS ctrl_age_bucket,
    COALESCE(em.ethnicity_group, 'UNKNOWN') AS ctrl_ethnicity_group
FROM eligible_cases ck
JOIN control_latest_adm cla
  ON cla.case_id = ck.subject_id
 AND cla.index_time = ck.index_time
 AND cla.rn = 1
LEFT JOIN eth_map em
  ON em.hadm_id = cla.hadm_id
WHERE EXISTS (
    SELECT 1
    FROM ADMISSIONS a
    WHERE a.subject_id = cla.ctrl_id
      AND a.admittime >= DATE_SUB(ck.index_time, INTERVAL 180 DAY)
      AND a.admittime <= ck.index_time
);
ALTER TABLE matched_controls
    ADD INDEX idx_matched_controls_case (case_id, index_time, ctrl_id),
    ADD INDEX idx_matched_controls_match (case_age_bucket, ctrl_age_bucket, ethnicity_group, ctrl_ethnicity_group),
    ADD INDEX idx_matched_controls_ctrl (ctrl_id, case_id);

DROP TEMPORARY TABLE IF EXISTS matched_controls_limited;
CREATE TEMPORARY TABLE matched_controls_limited AS
SELECT case_id, index_time, ctrl_id
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY ctrl_id) AS rn_match
    FROM matched_controls
    WHERE ctrl_age_bucket = case_age_bucket
      AND ctrl_ethnicity_group = ethnicity_group
) t
WHERE rn_match <= 3;
ALTER TABLE matched_controls_limited
    ADD INDEX idx_matched_controls_limited_case (case_id, ctrl_id),
    ADD INDEX idx_matched_controls_limited_time (case_id, index_time, ctrl_id);

DROP TEMPORARY TABLE IF EXISTS cohort_subjects;
DROP TEMPORARY TABLE IF EXISTS case_subjects;
CREATE TEMPORARY TABLE case_subjects AS
SELECT
    ck.subject_id,
    ck.index_time,
    ci.index_hadm_id,
    ck.gender,
    ck.age_bucket,
    ck.ethnicity_group
FROM eligible_cases ck
JOIN case_index ci
  ON ci.subject_id = ck.subject_id
 AND ci.index_time = ck.index_time;
ALTER TABLE case_subjects
    ADD INDEX idx_case_subjects_subject (subject_id),
    ADD INDEX idx_case_subjects_index (index_time, index_hadm_id);

CREATE TEMPORARY TABLE cohort_subjects (
    subject_id INT,
    index_time DATETIME,
    index_hadm_id INT,
    label TINYINT,
    matched_case_subject_id INT,
    gender VARCHAR(5),
    age_bucket INT,
    ethnicity_group VARCHAR(20)
);
ALTER TABLE cohort_subjects
    ADD INDEX idx_cohort_subjects_subject (subject_id),
    ADD INDEX idx_cohort_subjects_match (label, matched_case_subject_id, index_time),
    ADD INDEX idx_cohort_subjects_case (matched_case_subject_id, subject_id);

INSERT INTO cohort_subjects
SELECT
    cs.subject_id,
    cs.index_time,
    cs.index_hadm_id,
    1 AS label,
    CAST(NULL AS SIGNED) AS matched_case_subject_id,
    cs.gender,
    cs.age_bucket,
    cs.ethnicity_group
FROM case_subjects cs;

INSERT INTO cohort_subjects
SELECT
    mc.ctrl_id AS subject_id,
    mc.index_time,
    CAST(NULL AS SIGNED) AS index_hadm_id,
    0 AS label,
    mc.case_id AS matched_case_subject_id,
    cs.gender,
    cs.age_bucket,
    cs.ethnicity_group
FROM matched_controls_limited mc
JOIN case_subjects cs ON cs.subject_id = mc.case_id;

DROP TEMPORARY TABLE IF EXISTS cohort_visits;
CREATE TEMPORARY TABLE cohort_visits AS
SELECT
    cs.subject_id,
    cs.label,
    cs.matched_case_subject_id,
    cs.index_time,
    cs.gender,
    cs.age_bucket,
    cs.ethnicity_group,
    a.hadm_id,
    a.admittime AS visit_time
FROM cohort_subjects cs
JOIN ADMISSIONS a ON a.subject_id = cs.subject_id
WHERE a.admittime >= DATE_SUB(cs.index_time, INTERVAL 180 DAY)
  AND a.admittime <= cs.index_time
  AND (
      cs.label = 0
      OR a.hadm_id = cs.index_hadm_id
      OR a.admittime < cs.index_time
  );
ALTER TABLE cohort_visits
    ADD INDEX idx_cohort_visits_subject (subject_id, visit_time),
    ADD INDEX idx_cohort_visits_hadm (hadm_id),
    ADD INDEX idx_cohort_visits_index (index_time, subject_id);

DROP TEMPORARY TABLE IF EXISTS visit_codes;
CREATE TEMPORARY TABLE visit_codes AS
SELECT
    v.subject_id,
    v.hadm_id,
    v.visit_time,
    v.label,
    v.index_time,
    v.matched_case_subject_id,
    v.gender,
    v.age_bucket,
    v.ethnicity_group,
    GROUP_CONCAT(DISTINCT d.icd9_code ORDER BY d.icd9_code SEPARATOR ',') AS icd9_codes
FROM cohort_visits v
JOIN DIAGNOSES_ICD d ON d.hadm_id = v.hadm_id
GROUP BY v.subject_id, v.hadm_id, v.visit_time, v.label, v.index_time, v.matched_case_subject_id, v.gender, v.age_bucket, v.ethnicity_group;
ALTER TABLE visit_codes
    ADD INDEX idx_visit_codes_subject (subject_id, hadm_id),
    ADD INDEX idx_visit_codes_hadm (hadm_id);

CREATE TABLE meddiff_hf_cohort AS
SELECT *
FROM visit_codes;
ALTER TABLE meddiff_hf_cohort
    ADD INDEX idx_meddiff_hf_cohort_subject (subject_id, label),
    ADD INDEX idx_meddiff_hf_cohort_hadm (hadm_id);

CREATE TABLE meddiff_hf_stats AS
SELECT
totals.patients_total,
totals.patients_pos,
totals.patients_neg,
totals.control_instances,
visits.avg_visits_per_patient,
codes_avg.avg_codes_per_visit,
codes_unique.unique_icd9_codes
FROM (
SELECT
COUNT(DISTINCT subject_id) AS patients_total,
COUNT(DISTINCT CASE WHEN label=1 THEN subject_id END) AS patients_pos,
COUNT(DISTINCT CASE WHEN label=0 THEN subject_id END) AS patients_neg,
COUNT(DISTINCT CONCAT(matched_case_subject_id, ':', subject_id)) AS control_instances
FROM meddiff_hf_cohort
WHERE label IN (0, 1)
) totals
CROSS JOIN (
SELECT AVG(visit_count) AS avg_visits_per_patient
FROM (
SELECT instance_id, COUNT(*) AS visit_count
FROM (
SELECT
CASE
WHEN label = 1 THEN CONCAT('C:', subject_id, ':', index_time)
ELSE CONCAT('N:', matched_case_subject_id, ':', subject_id, ':', index_time)
END AS instance_id
FROM meddiff_hf_cohort
) inst
GROUP BY instance_id
) vc
) visits
CROSS JOIN (
SELECT AVG(code_count) AS avg_codes_per_visit
FROM (
SELECT v.hadm_id, COUNT(DISTINCT d.icd9_code) AS code_count
FROM meddiff_hf_cohort v
JOIN DIAGNOSES_ICD d ON d.hadm_id = v.hadm_id
GROUP BY v.hadm_id
) vc
) codes_avg
CROSS JOIN (
SELECT COUNT(DISTINCT d.icd9_code) AS unique_icd9_codes
FROM meddiff_hf_cohort v
JOIN DIAGNOSES_ICD d ON d.hadm_id = v.hadm_id
) codes_unique;

SELECT * FROM meddiff_hf_stats;
