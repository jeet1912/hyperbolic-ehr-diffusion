/* ============================================================
   Heart Failure (HF) Prediction Cohort (Leakage-Safe)
   - MIMIC-III style schema: PATIENTS, ADMISSIONS, DIAGNOSES_ICD, PROCEDURES_ICD
   - Goal: predict FIRST-EVER HF (ICD9 428%) from PRE-INDEX history only
   - Key anti-leak fixes:
       (1) Positives: first-ever HF admission only (no prior HF)
       (2) No post-outcome filters (no HOSPITAL_EXPIRE_FLAG filtering)
       (3) Negatives: indexed at a matched calendar time to positives (±30d),
           and must have >= lookback history and >=2 visits in lookback
       (4) Input codes exclude HF codes (428%) + exclude E-codes
       (5) Vocabulary built from PRE-INDEX window only
       (6) Matching uses WINDOW visit counts (not lifetime TOTAL_VISITS)
       (7) Avoid massive re-use of negatives (best-effort via neg_pick table)
   ============================================================ */

/* -------------------------
   Params (edit if needed)
-------------------------- */
SET @LOOKBACK_DAYS := 365;          -- 1 year lookback
SET @MATCH_CAL_DAYS := 30;          -- calendar-time matching window (+/- 30d)
SET @CONTROLS_PER_CASE := 2;        -- number of controls per positive

/* -------------------------
   Clean up old tmp tables
-------------------------- */
DROP TABLE IF EXISTS tmp_all_admissions;
DROP TABLE IF EXISTS tmp_patient_demo;
DROP TABLE IF EXISTS tmp_admission_race;
DROP TABLE IF EXISTS tmp_hf_index;
DROP TABLE IF EXISTS tmp_pos_pool;
DROP TABLE IF EXISTS tmp_neg_pool_base;
DROP TABLE IF EXISTS tmp_pos_with_window_counts;
DROP TABLE IF EXISTS tmp_neg_with_window_counts;
DROP TABLE IF EXISTS tmp_ranked_matches;
DROP TABLE IF EXISTS tmp_matched_cohort;
DROP TABLE IF EXISTS tmp_windowed_data;
DROP TABLE IF EXISTS tmp_windowed_diag;
DROP TABLE IF EXISTS tmp_windowed_proc;
DROP TABLE IF EXISTS tmp_windowed_codes;
DROP TABLE IF EXISTS tmp_global_vocabulary;

/* ============================================================
   1) Admissions (exclude newborn only)
   ============================================================ */
CREATE TABLE tmp_all_admissions AS
SELECT *
FROM ADMISSIONS
WHERE ADMISSION_TYPE <> 'NEWBORN';

/* ============================================================
   2) Demographics (race derived later from PRE-INDEX admissions)
   ============================================================ */
CREATE TABLE tmp_patient_demo AS
SELECT
    p.SUBJECT_ID,
    p.GENDER,
    p.DOB,
    /* Age at FIRST admission (approx; used only for matching) */
    TIMESTAMPDIFF(YEAR, p.DOB, MIN(a.ADMITTIME)) AS AGE_FIRST_ADMIT
FROM PATIENTS p
JOIN tmp_all_admissions a
  ON p.SUBJECT_ID = a.SUBJECT_ID
GROUP BY p.SUBJECT_ID, p.GENDER, p.DOB;

/* ============================================================
   2b) Admission-level race bucket (PRE-INDEX only)
       - derived from the most recent admission on or before that admit time
       - avoids using MAX(ETHNICITY) across all admissions
   ============================================================ */
CREATE TABLE tmp_admission_race AS
SELECT
    a.HADM_ID,
    a.SUBJECT_ID,
    CASE
        WHEN a_last.ETHNICITY LIKE '%WHITE%' THEN 'WHITE'
        WHEN a_last.ETHNICITY LIKE '%BLACK%' OR a_last.ETHNICITY LIKE '%AFRICAN%' THEN 'BLACK'
        WHEN a_last.ETHNICITY LIKE '%HISPANIC%' OR a_last.ETHNICITY LIKE '%LATINO%' THEN 'HISPANIC'
        WHEN a_last.ETHNICITY LIKE '%ASIAN%' THEN 'ASIAN'
        ELSE NULL
    END AS RACE_BUCKET
FROM tmp_all_admissions a
JOIN tmp_all_admissions a_last
  ON a_last.SUBJECT_ID = a.SUBJECT_ID
 AND a_last.ADMITTIME <= a.ADMITTIME
WHERE a_last.ADMITTIME = (
    SELECT MAX(a2.ADMITTIME)
    FROM tmp_all_admissions a2
    WHERE a2.SUBJECT_ID = a.SUBJECT_ID
      AND a2.ADMITTIME <= a.ADMITTIME
);

/* ============================================================
   3) First-ever HF index admission per patient (ICD9 428%)
      - index_date = admit time of first HF-coded admission
      - enforce: no earlier HF diagnosis before that admission
   ============================================================ */
CREATE TABLE tmp_hf_index AS
SELECT
    d.SUBJECT_ID,
    a.HADM_ID AS INDEX_HADM_ID,
    a.ADMITTIME AS INDEX_DATE
FROM DIAGNOSES_ICD d
JOIN tmp_all_admissions a
  ON d.HADM_ID = a.HADM_ID
WHERE d.ICD9_CODE LIKE '428%'
  AND a.ADMISSION_TYPE <> 'ELECTIVE'
  AND NOT EXISTS (
      SELECT 1
      FROM DIAGNOSES_ICD d2
      JOIN tmp_all_admissions a2
        ON d2.HADM_ID = a2.HADM_ID
      WHERE d2.SUBJECT_ID = d.SUBJECT_ID
        AND d2.ICD9_CODE LIKE '428%'
        AND a2.ADMITTIME < a.ADMITTIME
  )
;

/* One row per SUBJECT_ID (in case of multiple 428 codes on same HADM) */
DROP TABLE IF EXISTS tmp_pos_pool;
CREATE TABLE tmp_pos_pool AS
SELECT
    hi.SUBJECT_ID,
    1 AS LABEL,
    MIN(hi.INDEX_DATE) AS INDEX_DATE,
    SUBSTRING_INDEX(
      GROUP_CONCAT(hi.INDEX_HADM_ID ORDER BY hi.INDEX_DATE ASC),
      ',', 1
    ) AS INDEX_HADM_ID
FROM tmp_hf_index hi
GROUP BY hi.SUBJECT_ID;

/* ============================================================
   4) Build NEGATIVE base pool:
      - must have NO HF ever (ICD9 428%)
      - must have valid demographic buckets
      - age >= 18 (at first admit) to roughly align adult cohort
   ============================================================ */
CREATE TABLE tmp_neg_pool_base AS
SELECT
    pd.SUBJECT_ID,
    0 AS LABEL,
    pd.GENDER,
    pd.AGE_FIRST_ADMIT
FROM tmp_patient_demo pd
WHERE pd.AGE_FIRST_ADMIT >= 18
  AND NOT EXISTS (
      SELECT 1
      FROM DIAGNOSES_ICD d
      WHERE d.SUBJECT_ID = pd.SUBJECT_ID
        AND d.ICD9_CODE LIKE '428%'
  );

/* ============================================================
   5) Define PRE-INDEX windows and compute WINDOWED VISIT COUNTS
      - For positives: window = [index_date - lookback, index_date)
      - For negatives: index date will be assigned during matching,
        so compute window counts after assigning index per match.
      - Also: enforce >=2 visits in LOOKBACK window for BOTH classes.
   ============================================================ */

/* Positives: window visits count */
CREATE TABLE tmp_pos_with_window_counts AS
SELECT
    p.SUBJECT_ID,
    p.LABEL,
    p.INDEX_DATE,
    p.INDEX_HADM_ID,
    pd.GENDER,
    ar_pos.RACE_BUCKET,
    /* approximate age at index */
    TIMESTAMPDIFF(YEAR, pd.DOB, p.INDEX_DATE) AS AGE_AT_INDEX,
    COUNT(DISTINCT a.HADM_ID) AS WINDOW_VISITS
FROM tmp_pos_pool p
JOIN tmp_patient_demo pd
  ON p.SUBJECT_ID = pd.SUBJECT_ID
JOIN tmp_admission_race ar_pos
  ON ar_pos.HADM_ID = p.INDEX_HADM_ID
JOIN tmp_all_admissions a
  ON a.SUBJECT_ID = p.SUBJECT_ID
 AND a.ADMITTIME >= DATE_SUB(p.INDEX_DATE, INTERVAL @LOOKBACK_DAYS DAY)
 AND a.ADMITTIME < p.INDEX_DATE
 AND a.HADM_ID <> p.INDEX_HADM_ID
WHERE TIMESTAMPDIFF(YEAR, pd.DOB, p.INDEX_DATE) >= 18
  AND ar_pos.RACE_BUCKET IS NOT NULL
GROUP BY
    p.SUBJECT_ID, p.LABEL, p.INDEX_DATE, p.INDEX_HADM_ID,
    pd.GENDER, ar_pos.RACE_BUCKET, AGE_AT_INDEX
HAVING WINDOW_VISITS >= 2;

/* ============================================================
   6) Match negatives to positives by:
      - same gender, race bucket, exact age-at-index (years)
      - negative "index date" chosen as a real admission admit time
        within +/- MATCH_CAL_DAYS of the positive index date
      - negative must have >=2 visits in its own lookback window
      - negative window_visits roughly similar to positive (+0..+4)
      - sample CONTROLS_PER_CASE per positive
   ============================================================ */

/* Ranked candidate controls per positive */
CREATE TABLE tmp_ranked_matches AS
SELECT
    pos.SUBJECT_ID   AS POS_SUBJECT_ID,
    pos.INDEX_DATE   AS POS_INDEX_DATE,
    pos.WINDOW_VISITS AS POS_WINDOW_VISITS,
    neg.SUBJECT_ID   AS NEG_SUBJECT_ID,
    /* choose a negative index admission near pos index date */
    a_neg.HADM_ID    AS NEG_INDEX_HADM_ID,
    a_neg.ADMITTIME  AS NEG_INDEX_DATE,
    /* negative window visits in its own lookback */
    (
      SELECT COUNT(DISTINCT a2.HADM_ID)
      FROM tmp_all_admissions a2
      WHERE a2.SUBJECT_ID = neg.SUBJECT_ID
        AND a2.ADMITTIME >= DATE_SUB(a_neg.ADMITTIME, INTERVAL @LOOKBACK_DAYS DAY)
        AND a2.ADMITTIME < a_neg.ADMITTIME
    ) AS NEG_WINDOW_VISITS,
    ROW_NUMBER() OVER (
      PARTITION BY pos.SUBJECT_ID
      ORDER BY RAND()
    ) AS RN_POS,
    /* best-effort uniqueness pressure: rank per negative too */
    ROW_NUMBER() OVER (
      PARTITION BY neg.SUBJECT_ID
      ORDER BY pos.SUBJECT_ID
    ) AS RN_NEG
FROM tmp_pos_with_window_counts pos
JOIN tmp_neg_pool_base neg
  ON neg.GENDER = pos.GENDER
/* choose a real negative admission close in calendar time */
JOIN tmp_all_admissions a_neg
  ON a_neg.SUBJECT_ID = neg.SUBJECT_ID
 AND a_neg.ADMITTIME BETWEEN DATE_SUB(pos.INDEX_DATE, INTERVAL @MATCH_CAL_DAYS DAY)
                        AND DATE_ADD(pos.INDEX_DATE, INTERVAL @MATCH_CAL_DAYS DAY)
JOIN tmp_admission_race ar_neg
  ON ar_neg.HADM_ID = a_neg.HADM_ID
 AND ar_neg.RACE_BUCKET = pos.RACE_BUCKET
WHERE
  /* exact age match in years (at index) */
  TIMESTAMPDIFF(YEAR, (SELECT DOB FROM PATIENTS WHERE SUBJECT_ID = neg.SUBJECT_ID), a_neg.ADMITTIME)
    = pos.AGE_AT_INDEX
;

/* Now filter matches to enforce negative window visits constraints and visit similarity,
   then pick top CONTROLS_PER_CASE per positive, with soft uniqueness on negatives. */
DROP TABLE IF EXISTS tmp_matched_cohort;
CREATE TABLE tmp_matched_cohort AS
/* all positives */
SELECT
  pos.SUBJECT_ID,
  1 AS LABEL,
  pos.INDEX_DATE,
  pos.INDEX_HADM_ID
FROM tmp_pos_with_window_counts pos

UNION ALL

/* sampled controls */
SELECT
  rm.NEG_SUBJECT_ID AS SUBJECT_ID,
  0 AS LABEL,
  rm.NEG_INDEX_DATE AS INDEX_DATE,
  rm.NEG_INDEX_HADM_ID AS INDEX_HADM_ID
FROM tmp_ranked_matches rm
WHERE rm.NEG_WINDOW_VISITS >= 2
  AND rm.NEG_WINDOW_VISITS BETWEEN rm.POS_WINDOW_VISITS AND (rm.POS_WINDOW_VISITS + 4)
  AND rm.RN_NEG <= 1               /* best-effort reduce re-use */
  AND rm.RN_POS <= @CONTROLS_PER_CASE
;

/* ============================================================
   7) Windowed admissions (PRE-INDEX only)
   ============================================================ */
CREATE TABLE tmp_windowed_data AS
SELECT
  mc.SUBJECT_ID,
  mc.LABEL,
  mc.INDEX_DATE,
  a.HADM_ID,
  a.ADMITTIME
FROM tmp_matched_cohort mc
JOIN tmp_all_admissions a
  ON a.SUBJECT_ID = mc.SUBJECT_ID
WHERE a.ADMITTIME >= DATE_SUB(mc.INDEX_DATE, INTERVAL @LOOKBACK_DAYS DAY)
  AND a.ADMITTIME < mc.INDEX_DATE
  AND a.HADM_ID <> mc.INDEX_HADM_ID;

/* ============================================================
   8) Windowed DIAG + PROC codes
      - EXCLUDE:
          * E-codes
          * HF codes 428% (to avoid trivial leakage)
   ============================================================ */
CREATE TABLE tmp_windowed_diag AS
SELECT
  wd.SUBJECT_ID,
  wd.LABEL,
  wd.INDEX_DATE,
  wd.HADM_ID,
  d.ICD9_CODE AS CODE,
  'DX' AS CODE_TYPE
FROM tmp_windowed_data wd
JOIN DIAGNOSES_ICD d
  ON d.HADM_ID = wd.HADM_ID
WHERE d.ICD9_CODE NOT LIKE 'E%'
  AND d.ICD9_CODE NOT LIKE '428%';

CREATE TABLE tmp_windowed_proc AS
SELECT
  wd.SUBJECT_ID,
  wd.LABEL,
  wd.INDEX_DATE,
  wd.HADM_ID,
  p.ICD9_CODE AS CODE,
  'PX' AS CODE_TYPE
FROM tmp_windowed_data wd
JOIN PROCEDURES_ICD p
  ON p.HADM_ID = wd.HADM_ID;

/* unified windowed code table */
CREATE TABLE tmp_windowed_codes AS
SELECT * FROM tmp_windowed_diag
UNION ALL
SELECT * FROM tmp_windowed_proc;

/* ============================================================
   9) Global vocabulary from PRE-INDEX window only (leakage-safe)
   ============================================================ */
CREATE TABLE tmp_global_vocabulary AS
SELECT DISTINCT CODE
FROM tmp_windowed_codes;

/* ============================================================
   10) Summary stats (cohort-level)
   ============================================================ */
SELECT
  'MIMIC' AS Dataset,
  (SELECT COUNT(*) FROM tmp_matched_cohort WHERE LABEL = 1) AS Positive_Cases,
  (SELECT COUNT(*) FROM tmp_matched_cohort WHERE LABEL = 0) AS Negative_Cases,
  (
    SELECT ROUND(COUNT(*) / COUNT(DISTINCT SUBJECT_ID), 2)
    FROM (
      SELECT DISTINCT SUBJECT_ID, HADM_ID
      FROM tmp_windowed_data
    ) x
  ) AS Avg_Visits_Per_Patient,
  (
    SELECT ROUND(COUNT(*) / NULLIF(COUNT(DISTINCT HADM_ID), 0), 2)
    FROM tmp_windowed_codes
  ) AS Avg_Code_Per_Visit,
  (SELECT COUNT(*) FROM tmp_global_vocabulary) AS Unique_ICD9_Codes;

/* ============================================================
   Optional: Inspect class balance and window visit distributions
   ============================================================ */
-- SELECT LABEL, COUNT(*) AS N FROM tmp_matched_cohort GROUP BY LABEL;
-- SELECT LABEL, ROUND(AVG(vcnt),2) AS AVG_WINDOW_VISITS
-- FROM (
--   SELECT SUBJECT_ID, LABEL, COUNT(DISTINCT HADM_ID) AS vcnt
--   FROM tmp_windowed_data
--   GROUP BY SUBJECT_ID, LABEL
-- ) t
-- GROUP BY LABEL;
