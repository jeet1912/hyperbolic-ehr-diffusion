-- Active: 1753949249665@@127.0.0.1@3306@mimiciii4
WITH all_admissions AS (
    SELECT * FROM ADMISSIONS WHERE ADMISSION_TYPE != 'NEWBORN'
),
patient_features AS (
    SELECT 
        p.SUBJECT_ID,
        p.GENDER,
        p.DOB,
        TIMESTAMPDIFF(YEAR, p.DOB, MIN(a.ADMITTIME)) as AGE,
        COUNT(DISTINCT a.HADM_ID) as TOTAL_VISITS,
        CASE 
            WHEN MAX(a.ETHNICITY) LIKE '%WHITE%' THEN 'WHITE'
            WHEN MAX(a.ETHNICITY) LIKE '%BLACK%' OR MAX(a.ETHNICITY) LIKE '%AFRICAN%' THEN 'BLACK'
            WHEN MAX(a.ETHNICITY) LIKE '%HISPANIC%' OR MAX(a.ETHNICITY) LIKE '%LATINO%' THEN 'HISPANIC'
            WHEN MAX(a.ETHNICITY) LIKE '%ASIAN%' THEN 'ASIAN'
            ELSE NULL 
        END as RACE_BUCKET
    FROM PATIENTS p
    JOIN all_admissions a ON p.SUBJECT_ID = a.SUBJECT_ID
    GROUP BY p.SUBJECT_ID, p.GENDER, p.DOB
),
positive_candidates AS (
    SELECT 
        d.SUBJECT_ID,
        MIN(a.ADMITTIME) as INDEX_DATE,
        1 as LABEL,
        SUBSTRING_INDEX(GROUP_CONCAT(a.HADM_ID ORDER BY a.ADMITTIME ASC), ',', 1) as INDEX_HADM_ID
    FROM DIAGNOSES_ICD d
    JOIN all_admissions a ON d.HADM_ID = a.HADM_ID
    WHERE d.ICD9_CODE LIKE '428%' 
    AND a.ADMISSION_TYPE != 'ELECTIVE'
    GROUP BY d.SUBJECT_ID
),
final_positives AS (
    SELECT pc.* FROM positive_candidates pc
    JOIN all_admissions a ON pc.INDEX_HADM_ID = a.HADM_ID
    WHERE a.HOSPITAL_EXPIRE_FLAG = 0
),
negative_candidates AS (
    SELECT 
        pf.SUBJECT_ID,
        MAX(a.DISCHTIME) as INDEX_DATE,
        0 as LABEL
    FROM patient_features pf
    JOIN all_admissions a ON pf.SUBJECT_ID = a.SUBJECT_ID
    WHERE pf.SUBJECT_ID NOT IN (SELECT SUBJECT_ID FROM positive_candidates)
    GROUP BY pf.SUBJECT_ID
),
valid_pool AS (
    SELECT 
        pf.SUBJECT_ID,
        CASE WHEN pos.SUBJECT_ID IS NOT NULL THEN 1 ELSE 0 END as LABEL,
        COALESCE(pos.INDEX_DATE, neg.INDEX_DATE) as INDEX_DATE,
        pf.GENDER,
        pf.AGE,
        pf.RACE_BUCKET,
        pf.TOTAL_VISITS
    FROM patient_features pf
    LEFT JOIN final_positives pos ON pf.SUBJECT_ID = pos.SUBJECT_ID
    LEFT JOIN negative_candidates neg ON pf.SUBJECT_ID = neg.SUBJECT_ID
    WHERE pf.AGE >= 18 
      AND pf.TOTAL_VISITS >= 2 
      AND pf.RACE_BUCKET IS NOT NULL
      AND (pos.SUBJECT_ID IS NOT NULL OR neg.SUBJECT_ID IS NOT NULL)
),
matched_cohort AS (
    SELECT SUBJECT_ID, LABEL, INDEX_DATE FROM valid_pool WHERE LABEL = 1
    UNION ALL
    SELECT SUBJECT_ID, LABEL, INDEX_DATE FROM (
        SELECT 
            neg.SUBJECT_ID, neg.LABEL, neg.INDEX_DATE,
            ROW_NUMBER() OVER (PARTITION BY pos.SUBJECT_ID ORDER BY RAND()) as match_rank
        FROM valid_pool pos
        JOIN valid_pool neg 
          ON pos.LABEL = 1 AND neg.LABEL = 0
          AND pos.GENDER = neg.GENDER
          AND pos.RACE_BUCKET = neg.RACE_BUCKET
          AND neg.AGE = pos.AGE
          AND neg.TOTAL_VISITS BETWEEN pos.TOTAL_VISITS AND (pos.TOTAL_VISITS + 4)
    ) ranked
    WHERE match_rank <= 2
),
windowed_data AS (
    SELECT 
        mc.SUBJECT_ID,
        a.HADM_ID
    FROM matched_cohort mc
    JOIN all_admissions a ON mc.SUBJECT_ID = a.SUBJECT_ID
    WHERE a.ADMITTIME >= DATE_SUB(mc.INDEX_DATE, INTERVAL 1 YEAR)
      AND a.ADMITTIME <= mc.INDEX_DATE
),
windowed_codes AS (
    SELECT wd.HADM_ID, d.ICD9_CODE
    FROM windowed_data wd
    JOIN DIAGNOSES_ICD d ON wd.HADM_ID = d.HADM_ID
    WHERE d.ICD9_CODE NOT LIKE 'E%'
),
global_vocabulary AS (
    SELECT DISTINCT d.ICD9_CODE
    FROM matched_cohort mc
    JOIN all_admissions a ON mc.SUBJECT_ID = a.SUBJECT_ID
    JOIN DIAGNOSES_ICD d ON a.HADM_ID = d.HADM_ID
    WHERE d.ICD9_CODE NOT LIKE 'E%'
    UNION
    SELECT DISTINCT p.ICD9_CODE
    FROM matched_cohort mc
    JOIN all_admissions a ON mc.SUBJECT_ID = a.SUBJECT_ID
    JOIN PROCEDURES_ICD p ON a.HADM_ID = p.HADM_ID
)
SELECT 
    'MIMIC' as Dataset,
    (SELECT COUNT(*) FROM matched_cohort WHERE LABEL = 1) as Positive_Cases,
    (SELECT COUNT(*) FROM matched_cohort WHERE LABEL = 0) as Negative_Cases,
    (SELECT ROUND(COUNT(*) / COUNT(DISTINCT SUBJECT_ID), 2) FROM windowed_data) as Avg_Visits_Per_Patient,
    (SELECT ROUND(COUNT(*) / (SELECT COUNT(*) FROM windowed_data), 2) FROM windowed_codes) as Avg_Code_Per_Visit,
    (SELECT COUNT(*) FROM global_vocabulary) as Unique_ICD9_Codes;



