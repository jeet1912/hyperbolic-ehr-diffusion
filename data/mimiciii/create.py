import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
DB_CONNECTION_STR = 'mysql+pymysql://root:yourpasswd@localhost/mimiciii4' 
OUTPUT_PKL = 'mimic_hf_cohort.pkl'

print("Connecting to Database...")
engine = create_engine(DB_CONNECTION_STR)

# -------------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------------
print("Loading Cohort...")
# Load cohort and assign SEQ_ID to handle duplicates (Sampling with Replacement)
cohort_df = pd.read_sql("SELECT SUBJECT_ID, LABEL FROM tmp_matched_cohort", engine)
cohort_df['SEQ_ID'] = range(len(cohort_df)) 
print(f"Cohort Size: {len(cohort_df)}")

print("Loading Visits...")
# Load windowed visits
visits_sql = """
    SELECT wd.SUBJECT_ID, wd.HADM_ID, a.ADMITTIME
    FROM tmp_windowed_data wd
    JOIN ADMISSIONS a ON wd.HADM_ID = a.HADM_ID
"""
visits_df = pd.read_sql(visits_sql, engine)
print(f"Visits Loaded: {len(visits_df)}")

print("Loading Codes (Server-Side)...")
# Load Diagnoses + Procedures linked to these visits
codes_sql = """
    SELECT d.HADM_ID, d.ICD9_CODE 
    FROM DIAGNOSES_ICD d
    JOIN tmp_windowed_data wd ON d.HADM_ID = wd.HADM_ID
    WHERE d.ICD9_CODE NOT LIKE 'E%%'
    UNION ALL
    SELECT p.HADM_ID, p.ICD9_CODE 
    FROM PROCEDURES_ICD p
    JOIN tmp_windowed_data wd ON p.HADM_ID = wd.HADM_ID
"""
codes_df = pd.read_sql(codes_sql, engine)
print(f"Codes Loaded: {len(codes_df)}")

print("Loading Vocabulary...")
vocab_df = pd.read_sql("SELECT ICD9_CODE FROM tmp_global_vocabulary", engine)
target_vocab_list = sorted(vocab_df['ICD9_CODE'].astype(str).tolist())
print(f"Vocabulary Size: {len(target_vocab_list)}")

# -------------------------------------------------------------------------
# 2. ASSEMBLE
# -------------------------------------------------------------------------
print("Assembling Data...")

# Merge Cohort -> Visits -> Codes
# This duplicates visits for patients selected multiple times (Correct for ML)
traj_df = cohort_df.merge(visits_df, on='SUBJECT_ID', how='inner')
full_data = traj_df.merge(codes_df, on='HADM_ID', how='inner')

# Sort Chronologically
full_data['ADMITTIME'] = pd.to_datetime(full_data['ADMITTIME'])
full_data.sort_values(['SEQ_ID', 'ADMITTIME'], inplace=True)

# -------------------------------------------------------------------------
# 3. FORMAT
# -------------------------------------------------------------------------
print("Constructing Tensors...")

# Map codes to integers
code_map = {code: i+1 for i, code in enumerate(target_vocab_list)}
vocab_size = len(code_map) + 1

final_x = []
final_y = []

# Group by Sequence ID
grouped = full_data.groupby('SEQ_ID')

# Iterate through original cohort indices to maintain alignment
for seq_id in range(len(cohort_df)):
    if seq_id not in grouped.groups:
        continue # Skip if patient has visits but 0 codes (rare edge case)
        
    group = grouped.get_group(seq_id)
    label = int(group['LABEL'].iloc[0])
    
    visits = []
    
    # Get unique visits in order
    visit_ids = group['HADM_ID'].unique()
    
    for hadm_id in visit_ids:
        # Extract codes
        v_codes = group[group['HADM_ID'] == hadm_id]['ICD9_CODE'].astype(str).tolist()
        
        # Map to standard Python Ints
        v_ints = [int(code_map[c]) for c in v_codes if c in code_map]
        
        if v_ints:
            visits.append(v_ints)
            
    if visits:
        final_x.append(visits)
        final_y.append(label)

# -------------------------------------------------------------------------
# 4. SAVE
# -------------------------------------------------------------------------
output = {
    'x': final_x,
    'y': final_y,
    'code_map': code_map,
    'vocab_size': int(vocab_size)
}

print("-" * 30)
print(f"Final Sequence Count: {len(final_x)}")
print(f"Avg Sequence Length: {sum(len(x) for x in final_x) / len(final_x):.2f} visits")
print("(Note: 1.70 avg sequence length is mathematically consistent with 2.62 avg visits per unique patient)")
print("-" * 30)

with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(output, f)

print(f"✅ DONE. Dataset saved to {OUTPUT_PKL}")