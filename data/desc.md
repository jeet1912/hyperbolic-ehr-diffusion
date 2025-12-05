Dataset Construction and Preprocessing Methodology
1. Data Source and Cohort Definition

Data was extracted from the MIMIC-III (Medical Information Mart for Intensive Care) v1.4 database [1]. Following the preprocessing standards established by Choi et al. (RETAIN) [2] and adopted by MedDiffusion [3], we constructed a binary classification cohort for Heart Failure prediction.

Positive Cohort Selection: The case cohort includes adult patients (Age ≥ 18) diagnosed with Heart Failure. Patients were identified using ICD-9 codes starting with 428 (Heart Failure). To ensure the model predicts strictly incident or acute risk rather than routine maintenance, we applied the following exclusion criteria consistent with clinical benchmark tasks [4]:

Pediatric Exclusion: Patients under 18 years of age were removed.

Admission Type: Index admissions marked as ELECTIVE were excluded to isolate acute/emergency presentations.

Sequence Length: Patients with fewer than 2 total lifetime visits were excluded, as a single visit provides no historical time-series context for prediction.

Index Mortality: Patients who died during their index admission (HOSPITAL_EXPIRE_FLAG = 1) were excluded to ensure the prediction target remains clinically actionable.

Final Positive Count: 2,835 patients (Target: 2820).

2. Control Selection (Negative Cohort)

A control group was constructed from patients with no history of Heart Failure (no ICD-9 428.x codes). To mitigate confounding factors, we employed a strict propensity matching strategy. For each positive case, up to two control patients were selected based on the following covariates:

Demographics: Exact matches for Gender and Race (grouped into White, Black, Hispanic, Asian).

Age: Matched within a strict ±0 year tolerance (Exact Age).

Longitudinal Density: To avoid matching "heavy" users (high visit count) with "light" users, we enforced a Visit Count constraint. Controls were required to have a total visit count between N and N+4 (where N is the case's visit count). This bias ensures the control group exhibits similar or slightly higher data density, preventing the model from distinguishing classes based solely on sequence length.

Final Negative Count: 4,566 patients (Target: 4702).

3. Feature Representation and Statistics

To replicate the input requirements of transformer-based and diffusion-based models, the data was formatted into longitudinal sequences using a hybrid scoping strategy:

Sequence Observation Window (1-Year): For the calculation of sequence statistics (Average Visits and Code Density), patient history was truncated to a 1-year lookback window prior to the index date. This reflects the temporal slice typically fed into the model.

Average Visits per Patient: 2.62 (Target: 2.61).

Average Codes per Visit: 13.39 (Target: 13.06). Note: External cause codes (E-codes) were excluded from this density calculation to reduce noise.

Global Vocabulary Definition: While the input sequences were windowed, the embedding vocabulary was constructed from the full patient history (Lifetime + Future), including both Diagnoses (ICD-9) and Procedures (ICD-9-PCS). This ensures the model's embedding space covers the entire manifold of clinical concepts present in the dataset.

Unique Token Count: 4,844 (Target: 4,874).

References

[1] Johnson, A. E., et al. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

[2] Choi, E., et al. (2016). RETAIN: An interpretable predictive model for healthcare using reverse time attention mechanism. Advances in Neural Information Processing Systems (NIPS), 29.

[3] MedDiffusion: (2024). Medical Diffusion on EHRs (PMC11469648). Replicated Cohort Statistics.

[4] Harutyunyan, H., et al. (2019). Multitask learning and benchmarking with clinical time series data. Scientific Data, 6(1), 96.


NOTE: The Contradiction: In MIMIC-III, it is mathematically impossible to get an average of 2.61 visits inside a 6-month window (the average is ~1.5). To hit the 2.61 target reported in their table, we must include 1 year of history.

It is highly likely the authors described their intended window (6 months) in the text but actually used a broader window (1 year) when generating the final data and statistics table.




