Using device: mps
[MIMIC] Loading data/mimiciii/mimic_hf_cohort.pkl ...
[MIMIC] Patients: 7397 | Vocab size: 4817
[HyperMedDiff-Risk] Real trajectory stats: {
  "patients": 7397,
  "avg_visits_per_patient": 1.7,
  "avg_codes_per_visit": 424.04,
  "max_visits": 12,
  "max_codes": 9261
}
[HyperMedDiff-Risk] ICD tree source: cms | codes: 10248
[HyperMedDiff-Risk] Running 18 ablation configurations.
[HyperMedDiff-Risk] ===== Experiment 1/18: 01_Baseline =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0238 | val=0.0248
[Pretrain] Epoch 02 | train=0.0233 | val=0.0254
[Pretrain] Epoch 03 | train=0.0250 | val=0.0238
[Pretrain] Epoch 04 | train=0.0221 | val=0.0230
[Pretrain] Epoch 05 | train=0.0211 | val=0.0220
[Pretrain] Epoch 06 | train=0.0228 | val=0.0211
[Pretrain] Epoch 07 | train=0.0220 | val=0.0209
[Pretrain] Epoch 08 | train=0.0203 | val=0.0201
[Pretrain] Epoch 09 | train=0.0205 | val=0.0199
[Pretrain] Epoch 10 | train=0.0190 | val=0.0197
[Pretrain] Epoch 11 | train=0.0200 | val=0.0202
[Pretrain] Epoch 12 | train=0.0198 | val=0.0193
[Pretrain] Epoch 13 | train=0.0197 | val=0.0192
[Pretrain] Epoch 14 | train=0.0193 | val=0.0189
[Pretrain] Epoch 15 | train=0.0192 | val=0.0181
[Pretrain] Epoch 16 | train=0.0189 | val=0.0180
[Pretrain] Epoch 17 | train=0.0186 | val=0.0179
[Pretrain] Epoch 18 | train=0.0173 | val=0.0173
[Pretrain] Epoch 19 | train=0.0177 | val=0.0182
[Pretrain] Epoch 20 | train=0.0174 | val=0.0173
[Pretrain] Epoch 21 | train=0.0167 | val=0.0169
[Pretrain] Epoch 22 | train=0.0162 | val=0.0162
[Pretrain] Epoch 23 | train=0.0169 | val=0.0167
[Pretrain] Epoch 24 | train=0.0164 | val=0.0151
[Pretrain] Epoch 25 | train=0.0160 | val=0.0163
[Pretrain] Epoch 26 | train=0.0162 | val=0.0159
[Pretrain] Epoch 27 | train=0.0151 | val=0.0151
[Pretrain] Epoch 28 | train=0.0147 | val=0.0154
[Pretrain] Epoch 29 | train=0.0146 | val=0.0156
[Pretrain] Epoch 30 | train=0.0148 | val=0.0153
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.6978 ± 0.0073
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.6994 | Val 10.9978
[HyperMedDiff-Risk] Epoch 002 | Train 11.3789 | Val 10.8173
[HyperMedDiff-Risk] Epoch 003 | Train 11.1403 | Val 10.4876
[HyperMedDiff-Risk] Epoch 004 | Train 10.6012 | Val 9.6873
[HyperMedDiff-Risk] Epoch 005 | Train 9.8835 | Val 8.8391
[HyperMedDiff-Risk] Epoch 006 | Train 9.1969 | Val 8.1572
[HyperMedDiff-Risk] Epoch 007 | Train 8.6410 | Val 7.5250
[HyperMedDiff-Risk] Epoch 008 | Train 8.1347 | Val 6.9909
[HyperMedDiff-Risk] Epoch 009 | Train 7.7856 | Val 6.6440
[HyperMedDiff-Risk] Epoch 010 | Train 7.3876 | Val 6.2020
[HyperMedDiff-Risk] Epoch 011 | Train 7.1108 | Val 5.9307
[HyperMedDiff-Risk] Epoch 012 | Train 6.8124 | Val 5.5421
[HyperMedDiff-Risk] Epoch 013 | Train 6.5472 | Val 5.4145
[HyperMedDiff-Risk] Epoch 014 | Train 6.3320 | Val 5.0929
[HyperMedDiff-Risk] Epoch 015 | Train 6.1146 | Val 4.8816
[HyperMedDiff-Risk] Epoch 016 | Train 5.9286 | Val 4.6805
[HyperMedDiff-Risk] Epoch 017 | Train 5.8152 | Val 4.4695
[HyperMedDiff-Risk] Epoch 018 | Train 5.6033 | Val 4.2899
[HyperMedDiff-Risk] Epoch 019 | Train 5.4286 | Val 4.2217
[HyperMedDiff-Risk] Epoch 020 | Train 5.3177 | Val 4.1209
[HyperMedDiff-Risk] Epoch 021 | Train 5.2020 | Val 4.0213
[HyperMedDiff-Risk] Epoch 022 | Train 5.0459 | Val 3.7837
[HyperMedDiff-Risk] Epoch 023 | Train 4.9349 | Val 3.7846
[HyperMedDiff-Risk] Epoch 024 | Train 4.8495 | Val 3.5849
[HyperMedDiff-Risk] Epoch 025 | Train 4.7492 | Val 3.4251
[HyperMedDiff-Risk] Epoch 026 | Train 4.6266 | Val 3.3322
[HyperMedDiff-Risk] Epoch 027 | Train 4.5137 | Val 3.3181
[HyperMedDiff-Risk] Epoch 028 | Train 4.4669 | Val 3.2240
[HyperMedDiff-Risk] Epoch 029 | Train 4.3705 | Val 3.0634
[HyperMedDiff-Risk] Epoch 030 | Train 4.2763 | Val 3.0684
[HyperMedDiff-Risk] Epoch 031 | Train 4.2323 | Val 3.0065
[HyperMedDiff-Risk] Epoch 032 | Train 4.1418 | Val 2.9384
[HyperMedDiff-Risk] Epoch 033 | Train 4.1122 | Val 2.9267
[HyperMedDiff-Risk] Epoch 034 | Train 4.0630 | Val 2.8542
[HyperMedDiff-Risk] Epoch 035 | Train 3.9917 | Val 2.7694
[HyperMedDiff-Risk] Epoch 036 | Train 3.9305 | Val 2.7091
[HyperMedDiff-Risk] Epoch 037 | Train 3.8724 | Val 2.6749
[HyperMedDiff-Risk] Epoch 038 | Train 3.7841 | Val 2.5532
[HyperMedDiff-Risk] Epoch 039 | Train 3.7931 | Val 2.5482
[HyperMedDiff-Risk] Epoch 040 | Train 3.7394 | Val 2.4666
[HyperMedDiff-Risk] Epoch 041 | Train 3.6711 | Val 2.4856
[HyperMedDiff-Risk] Epoch 042 | Train 3.6176 | Val 2.4242
[HyperMedDiff-Risk] Epoch 043 | Train 3.5841 | Val 2.3400
[HyperMedDiff-Risk] Epoch 044 | Train 3.5297 | Val 2.3943
[HyperMedDiff-Risk] Epoch 045 | Train 3.4951 | Val 2.3185
[HyperMedDiff-Risk] Epoch 046 | Train 3.4932 | Val 2.3412
[HyperMedDiff-Risk] Epoch 047 | Train 3.4656 | Val 2.2793
[HyperMedDiff-Risk] Epoch 048 | Train 3.4234 | Val 2.2234
[HyperMedDiff-Risk] Epoch 049 | Train 3.4013 | Val 2.1803
[HyperMedDiff-Risk] Epoch 050 | Train 3.3709 | Val 2.1601
[HyperMedDiff-Risk] Epoch 051 | Train 3.3477 | Val 2.1013
[HyperMedDiff-Risk] Epoch 052 | Train 3.3083 | Val 2.1598
[HyperMedDiff-Risk] Epoch 053 | Train 3.2878 | Val 2.1391
[HyperMedDiff-Risk] Epoch 054 | Train 3.2616 | Val 2.1636
[HyperMedDiff-Risk] Epoch 055 | Train 3.2533 | Val 2.0353
[HyperMedDiff-Risk] Epoch 056 | Train 3.2435 | Val 2.0508
[HyperMedDiff-Risk] Epoch 057 | Train 3.1995 | Val 2.0458
[HyperMedDiff-Risk] Epoch 058 | Train 3.2315 | Val 1.9811
[HyperMedDiff-Risk] Epoch 059 | Train 3.1705 | Val 2.0101
[HyperMedDiff-Risk] Epoch 060 | Train 3.1441 | Val 1.9910
[HyperMedDiff-Risk] Epoch 061 | Train 3.1659 | Val 1.9542
[HyperMedDiff-Risk] Epoch 062 | Train 3.1492 | Val 1.9647
[HyperMedDiff-Risk] Epoch 063 | Train 3.1022 | Val 1.8921
[HyperMedDiff-Risk] Epoch 064 | Train 3.0942 | Val 1.9167
[HyperMedDiff-Risk] Epoch 065 | Train 3.0872 | Val 1.9483
[HyperMedDiff-Risk] Epoch 066 | Train 3.0878 | Val 1.9129
[HyperMedDiff-Risk] Epoch 067 | Train 3.0460 | Val 2.0040
[HyperMedDiff-Risk] Epoch 068 | Train 3.0372 | Val 1.8701
[HyperMedDiff-Risk] Epoch 069 | Train 3.0369 | Val 1.8567
[HyperMedDiff-Risk] Epoch 070 | Train 3.0215 | Val 1.8495
[HyperMedDiff-Risk] Epoch 071 | Train 2.9895 | Val 1.8315
[HyperMedDiff-Risk] Epoch 072 | Train 3.0085 | Val 1.8355
[HyperMedDiff-Risk] Epoch 073 | Train 2.9838 | Val 1.8510
[HyperMedDiff-Risk] Epoch 074 | Train 2.9772 | Val 1.8110
[HyperMedDiff-Risk] Epoch 075 | Train 3.0158 | Val 1.8015
[HyperMedDiff-Risk] Epoch 076 | Train 2.9761 | Val 1.8364
[HyperMedDiff-Risk] Epoch 077 | Train 2.9761 | Val 1.8060
[HyperMedDiff-Risk] Epoch 078 | Train 2.9574 | Val 1.7922
[HyperMedDiff-Risk] Epoch 079 | Train 2.9505 | Val 1.8027
[HyperMedDiff-Risk] Epoch 080 | Train 2.9414 | Val 1.8438
[HyperMedDiff-Risk] Epoch 081 | Train 2.9310 | Val 1.7551
[HyperMedDiff-Risk] Epoch 082 | Train 2.9641 | Val 1.7817
[HyperMedDiff-Risk] Epoch 083 | Train 2.9290 | Val 1.8178
[HyperMedDiff-Risk] Epoch 084 | Train 2.9355 | Val 1.8060
[HyperMedDiff-Risk] Epoch 085 | Train 2.9319 | Val 1.7283
[HyperMedDiff-Risk] Epoch 086 | Train 2.9655 | Val 1.7615
[HyperMedDiff-Risk] Epoch 087 | Train 2.9131 | Val 1.7828
[HyperMedDiff-Risk] Epoch 088 | Train 2.9183 | Val 1.7712
[HyperMedDiff-Risk] Epoch 089 | Train 2.9169 | Val 1.7456
[HyperMedDiff-Risk] Epoch 090 | Train 2.9070 | Val 1.7513
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 1): 1.7283
[HyperMedDiff-Risk] Saved training curve plot to results/plots/01_Baseline.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.873267021094152,
  "auprc": 0.8056978897369395
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.6978 ± 0.0073
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7356
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): -0.0156
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.1953 std=0.0349
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/01_Baseline_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/01_Baseline_umap.png
Saved checkpoint to results/checkpoints/01_Baseline.pt
[HyperMedDiff-Risk] ===== Experiment 2/18: 02_NoDiffusion =====
{
  "diffusion_steps": [
    1
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0086 | val=0.0083
[Pretrain] Epoch 02 | train=0.0081 | val=0.0077
[Pretrain] Epoch 03 | train=0.0083 | val=0.0076
[Pretrain] Epoch 04 | train=0.0075 | val=0.0070
[Pretrain] Epoch 05 | train=0.0069 | val=0.0068
[Pretrain] Epoch 06 | train=0.0067 | val=0.0067
[Pretrain] Epoch 07 | train=0.0064 | val=0.0064
[Pretrain] Epoch 08 | train=0.0062 | val=0.0061
[Pretrain] Epoch 09 | train=0.0060 | val=0.0056
[Pretrain] Epoch 10 | train=0.0057 | val=0.0055
[Pretrain] Epoch 11 | train=0.0053 | val=0.0054
[Pretrain] Epoch 12 | train=0.0052 | val=0.0051
[Pretrain] Epoch 13 | train=0.0052 | val=0.0050
[Pretrain] Epoch 14 | train=0.0052 | val=0.0049
[Pretrain] Epoch 15 | train=0.0050 | val=0.0049
[Pretrain] Epoch 16 | train=0.0049 | val=0.0048
[Pretrain] Epoch 17 | train=0.0048 | val=0.0047
[Pretrain] Epoch 18 | train=0.0046 | val=0.0047
[Pretrain] Epoch 19 | train=0.0043 | val=0.0046
[Pretrain] Epoch 20 | train=0.0045 | val=0.0043
[Pretrain] Epoch 21 | train=0.0043 | val=0.0044
[Pretrain] Epoch 22 | train=0.0042 | val=0.0042
[Pretrain] Epoch 23 | train=0.0042 | val=0.0040
[Pretrain] Epoch 24 | train=0.0042 | val=0.0040
[Pretrain] Epoch 25 | train=0.0040 | val=0.0040
[Pretrain] Epoch 26 | train=0.0040 | val=0.0040
[Pretrain] Epoch 27 | train=0.0039 | val=0.0038
[Pretrain] Epoch 28 | train=0.0037 | val=0.0039
[Pretrain] Epoch 29 | train=0.0037 | val=0.0036
[Pretrain] Epoch 30 | train=0.0037 | val=0.0036
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.8260 ± 0.0046
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.5191 | Val 10.9696
[HyperMedDiff-Risk] Epoch 002 | Train 11.3851 | Val 10.8395
[HyperMedDiff-Risk] Epoch 003 | Train 11.2008 | Val 10.6478
[HyperMedDiff-Risk] Epoch 004 | Train 10.8365 | Val 10.0774
[HyperMedDiff-Risk] Epoch 005 | Train 10.1730 | Val 9.1565
[HyperMedDiff-Risk] Epoch 006 | Train 9.4600 | Val 8.4206
[HyperMedDiff-Risk] Epoch 007 | Train 8.8520 | Val 7.7731
[HyperMedDiff-Risk] Epoch 008 | Train 8.3332 | Val 7.1835
[HyperMedDiff-Risk] Epoch 009 | Train 7.8905 | Val 6.8035
[HyperMedDiff-Risk] Epoch 010 | Train 7.4876 | Val 6.3354
[HyperMedDiff-Risk] Epoch 011 | Train 7.1877 | Val 5.9109
[HyperMedDiff-Risk] Epoch 012 | Train 6.9079 | Val 5.8096
[HyperMedDiff-Risk] Epoch 013 | Train 6.6218 | Val 5.4442
[HyperMedDiff-Risk] Epoch 014 | Train 6.3668 | Val 5.1838
[HyperMedDiff-Risk] Epoch 015 | Train 6.1763 | Val 4.9515
[HyperMedDiff-Risk] Epoch 016 | Train 6.0108 | Val 4.7523
[HyperMedDiff-Risk] Epoch 017 | Train 5.7945 | Val 4.6346
[HyperMedDiff-Risk] Epoch 018 | Train 5.6109 | Val 4.3495
[HyperMedDiff-Risk] Epoch 019 | Train 5.4820 | Val 4.1584
[HyperMedDiff-Risk] Epoch 020 | Train 5.2937 | Val 4.0340
[HyperMedDiff-Risk] Epoch 021 | Train 5.1821 | Val 3.9048
[HyperMedDiff-Risk] Epoch 022 | Train 5.0514 | Val 3.8015
[HyperMedDiff-Risk] Epoch 023 | Train 4.9827 | Val 3.7514
[HyperMedDiff-Risk] Epoch 024 | Train 4.8769 | Val 3.5647
[HyperMedDiff-Risk] Epoch 025 | Train 4.7563 | Val 3.4380
[HyperMedDiff-Risk] Epoch 026 | Train 4.6384 | Val 3.3470
[HyperMedDiff-Risk] Epoch 027 | Train 4.5398 | Val 3.3347
[HyperMedDiff-Risk] Epoch 028 | Train 4.4725 | Val 3.2773
[HyperMedDiff-Risk] Epoch 029 | Train 4.3287 | Val 3.1037
[HyperMedDiff-Risk] Epoch 030 | Train 4.2807 | Val 3.0676
[HyperMedDiff-Risk] Epoch 031 | Train 4.2245 | Val 2.9796
[HyperMedDiff-Risk] Epoch 032 | Train 4.1459 | Val 2.9434
[HyperMedDiff-Risk] Epoch 033 | Train 4.0881 | Val 2.8398
[HyperMedDiff-Risk] Epoch 034 | Train 4.0408 | Val 2.7472
[HyperMedDiff-Risk] Epoch 035 | Train 3.9751 | Val 2.6730
[HyperMedDiff-Risk] Epoch 036 | Train 3.8768 | Val 2.7846
[HyperMedDiff-Risk] Epoch 037 | Train 3.8339 | Val 2.5904
[HyperMedDiff-Risk] Epoch 038 | Train 3.8102 | Val 2.6311
[HyperMedDiff-Risk] Epoch 039 | Train 3.7433 | Val 2.4817
[HyperMedDiff-Risk] Epoch 040 | Train 3.7197 | Val 2.4445
[HyperMedDiff-Risk] Epoch 041 | Train 3.6681 | Val 2.4800
[HyperMedDiff-Risk] Epoch 042 | Train 3.6044 | Val 2.4348
[HyperMedDiff-Risk] Epoch 043 | Train 3.5807 | Val 2.3110
[HyperMedDiff-Risk] Epoch 044 | Train 3.5421 | Val 2.3173
[HyperMedDiff-Risk] Epoch 045 | Train 3.5054 | Val 2.3467
[HyperMedDiff-Risk] Epoch 046 | Train 3.4806 | Val 2.2365
[HyperMedDiff-Risk] Epoch 047 | Train 3.4528 | Val 2.2871
[HyperMedDiff-Risk] Epoch 048 | Train 3.3886 | Val 2.2426
[HyperMedDiff-Risk] Epoch 049 | Train 3.3740 | Val 2.2691
[HyperMedDiff-Risk] Epoch 050 | Train 3.3716 | Val 2.1904
[HyperMedDiff-Risk] Epoch 051 | Train 3.3060 | Val 2.1347
[HyperMedDiff-Risk] Epoch 052 | Train 3.2794 | Val 2.0661
[HyperMedDiff-Risk] Epoch 053 | Train 3.2697 | Val 2.1191
[HyperMedDiff-Risk] Epoch 054 | Train 3.2683 | Val 2.0440
[HyperMedDiff-Risk] Epoch 055 | Train 3.2212 | Val 1.9841
[HyperMedDiff-Risk] Epoch 056 | Train 3.1850 | Val 2.0449
[HyperMedDiff-Risk] Epoch 057 | Train 3.2046 | Val 1.9919
[HyperMedDiff-Risk] Epoch 058 | Train 3.1948 | Val 2.0484
[HyperMedDiff-Risk] Epoch 059 | Train 3.1439 | Val 1.9980
[HyperMedDiff-Risk] Epoch 060 | Train 3.1235 | Val 1.9502
[HyperMedDiff-Risk] Epoch 061 | Train 3.1364 | Val 1.9442
[HyperMedDiff-Risk] Epoch 062 | Train 3.1050 | Val 1.9182
[HyperMedDiff-Risk] Epoch 063 | Train 3.0976 | Val 1.9846
[HyperMedDiff-Risk] Epoch 064 | Train 3.0959 | Val 1.9583
[HyperMedDiff-Risk] Epoch 065 | Train 3.0975 | Val 1.9355
[HyperMedDiff-Risk] Epoch 066 | Train 3.0780 | Val 1.8815
[HyperMedDiff-Risk] Epoch 067 | Train 3.0530 | Val 1.9257
[HyperMedDiff-Risk] Epoch 068 | Train 3.0528 | Val 1.8833
[HyperMedDiff-Risk] Epoch 069 | Train 3.0607 | Val 1.8714
[HyperMedDiff-Risk] Epoch 070 | Train 3.0190 | Val 1.8864
[HyperMedDiff-Risk] Epoch 071 | Train 3.0242 | Val 1.8616
[HyperMedDiff-Risk] Epoch 072 | Train 3.0003 | Val 1.8643
[HyperMedDiff-Risk] Epoch 073 | Train 2.9933 | Val 1.8642
[HyperMedDiff-Risk] Epoch 074 | Train 3.0079 | Val 1.8295
[HyperMedDiff-Risk] Epoch 075 | Train 2.9976 | Val 1.8724
[HyperMedDiff-Risk] Epoch 076 | Train 2.9751 | Val 1.8332
[HyperMedDiff-Risk] Epoch 077 | Train 2.9866 | Val 1.8347
[HyperMedDiff-Risk] Epoch 078 | Train 3.0173 | Val 1.8181
[HyperMedDiff-Risk] Epoch 079 | Train 2.9807 | Val 1.8506
[HyperMedDiff-Risk] Epoch 080 | Train 2.9826 | Val 1.8129
[HyperMedDiff-Risk] Epoch 081 | Train 2.9548 | Val 1.7968
[HyperMedDiff-Risk] Epoch 082 | Train 2.9584 | Val 1.8374
[HyperMedDiff-Risk] Epoch 083 | Train 2.9745 | Val 1.8351
[HyperMedDiff-Risk] Epoch 084 | Train 2.9562 | Val 1.8078
[HyperMedDiff-Risk] Epoch 085 | Train 2.9470 | Val 1.8053
[HyperMedDiff-Risk] Epoch 086 | Train 2.9533 | Val 1.7779
[HyperMedDiff-Risk] Epoch 087 | Train 2.9344 | Val 1.7951
[HyperMedDiff-Risk] Epoch 088 | Train 2.9624 | Val 1.8078
[HyperMedDiff-Risk] Epoch 089 | Train 2.9428 | Val 1.7886
[HyperMedDiff-Risk] Epoch 090 | Train 2.9310 | Val 1.7493
[HyperMedDiff-Risk] Epoch 091 | Train 2.9212 | Val 1.8007
[HyperMedDiff-Risk] Epoch 092 | Train 2.9434 | Val 1.8056
[HyperMedDiff-Risk] Epoch 093 | Train 2.9349 | Val 1.8074
[HyperMedDiff-Risk] Epoch 094 | Train 2.9391 | Val 1.7682
[HyperMedDiff-Risk] Epoch 095 | Train 2.9225 | Val 1.7600
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 2): 1.7493
[HyperMedDiff-Risk] Saved training curve plot to results/plots/02_NoDiffusion.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8737232035671412,
  "auprc": 0.8045036223250879
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.8260 ± 0.0046
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.8411
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0210
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.1961 std=0.0417
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/02_NoDiffusion_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/02_NoDiffusion_umap.png
Saved checkpoint to results/checkpoints/02_NoDiffusion.pt
[HyperMedDiff-Risk] ===== Experiment 3/18: 03_LocalDiff =====
{
  "diffusion_steps": [
    1,
    2
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0140 | val=0.0133
[Pretrain] Epoch 02 | train=0.0133 | val=0.0126
[Pretrain] Epoch 03 | train=0.0119 | val=0.0121
[Pretrain] Epoch 04 | train=0.0126 | val=0.0122
[Pretrain] Epoch 05 | train=0.0114 | val=0.0126
[Pretrain] Epoch 06 | train=0.0113 | val=0.0116
[Pretrain] Epoch 07 | train=0.0118 | val=0.0105
[Pretrain] Epoch 08 | train=0.0110 | val=0.0110
[Pretrain] Epoch 09 | train=0.0101 | val=0.0103
[Pretrain] Epoch 10 | train=0.0107 | val=0.0096
[Pretrain] Epoch 11 | train=0.0100 | val=0.0097
[Pretrain] Epoch 12 | train=0.0096 | val=0.0096
[Pretrain] Epoch 13 | train=0.0099 | val=0.0093
[Pretrain] Epoch 14 | train=0.0089 | val=0.0091
[Pretrain] Epoch 15 | train=0.0093 | val=0.0090
[Pretrain] Epoch 16 | train=0.0086 | val=0.0085
[Pretrain] Epoch 17 | train=0.0090 | val=0.0089
[Pretrain] Epoch 18 | train=0.0084 | val=0.0089
[Pretrain] Epoch 19 | train=0.0084 | val=0.0083
[Pretrain] Epoch 20 | train=0.0080 | val=0.0080
[Pretrain] Epoch 21 | train=0.0082 | val=0.0082
[Pretrain] Epoch 22 | train=0.0075 | val=0.0086
[Pretrain] Epoch 23 | train=0.0083 | val=0.0075
[Pretrain] Epoch 24 | train=0.0077 | val=0.0078
[Pretrain] Epoch 25 | train=0.0071 | val=0.0076
[Pretrain] Epoch 26 | train=0.0076 | val=0.0074
[Pretrain] Epoch 27 | train=0.0074 | val=0.0072
[Pretrain] Epoch 28 | train=0.0075 | val=0.0073
[Pretrain] Epoch 29 | train=0.0071 | val=0.0071
[Pretrain] Epoch 30 | train=0.0068 | val=0.0071
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7797 ± 0.0051
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.7554 | Val 10.9591
[HyperMedDiff-Risk] Epoch 002 | Train 11.3434 | Val 10.7938
[HyperMedDiff-Risk] Epoch 003 | Train 11.0909 | Val 10.4262
[HyperMedDiff-Risk] Epoch 004 | Train 10.5312 | Val 9.6082
[HyperMedDiff-Risk] Epoch 005 | Train 9.7928 | Val 8.7861
[HyperMedDiff-Risk] Epoch 006 | Train 9.1103 | Val 8.1209
[HyperMedDiff-Risk] Epoch 007 | Train 8.5315 | Val 7.3938
[HyperMedDiff-Risk] Epoch 008 | Train 8.0692 | Val 6.8877
[HyperMedDiff-Risk] Epoch 009 | Train 7.6501 | Val 6.5901
[HyperMedDiff-Risk] Epoch 010 | Train 7.3124 | Val 6.1891
[HyperMedDiff-Risk] Epoch 011 | Train 6.9974 | Val 5.8274
[HyperMedDiff-Risk] Epoch 012 | Train 6.7460 | Val 5.5182
[HyperMedDiff-Risk] Epoch 013 | Train 6.4935 | Val 5.3149
[HyperMedDiff-Risk] Epoch 014 | Train 6.2325 | Val 5.0537
[HyperMedDiff-Risk] Epoch 015 | Train 6.0085 | Val 4.8367
[HyperMedDiff-Risk] Epoch 016 | Train 5.8699 | Val 4.6088
[HyperMedDiff-Risk] Epoch 017 | Train 5.7041 | Val 4.3832
[HyperMedDiff-Risk] Epoch 018 | Train 5.4816 | Val 4.3145
[HyperMedDiff-Risk] Epoch 019 | Train 5.3227 | Val 4.2107
[HyperMedDiff-Risk] Epoch 020 | Train 5.2257 | Val 4.0827
[HyperMedDiff-Risk] Epoch 021 | Train 5.0809 | Val 3.9449
[HyperMedDiff-Risk] Epoch 022 | Train 4.9321 | Val 3.6641
[HyperMedDiff-Risk] Epoch 023 | Train 4.8108 | Val 3.6077
[HyperMedDiff-Risk] Epoch 024 | Train 4.6937 | Val 3.4546
[HyperMedDiff-Risk] Epoch 025 | Train 4.5953 | Val 3.3418
[HyperMedDiff-Risk] Epoch 026 | Train 4.4678 | Val 3.2152
[HyperMedDiff-Risk] Epoch 027 | Train 4.4146 | Val 3.1091
[HyperMedDiff-Risk] Epoch 028 | Train 4.2899 | Val 3.0406
[HyperMedDiff-Risk] Epoch 029 | Train 4.2334 | Val 2.9749
[HyperMedDiff-Risk] Epoch 030 | Train 4.1595 | Val 2.9531
[HyperMedDiff-Risk] Epoch 031 | Train 4.0940 | Val 2.9178
[HyperMedDiff-Risk] Epoch 032 | Train 4.0255 | Val 2.7986
[HyperMedDiff-Risk] Epoch 033 | Train 4.0200 | Val 2.8272
[HyperMedDiff-Risk] Epoch 034 | Train 3.9610 | Val 2.7221
[HyperMedDiff-Risk] Epoch 035 | Train 3.8822 | Val 2.6556
[HyperMedDiff-Risk] Epoch 036 | Train 3.8162 | Val 2.6538
[HyperMedDiff-Risk] Epoch 037 | Train 3.8299 | Val 2.5311
[HyperMedDiff-Risk] Epoch 038 | Train 3.7732 | Val 2.6040
[HyperMedDiff-Risk] Epoch 039 | Train 3.7015 | Val 2.5518
[HyperMedDiff-Risk] Epoch 040 | Train 3.6537 | Val 2.5253
[HyperMedDiff-Risk] Epoch 041 | Train 3.6041 | Val 2.4232
[HyperMedDiff-Risk] Epoch 042 | Train 3.5802 | Val 2.3850
[HyperMedDiff-Risk] Epoch 043 | Train 3.5483 | Val 2.4179
[HyperMedDiff-Risk] Epoch 044 | Train 3.5138 | Val 2.2775
[HyperMedDiff-Risk] Epoch 045 | Train 3.4580 | Val 2.2234
[HyperMedDiff-Risk] Epoch 046 | Train 3.4608 | Val 2.2547
[HyperMedDiff-Risk] Epoch 047 | Train 3.4229 | Val 2.2873
[HyperMedDiff-Risk] Epoch 048 | Train 3.3730 | Val 2.2029
[HyperMedDiff-Risk] Epoch 049 | Train 3.3270 | Val 2.0984
[HyperMedDiff-Risk] Epoch 050 | Train 3.3112 | Val 2.1983
[HyperMedDiff-Risk] Epoch 051 | Train 3.2984 | Val 2.1634
[HyperMedDiff-Risk] Epoch 052 | Train 3.2925 | Val 2.0834
[HyperMedDiff-Risk] Epoch 053 | Train 3.2597 | Val 2.0650
[HyperMedDiff-Risk] Epoch 054 | Train 3.2493 | Val 2.1361
[HyperMedDiff-Risk] Epoch 055 | Train 3.2254 | Val 2.0786
[HyperMedDiff-Risk] Epoch 056 | Train 3.1973 | Val 2.1029
[HyperMedDiff-Risk] Epoch 057 | Train 3.1990 | Val 2.0213
[HyperMedDiff-Risk] Epoch 058 | Train 3.1690 | Val 2.1053
[HyperMedDiff-Risk] Epoch 059 | Train 3.1557 | Val 2.0193
[HyperMedDiff-Risk] Epoch 060 | Train 3.1465 | Val 2.0082
[HyperMedDiff-Risk] Epoch 061 | Train 3.1385 | Val 1.9972
[HyperMedDiff-Risk] Epoch 062 | Train 3.1324 | Val 1.9695
[HyperMedDiff-Risk] Epoch 063 | Train 3.1021 | Val 1.9714
[HyperMedDiff-Risk] Epoch 064 | Train 3.0766 | Val 1.8890
[HyperMedDiff-Risk] Epoch 065 | Train 3.0711 | Val 1.9806
[HyperMedDiff-Risk] Epoch 066 | Train 3.0837 | Val 1.9506
[HyperMedDiff-Risk] Epoch 067 | Train 3.0535 | Val 1.9292
[HyperMedDiff-Risk] Epoch 068 | Train 3.0699 | Val 1.9023
[HyperMedDiff-Risk] Epoch 069 | Train 3.0378 | Val 1.9140
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 3): 1.8890
[HyperMedDiff-Risk] Saved training curve plot to results/plots/03_LocalDiff.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8757537300634539,
  "auprc": 0.8127593753364201
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7797 ± 0.0051
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.8023
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0181
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2011 std=0.0406
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/03_LocalDiff_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/03_LocalDiff_umap.png
Saved checkpoint to results/checkpoints/03_LocalDiff.pt
[HyperMedDiff-Risk] ===== Experiment 4/18: 04_GlobalDiff_Stress =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8,
    16
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0316 | val=0.0291
[Pretrain] Epoch 02 | train=0.0281 | val=0.0297
[Pretrain] Epoch 03 | train=0.0278 | val=0.0286
[Pretrain] Epoch 04 | train=0.0264 | val=0.0286
[Pretrain] Epoch 05 | train=0.0276 | val=0.0274
[Pretrain] Epoch 06 | train=0.0277 | val=0.0270
[Pretrain] Epoch 07 | train=0.0267 | val=0.0250
[Pretrain] Epoch 08 | train=0.0255 | val=0.0253
[Pretrain] Epoch 09 | train=0.0238 | val=0.0261
[Pretrain] Epoch 10 | train=0.0255 | val=0.0258
[Pretrain] Epoch 11 | train=0.0240 | val=0.0257
[Pretrain] Epoch 12 | train=0.0247 | val=0.0241
[Pretrain] Epoch 13 | train=0.0237 | val=0.0240
[Pretrain] Epoch 14 | train=0.0246 | val=0.0227
[Pretrain] Epoch 15 | train=0.0222 | val=0.0239
[Pretrain] Epoch 16 | train=0.0235 | val=0.0224
[Pretrain] Epoch 17 | train=0.0230 | val=0.0222
[Pretrain] Epoch 18 | train=0.0235 | val=0.0212
[Pretrain] Epoch 19 | train=0.0238 | val=0.0217
[Pretrain] Epoch 20 | train=0.0218 | val=0.0231
[Pretrain] Epoch 21 | train=0.0217 | val=0.0225
[Pretrain] Epoch 22 | train=0.0207 | val=0.0207
[Pretrain] Epoch 23 | train=0.0205 | val=0.0212
[Pretrain] Epoch 24 | train=0.0207 | val=0.0205
[Pretrain] Epoch 25 | train=0.0210 | val=0.0203
[Pretrain] Epoch 26 | train=0.0210 | val=0.0198
[Pretrain] Epoch 27 | train=0.0194 | val=0.0192
[Pretrain] Epoch 28 | train=0.0196 | val=0.0206
[Pretrain] Epoch 29 | train=0.0196 | val=0.0202
[Pretrain] Epoch 30 | train=0.0200 | val=0.0193
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.6976 ± 0.0065
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.5029 | Val 10.9799
[HyperMedDiff-Risk] Epoch 002 | Train 11.3621 | Val 10.8329
[HyperMedDiff-Risk] Epoch 003 | Train 11.1184 | Val 10.4210
[HyperMedDiff-Risk] Epoch 004 | Train 10.5534 | Val 9.6471
[HyperMedDiff-Risk] Epoch 005 | Train 9.8233 | Val 8.8172
[HyperMedDiff-Risk] Epoch 006 | Train 9.1791 | Val 8.0693
[HyperMedDiff-Risk] Epoch 007 | Train 8.5878 | Val 7.5444
[HyperMedDiff-Risk] Epoch 008 | Train 8.1256 | Val 7.0470
[HyperMedDiff-Risk] Epoch 009 | Train 7.6916 | Val 6.5804
[HyperMedDiff-Risk] Epoch 010 | Train 7.3861 | Val 6.3296
[HyperMedDiff-Risk] Epoch 011 | Train 7.0573 | Val 5.8358
[HyperMedDiff-Risk] Epoch 012 | Train 6.7497 | Val 5.5892
[HyperMedDiff-Risk] Epoch 013 | Train 6.4703 | Val 5.3482
[HyperMedDiff-Risk] Epoch 014 | Train 6.2861 | Val 5.0069
[HyperMedDiff-Risk] Epoch 015 | Train 6.0632 | Val 4.9159
[HyperMedDiff-Risk] Epoch 016 | Train 5.8631 | Val 4.6581
[HyperMedDiff-Risk] Epoch 017 | Train 5.7374 | Val 4.5611
[HyperMedDiff-Risk] Epoch 018 | Train 5.5302 | Val 4.4293
[HyperMedDiff-Risk] Epoch 019 | Train 5.4058 | Val 4.2271
[HyperMedDiff-Risk] Epoch 020 | Train 5.2730 | Val 4.0469
[HyperMedDiff-Risk] Epoch 021 | Train 5.1267 | Val 3.8418
[HyperMedDiff-Risk] Epoch 022 | Train 5.0634 | Val 3.8806
[HyperMedDiff-Risk] Epoch 023 | Train 4.8777 | Val 3.6022
[HyperMedDiff-Risk] Epoch 024 | Train 4.8125 | Val 3.5118
[HyperMedDiff-Risk] Epoch 025 | Train 4.6266 | Val 3.4321
[HyperMedDiff-Risk] Epoch 026 | Train 4.5692 | Val 3.3546
[HyperMedDiff-Risk] Epoch 027 | Train 4.4949 | Val 3.3135
[HyperMedDiff-Risk] Epoch 028 | Train 4.4483 | Val 3.1638
[HyperMedDiff-Risk] Epoch 029 | Train 4.2966 | Val 3.1168
[HyperMedDiff-Risk] Epoch 030 | Train 4.2681 | Val 3.0894
[HyperMedDiff-Risk] Epoch 031 | Train 4.1788 | Val 3.0508
[HyperMedDiff-Risk] Epoch 032 | Train 4.0697 | Val 2.8822
[HyperMedDiff-Risk] Epoch 033 | Train 4.0367 | Val 2.8505
[HyperMedDiff-Risk] Epoch 034 | Train 3.9792 | Val 2.7775
[HyperMedDiff-Risk] Epoch 035 | Train 3.9316 | Val 2.7355
[HyperMedDiff-Risk] Epoch 036 | Train 3.8844 | Val 2.7002
[HyperMedDiff-Risk] Epoch 037 | Train 3.8514 | Val 2.5427
[HyperMedDiff-Risk] Epoch 038 | Train 3.8116 | Val 2.5473
[HyperMedDiff-Risk] Epoch 039 | Train 3.7458 | Val 2.5904
[HyperMedDiff-Risk] Epoch 040 | Train 3.7252 | Val 2.4508
[HyperMedDiff-Risk] Epoch 041 | Train 3.6411 | Val 2.4546
[HyperMedDiff-Risk] Epoch 042 | Train 3.6287 | Val 2.4700
[HyperMedDiff-Risk] Epoch 043 | Train 3.6330 | Val 2.4348
[HyperMedDiff-Risk] Epoch 044 | Train 3.5522 | Val 2.4444
[HyperMedDiff-Risk] Epoch 045 | Train 3.5262 | Val 2.3588
[HyperMedDiff-Risk] Epoch 046 | Train 3.4915 | Val 2.3430
[HyperMedDiff-Risk] Epoch 047 | Train 3.4904 | Val 2.3268
[HyperMedDiff-Risk] Epoch 048 | Train 3.4653 | Val 2.2781
[HyperMedDiff-Risk] Epoch 049 | Train 3.4192 | Val 2.2741
[HyperMedDiff-Risk] Epoch 050 | Train 3.3851 | Val 2.2150
[HyperMedDiff-Risk] Epoch 051 | Train 3.4033 | Val 2.2292
[HyperMedDiff-Risk] Epoch 052 | Train 3.3809 | Val 2.1451
[HyperMedDiff-Risk] Epoch 053 | Train 3.3605 | Val 2.1701
[HyperMedDiff-Risk] Epoch 054 | Train 3.3101 | Val 2.2182
[HyperMedDiff-Risk] Epoch 055 | Train 3.3022 | Val 2.1074
[HyperMedDiff-Risk] Epoch 056 | Train 3.2891 | Val 2.1748
[HyperMedDiff-Risk] Epoch 057 | Train 3.2820 | Val 2.1790
[HyperMedDiff-Risk] Epoch 058 | Train 3.2749 | Val 2.0788
[HyperMedDiff-Risk] Epoch 059 | Train 3.2530 | Val 2.0742
[HyperMedDiff-Risk] Epoch 060 | Train 3.2435 | Val 2.0792
[HyperMedDiff-Risk] Epoch 061 | Train 3.1979 | Val 2.0465
[HyperMedDiff-Risk] Epoch 062 | Train 3.1918 | Val 2.0308
[HyperMedDiff-Risk] Epoch 063 | Train 3.1757 | Val 2.0503
[HyperMedDiff-Risk] Epoch 064 | Train 3.1852 | Val 1.9912
[HyperMedDiff-Risk] Epoch 065 | Train 3.1656 | Val 2.0575
[HyperMedDiff-Risk] Epoch 066 | Train 3.1449 | Val 1.9893
[HyperMedDiff-Risk] Epoch 067 | Train 3.1384 | Val 2.0077
[HyperMedDiff-Risk] Epoch 068 | Train 3.1266 | Val 1.9917
[HyperMedDiff-Risk] Epoch 069 | Train 3.1167 | Val 1.9629
[HyperMedDiff-Risk] Epoch 070 | Train 3.0904 | Val 1.9831
[HyperMedDiff-Risk] Epoch 071 | Train 3.1017 | Val 1.9065
[HyperMedDiff-Risk] Epoch 072 | Train 3.0879 | Val 1.9755
[HyperMedDiff-Risk] Epoch 073 | Train 3.0654 | Val 1.9185
[HyperMedDiff-Risk] Epoch 074 | Train 3.0562 | Val 1.9919
[HyperMedDiff-Risk] Epoch 075 | Train 3.0508 | Val 1.9304
[HyperMedDiff-Risk] Epoch 076 | Train 3.0218 | Val 1.9137
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 4): 1.9065
[HyperMedDiff-Risk] Saved training curve plot to results/plots/04_GlobalDiff_Stress.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8763608300463043,
  "auprc": 0.8160355078540693
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.6976 ± 0.0065
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7246
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): -0.0011
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.1956 std=0.0347
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/04_GlobalDiff_Stress_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/04_GlobalDiff_Stress_umap.png
Saved checkpoint to results/checkpoints/04_GlobalDiff_Stress.pt
[HyperMedDiff-Risk] ===== Experiment 5/18: 05_NoHDD =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.0,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0029 | val=0.0028
[Pretrain] Epoch 02 | train=0.0028 | val=0.0026
[Pretrain] Epoch 03 | train=0.0026 | val=0.0025
[Pretrain] Epoch 04 | train=0.0025 | val=0.0024
[Pretrain] Epoch 05 | train=0.0024 | val=0.0023
[Pretrain] Epoch 06 | train=0.0023 | val=0.0022
[Pretrain] Epoch 07 | train=0.0022 | val=0.0021
[Pretrain] Epoch 08 | train=0.0021 | val=0.0020
[Pretrain] Epoch 09 | train=0.0020 | val=0.0019
[Pretrain] Epoch 10 | train=0.0019 | val=0.0018
[Pretrain] Epoch 11 | train=0.0018 | val=0.0017
[Pretrain] Epoch 12 | train=0.0017 | val=0.0016
[Pretrain] Epoch 13 | train=0.0016 | val=0.0016
[Pretrain] Epoch 14 | train=0.0016 | val=0.0015
[Pretrain] Epoch 15 | train=0.0015 | val=0.0014
[Pretrain] Epoch 16 | train=0.0014 | val=0.0014
[Pretrain] Epoch 17 | train=0.0014 | val=0.0013
[Pretrain] Epoch 18 | train=0.0013 | val=0.0013
[Pretrain] Epoch 19 | train=0.0013 | val=0.0012
[Pretrain] Epoch 20 | train=0.0012 | val=0.0012
[Pretrain] Epoch 21 | train=0.0012 | val=0.0012
[Pretrain] Epoch 22 | train=0.0012 | val=0.0011
[Pretrain] Epoch 23 | train=0.0011 | val=0.0011
[Pretrain] Epoch 24 | train=0.0011 | val=0.0011
[Pretrain] Epoch 25 | train=0.0011 | val=0.0011
[Pretrain] Epoch 26 | train=0.0011 | val=0.0011
[Pretrain] Epoch 27 | train=0.0011 | val=0.0011
[Pretrain] Epoch 28 | train=0.0011 | val=0.0011
[Pretrain] Epoch 29 | train=0.0011 | val=0.0011
[Pretrain] Epoch 30 | train=0.0011 | val=0.0011
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.0059 ± 0.0031
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.2857 | Val 10.9790
[HyperMedDiff-Risk] Epoch 002 | Train 11.3361 | Val 10.7746
[HyperMedDiff-Risk] Epoch 003 | Train 11.0178 | Val 10.2768
[HyperMedDiff-Risk] Epoch 004 | Train 10.3961 | Val 9.4647
[HyperMedDiff-Risk] Epoch 005 | Train 9.6596 | Val 8.6833
[HyperMedDiff-Risk] Epoch 006 | Train 9.0419 | Val 7.9940
[HyperMedDiff-Risk] Epoch 007 | Train 8.4624 | Val 7.3957
[HyperMedDiff-Risk] Epoch 008 | Train 8.0325 | Val 6.9240
[HyperMedDiff-Risk] Epoch 009 | Train 7.6434 | Val 6.4432
[HyperMedDiff-Risk] Epoch 010 | Train 7.3003 | Val 6.1582
[HyperMedDiff-Risk] Epoch 011 | Train 6.9485 | Val 5.6915
[HyperMedDiff-Risk] Epoch 012 | Train 6.6877 | Val 5.5095
[HyperMedDiff-Risk] Epoch 013 | Train 6.4575 | Val 5.1941
[HyperMedDiff-Risk] Epoch 014 | Train 6.2237 | Val 5.0571
[HyperMedDiff-Risk] Epoch 015 | Train 6.0259 | Val 4.8011
[HyperMedDiff-Risk] Epoch 016 | Train 5.8211 | Val 4.6139
[HyperMedDiff-Risk] Epoch 017 | Train 5.6652 | Val 4.4341
[HyperMedDiff-Risk] Epoch 018 | Train 5.4966 | Val 4.2676
[HyperMedDiff-Risk] Epoch 019 | Train 5.3600 | Val 4.1689
[HyperMedDiff-Risk] Epoch 020 | Train 5.2163 | Val 3.8819
[HyperMedDiff-Risk] Epoch 021 | Train 5.1180 | Val 3.7955
[HyperMedDiff-Risk] Epoch 022 | Train 4.9558 | Val 3.7163
[HyperMedDiff-Risk] Epoch 023 | Train 4.8336 | Val 3.6305
[HyperMedDiff-Risk] Epoch 024 | Train 4.7286 | Val 3.4511
[HyperMedDiff-Risk] Epoch 025 | Train 4.6656 | Val 3.4675
[HyperMedDiff-Risk] Epoch 026 | Train 4.5198 | Val 3.3132
[HyperMedDiff-Risk] Epoch 027 | Train 4.4494 | Val 3.2567
[HyperMedDiff-Risk] Epoch 028 | Train 4.3719 | Val 3.1403
[HyperMedDiff-Risk] Epoch 029 | Train 4.2761 | Val 3.0617
[HyperMedDiff-Risk] Epoch 030 | Train 4.2067 | Val 2.9203
[HyperMedDiff-Risk] Epoch 031 | Train 4.1623 | Val 2.9770
[HyperMedDiff-Risk] Epoch 032 | Train 4.0623 | Val 2.7834
[HyperMedDiff-Risk] Epoch 033 | Train 4.0199 | Val 2.7083
[HyperMedDiff-Risk] Epoch 034 | Train 3.9668 | Val 2.7109
[HyperMedDiff-Risk] Epoch 035 | Train 3.9072 | Val 2.7117
[HyperMedDiff-Risk] Epoch 036 | Train 3.8519 | Val 2.6372
[HyperMedDiff-Risk] Epoch 037 | Train 3.8054 | Val 2.5867
[HyperMedDiff-Risk] Epoch 038 | Train 3.7534 | Val 2.4891
[HyperMedDiff-Risk] Epoch 039 | Train 3.6979 | Val 2.4679
[HyperMedDiff-Risk] Epoch 040 | Train 3.6714 | Val 2.4747
[HyperMedDiff-Risk] Epoch 041 | Train 3.6439 | Val 2.3826
[HyperMedDiff-Risk] Epoch 042 | Train 3.5505 | Val 2.3643
[HyperMedDiff-Risk] Epoch 043 | Train 3.5486 | Val 2.3563
[HyperMedDiff-Risk] Epoch 044 | Train 3.4987 | Val 2.2981
[HyperMedDiff-Risk] Epoch 045 | Train 3.4432 | Val 2.3230
[HyperMedDiff-Risk] Epoch 046 | Train 3.4184 | Val 2.2512
[HyperMedDiff-Risk] Epoch 047 | Train 3.3762 | Val 2.2797
[HyperMedDiff-Risk] Epoch 048 | Train 3.3735 | Val 2.2383
[HyperMedDiff-Risk] Epoch 049 | Train 3.3175 | Val 2.0694
[HyperMedDiff-Risk] Epoch 050 | Train 3.3127 | Val 2.1283
[HyperMedDiff-Risk] Epoch 051 | Train 3.2746 | Val 2.0645
[HyperMedDiff-Risk] Epoch 052 | Train 3.2757 | Val 2.0098
[HyperMedDiff-Risk] Epoch 053 | Train 3.2542 | Val 2.0981
[HyperMedDiff-Risk] Epoch 054 | Train 3.2148 | Val 2.0711
[HyperMedDiff-Risk] Epoch 055 | Train 3.1840 | Val 2.0687
[HyperMedDiff-Risk] Epoch 056 | Train 3.2011 | Val 2.0419
[HyperMedDiff-Risk] Epoch 057 | Train 3.1533 | Val 1.9894
[HyperMedDiff-Risk] Epoch 058 | Train 3.1538 | Val 2.0025
[HyperMedDiff-Risk] Epoch 059 | Train 3.1290 | Val 1.9669
[HyperMedDiff-Risk] Epoch 060 | Train 3.0982 | Val 1.8788
[HyperMedDiff-Risk] Epoch 061 | Train 3.1094 | Val 1.9588
[HyperMedDiff-Risk] Epoch 062 | Train 3.0766 | Val 1.9351
[HyperMedDiff-Risk] Epoch 063 | Train 3.0727 | Val 1.8999
[HyperMedDiff-Risk] Epoch 064 | Train 3.0555 | Val 1.9895
[HyperMedDiff-Risk] Epoch 065 | Train 3.0326 | Val 1.9839
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 5): 1.8788
[HyperMedDiff-Risk] Saved training curve plot to results/plots/05_NoHDD.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8764843080089179,
  "auprc": 0.8159612676584995
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.0059 ± 0.0031
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.0093
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0025
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2830 std=0.0248
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/05_NoHDD_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/05_NoHDD_umap.png
Saved checkpoint to results/checkpoints/05_NoHDD.pt
[HyperMedDiff-Risk] ===== Experiment 6/18: 06_StrongHDD =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.1,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.1118 | val=0.1050
[Pretrain] Epoch 02 | train=0.1134 | val=0.1037
[Pretrain] Epoch 03 | train=0.1071 | val=0.1107
[Pretrain] Epoch 04 | train=0.1094 | val=0.1028
[Pretrain] Epoch 05 | train=0.1054 | val=0.1010
[Pretrain] Epoch 06 | train=0.1045 | val=0.0985
[Pretrain] Epoch 07 | train=0.0990 | val=0.0932
[Pretrain] Epoch 08 | train=0.0962 | val=0.0911
[Pretrain] Epoch 09 | train=0.0957 | val=0.0944
[Pretrain] Epoch 10 | train=0.0917 | val=0.0933
[Pretrain] Epoch 11 | train=0.0906 | val=0.0921
[Pretrain] Epoch 12 | train=0.0925 | val=0.0887
[Pretrain] Epoch 13 | train=0.0865 | val=0.0876
[Pretrain] Epoch 14 | train=0.0868 | val=0.0912
[Pretrain] Epoch 15 | train=0.0929 | val=0.0844
[Pretrain] Epoch 16 | train=0.0771 | val=0.0845
[Pretrain] Epoch 17 | train=0.0839 | val=0.0839
[Pretrain] Epoch 18 | train=0.0808 | val=0.0784
[Pretrain] Epoch 19 | train=0.0818 | val=0.0798
[Pretrain] Epoch 20 | train=0.0781 | val=0.0785
[Pretrain] Epoch 21 | train=0.0829 | val=0.0795
[Pretrain] Epoch 22 | train=0.0777 | val=0.0750
[Pretrain] Epoch 23 | train=0.0756 | val=0.0718
[Pretrain] Epoch 24 | train=0.0746 | val=0.0743
[Pretrain] Epoch 25 | train=0.0718 | val=0.0713
[Pretrain] Epoch 26 | train=0.0688 | val=0.0653
[Pretrain] Epoch 27 | train=0.0680 | val=0.0713
[Pretrain] Epoch 28 | train=0.0706 | val=0.0736
[Pretrain] Epoch 29 | train=0.0763 | val=0.0679
[Pretrain] Epoch 30 | train=0.0720 | val=0.0710
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7856 ± 0.0053
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.7339 | Val 10.9764
[HyperMedDiff-Risk] Epoch 002 | Train 11.3693 | Val 10.8176
[HyperMedDiff-Risk] Epoch 003 | Train 11.1239 | Val 10.4643
[HyperMedDiff-Risk] Epoch 004 | Train 10.6206 | Val 9.6973
[HyperMedDiff-Risk] Epoch 005 | Train 9.8724 | Val 8.8781
[HyperMedDiff-Risk] Epoch 006 | Train 9.1999 | Val 8.1196
[HyperMedDiff-Risk] Epoch 007 | Train 8.6734 | Val 7.6111
[HyperMedDiff-Risk] Epoch 008 | Train 8.1699 | Val 7.1171
[HyperMedDiff-Risk] Epoch 009 | Train 7.7627 | Val 6.6884
[HyperMedDiff-Risk] Epoch 010 | Train 7.3768 | Val 6.3654
[HyperMedDiff-Risk] Epoch 011 | Train 7.0930 | Val 5.9725
[HyperMedDiff-Risk] Epoch 012 | Train 6.8265 | Val 5.5569
[HyperMedDiff-Risk] Epoch 013 | Train 6.5597 | Val 5.4184
[HyperMedDiff-Risk] Epoch 014 | Train 6.3325 | Val 5.1488
[HyperMedDiff-Risk] Epoch 015 | Train 6.0915 | Val 4.8862
[HyperMedDiff-Risk] Epoch 016 | Train 5.8954 | Val 4.6799
[HyperMedDiff-Risk] Epoch 017 | Train 5.7411 | Val 4.5374
[HyperMedDiff-Risk] Epoch 018 | Train 5.5633 | Val 4.4216
[HyperMedDiff-Risk] Epoch 019 | Train 5.4088 | Val 4.2412
[HyperMedDiff-Risk] Epoch 020 | Train 5.2604 | Val 4.0687
[HyperMedDiff-Risk] Epoch 021 | Train 5.1363 | Val 3.8364
[HyperMedDiff-Risk] Epoch 022 | Train 5.0072 | Val 3.7593
[HyperMedDiff-Risk] Epoch 023 | Train 4.8763 | Val 3.5996
[HyperMedDiff-Risk] Epoch 024 | Train 4.7630 | Val 3.4855
[HyperMedDiff-Risk] Epoch 025 | Train 4.6250 | Val 3.4255
[HyperMedDiff-Risk] Epoch 026 | Train 4.5782 | Val 3.2824
[HyperMedDiff-Risk] Epoch 027 | Train 4.4385 | Val 3.2063
[HyperMedDiff-Risk] Epoch 028 | Train 4.3838 | Val 3.1007
[HyperMedDiff-Risk] Epoch 029 | Train 4.2872 | Val 3.0081
[HyperMedDiff-Risk] Epoch 030 | Train 4.1928 | Val 2.9589
[HyperMedDiff-Risk] Epoch 031 | Train 4.1201 | Val 2.8511
[HyperMedDiff-Risk] Epoch 032 | Train 4.0972 | Val 2.8839
[HyperMedDiff-Risk] Epoch 033 | Train 4.0255 | Val 2.7933
[HyperMedDiff-Risk] Epoch 034 | Train 3.9467 | Val 2.7824
[HyperMedDiff-Risk] Epoch 035 | Train 3.8850 | Val 2.6485
[HyperMedDiff-Risk] Epoch 036 | Train 3.8229 | Val 2.5658
[HyperMedDiff-Risk] Epoch 037 | Train 3.7930 | Val 2.6202
[HyperMedDiff-Risk] Epoch 038 | Train 3.7750 | Val 2.4807
[HyperMedDiff-Risk] Epoch 039 | Train 3.7121 | Val 2.5250
[HyperMedDiff-Risk] Epoch 040 | Train 3.6746 | Val 2.5014
[HyperMedDiff-Risk] Epoch 041 | Train 3.6344 | Val 2.3908
[HyperMedDiff-Risk] Epoch 042 | Train 3.5831 | Val 2.4114
[HyperMedDiff-Risk] Epoch 043 | Train 3.5282 | Val 2.3698
[HyperMedDiff-Risk] Epoch 044 | Train 3.5127 | Val 2.3647
[HyperMedDiff-Risk] Epoch 045 | Train 3.4918 | Val 2.3028
[HyperMedDiff-Risk] Epoch 046 | Train 3.4573 | Val 2.3391
[HyperMedDiff-Risk] Epoch 047 | Train 3.4215 | Val 2.3362
[HyperMedDiff-Risk] Epoch 048 | Train 3.4114 | Val 2.2683
[HyperMedDiff-Risk] Epoch 049 | Train 3.3804 | Val 2.2063
[HyperMedDiff-Risk] Epoch 050 | Train 3.3936 | Val 2.1879
[HyperMedDiff-Risk] Epoch 051 | Train 3.3636 | Val 2.2258
[HyperMedDiff-Risk] Epoch 052 | Train 3.3537 | Val 2.1906
[HyperMedDiff-Risk] Epoch 053 | Train 3.3180 | Val 2.1614
[HyperMedDiff-Risk] Epoch 054 | Train 3.3099 | Val 2.1577
[HyperMedDiff-Risk] Epoch 055 | Train 3.2724 | Val 2.0999
[HyperMedDiff-Risk] Epoch 056 | Train 3.2567 | Val 2.1215
[HyperMedDiff-Risk] Epoch 057 | Train 3.2238 | Val 2.0739
[HyperMedDiff-Risk] Epoch 058 | Train 3.2012 | Val 2.1038
[HyperMedDiff-Risk] Epoch 059 | Train 3.2101 | Val 2.0606
[HyperMedDiff-Risk] Epoch 060 | Train 3.1931 | Val 2.0509
[HyperMedDiff-Risk] Epoch 061 | Train 3.1789 | Val 2.0121
[HyperMedDiff-Risk] Epoch 062 | Train 3.1717 | Val 2.0787
[HyperMedDiff-Risk] Epoch 063 | Train 3.1385 | Val 1.9743
[HyperMedDiff-Risk] Epoch 064 | Train 3.1608 | Val 1.9953
[HyperMedDiff-Risk] Epoch 065 | Train 3.1173 | Val 1.9886
[HyperMedDiff-Risk] Epoch 066 | Train 3.1128 | Val 1.9283
[HyperMedDiff-Risk] Epoch 067 | Train 3.0912 | Val 1.9161
[HyperMedDiff-Risk] Epoch 068 | Train 3.0972 | Val 1.9605
[HyperMedDiff-Risk] Epoch 069 | Train 3.0616 | Val 1.9153
[HyperMedDiff-Risk] Epoch 070 | Train 3.0392 | Val 1.9236
[HyperMedDiff-Risk] Epoch 071 | Train 3.0499 | Val 1.9535
[HyperMedDiff-Risk] Epoch 072 | Train 3.0402 | Val 1.9066
[HyperMedDiff-Risk] Epoch 073 | Train 3.0124 | Val 1.8549
[HyperMedDiff-Risk] Epoch 074 | Train 3.0027 | Val 1.8996
[HyperMedDiff-Risk] Epoch 075 | Train 3.0289 | Val 1.8998
[HyperMedDiff-Risk] Epoch 076 | Train 2.9833 | Val 1.8658
[HyperMedDiff-Risk] Epoch 077 | Train 2.9670 | Val 1.8185
[HyperMedDiff-Risk] Epoch 078 | Train 2.9965 | Val 1.8157
[HyperMedDiff-Risk] Epoch 079 | Train 2.9829 | Val 1.8273
[HyperMedDiff-Risk] Epoch 080 | Train 2.9634 | Val 1.7971
[HyperMedDiff-Risk] Epoch 081 | Train 2.9648 | Val 1.8234
[HyperMedDiff-Risk] Epoch 082 | Train 2.9673 | Val 1.8164
[HyperMedDiff-Risk] Epoch 083 | Train 2.9560 | Val 1.7873
[HyperMedDiff-Risk] Epoch 084 | Train 2.9756 | Val 1.8094
[HyperMedDiff-Risk] Epoch 085 | Train 2.9589 | Val 1.8014
[HyperMedDiff-Risk] Epoch 086 | Train 2.9552 | Val 1.8095
[HyperMedDiff-Risk] Epoch 087 | Train 2.9593 | Val 1.7957
[HyperMedDiff-Risk] Epoch 088 | Train 2.9450 | Val 1.7984
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 6): 1.7873
[HyperMedDiff-Risk] Saved training curve plot to results/plots/06_StrongHDD.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8711953352769679,
  "auprc": 0.7991750810084123
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7856 ± 0.0053
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.8154
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0119
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.1731 std=0.0369
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/06_StrongHDD_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/06_StrongHDD_umap.png
Saved checkpoint to results/checkpoints/06_StrongHDD.pt
[HyperMedDiff-Risk] ===== Experiment 7/18: 07_HighDropout =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.5,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0244 | val=0.0233
[Pretrain] Epoch 02 | train=0.0255 | val=0.0243
[Pretrain] Epoch 03 | train=0.0236 | val=0.0232
[Pretrain] Epoch 04 | train=0.0223 | val=0.0219
[Pretrain] Epoch 05 | train=0.0212 | val=0.0217
[Pretrain] Epoch 06 | train=0.0215 | val=0.0220
[Pretrain] Epoch 07 | train=0.0220 | val=0.0206
[Pretrain] Epoch 08 | train=0.0202 | val=0.0202
[Pretrain] Epoch 09 | train=0.0203 | val=0.0201
[Pretrain] Epoch 10 | train=0.0201 | val=0.0203
[Pretrain] Epoch 11 | train=0.0191 | val=0.0193
[Pretrain] Epoch 12 | train=0.0203 | val=0.0197
[Pretrain] Epoch 13 | train=0.0193 | val=0.0185
[Pretrain] Epoch 14 | train=0.0185 | val=0.0176
[Pretrain] Epoch 15 | train=0.0194 | val=0.0175
[Pretrain] Epoch 16 | train=0.0177 | val=0.0186
[Pretrain] Epoch 17 | train=0.0173 | val=0.0160
[Pretrain] Epoch 18 | train=0.0177 | val=0.0171
[Pretrain] Epoch 19 | train=0.0171 | val=0.0161
[Pretrain] Epoch 20 | train=0.0168 | val=0.0181
[Pretrain] Epoch 21 | train=0.0164 | val=0.0161
[Pretrain] Epoch 22 | train=0.0159 | val=0.0166
[Pretrain] Epoch 23 | train=0.0171 | val=0.0171
[Pretrain] Epoch 24 | train=0.0162 | val=0.0152
[Pretrain] Epoch 25 | train=0.0167 | val=0.0168
[Pretrain] Epoch 26 | train=0.0155 | val=0.0157
[Pretrain] Epoch 27 | train=0.0154 | val=0.0153
[Pretrain] Epoch 28 | train=0.0149 | val=0.0152
[Pretrain] Epoch 29 | train=0.0158 | val=0.0160
[Pretrain] Epoch 30 | train=0.0158 | val=0.0145
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7698 ± 0.0061
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.7588 | Val 10.9736
[HyperMedDiff-Risk] Epoch 002 | Train 11.3870 | Val 10.8117
[HyperMedDiff-Risk] Epoch 003 | Train 11.1034 | Val 10.4065
[HyperMedDiff-Risk] Epoch 004 | Train 10.5650 | Val 9.6665
[HyperMedDiff-Risk] Epoch 005 | Train 9.8681 | Val 8.8468
[HyperMedDiff-Risk] Epoch 006 | Train 9.2182 | Val 8.1446
[HyperMedDiff-Risk] Epoch 007 | Train 8.6712 | Val 7.6273
[HyperMedDiff-Risk] Epoch 008 | Train 8.2020 | Val 7.1252
[HyperMedDiff-Risk] Epoch 009 | Train 7.8068 | Val 6.6892
[HyperMedDiff-Risk] Epoch 010 | Train 7.4104 | Val 6.2161
[HyperMedDiff-Risk] Epoch 011 | Train 7.1306 | Val 5.8933
[HyperMedDiff-Risk] Epoch 012 | Train 6.8646 | Val 5.7078
[HyperMedDiff-Risk] Epoch 013 | Train 6.5607 | Val 5.4213
[HyperMedDiff-Risk] Epoch 014 | Train 6.3701 | Val 5.1415
[HyperMedDiff-Risk] Epoch 015 | Train 6.1412 | Val 5.0051
[HyperMedDiff-Risk] Epoch 016 | Train 6.0064 | Val 4.8108
[HyperMedDiff-Risk] Epoch 017 | Train 5.8525 | Val 4.6158
[HyperMedDiff-Risk] Epoch 018 | Train 5.6611 | Val 4.4530
[HyperMedDiff-Risk] Epoch 019 | Train 5.5408 | Val 4.2513
[HyperMedDiff-Risk] Epoch 020 | Train 5.4118 | Val 4.1834
[HyperMedDiff-Risk] Epoch 021 | Train 5.2534 | Val 3.9930
[HyperMedDiff-Risk] Epoch 022 | Train 5.1204 | Val 3.9012
[HyperMedDiff-Risk] Epoch 023 | Train 5.0298 | Val 3.7956
[HyperMedDiff-Risk] Epoch 024 | Train 4.9646 | Val 3.6371
[HyperMedDiff-Risk] Epoch 025 | Train 4.7908 | Val 3.4863
[HyperMedDiff-Risk] Epoch 026 | Train 4.6925 | Val 3.5095
[HyperMedDiff-Risk] Epoch 027 | Train 4.6168 | Val 3.3526
[HyperMedDiff-Risk] Epoch 028 | Train 4.5201 | Val 3.3067
[HyperMedDiff-Risk] Epoch 029 | Train 4.4002 | Val 3.0980
[HyperMedDiff-Risk] Epoch 030 | Train 4.3341 | Val 3.1407
[HyperMedDiff-Risk] Epoch 031 | Train 4.2866 | Val 3.0125
[HyperMedDiff-Risk] Epoch 032 | Train 4.1731 | Val 2.9451
[HyperMedDiff-Risk] Epoch 033 | Train 4.1510 | Val 2.8927
[HyperMedDiff-Risk] Epoch 034 | Train 4.0349 | Val 2.8396
[HyperMedDiff-Risk] Epoch 035 | Train 4.0167 | Val 2.8546
[HyperMedDiff-Risk] Epoch 036 | Train 3.9022 | Val 2.7122
[HyperMedDiff-Risk] Epoch 037 | Train 3.9030 | Val 2.6977
[HyperMedDiff-Risk] Epoch 038 | Train 3.8609 | Val 2.6528
[HyperMedDiff-Risk] Epoch 039 | Train 3.7994 | Val 2.6115
[HyperMedDiff-Risk] Epoch 040 | Train 3.7549 | Val 2.6218
[HyperMedDiff-Risk] Epoch 041 | Train 3.6759 | Val 2.5505
[HyperMedDiff-Risk] Epoch 042 | Train 3.6914 | Val 2.4641
[HyperMedDiff-Risk] Epoch 043 | Train 3.6185 | Val 2.4511
[HyperMedDiff-Risk] Epoch 044 | Train 3.6374 | Val 2.4338
[HyperMedDiff-Risk] Epoch 045 | Train 3.5478 | Val 2.4765
[HyperMedDiff-Risk] Epoch 046 | Train 3.5341 | Val 2.4124
[HyperMedDiff-Risk] Epoch 047 | Train 3.4697 | Val 2.2356
[HyperMedDiff-Risk] Epoch 048 | Train 3.4388 | Val 2.3148
[HyperMedDiff-Risk] Epoch 049 | Train 3.4229 | Val 2.2520
[HyperMedDiff-Risk] Epoch 050 | Train 3.3905 | Val 2.2104
[HyperMedDiff-Risk] Epoch 051 | Train 3.3534 | Val 2.2158
[HyperMedDiff-Risk] Epoch 052 | Train 3.3422 | Val 2.1855
[HyperMedDiff-Risk] Epoch 053 | Train 3.3305 | Val 2.1559
[HyperMedDiff-Risk] Epoch 054 | Train 3.3169 | Val 2.1513
[HyperMedDiff-Risk] Epoch 055 | Train 3.2909 | Val 2.1107
[HyperMedDiff-Risk] Epoch 056 | Train 3.2731 | Val 2.1067
[HyperMedDiff-Risk] Epoch 057 | Train 3.2531 | Val 2.0868
[HyperMedDiff-Risk] Epoch 058 | Train 3.2196 | Val 2.0565
[HyperMedDiff-Risk] Epoch 059 | Train 3.1992 | Val 2.1368
[HyperMedDiff-Risk] Epoch 060 | Train 3.1744 | Val 2.0520
[HyperMedDiff-Risk] Epoch 061 | Train 3.1350 | Val 1.9720
[HyperMedDiff-Risk] Epoch 062 | Train 3.1360 | Val 2.1005
[HyperMedDiff-Risk] Epoch 063 | Train 3.1308 | Val 2.0277
[HyperMedDiff-Risk] Epoch 064 | Train 3.1333 | Val 1.9601
[HyperMedDiff-Risk] Epoch 065 | Train 3.0897 | Val 1.9984
[HyperMedDiff-Risk] Epoch 066 | Train 3.0871 | Val 1.9267
[HyperMedDiff-Risk] Epoch 067 | Train 3.0803 | Val 1.9643
[HyperMedDiff-Risk] Epoch 068 | Train 3.0527 | Val 1.9128
[HyperMedDiff-Risk] Epoch 069 | Train 3.0589 | Val 1.9197
[HyperMedDiff-Risk] Epoch 070 | Train 3.0103 | Val 1.9107
[HyperMedDiff-Risk] Epoch 071 | Train 3.0372 | Val 1.9111
[HyperMedDiff-Risk] Epoch 072 | Train 3.0292 | Val 1.8693
[HyperMedDiff-Risk] Epoch 073 | Train 3.0222 | Val 1.8979
[HyperMedDiff-Risk] Epoch 074 | Train 3.0009 | Val 1.8403
[HyperMedDiff-Risk] Epoch 075 | Train 2.9915 | Val 1.8527
[HyperMedDiff-Risk] Epoch 076 | Train 3.0159 | Val 1.8216
[HyperMedDiff-Risk] Epoch 077 | Train 3.0116 | Val 1.8220
[HyperMedDiff-Risk] Epoch 078 | Train 2.9624 | Val 1.8406
[HyperMedDiff-Risk] Epoch 079 | Train 2.9957 | Val 1.8386
[HyperMedDiff-Risk] Epoch 080 | Train 2.9505 | Val 1.8763
[HyperMedDiff-Risk] Epoch 081 | Train 2.9801 | Val 1.8651
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 7): 1.8216
[HyperMedDiff-Risk] Saved training curve plot to results/plots/07_HighDropout.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8738775510204082,
  "auprc": 0.8089923549193165
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7698 ± 0.0061
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7633
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0167
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2035 std=0.0400
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/07_HighDropout_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/07_HighDropout_umap.png
Saved checkpoint to results/checkpoints/07_HighDropout.pt
[HyperMedDiff-Risk] ===== Experiment 8/18: 08_SmallDim =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 64,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0261 | val=0.0242
[Pretrain] Epoch 02 | train=0.0246 | val=0.0234
[Pretrain] Epoch 03 | train=0.0229 | val=0.0234
[Pretrain] Epoch 04 | train=0.0241 | val=0.0232
[Pretrain] Epoch 05 | train=0.0238 | val=0.0215
[Pretrain] Epoch 06 | train=0.0234 | val=0.0227
[Pretrain] Epoch 07 | train=0.0227 | val=0.0224
[Pretrain] Epoch 08 | train=0.0235 | val=0.0214
[Pretrain] Epoch 09 | train=0.0221 | val=0.0224
[Pretrain] Epoch 10 | train=0.0216 | val=0.0211
[Pretrain] Epoch 11 | train=0.0210 | val=0.0208
[Pretrain] Epoch 12 | train=0.0202 | val=0.0211
[Pretrain] Epoch 13 | train=0.0209 | val=0.0199
[Pretrain] Epoch 14 | train=0.0212 | val=0.0190
[Pretrain] Epoch 15 | train=0.0215 | val=0.0198
[Pretrain] Epoch 16 | train=0.0206 | val=0.0195
[Pretrain] Epoch 17 | train=0.0201 | val=0.0191
[Pretrain] Epoch 18 | train=0.0183 | val=0.0209
[Pretrain] Epoch 19 | train=0.0195 | val=0.0199
[Pretrain] Epoch 20 | train=0.0194 | val=0.0188
[Pretrain] Epoch 21 | train=0.0199 | val=0.0172
[Pretrain] Epoch 22 | train=0.0185 | val=0.0201
[Pretrain] Epoch 23 | train=0.0178 | val=0.0183
[Pretrain] Epoch 24 | train=0.0185 | val=0.0183
[Pretrain] Epoch 25 | train=0.0186 | val=0.0191
[Pretrain] Epoch 26 | train=0.0170 | val=0.0181
[Pretrain] Epoch 27 | train=0.0184 | val=0.0170
[Pretrain] Epoch 28 | train=0.0166 | val=0.0166
[Pretrain] Epoch 29 | train=0.0174 | val=0.0175
[Pretrain] Epoch 30 | train=0.0174 | val=0.0162
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.5384 ± 0.0134
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 12.8336 | Val 10.8390
[HyperMedDiff-Risk] Epoch 002 | Train 10.8911 | Val 10.1104
[HyperMedDiff-Risk] Epoch 003 | Train 10.0594 | Val 9.0758
[HyperMedDiff-Risk] Epoch 004 | Train 9.2420 | Val 8.1660
[HyperMedDiff-Risk] Epoch 005 | Train 8.6182 | Val 7.5426
[HyperMedDiff-Risk] Epoch 006 | Train 8.0776 | Val 6.9721
[HyperMedDiff-Risk] Epoch 007 | Train 7.6084 | Val 6.5511
[HyperMedDiff-Risk] Epoch 008 | Train 7.2319 | Val 6.0418
[HyperMedDiff-Risk] Epoch 009 | Train 6.8630 | Val 5.7452
[HyperMedDiff-Risk] Epoch 010 | Train 6.6300 | Val 5.5164
[HyperMedDiff-Risk] Epoch 011 | Train 6.3656 | Val 5.1391
[HyperMedDiff-Risk] Epoch 012 | Train 6.1070 | Val 4.9973
[HyperMedDiff-Risk] Epoch 013 | Train 5.8698 | Val 4.7160
[HyperMedDiff-Risk] Epoch 014 | Train 5.7455 | Val 4.5854
[HyperMedDiff-Risk] Epoch 015 | Train 5.5518 | Val 4.3477
[HyperMedDiff-Risk] Epoch 016 | Train 5.4020 | Val 4.2071
[HyperMedDiff-Risk] Epoch 017 | Train 5.2359 | Val 4.1028
[HyperMedDiff-Risk] Epoch 018 | Train 5.1276 | Val 3.9395
[HyperMedDiff-Risk] Epoch 019 | Train 4.9936 | Val 3.8010
[HyperMedDiff-Risk] Epoch 020 | Train 4.9036 | Val 3.7117
[HyperMedDiff-Risk] Epoch 021 | Train 4.8011 | Val 3.5782
[HyperMedDiff-Risk] Epoch 022 | Train 4.6757 | Val 3.4286
[HyperMedDiff-Risk] Epoch 023 | Train 4.5607 | Val 3.3851
[HyperMedDiff-Risk] Epoch 024 | Train 4.4219 | Val 3.3299
[HyperMedDiff-Risk] Epoch 025 | Train 4.3784 | Val 3.2240
[HyperMedDiff-Risk] Epoch 026 | Train 4.2833 | Val 3.0710
[HyperMedDiff-Risk] Epoch 027 | Train 4.1818 | Val 3.0122
[HyperMedDiff-Risk] Epoch 028 | Train 4.1237 | Val 2.9091
[HyperMedDiff-Risk] Epoch 029 | Train 4.0490 | Val 2.8125
[HyperMedDiff-Risk] Epoch 030 | Train 3.9708 | Val 2.7381
[HyperMedDiff-Risk] Epoch 031 | Train 3.9094 | Val 2.7095
[HyperMedDiff-Risk] Epoch 032 | Train 3.8710 | Val 2.7272
[HyperMedDiff-Risk] Epoch 033 | Train 3.7499 | Val 2.6350
[HyperMedDiff-Risk] Epoch 034 | Train 3.7147 | Val 2.5440
[HyperMedDiff-Risk] Epoch 035 | Train 3.6554 | Val 2.5098
[HyperMedDiff-Risk] Epoch 036 | Train 3.6509 | Val 2.5460
[HyperMedDiff-Risk] Epoch 037 | Train 3.6062 | Val 2.4905
[HyperMedDiff-Risk] Epoch 038 | Train 3.5706 | Val 2.4579
[HyperMedDiff-Risk] Epoch 039 | Train 3.5342 | Val 2.3983
[HyperMedDiff-Risk] Epoch 040 | Train 3.4692 | Val 2.3588
[HyperMedDiff-Risk] Epoch 041 | Train 3.4804 | Val 2.3533
[HyperMedDiff-Risk] Epoch 042 | Train 3.4382 | Val 2.3311
[HyperMedDiff-Risk] Epoch 043 | Train 3.4075 | Val 2.3189
[HyperMedDiff-Risk] Epoch 044 | Train 3.3676 | Val 2.2738
[HyperMedDiff-Risk] Epoch 045 | Train 3.3887 | Val 2.2226
[HyperMedDiff-Risk] Epoch 046 | Train 3.3068 | Val 2.1992
[HyperMedDiff-Risk] Epoch 047 | Train 3.2866 | Val 2.2309
[HyperMedDiff-Risk] Epoch 048 | Train 3.2788 | Val 2.1427
[HyperMedDiff-Risk] Epoch 049 | Train 3.2364 | Val 2.1218
[HyperMedDiff-Risk] Epoch 050 | Train 3.2125 | Val 2.0699
[HyperMedDiff-Risk] Epoch 051 | Train 3.2085 | Val 2.0476
[HyperMedDiff-Risk] Epoch 052 | Train 3.1876 | Val 2.0586
[HyperMedDiff-Risk] Epoch 053 | Train 3.1606 | Val 2.1002
[HyperMedDiff-Risk] Epoch 054 | Train 3.1422 | Val 2.0382
[HyperMedDiff-Risk] Epoch 055 | Train 3.1187 | Val 2.0730
[HyperMedDiff-Risk] Epoch 056 | Train 3.0911 | Val 1.9983
[HyperMedDiff-Risk] Epoch 057 | Train 3.0962 | Val 1.9715
[HyperMedDiff-Risk] Epoch 058 | Train 3.0636 | Val 2.0064
[HyperMedDiff-Risk] Epoch 059 | Train 3.0508 | Val 1.9718
[HyperMedDiff-Risk] Epoch 060 | Train 3.0582 | Val 1.9961
[HyperMedDiff-Risk] Epoch 061 | Train 3.0297 | Val 1.9148
[HyperMedDiff-Risk] Epoch 062 | Train 2.9981 | Val 1.9288
[HyperMedDiff-Risk] Epoch 063 | Train 3.0101 | Val 1.9202
[HyperMedDiff-Risk] Epoch 064 | Train 2.9713 | Val 1.9369
[HyperMedDiff-Risk] Epoch 065 | Train 2.9916 | Val 1.9575
[HyperMedDiff-Risk] Epoch 066 | Train 2.9590 | Val 1.8715
[HyperMedDiff-Risk] Epoch 067 | Train 2.9757 | Val 1.8737
[HyperMedDiff-Risk] Epoch 068 | Train 2.9458 | Val 1.8734
[HyperMedDiff-Risk] Epoch 069 | Train 2.9179 | Val 1.9211
[HyperMedDiff-Risk] Epoch 070 | Train 2.9453 | Val 1.9092
[HyperMedDiff-Risk] Epoch 071 | Train 2.9252 | Val 1.8736
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 8): 1.8715
[HyperMedDiff-Risk] Saved training curve plot to results/plots/08_SmallDim.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.862435259818213,
  "auprc": 0.790687523598171
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.5384 ± 0.0134
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.5827
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0254
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.1648 std=0.0254
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/08_SmallDim_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/08_SmallDim_umap.png
Saved checkpoint to results/checkpoints/08_SmallDim.pt
[HyperMedDiff-Risk] ===== Experiment 9/18: 09_NoSynthRisk =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 0.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0243 | val=0.0254
[Pretrain] Epoch 02 | train=0.0241 | val=0.0246
[Pretrain] Epoch 03 | train=0.0244 | val=0.0227
[Pretrain] Epoch 04 | train=0.0240 | val=0.0219
[Pretrain] Epoch 05 | train=0.0227 | val=0.0218
[Pretrain] Epoch 06 | train=0.0231 | val=0.0221
[Pretrain] Epoch 07 | train=0.0226 | val=0.0209
[Pretrain] Epoch 08 | train=0.0218 | val=0.0207
[Pretrain] Epoch 09 | train=0.0201 | val=0.0205
[Pretrain] Epoch 10 | train=0.0204 | val=0.0198
[Pretrain] Epoch 11 | train=0.0197 | val=0.0195
[Pretrain] Epoch 12 | train=0.0192 | val=0.0191
[Pretrain] Epoch 13 | train=0.0189 | val=0.0191
[Pretrain] Epoch 14 | train=0.0187 | val=0.0190
[Pretrain] Epoch 15 | train=0.0191 | val=0.0172
[Pretrain] Epoch 16 | train=0.0184 | val=0.0174
[Pretrain] Epoch 17 | train=0.0180 | val=0.0195
[Pretrain] Epoch 18 | train=0.0180 | val=0.0187
[Pretrain] Epoch 19 | train=0.0164 | val=0.0174
[Pretrain] Epoch 20 | train=0.0171 | val=0.0175
[Pretrain] Epoch 21 | train=0.0169 | val=0.0168
[Pretrain] Epoch 22 | train=0.0160 | val=0.0152
[Pretrain] Epoch 23 | train=0.0163 | val=0.0159
[Pretrain] Epoch 24 | train=0.0165 | val=0.0153
[Pretrain] Epoch 25 | train=0.0148 | val=0.0159
[Pretrain] Epoch 26 | train=0.0157 | val=0.0161
[Pretrain] Epoch 27 | train=0.0156 | val=0.0151
[Pretrain] Epoch 28 | train=0.0142 | val=0.0148
[Pretrain] Epoch 29 | train=0.0145 | val=0.0155
[Pretrain] Epoch 30 | train=0.0141 | val=0.0139
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7632 ± 0.0028
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 12.9369 | Val 10.2953
[HyperMedDiff-Risk] Epoch 002 | Train 10.7253 | Val 10.1783
[HyperMedDiff-Risk] Epoch 003 | Train 10.5132 | Val 9.8314
[HyperMedDiff-Risk] Epoch 004 | Train 9.9639 | Val 9.0511
[HyperMedDiff-Risk] Epoch 005 | Train 9.2659 | Val 8.2512
[HyperMedDiff-Risk] Epoch 006 | Train 8.5960 | Val 7.5199
[HyperMedDiff-Risk] Epoch 007 | Train 8.0149 | Val 6.9410
[HyperMedDiff-Risk] Epoch 008 | Train 7.5600 | Val 6.3751
[HyperMedDiff-Risk] Epoch 009 | Train 7.1149 | Val 5.9807
[HyperMedDiff-Risk] Epoch 010 | Train 6.7344 | Val 5.6377
[HyperMedDiff-Risk] Epoch 011 | Train 6.4409 | Val 5.2551
[HyperMedDiff-Risk] Epoch 012 | Train 6.1921 | Val 4.9415
[HyperMedDiff-Risk] Epoch 013 | Train 5.9376 | Val 4.6534
[HyperMedDiff-Risk] Epoch 014 | Train 5.6859 | Val 4.4496
[HyperMedDiff-Risk] Epoch 015 | Train 5.4506 | Val 4.2694
[HyperMedDiff-Risk] Epoch 016 | Train 5.2287 | Val 3.9993
[HyperMedDiff-Risk] Epoch 017 | Train 5.0784 | Val 3.8703
[HyperMedDiff-Risk] Epoch 018 | Train 4.9452 | Val 3.6586
[HyperMedDiff-Risk] Epoch 019 | Train 4.7757 | Val 3.4795
[HyperMedDiff-Risk] Epoch 020 | Train 4.5998 | Val 3.3086
[HyperMedDiff-Risk] Epoch 021 | Train 4.5131 | Val 3.2224
[HyperMedDiff-Risk] Epoch 022 | Train 4.3806 | Val 3.1470
[HyperMedDiff-Risk] Epoch 023 | Train 4.2744 | Val 3.0065
[HyperMedDiff-Risk] Epoch 024 | Train 4.1456 | Val 2.9827
[HyperMedDiff-Risk] Epoch 025 | Train 4.0346 | Val 2.8051
[HyperMedDiff-Risk] Epoch 026 | Train 3.9755 | Val 2.7506
[HyperMedDiff-Risk] Epoch 027 | Train 3.8705 | Val 2.5700
[HyperMedDiff-Risk] Epoch 028 | Train 3.8000 | Val 2.5249
[HyperMedDiff-Risk] Epoch 029 | Train 3.7000 | Val 2.4815
[HyperMedDiff-Risk] Epoch 030 | Train 3.6889 | Val 2.3499
[HyperMedDiff-Risk] Epoch 031 | Train 3.5932 | Val 2.3194
[HyperMedDiff-Risk] Epoch 032 | Train 3.5095 | Val 2.2400
[HyperMedDiff-Risk] Epoch 033 | Train 3.4735 | Val 2.1417
[HyperMedDiff-Risk] Epoch 034 | Train 3.4167 | Val 2.1072
[HyperMedDiff-Risk] Epoch 035 | Train 3.3171 | Val 2.1203
[HyperMedDiff-Risk] Epoch 036 | Train 3.3450 | Val 2.0619
[HyperMedDiff-Risk] Epoch 037 | Train 3.2165 | Val 2.0374
[HyperMedDiff-Risk] Epoch 038 | Train 3.2046 | Val 2.0482
[HyperMedDiff-Risk] Epoch 039 | Train 3.1290 | Val 1.9197
[HyperMedDiff-Risk] Epoch 040 | Train 3.1399 | Val 1.9286
[HyperMedDiff-Risk] Epoch 041 | Train 3.0846 | Val 1.7824
[HyperMedDiff-Risk] Epoch 042 | Train 3.0591 | Val 1.8918
[HyperMedDiff-Risk] Epoch 043 | Train 2.9945 | Val 1.8336
[HyperMedDiff-Risk] Epoch 044 | Train 2.9764 | Val 1.7742
[HyperMedDiff-Risk] Epoch 045 | Train 2.9469 | Val 1.7455
[HyperMedDiff-Risk] Epoch 046 | Train 2.9001 | Val 1.7737
[HyperMedDiff-Risk] Epoch 047 | Train 2.8726 | Val 1.6609
[HyperMedDiff-Risk] Epoch 048 | Train 2.8331 | Val 1.7187
[HyperMedDiff-Risk] Epoch 049 | Train 2.8285 | Val 1.6986
[HyperMedDiff-Risk] Epoch 050 | Train 2.8100 | Val 1.6562
[HyperMedDiff-Risk] Epoch 051 | Train 2.7539 | Val 1.6365
[HyperMedDiff-Risk] Epoch 052 | Train 2.7279 | Val 1.5759
[HyperMedDiff-Risk] Epoch 053 | Train 2.7136 | Val 1.4928
[HyperMedDiff-Risk] Epoch 054 | Train 2.6781 | Val 1.5220
[HyperMedDiff-Risk] Epoch 055 | Train 2.6787 | Val 1.5862
[HyperMedDiff-Risk] Epoch 056 | Train 2.6528 | Val 1.5111
[HyperMedDiff-Risk] Epoch 057 | Train 2.6477 | Val 1.4672
[HyperMedDiff-Risk] Epoch 058 | Train 2.6203 | Val 1.4784
[HyperMedDiff-Risk] Epoch 059 | Train 2.6289 | Val 1.4780
[HyperMedDiff-Risk] Epoch 060 | Train 2.6126 | Val 1.4415
[HyperMedDiff-Risk] Epoch 061 | Train 2.5933 | Val 1.4335
[HyperMedDiff-Risk] Epoch 062 | Train 2.5877 | Val 1.4952
[HyperMedDiff-Risk] Epoch 063 | Train 2.5574 | Val 1.4565
[HyperMedDiff-Risk] Epoch 064 | Train 2.5481 | Val 1.4140
[HyperMedDiff-Risk] Epoch 065 | Train 2.5140 | Val 1.4278
[HyperMedDiff-Risk] Epoch 066 | Train 2.5319 | Val 1.3840
[HyperMedDiff-Risk] Epoch 067 | Train 2.5292 | Val 1.3718
[HyperMedDiff-Risk] Epoch 068 | Train 2.4874 | Val 1.3236
[HyperMedDiff-Risk] Epoch 069 | Train 2.4972 | Val 1.3477
[HyperMedDiff-Risk] Epoch 070 | Train 2.4644 | Val 1.3113
[HyperMedDiff-Risk] Epoch 071 | Train 2.4780 | Val 1.3159
[HyperMedDiff-Risk] Epoch 072 | Train 2.4663 | Val 1.2769
[HyperMedDiff-Risk] Epoch 073 | Train 2.4387 | Val 1.3070
[HyperMedDiff-Risk] Epoch 074 | Train 2.4712 | Val 1.3200
[HyperMedDiff-Risk] Epoch 075 | Train 2.4095 | Val 1.3276
[HyperMedDiff-Risk] Epoch 076 | Train 2.4278 | Val 1.2273
[HyperMedDiff-Risk] Epoch 077 | Train 2.4138 | Val 1.2696
[HyperMedDiff-Risk] Epoch 078 | Train 2.4106 | Val 1.3230
[HyperMedDiff-Risk] Epoch 079 | Train 2.4115 | Val 1.2502
[HyperMedDiff-Risk] Epoch 080 | Train 2.3995 | Val 1.2222
[HyperMedDiff-Risk] Epoch 081 | Train 2.3844 | Val 1.2760
[HyperMedDiff-Risk] Epoch 082 | Train 2.3754 | Val 1.2264
[HyperMedDiff-Risk] Epoch 083 | Train 2.3883 | Val 1.2419
[HyperMedDiff-Risk] Epoch 084 | Train 2.3954 | Val 1.2302
[HyperMedDiff-Risk] Epoch 085 | Train 2.3581 | Val 1.2509
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 9): 1.2222
[HyperMedDiff-Risk] Saved training curve plot to results/plots/09_NoSynthRisk.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.8307830783078308,
  "f1": 0.7729468599033817,
  "kappa": 0.6382348560165444,
  "auroc": 0.8844486365974962,
  "auprc": 0.8207105777959087
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7632 ± 0.0028
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7659
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0010
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2038 std=0.0401
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/09_NoSynthRisk_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/09_NoSynthRisk_umap.png
Saved checkpoint to results/checkpoints/09_NoSynthRisk.pt
[HyperMedDiff-Risk] ===== Experiment 10/18: 10_GenFocus =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 2.0,
  "lambda_d": 2.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0238 | val=0.0231
[Pretrain] Epoch 02 | train=0.0233 | val=0.0244
[Pretrain] Epoch 03 | train=0.0231 | val=0.0218
[Pretrain] Epoch 04 | train=0.0232 | val=0.0228
[Pretrain] Epoch 05 | train=0.0223 | val=0.0207
[Pretrain] Epoch 06 | train=0.0216 | val=0.0208
[Pretrain] Epoch 07 | train=0.0220 | val=0.0204
[Pretrain] Epoch 08 | train=0.0208 | val=0.0207
[Pretrain] Epoch 09 | train=0.0207 | val=0.0189
[Pretrain] Epoch 10 | train=0.0201 | val=0.0203
[Pretrain] Epoch 11 | train=0.0207 | val=0.0201
[Pretrain] Epoch 12 | train=0.0199 | val=0.0184
[Pretrain] Epoch 13 | train=0.0197 | val=0.0191
[Pretrain] Epoch 14 | train=0.0186 | val=0.0182
[Pretrain] Epoch 15 | train=0.0194 | val=0.0188
[Pretrain] Epoch 16 | train=0.0187 | val=0.0187
[Pretrain] Epoch 17 | train=0.0177 | val=0.0172
[Pretrain] Epoch 18 | train=0.0164 | val=0.0176
[Pretrain] Epoch 19 | train=0.0177 | val=0.0172
[Pretrain] Epoch 20 | train=0.0173 | val=0.0168
[Pretrain] Epoch 21 | train=0.0171 | val=0.0174
[Pretrain] Epoch 22 | train=0.0166 | val=0.0174
[Pretrain] Epoch 23 | train=0.0162 | val=0.0161
[Pretrain] Epoch 24 | train=0.0169 | val=0.0159
[Pretrain] Epoch 25 | train=0.0165 | val=0.0152
[Pretrain] Epoch 26 | train=0.0154 | val=0.0160
[Pretrain] Epoch 27 | train=0.0157 | val=0.0156
[Pretrain] Epoch 28 | train=0.0149 | val=0.0152
[Pretrain] Epoch 29 | train=0.0153 | val=0.0144
[Pretrain] Epoch 30 | train=0.0152 | val=0.0152
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7500 ± 0.0058
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 26.4283 | Val 21.3124
[HyperMedDiff-Risk] Epoch 002 | Train 22.1257 | Val 20.9328
[HyperMedDiff-Risk] Epoch 003 | Train 21.5551 | Val 20.1758
[HyperMedDiff-Risk] Epoch 004 | Train 20.4643 | Val 18.6093
[HyperMedDiff-Risk] Epoch 005 | Train 19.0432 | Val 17.1095
[HyperMedDiff-Risk] Epoch 006 | Train 17.8042 | Val 15.5424
[HyperMedDiff-Risk] Epoch 007 | Train 16.6511 | Val 14.5761
[HyperMedDiff-Risk] Epoch 008 | Train 15.7720 | Val 13.4073
[HyperMedDiff-Risk] Epoch 009 | Train 14.8494 | Val 12.6686
[HyperMedDiff-Risk] Epoch 010 | Train 14.2181 | Val 11.9207
[HyperMedDiff-Risk] Epoch 011 | Train 13.5906 | Val 11.3440
[HyperMedDiff-Risk] Epoch 012 | Train 12.9994 | Val 10.5828
[HyperMedDiff-Risk] Epoch 013 | Train 12.5260 | Val 10.0242
[HyperMedDiff-Risk] Epoch 014 | Train 12.0562 | Val 9.5489
[HyperMedDiff-Risk] Epoch 015 | Train 11.5403 | Val 9.2248
[HyperMedDiff-Risk] Epoch 016 | Train 11.2618 | Val 8.7093
[HyperMedDiff-Risk] Epoch 017 | Train 10.8185 | Val 8.4749
[HyperMedDiff-Risk] Epoch 018 | Train 10.6038 | Val 8.1792
[HyperMedDiff-Risk] Epoch 019 | Train 10.1982 | Val 7.6340
[HyperMedDiff-Risk] Epoch 020 | Train 9.9458 | Val 7.4970
[HyperMedDiff-Risk] Epoch 021 | Train 9.6344 | Val 7.0847
[HyperMedDiff-Risk] Epoch 022 | Train 9.3807 | Val 6.9768
[HyperMedDiff-Risk] Epoch 023 | Train 9.1275 | Val 6.6850
[HyperMedDiff-Risk] Epoch 024 | Train 8.9623 | Val 6.5112
[HyperMedDiff-Risk] Epoch 025 | Train 8.7338 | Val 6.2781
[HyperMedDiff-Risk] Epoch 026 | Train 8.5219 | Val 6.0141
[HyperMedDiff-Risk] Epoch 027 | Train 8.3218 | Val 6.0359
[HyperMedDiff-Risk] Epoch 028 | Train 8.1378 | Val 5.5172
[HyperMedDiff-Risk] Epoch 029 | Train 7.9540 | Val 5.5048
[HyperMedDiff-Risk] Epoch 030 | Train 7.8116 | Val 5.3214
[HyperMedDiff-Risk] Epoch 031 | Train 7.6829 | Val 5.2414
[HyperMedDiff-Risk] Epoch 032 | Train 7.6170 | Val 5.0261
[HyperMedDiff-Risk] Epoch 033 | Train 7.4547 | Val 5.0291
[HyperMedDiff-Risk] Epoch 034 | Train 7.3511 | Val 4.8666
[HyperMedDiff-Risk] Epoch 035 | Train 7.2567 | Val 4.7043
[HyperMedDiff-Risk] Epoch 036 | Train 7.1598 | Val 4.5972
[HyperMedDiff-Risk] Epoch 037 | Train 6.9762 | Val 4.4780
[HyperMedDiff-Risk] Epoch 038 | Train 6.9306 | Val 4.4534
[HyperMedDiff-Risk] Epoch 039 | Train 6.8370 | Val 4.3026
[HyperMedDiff-Risk] Epoch 040 | Train 6.7896 | Val 4.3879
[HyperMedDiff-Risk] Epoch 041 | Train 6.6668 | Val 4.3393
[HyperMedDiff-Risk] Epoch 042 | Train 6.6233 | Val 4.2033
[HyperMedDiff-Risk] Epoch 043 | Train 6.6044 | Val 4.1005
[HyperMedDiff-Risk] Epoch 044 | Train 6.4227 | Val 4.0134
[HyperMedDiff-Risk] Epoch 045 | Train 6.4546 | Val 3.9999
[HyperMedDiff-Risk] Epoch 046 | Train 6.3880 | Val 4.0245
[HyperMedDiff-Risk] Epoch 047 | Train 6.2971 | Val 3.9371
[HyperMedDiff-Risk] Epoch 048 | Train 6.2387 | Val 4.0587
[HyperMedDiff-Risk] Epoch 049 | Train 6.2943 | Val 3.7879
[HyperMedDiff-Risk] Epoch 050 | Train 6.1463 | Val 3.7667
[HyperMedDiff-Risk] Epoch 051 | Train 6.0959 | Val 3.7971
[HyperMedDiff-Risk] Epoch 052 | Train 6.0864 | Val 3.7219
[HyperMedDiff-Risk] Epoch 053 | Train 6.0227 | Val 3.6957
[HyperMedDiff-Risk] Epoch 054 | Train 5.9872 | Val 3.6644
[HyperMedDiff-Risk] Epoch 055 | Train 5.9173 | Val 3.6625
[HyperMedDiff-Risk] Epoch 056 | Train 5.8975 | Val 3.5471
[HyperMedDiff-Risk] Epoch 057 | Train 5.8668 | Val 3.5027
[HyperMedDiff-Risk] Epoch 058 | Train 5.8229 | Val 3.5582
[HyperMedDiff-Risk] Epoch 059 | Train 5.8368 | Val 3.5644
[HyperMedDiff-Risk] Epoch 060 | Train 5.8149 | Val 3.5348
[HyperMedDiff-Risk] Epoch 061 | Train 5.7315 | Val 3.5173
[HyperMedDiff-Risk] Epoch 062 | Train 5.7331 | Val 3.5602
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 10): 3.5027
[HyperMedDiff-Risk] Saved training curve plot to results/plots/10_GenFocus.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8710821471445722,
  "auprc": 0.8035940239808055
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7500 ± 0.0058
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7533
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0130
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2018 std=0.0386
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/10_GenFocus_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/10_GenFocus_umap.png
Saved checkpoint to results/checkpoints/10_GenFocus.pt
[HyperMedDiff-Risk] ===== Experiment 11/18: 11_NoAttention =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": false,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0262 | val=0.0245
[Pretrain] Epoch 02 | train=0.0238 | val=0.0216
[Pretrain] Epoch 03 | train=0.0239 | val=0.0231
[Pretrain] Epoch 04 | train=0.0230 | val=0.0229
[Pretrain] Epoch 05 | train=0.0214 | val=0.0214
[Pretrain] Epoch 06 | train=0.0215 | val=0.0218
[Pretrain] Epoch 07 | train=0.0200 | val=0.0214
[Pretrain] Epoch 08 | train=0.0207 | val=0.0218
[Pretrain] Epoch 09 | train=0.0211 | val=0.0207
[Pretrain] Epoch 10 | train=0.0208 | val=0.0203
[Pretrain] Epoch 11 | train=0.0207 | val=0.0187
[Pretrain] Epoch 12 | train=0.0195 | val=0.0184
[Pretrain] Epoch 13 | train=0.0198 | val=0.0178
[Pretrain] Epoch 14 | train=0.0178 | val=0.0194
[Pretrain] Epoch 15 | train=0.0177 | val=0.0181
[Pretrain] Epoch 16 | train=0.0187 | val=0.0185
[Pretrain] Epoch 17 | train=0.0180 | val=0.0181
[Pretrain] Epoch 18 | train=0.0173 | val=0.0181
[Pretrain] Epoch 19 | train=0.0171 | val=0.0171
[Pretrain] Epoch 20 | train=0.0170 | val=0.0184
[Pretrain] Epoch 21 | train=0.0173 | val=0.0159
[Pretrain] Epoch 22 | train=0.0169 | val=0.0163
[Pretrain] Epoch 23 | train=0.0162 | val=0.0162
[Pretrain] Epoch 24 | train=0.0154 | val=0.0153
[Pretrain] Epoch 25 | train=0.0161 | val=0.0164
[Pretrain] Epoch 26 | train=0.0170 | val=0.0167
[Pretrain] Epoch 27 | train=0.0150 | val=0.0154
[Pretrain] Epoch 28 | train=0.0153 | val=0.0147
[Pretrain] Epoch 29 | train=0.0155 | val=0.0157
[Pretrain] Epoch 30 | train=0.0144 | val=0.0157
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7187 ± 0.0084
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 14.2731 | Val 10.9836
[HyperMedDiff-Risk] Epoch 002 | Train 11.3441 | Val 10.8031
[HyperMedDiff-Risk] Epoch 003 | Train 11.1537 | Val 10.5717
[HyperMedDiff-Risk] Epoch 004 | Train 10.7176 | Val 9.8836
[HyperMedDiff-Risk] Epoch 005 | Train 10.0555 | Val 9.0494
[HyperMedDiff-Risk] Epoch 006 | Train 9.3707 | Val 8.3451
[HyperMedDiff-Risk] Epoch 007 | Train 8.7931 | Val 7.7683
[HyperMedDiff-Risk] Epoch 008 | Train 8.3086 | Val 7.2057
[HyperMedDiff-Risk] Epoch 009 | Train 7.8732 | Val 6.6590
[HyperMedDiff-Risk] Epoch 010 | Train 7.4838 | Val 6.3804
[HyperMedDiff-Risk] Epoch 011 | Train 7.1925 | Val 6.1066
[HyperMedDiff-Risk] Epoch 012 | Train 6.9463 | Val 5.7134
[HyperMedDiff-Risk] Epoch 013 | Train 6.6807 | Val 5.4980
[HyperMedDiff-Risk] Epoch 014 | Train 6.4321 | Val 5.3090
[HyperMedDiff-Risk] Epoch 015 | Train 6.2076 | Val 5.0089
[HyperMedDiff-Risk] Epoch 016 | Train 6.0582 | Val 4.7780
[HyperMedDiff-Risk] Epoch 017 | Train 5.8339 | Val 4.5629
[HyperMedDiff-Risk] Epoch 018 | Train 5.6678 | Val 4.4836
[HyperMedDiff-Risk] Epoch 019 | Train 5.5077 | Val 4.2701
[HyperMedDiff-Risk] Epoch 020 | Train 5.3530 | Val 4.0730
[HyperMedDiff-Risk] Epoch 021 | Train 5.1973 | Val 3.9836
[HyperMedDiff-Risk] Epoch 022 | Train 5.0836 | Val 3.7168
[HyperMedDiff-Risk] Epoch 023 | Train 4.9759 | Val 3.7497
[HyperMedDiff-Risk] Epoch 024 | Train 4.8588 | Val 3.6621
[HyperMedDiff-Risk] Epoch 025 | Train 4.7386 | Val 3.5614
[HyperMedDiff-Risk] Epoch 026 | Train 4.6343 | Val 3.3925
[HyperMedDiff-Risk] Epoch 027 | Train 4.5159 | Val 3.3505
[HyperMedDiff-Risk] Epoch 028 | Train 4.4157 | Val 3.1834
[HyperMedDiff-Risk] Epoch 029 | Train 4.3560 | Val 3.1362
[HyperMedDiff-Risk] Epoch 030 | Train 4.2648 | Val 3.0469
[HyperMedDiff-Risk] Epoch 031 | Train 4.1834 | Val 2.9676
[HyperMedDiff-Risk] Epoch 032 | Train 4.0768 | Val 2.8730
[HyperMedDiff-Risk] Epoch 033 | Train 4.0459 | Val 2.8051
[HyperMedDiff-Risk] Epoch 034 | Train 3.9655 | Val 2.8299
[HyperMedDiff-Risk] Epoch 035 | Train 3.9014 | Val 2.7076
[HyperMedDiff-Risk] Epoch 036 | Train 3.8700 | Val 2.6821
[HyperMedDiff-Risk] Epoch 037 | Train 3.7994 | Val 2.5985
[HyperMedDiff-Risk] Epoch 038 | Train 3.7626 | Val 2.5764
[HyperMedDiff-Risk] Epoch 039 | Train 3.7193 | Val 2.4946
[HyperMedDiff-Risk] Epoch 040 | Train 3.6542 | Val 2.4735
[HyperMedDiff-Risk] Epoch 041 | Train 3.6418 | Val 2.4083
[HyperMedDiff-Risk] Epoch 042 | Train 3.6169 | Val 2.3932
[HyperMedDiff-Risk] Epoch 043 | Train 3.5550 | Val 2.3959
[HyperMedDiff-Risk] Epoch 044 | Train 3.5214 | Val 2.2958
[HyperMedDiff-Risk] Epoch 045 | Train 3.4898 | Val 2.3446
[HyperMedDiff-Risk] Epoch 046 | Train 3.4084 | Val 2.2624
[HyperMedDiff-Risk] Epoch 047 | Train 3.3938 | Val 2.2862
[HyperMedDiff-Risk] Epoch 048 | Train 3.4078 | Val 2.2422
[HyperMedDiff-Risk] Epoch 049 | Train 3.3660 | Val 2.2627
[HyperMedDiff-Risk] Epoch 050 | Train 3.3253 | Val 2.1946
[HyperMedDiff-Risk] Epoch 051 | Train 3.3235 | Val 2.1481
[HyperMedDiff-Risk] Epoch 052 | Train 3.3066 | Val 2.1857
[HyperMedDiff-Risk] Epoch 053 | Train 3.2529 | Val 2.1452
[HyperMedDiff-Risk] Epoch 054 | Train 3.2443 | Val 2.1305
[HyperMedDiff-Risk] Epoch 055 | Train 3.2570 | Val 2.0901
[HyperMedDiff-Risk] Epoch 056 | Train 3.2085 | Val 2.0373
[HyperMedDiff-Risk] Epoch 057 | Train 3.2149 | Val 2.0609
[HyperMedDiff-Risk] Epoch 058 | Train 3.2004 | Val 2.0709
[HyperMedDiff-Risk] Epoch 059 | Train 3.1763 | Val 1.9960
[HyperMedDiff-Risk] Epoch 060 | Train 3.1548 | Val 2.0465
[HyperMedDiff-Risk] Epoch 061 | Train 3.1301 | Val 2.0176
[HyperMedDiff-Risk] Epoch 062 | Train 3.1148 | Val 1.9823
[HyperMedDiff-Risk] Epoch 063 | Train 3.1031 | Val 1.9437
[HyperMedDiff-Risk] Epoch 064 | Train 3.1094 | Val 1.9501
[HyperMedDiff-Risk] Epoch 065 | Train 3.0977 | Val 1.9801
[HyperMedDiff-Risk] Epoch 066 | Train 3.0826 | Val 1.9021
[HyperMedDiff-Risk] Epoch 067 | Train 3.0596 | Val 1.8854
[HyperMedDiff-Risk] Epoch 068 | Train 3.0625 | Val 1.9159
[HyperMedDiff-Risk] Epoch 069 | Train 3.0396 | Val 1.9399
[HyperMedDiff-Risk] Epoch 070 | Train 3.0437 | Val 1.9084
[HyperMedDiff-Risk] Epoch 071 | Train 3.0362 | Val 1.9216
[HyperMedDiff-Risk] Epoch 072 | Train 3.0231 | Val 1.8962
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 11): 1.8854
[HyperMedDiff-Risk] Saved training curve plot to results/plots/11_NoAttention.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8452718230149203,
  "auprc": 0.7568488946709361
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7187 ± 0.0084
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7558
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): -0.0019
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.1978 std=0.0362
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/11_NoAttention_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/11_NoAttention_umap.png
Saved checkpoint to results/checkpoints/11_NoAttention.pt
[HyperMedDiff-Risk] ===== Experiment 12/18: 12_NoConsistency =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.0,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0251 | val=0.0231
[Pretrain] Epoch 02 | train=0.0247 | val=0.0240
[Pretrain] Epoch 03 | train=0.0241 | val=0.0241
[Pretrain] Epoch 04 | train=0.0223 | val=0.0227
[Pretrain] Epoch 05 | train=0.0236 | val=0.0226
[Pretrain] Epoch 06 | train=0.0208 | val=0.0214
[Pretrain] Epoch 07 | train=0.0213 | val=0.0223
[Pretrain] Epoch 08 | train=0.0208 | val=0.0201
[Pretrain] Epoch 09 | train=0.0212 | val=0.0203
[Pretrain] Epoch 10 | train=0.0201 | val=0.0194
[Pretrain] Epoch 11 | train=0.0202 | val=0.0190
[Pretrain] Epoch 12 | train=0.0183 | val=0.0194
[Pretrain] Epoch 13 | train=0.0192 | val=0.0186
[Pretrain] Epoch 14 | train=0.0189 | val=0.0187
[Pretrain] Epoch 15 | train=0.0173 | val=0.0184
[Pretrain] Epoch 16 | train=0.0180 | val=0.0177
[Pretrain] Epoch 17 | train=0.0185 | val=0.0171
[Pretrain] Epoch 18 | train=0.0198 | val=0.0171
[Pretrain] Epoch 19 | train=0.0180 | val=0.0168
[Pretrain] Epoch 20 | train=0.0172 | val=0.0166
[Pretrain] Epoch 21 | train=0.0164 | val=0.0172
[Pretrain] Epoch 22 | train=0.0176 | val=0.0178
[Pretrain] Epoch 23 | train=0.0170 | val=0.0165
[Pretrain] Epoch 24 | train=0.0154 | val=0.0161
[Pretrain] Epoch 25 | train=0.0160 | val=0.0160
[Pretrain] Epoch 26 | train=0.0156 | val=0.0155
[Pretrain] Epoch 27 | train=0.0162 | val=0.0152
[Pretrain] Epoch 28 | train=0.0154 | val=0.0151
[Pretrain] Epoch 29 | train=0.0154 | val=0.0149
[Pretrain] Epoch 30 | train=0.0148 | val=0.0145
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7614 ± 0.0070
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.7864 | Val 10.9485
[HyperMedDiff-Risk] Epoch 002 | Train 11.3673 | Val 10.8511
[HyperMedDiff-Risk] Epoch 003 | Train 11.1631 | Val 10.5744
[HyperMedDiff-Risk] Epoch 004 | Train 10.6969 | Val 9.8123
[HyperMedDiff-Risk] Epoch 005 | Train 9.9652 | Val 8.9383
[HyperMedDiff-Risk] Epoch 006 | Train 9.3005 | Val 8.2199
[HyperMedDiff-Risk] Epoch 007 | Train 8.6861 | Val 7.5023
[HyperMedDiff-Risk] Epoch 008 | Train 8.1911 | Val 7.0255
[HyperMedDiff-Risk] Epoch 009 | Train 7.7705 | Val 6.7392
[HyperMedDiff-Risk] Epoch 010 | Train 7.4038 | Val 6.2827
[HyperMedDiff-Risk] Epoch 011 | Train 7.0989 | Val 5.9536
[HyperMedDiff-Risk] Epoch 012 | Train 6.8535 | Val 5.5859
[HyperMedDiff-Risk] Epoch 013 | Train 6.5866 | Val 5.3315
[HyperMedDiff-Risk] Epoch 014 | Train 6.2986 | Val 5.1163
[HyperMedDiff-Risk] Epoch 015 | Train 6.1411 | Val 4.8352
[HyperMedDiff-Risk] Epoch 016 | Train 5.9444 | Val 4.6686
[HyperMedDiff-Risk] Epoch 017 | Train 5.7318 | Val 4.4900
[HyperMedDiff-Risk] Epoch 018 | Train 5.5864 | Val 4.3780
[HyperMedDiff-Risk] Epoch 019 | Train 5.4453 | Val 4.1616
[HyperMedDiff-Risk] Epoch 020 | Train 5.3248 | Val 4.0498
[HyperMedDiff-Risk] Epoch 021 | Train 5.1476 | Val 3.8970
[HyperMedDiff-Risk] Epoch 022 | Train 5.0316 | Val 3.7553
[HyperMedDiff-Risk] Epoch 023 | Train 4.9049 | Val 3.6605
[HyperMedDiff-Risk] Epoch 024 | Train 4.8086 | Val 3.6039
[HyperMedDiff-Risk] Epoch 025 | Train 4.7215 | Val 3.4017
[HyperMedDiff-Risk] Epoch 026 | Train 4.6128 | Val 3.3591
[HyperMedDiff-Risk] Epoch 027 | Train 4.5499 | Val 3.2452
[HyperMedDiff-Risk] Epoch 028 | Train 4.4143 | Val 3.2060
[HyperMedDiff-Risk] Epoch 029 | Train 4.3700 | Val 3.1585
[HyperMedDiff-Risk] Epoch 030 | Train 4.2318 | Val 3.0470
[HyperMedDiff-Risk] Epoch 031 | Train 4.1562 | Val 2.9491
[HyperMedDiff-Risk] Epoch 032 | Train 4.1522 | Val 2.8048
[HyperMedDiff-Risk] Epoch 033 | Train 4.0473 | Val 2.7599
[HyperMedDiff-Risk] Epoch 034 | Train 4.0071 | Val 2.8387
[HyperMedDiff-Risk] Epoch 035 | Train 3.9397 | Val 2.6474
[HyperMedDiff-Risk] Epoch 036 | Train 3.8682 | Val 2.6683
[HyperMedDiff-Risk] Epoch 037 | Train 3.8000 | Val 2.5443
[HyperMedDiff-Risk] Epoch 038 | Train 3.7677 | Val 2.5360
[HyperMedDiff-Risk] Epoch 039 | Train 3.7353 | Val 2.4755
[HyperMedDiff-Risk] Epoch 040 | Train 3.6659 | Val 2.4878
[HyperMedDiff-Risk] Epoch 041 | Train 3.6393 | Val 2.4547
[HyperMedDiff-Risk] Epoch 042 | Train 3.5765 | Val 2.4191
[HyperMedDiff-Risk] Epoch 043 | Train 3.5759 | Val 2.3762
[HyperMedDiff-Risk] Epoch 044 | Train 3.5338 | Val 2.3235
[HyperMedDiff-Risk] Epoch 045 | Train 3.4956 | Val 2.3455
[HyperMedDiff-Risk] Epoch 046 | Train 3.4674 | Val 2.2478
[HyperMedDiff-Risk] Epoch 047 | Train 3.4422 | Val 2.2893
[HyperMedDiff-Risk] Epoch 048 | Train 3.4174 | Val 2.2090
[HyperMedDiff-Risk] Epoch 049 | Train 3.3914 | Val 2.1454
[HyperMedDiff-Risk] Epoch 050 | Train 3.3522 | Val 2.1528
[HyperMedDiff-Risk] Epoch 051 | Train 3.3239 | Val 2.1247
[HyperMedDiff-Risk] Epoch 052 | Train 3.3085 | Val 2.1653
[HyperMedDiff-Risk] Epoch 053 | Train 3.2903 | Val 2.0742
[HyperMedDiff-Risk] Epoch 054 | Train 3.2852 | Val 2.0704
[HyperMedDiff-Risk] Epoch 055 | Train 3.2478 | Val 2.0598
[HyperMedDiff-Risk] Epoch 056 | Train 3.2191 | Val 2.0203
[HyperMedDiff-Risk] Epoch 057 | Train 3.2082 | Val 2.0340
[HyperMedDiff-Risk] Epoch 058 | Train 3.1871 | Val 1.9868
[HyperMedDiff-Risk] Epoch 059 | Train 3.1872 | Val 1.9736
[HyperMedDiff-Risk] Epoch 060 | Train 3.1435 | Val 2.0158
[HyperMedDiff-Risk] Epoch 061 | Train 3.1217 | Val 1.9798
[HyperMedDiff-Risk] Epoch 062 | Train 3.1064 | Val 1.9033
[HyperMedDiff-Risk] Epoch 063 | Train 3.0932 | Val 1.8849
[HyperMedDiff-Risk] Epoch 064 | Train 3.0663 | Val 1.9656
[HyperMedDiff-Risk] Epoch 065 | Train 3.0901 | Val 1.9054
[HyperMedDiff-Risk] Epoch 066 | Train 3.0598 | Val 1.8218
[HyperMedDiff-Risk] Epoch 067 | Train 3.0322 | Val 1.8792
[HyperMedDiff-Risk] Epoch 068 | Train 3.0506 | Val 1.8256
[HyperMedDiff-Risk] Epoch 069 | Train 3.0096 | Val 1.8773
[HyperMedDiff-Risk] Epoch 070 | Train 3.0040 | Val 1.8662
[HyperMedDiff-Risk] Epoch 071 | Train 3.0221 | Val 1.8433
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 12): 1.8218
[HyperMedDiff-Risk] Saved training curve plot to results/plots/12_NoConsistency.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8761893328760075,
  "auprc": 0.8142730407653935
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7614 ± 0.0070
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7590
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0003
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2043 std=0.0403
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/12_NoConsistency_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/12_NoConsistency_umap.png
Saved checkpoint to results/checkpoints/12_NoConsistency.pt
[HyperMedDiff-Risk] ===== Experiment 13/18: 13_NoFlow =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0247 | val=0.0247
[Pretrain] Epoch 02 | train=0.0241 | val=0.0247
[Pretrain] Epoch 03 | train=0.0225 | val=0.0240
[Pretrain] Epoch 04 | train=0.0230 | val=0.0220
[Pretrain] Epoch 05 | train=0.0233 | val=0.0222
[Pretrain] Epoch 06 | train=0.0231 | val=0.0221
[Pretrain] Epoch 07 | train=0.0226 | val=0.0208
[Pretrain] Epoch 08 | train=0.0205 | val=0.0205
[Pretrain] Epoch 09 | train=0.0214 | val=0.0197
[Pretrain] Epoch 10 | train=0.0187 | val=0.0213
[Pretrain] Epoch 11 | train=0.0201 | val=0.0199
[Pretrain] Epoch 12 | train=0.0198 | val=0.0194
[Pretrain] Epoch 13 | train=0.0197 | val=0.0181
[Pretrain] Epoch 14 | train=0.0202 | val=0.0181
[Pretrain] Epoch 15 | train=0.0199 | val=0.0177
[Pretrain] Epoch 16 | train=0.0181 | val=0.0182
[Pretrain] Epoch 17 | train=0.0181 | val=0.0170
[Pretrain] Epoch 18 | train=0.0176 | val=0.0178
[Pretrain] Epoch 19 | train=0.0175 | val=0.0165
[Pretrain] Epoch 20 | train=0.0182 | val=0.0168
[Pretrain] Epoch 21 | train=0.0173 | val=0.0161
[Pretrain] Epoch 22 | train=0.0173 | val=0.0182
[Pretrain] Epoch 23 | train=0.0165 | val=0.0151
[Pretrain] Epoch 24 | train=0.0165 | val=0.0156
[Pretrain] Epoch 25 | train=0.0158 | val=0.0167
[Pretrain] Epoch 26 | train=0.0162 | val=0.0150
[Pretrain] Epoch 27 | train=0.0164 | val=0.0153
[Pretrain] Epoch 28 | train=0.0163 | val=0.0155
[Pretrain] Epoch 29 | train=0.0152 | val=0.0152
[Pretrain] Epoch 30 | train=0.0141 | val=0.0136
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7651 ± 0.0075
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 0.5957 | Val 0.5648
[HyperMedDiff-Risk] Epoch 002 | Train 0.5539 | Val 0.5626
[HyperMedDiff-Risk] Epoch 003 | Train 0.5557 | Val 0.5641
[HyperMedDiff-Risk] Epoch 004 | Train 0.5541 | Val 0.5626
[HyperMedDiff-Risk] Epoch 005 | Train 0.5541 | Val 0.5639
[HyperMedDiff-Risk] Epoch 006 | Train 0.5546 | Val 0.5618
[HyperMedDiff-Risk] Epoch 007 | Train 0.5542 | Val 0.5624
[HyperMedDiff-Risk] Epoch 008 | Train 0.5534 | Val 0.5640
[HyperMedDiff-Risk] Epoch 009 | Train 0.5401 | Val 0.5556
[HyperMedDiff-Risk] Epoch 010 | Train 0.5042 | Val 0.4697
[HyperMedDiff-Risk] Epoch 011 | Train 0.4882 | Val 0.4764
[HyperMedDiff-Risk] Epoch 012 | Train 0.4683 | Val 0.4381
[HyperMedDiff-Risk] Epoch 013 | Train 0.4497 | Val 0.4555
[HyperMedDiff-Risk] Epoch 014 | Train 0.4419 | Val 0.4205
[HyperMedDiff-Risk] Epoch 015 | Train 0.4321 | Val 0.4120
[HyperMedDiff-Risk] Epoch 016 | Train 0.4208 | Val 0.4105
[HyperMedDiff-Risk] Epoch 017 | Train 0.4230 | Val 0.4042
[HyperMedDiff-Risk] Epoch 018 | Train 0.4118 | Val 0.4142
[HyperMedDiff-Risk] Epoch 019 | Train 0.4104 | Val 0.3949
[HyperMedDiff-Risk] Epoch 020 | Train 0.4036 | Val 0.4541
[HyperMedDiff-Risk] Epoch 021 | Train 0.4004 | Val 0.3891
[HyperMedDiff-Risk] Epoch 022 | Train 0.3879 | Val 0.3750
[HyperMedDiff-Risk] Epoch 023 | Train 0.3824 | Val 0.3668
[HyperMedDiff-Risk] Epoch 024 | Train 0.3720 | Val 0.3680
[HyperMedDiff-Risk] Epoch 025 | Train 0.3652 | Val 0.3473
[HyperMedDiff-Risk] Epoch 026 | Train 0.3528 | Val 0.4348
[HyperMedDiff-Risk] Epoch 027 | Train 0.3550 | Val 0.3550
[HyperMedDiff-Risk] Epoch 028 | Train 0.3487 | Val 0.3409
[HyperMedDiff-Risk] Epoch 029 | Train 0.3343 | Val 0.3515
[HyperMedDiff-Risk] Epoch 030 | Train 0.3290 | Val 0.3315
[HyperMedDiff-Risk] Epoch 031 | Train 0.3268 | Val 0.3432
[HyperMedDiff-Risk] Epoch 032 | Train 0.3242 | Val 0.3371
[HyperMedDiff-Risk] Epoch 033 | Train 0.3224 | Val 0.3535
[HyperMedDiff-Risk] Epoch 034 | Train 0.3271 | Val 0.3490
[HyperMedDiff-Risk] Epoch 035 | Train 0.3137 | Val 0.3216
[HyperMedDiff-Risk] Epoch 036 | Train 0.3107 | Val 0.3206
[HyperMedDiff-Risk] Epoch 037 | Train 0.3093 | Val 0.3562
[HyperMedDiff-Risk] Epoch 038 | Train 0.3013 | Val 0.3257
[HyperMedDiff-Risk] Epoch 039 | Train 0.3032 | Val 0.3151
[HyperMedDiff-Risk] Epoch 040 | Train 0.2955 | Val 0.3091
[HyperMedDiff-Risk] Epoch 041 | Train 0.2934 | Val 0.3469
[HyperMedDiff-Risk] Epoch 042 | Train 0.2951 | Val 0.3148
[HyperMedDiff-Risk] Epoch 043 | Train 0.2959 | Val 0.3336
[HyperMedDiff-Risk] Epoch 044 | Train 0.2861 | Val 0.3150
[HyperMedDiff-Risk] Epoch 045 | Train 0.2854 | Val 0.3002
[HyperMedDiff-Risk] Epoch 046 | Train 0.2775 | Val 0.2953
[HyperMedDiff-Risk] Epoch 047 | Train 0.2682 | Val 0.3370
[HyperMedDiff-Risk] Epoch 048 | Train 0.2768 | Val 0.3186
[HyperMedDiff-Risk] Epoch 049 | Train 0.2728 | Val 0.3277
[HyperMedDiff-Risk] Epoch 050 | Train 0.2725 | Val 0.2981
[HyperMedDiff-Risk] Epoch 051 | Train 0.2702 | Val 0.2831
[HyperMedDiff-Risk] Epoch 052 | Train 0.2689 | Val 0.2930
[HyperMedDiff-Risk] Epoch 053 | Train 0.2574 | Val 0.3149
[HyperMedDiff-Risk] Epoch 054 | Train 0.2572 | Val 0.3436
[HyperMedDiff-Risk] Epoch 055 | Train 0.2579 | Val 0.3360
[HyperMedDiff-Risk] Epoch 056 | Train 0.2464 | Val 0.2783
[HyperMedDiff-Risk] Epoch 057 | Train 0.2401 | Val 0.3043
[HyperMedDiff-Risk] Epoch 058 | Train 0.2346 | Val 0.2983
[HyperMedDiff-Risk] Epoch 059 | Train 0.2330 | Val 0.2936
[HyperMedDiff-Risk] Epoch 060 | Train 0.2312 | Val 0.2931
[HyperMedDiff-Risk] Epoch 061 | Train 0.2240 | Val 0.2854
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 13): 0.2783
[HyperMedDiff-Risk] Saved training curve plot to results/plots/13_NoFlow.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.8820882088208821,
  "f1": 0.8446026097271648,
  "kappa": 0.7496167893571524,
  "auroc": 0.9564157091407991,
  "auprc": 0.916660159317843
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7651 ± 0.0075
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7769
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0116
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2044 std=0.0406
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/13_NoFlow_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/13_NoFlow_umap.png
Saved checkpoint to results/checkpoints/13_NoFlow.pt
[HyperMedDiff-Risk] ===== Experiment 14/18: 14_RealOnlyRisk =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 0.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.0,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0247 | val=0.0244
[Pretrain] Epoch 02 | train=0.0250 | val=0.0242
[Pretrain] Epoch 03 | train=0.0228 | val=0.0225
[Pretrain] Epoch 04 | train=0.0229 | val=0.0239
[Pretrain] Epoch 05 | train=0.0213 | val=0.0221
[Pretrain] Epoch 06 | train=0.0211 | val=0.0199
[Pretrain] Epoch 07 | train=0.0201 | val=0.0214
[Pretrain] Epoch 08 | train=0.0206 | val=0.0209
[Pretrain] Epoch 09 | train=0.0202 | val=0.0207
[Pretrain] Epoch 10 | train=0.0193 | val=0.0188
[Pretrain] Epoch 11 | train=0.0207 | val=0.0202
[Pretrain] Epoch 12 | train=0.0185 | val=0.0195
[Pretrain] Epoch 13 | train=0.0207 | val=0.0187
[Pretrain] Epoch 14 | train=0.0179 | val=0.0184
[Pretrain] Epoch 15 | train=0.0192 | val=0.0181
[Pretrain] Epoch 16 | train=0.0166 | val=0.0175
[Pretrain] Epoch 17 | train=0.0183 | val=0.0181
[Pretrain] Epoch 18 | train=0.0168 | val=0.0177
[Pretrain] Epoch 19 | train=0.0172 | val=0.0183
[Pretrain] Epoch 20 | train=0.0181 | val=0.0163
[Pretrain] Epoch 21 | train=0.0171 | val=0.0161
[Pretrain] Epoch 22 | train=0.0164 | val=0.0173
[Pretrain] Epoch 23 | train=0.0169 | val=0.0156
[Pretrain] Epoch 24 | train=0.0165 | val=0.0161
[Pretrain] Epoch 25 | train=0.0168 | val=0.0164
[Pretrain] Epoch 26 | train=0.0169 | val=0.0159
[Pretrain] Epoch 27 | train=0.0153 | val=0.0161
[Pretrain] Epoch 28 | train=0.0154 | val=0.0151
[Pretrain] Epoch 29 | train=0.0146 | val=0.0152
[Pretrain] Epoch 30 | train=0.0156 | val=0.0149
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.7576 ± 0.0032
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 0.5956 | Val 0.5617
[HyperMedDiff-Risk] Epoch 002 | Train 0.5560 | Val 0.5690
[HyperMedDiff-Risk] Epoch 003 | Train 0.5542 | Val 0.5619
[HyperMedDiff-Risk] Epoch 004 | Train 0.5547 | Val 0.5630
[HyperMedDiff-Risk] Epoch 005 | Train 0.5548 | Val 0.5653
[HyperMedDiff-Risk] Epoch 006 | Train 0.5535 | Val 0.5629
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 14): 0.5617
[HyperMedDiff-Risk] Saved training curve plot to results/plots/14_RealOnlyRisk.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8667226890756302,
  "auprc": 0.7949335895755534
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.7576 ± 0.0032
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7622
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0057
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2042 std=0.0401
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/14_RealOnlyRisk_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/14_RealOnlyRisk_umap.png
Saved checkpoint to results/checkpoints/14_RealOnlyRisk.pt
[HyperMedDiff-Risk] ===== Experiment 15/18: 15_NoPretrain_RandomInit =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.0,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.0,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": false,
  "freeze_code_emb": true
}
[HyperMedDiff-Risk] Code embedding pretraining disabled (random init).
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.0120 ± 0.0100
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.2884 | Val 10.9732
[HyperMedDiff-Risk] Epoch 002 | Train 11.3617 | Val 10.8430
[HyperMedDiff-Risk] Epoch 003 | Train 11.1536 | Val 10.5343
[HyperMedDiff-Risk] Epoch 004 | Train 10.6357 | Val 9.7487
[HyperMedDiff-Risk] Epoch 005 | Train 9.8717 | Val 8.8638
[HyperMedDiff-Risk] Epoch 006 | Train 9.1735 | Val 8.0955
[HyperMedDiff-Risk] Epoch 007 | Train 8.6185 | Val 7.5112
[HyperMedDiff-Risk] Epoch 008 | Train 8.1261 | Val 7.0559
[HyperMedDiff-Risk] Epoch 009 | Train 7.7032 | Val 6.5537
[HyperMedDiff-Risk] Epoch 010 | Train 7.3948 | Val 6.2650
[HyperMedDiff-Risk] Epoch 011 | Train 7.0710 | Val 5.9577
[HyperMedDiff-Risk] Epoch 012 | Train 6.8018 | Val 5.6156
[HyperMedDiff-Risk] Epoch 013 | Train 6.5467 | Val 5.4065
[HyperMedDiff-Risk] Epoch 014 | Train 6.3116 | Val 5.1959
[HyperMedDiff-Risk] Epoch 015 | Train 6.1245 | Val 4.8819
[HyperMedDiff-Risk] Epoch 016 | Train 5.9355 | Val 4.6816
[HyperMedDiff-Risk] Epoch 017 | Train 5.7646 | Val 4.5473
[HyperMedDiff-Risk] Epoch 018 | Train 5.6309 | Val 4.4326
[HyperMedDiff-Risk] Epoch 019 | Train 5.4344 | Val 4.2009
[HyperMedDiff-Risk] Epoch 020 | Train 5.2926 | Val 4.0289
[HyperMedDiff-Risk] Epoch 021 | Train 5.1762 | Val 3.8589
[HyperMedDiff-Risk] Epoch 022 | Train 5.0483 | Val 3.7471
[HyperMedDiff-Risk] Epoch 023 | Train 4.9250 | Val 3.6692
[HyperMedDiff-Risk] Epoch 024 | Train 4.7855 | Val 3.5849
[HyperMedDiff-Risk] Epoch 025 | Train 4.7079 | Val 3.4972
[HyperMedDiff-Risk] Epoch 026 | Train 4.5808 | Val 3.3354
[HyperMedDiff-Risk] Epoch 027 | Train 4.4995 | Val 3.2633
[HyperMedDiff-Risk] Epoch 028 | Train 4.4361 | Val 3.2614
[HyperMedDiff-Risk] Epoch 029 | Train 4.3266 | Val 3.1507
[HyperMedDiff-Risk] Epoch 030 | Train 4.2673 | Val 3.0959
[HyperMedDiff-Risk] Epoch 031 | Train 4.2013 | Val 3.0017
[HyperMedDiff-Risk] Epoch 032 | Train 4.0917 | Val 2.8554
[HyperMedDiff-Risk] Epoch 033 | Train 4.0400 | Val 2.8505
[HyperMedDiff-Risk] Epoch 034 | Train 3.9632 | Val 2.7977
[HyperMedDiff-Risk] Epoch 035 | Train 3.9121 | Val 2.6824
[HyperMedDiff-Risk] Epoch 036 | Train 3.8423 | Val 2.6243
[HyperMedDiff-Risk] Epoch 037 | Train 3.8276 | Val 2.5342
[HyperMedDiff-Risk] Epoch 038 | Train 3.7442 | Val 2.5717
[HyperMedDiff-Risk] Epoch 039 | Train 3.7195 | Val 2.5180
[HyperMedDiff-Risk] Epoch 040 | Train 3.6704 | Val 2.5094
[HyperMedDiff-Risk] Epoch 041 | Train 3.6374 | Val 2.4310
[HyperMedDiff-Risk] Epoch 042 | Train 3.5738 | Val 2.3770
[HyperMedDiff-Risk] Epoch 043 | Train 3.5440 | Val 2.3694
[HyperMedDiff-Risk] Epoch 044 | Train 3.5168 | Val 2.3248
[HyperMedDiff-Risk] Epoch 045 | Train 3.4790 | Val 2.2544
[HyperMedDiff-Risk] Epoch 046 | Train 3.4863 | Val 2.2848
[HyperMedDiff-Risk] Epoch 047 | Train 3.4473 | Val 2.2648
[HyperMedDiff-Risk] Epoch 048 | Train 3.4013 | Val 2.2095
[HyperMedDiff-Risk] Epoch 049 | Train 3.3862 | Val 2.1672
[HyperMedDiff-Risk] Epoch 050 | Train 3.3378 | Val 2.2440
[HyperMedDiff-Risk] Epoch 051 | Train 3.3214 | Val 2.1605
[HyperMedDiff-Risk] Epoch 052 | Train 3.2910 | Val 2.1518
[HyperMedDiff-Risk] Epoch 053 | Train 3.2860 | Val 2.0852
[HyperMedDiff-Risk] Epoch 054 | Train 3.2501 | Val 2.0874
[HyperMedDiff-Risk] Epoch 055 | Train 3.2344 | Val 2.0548
[HyperMedDiff-Risk] Epoch 056 | Train 3.2320 | Val 2.0584
[HyperMedDiff-Risk] Epoch 057 | Train 3.2185 | Val 2.0160
[HyperMedDiff-Risk] Epoch 058 | Train 3.1763 | Val 2.0058
[HyperMedDiff-Risk] Epoch 059 | Train 3.1649 | Val 1.9891
[HyperMedDiff-Risk] Epoch 060 | Train 3.1736 | Val 1.9899
[HyperMedDiff-Risk] Epoch 061 | Train 3.1655 | Val 1.9898
[HyperMedDiff-Risk] Epoch 062 | Train 3.1212 | Val 2.0201
[HyperMedDiff-Risk] Epoch 063 | Train 3.1126 | Val 1.9698
[HyperMedDiff-Risk] Epoch 064 | Train 3.1272 | Val 1.9293
[HyperMedDiff-Risk] Epoch 065 | Train 3.0999 | Val 1.9648
[HyperMedDiff-Risk] Epoch 066 | Train 3.0465 | Val 1.9226
[HyperMedDiff-Risk] Epoch 067 | Train 3.0399 | Val 1.8987
[HyperMedDiff-Risk] Epoch 068 | Train 3.0625 | Val 1.9000
[HyperMedDiff-Risk] Epoch 069 | Train 3.0528 | Val 1.9522
[HyperMedDiff-Risk] Epoch 070 | Train 3.0130 | Val 1.9211
[HyperMedDiff-Risk] Epoch 071 | Train 3.0463 | Val 1.9075
[HyperMedDiff-Risk] Epoch 072 | Train 3.0151 | Val 1.8710
[HyperMedDiff-Risk] Epoch 073 | Train 2.9983 | Val 1.8852
[HyperMedDiff-Risk] Epoch 074 | Train 3.0252 | Val 1.8417
[HyperMedDiff-Risk] Epoch 075 | Train 2.9591 | Val 1.8608
[HyperMedDiff-Risk] Epoch 076 | Train 3.0012 | Val 1.8232
[HyperMedDiff-Risk] Epoch 077 | Train 2.9849 | Val 1.8497
[HyperMedDiff-Risk] Epoch 078 | Train 2.9757 | Val 1.8310
[HyperMedDiff-Risk] Epoch 079 | Train 2.9884 | Val 1.8105
[HyperMedDiff-Risk] Epoch 080 | Train 2.9574 | Val 1.8750
[HyperMedDiff-Risk] Epoch 081 | Train 2.9596 | Val 1.8055
[HyperMedDiff-Risk] Epoch 082 | Train 2.9577 | Val 1.8045
[HyperMedDiff-Risk] Epoch 083 | Train 2.9454 | Val 1.8069
[HyperMedDiff-Risk] Epoch 084 | Train 2.9625 | Val 1.7912
[HyperMedDiff-Risk] Epoch 085 | Train 2.9393 | Val 1.8166
[HyperMedDiff-Risk] Epoch 086 | Train 2.9355 | Val 1.8068
[HyperMedDiff-Risk] Epoch 087 | Train 2.9387 | Val 1.8110
[HyperMedDiff-Risk] Epoch 088 | Train 2.9463 | Val 1.7981
[HyperMedDiff-Risk] Epoch 089 | Train 2.9509 | Val 1.7838
[HyperMedDiff-Risk] Epoch 090 | Train 2.9322 | Val 1.7829
[HyperMedDiff-Risk] Epoch 091 | Train 2.9135 | Val 1.7789
[HyperMedDiff-Risk] Epoch 092 | Train 2.9190 | Val 1.7976
[HyperMedDiff-Risk] Epoch 093 | Train 2.9300 | Val 1.7487
[HyperMedDiff-Risk] Epoch 094 | Train 2.9269 | Val 1.7544
[HyperMedDiff-Risk] Epoch 095 | Train 2.9270 | Val 1.7907
[HyperMedDiff-Risk] Epoch 096 | Train 2.9111 | Val 1.7522
[HyperMedDiff-Risk] Epoch 097 | Train 2.9154 | Val 1.7610
[HyperMedDiff-Risk] Epoch 098 | Train 2.9178 | Val 1.7704
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 15): 1.7487
[HyperMedDiff-Risk] Saved training curve plot to results/plots/15_NoPretrain_RandomInit.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.867460126907906,
  "auprc": 0.7933409054522451
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.0120 ± 0.0100
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: -0.0042
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0056
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.0157 std=0.0015
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/15_NoPretrain_RandomInit_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/15_NoPretrain_RandomInit_umap.png
Saved checkpoint to results/checkpoints/15_NoPretrain_RandomInit.pt
[HyperMedDiff-Risk] ===== Experiment 16/18: 16_UnfreezeEmbeddings =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": false
}
[Pretrain] Epoch 01 | train=0.0257 | val=0.0246
[Pretrain] Epoch 02 | train=0.0244 | val=0.0241
[Pretrain] Epoch 03 | train=0.0228 | val=0.0233
[Pretrain] Epoch 04 | train=0.0222 | val=0.0230
[Pretrain] Epoch 05 | train=0.0218 | val=0.0236
[Pretrain] Epoch 06 | train=0.0228 | val=0.0210
[Pretrain] Epoch 07 | train=0.0215 | val=0.0208
[Pretrain] Epoch 08 | train=0.0198 | val=0.0207
[Pretrain] Epoch 09 | train=0.0205 | val=0.0203
[Pretrain] Epoch 10 | train=0.0199 | val=0.0193
[Pretrain] Epoch 11 | train=0.0191 | val=0.0198
[Pretrain] Epoch 12 | train=0.0198 | val=0.0198
[Pretrain] Epoch 13 | train=0.0183 | val=0.0200
[Pretrain] Epoch 14 | train=0.0190 | val=0.0185
[Pretrain] Epoch 15 | train=0.0189 | val=0.0185
[Pretrain] Epoch 16 | train=0.0167 | val=0.0174
[Pretrain] Epoch 17 | train=0.0183 | val=0.0165
[Pretrain] Epoch 18 | train=0.0176 | val=0.0174
[Pretrain] Epoch 19 | train=0.0172 | val=0.0174
[Pretrain] Epoch 20 | train=0.0173 | val=0.0162
[Pretrain] Epoch 21 | train=0.0168 | val=0.0177
[Pretrain] Epoch 22 | train=0.0168 | val=0.0160
[Pretrain] Epoch 23 | train=0.0165 | val=0.0159
[Pretrain] Epoch 24 | train=0.0171 | val=0.0168
[Pretrain] Epoch 25 | train=0.0161 | val=0.0163
[Pretrain] Epoch 26 | train=0.0171 | val=0.0158
[Pretrain] Epoch 27 | train=0.0156 | val=0.0152
[Pretrain] Epoch 28 | train=0.0162 | val=0.0150
[Pretrain] Epoch 29 | train=0.0156 | val=0.0154
[Pretrain] Epoch 30 | train=0.0145 | val=0.0149
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Code diffusion faithfulness (post-pretrain eval): 0.7604 ± 0.0053
[Check] code_emb.requires_grad = True
[HyperMedDiff-Risk] Epoch 001 | Train 13.6760 | Val 10.9614
[HyperMedDiff-Risk] Epoch 002 | Train 11.3633 | Val 10.8602
[HyperMedDiff-Risk] Epoch 003 | Train 11.1143 | Val 10.4536
[HyperMedDiff-Risk] Epoch 004 | Train 10.5692 | Val 9.6338
[HyperMedDiff-Risk] Epoch 005 | Train 9.8315 | Val 8.8196
[HyperMedDiff-Risk] Epoch 006 | Train 9.1996 | Val 8.1044
[HyperMedDiff-Risk] Epoch 007 | Train 8.6412 | Val 7.6347
[HyperMedDiff-Risk] Epoch 008 | Train 8.2036 | Val 7.1011
[HyperMedDiff-Risk] Epoch 009 | Train 7.7678 | Val 6.6244
[HyperMedDiff-Risk] Epoch 010 | Train 7.4035 | Val 6.2565
[HyperMedDiff-Risk] Epoch 011 | Train 7.0569 | Val 5.9052
[HyperMedDiff-Risk] Epoch 012 | Train 6.8075 | Val 5.7040
[HyperMedDiff-Risk] Epoch 013 | Train 6.5396 | Val 5.3997
[HyperMedDiff-Risk] Epoch 014 | Train 6.3066 | Val 5.1873
[HyperMedDiff-Risk] Epoch 015 | Train 6.1229 | Val 4.9083
[HyperMedDiff-Risk] Epoch 016 | Train 5.9266 | Val 4.6868
[HyperMedDiff-Risk] Epoch 017 | Train 5.7349 | Val 4.4557
[HyperMedDiff-Risk] Epoch 018 | Train 5.5648 | Val 4.2933
[HyperMedDiff-Risk] Epoch 019 | Train 5.3891 | Val 4.1599
[HyperMedDiff-Risk] Epoch 020 | Train 5.2282 | Val 4.0045
[HyperMedDiff-Risk] Epoch 021 | Train 5.0681 | Val 3.8421
[HyperMedDiff-Risk] Epoch 022 | Train 4.9094 | Val 3.6273
[HyperMedDiff-Risk] Epoch 023 | Train 4.7718 | Val 3.6052
[HyperMedDiff-Risk] Epoch 024 | Train 4.6735 | Val 3.4132
[HyperMedDiff-Risk] Epoch 025 | Train 4.5391 | Val 3.3334
[HyperMedDiff-Risk] Epoch 026 | Train 4.4313 | Val 3.1351
[HyperMedDiff-Risk] Epoch 027 | Train 4.3026 | Val 3.1401
[HyperMedDiff-Risk] Epoch 028 | Train 4.1885 | Val 3.0921
[HyperMedDiff-Risk] Epoch 029 | Train 4.0487 | Val 2.8592
[HyperMedDiff-Risk] Epoch 030 | Train 4.0116 | Val 2.8818
[HyperMedDiff-Risk] Epoch 031 | Train 3.9288 | Val 2.7315
[HyperMedDiff-Risk] Epoch 032 | Train 3.8180 | Val 2.7044
[HyperMedDiff-Risk] Epoch 033 | Train 3.7572 | Val 2.5717
[HyperMedDiff-Risk] Epoch 034 | Train 3.7091 | Val 2.5314
[HyperMedDiff-Risk] Epoch 035 | Train 3.5659 | Val 2.4205
[HyperMedDiff-Risk] Epoch 036 | Train 3.5469 | Val 2.3381
[HyperMedDiff-Risk] Epoch 037 | Train 3.4778 | Val 2.4129
[HyperMedDiff-Risk] Epoch 038 | Train 3.4316 | Val 2.2823
[HyperMedDiff-Risk] Epoch 039 | Train 3.3541 | Val 2.1906
[HyperMedDiff-Risk] Epoch 040 | Train 3.3333 | Val 2.1533
[HyperMedDiff-Risk] Epoch 041 | Train 3.2549 | Val 2.1034
[HyperMedDiff-Risk] Epoch 042 | Train 3.2444 | Val 2.0625
[HyperMedDiff-Risk] Epoch 043 | Train 3.1848 | Val 2.1208
[HyperMedDiff-Risk] Epoch 044 | Train 3.1559 | Val 2.0185
[HyperMedDiff-Risk] Epoch 045 | Train 3.1111 | Val 2.0552
[HyperMedDiff-Risk] Epoch 046 | Train 3.0651 | Val 2.0083
[HyperMedDiff-Risk] Epoch 047 | Train 3.0429 | Val 1.9758
[HyperMedDiff-Risk] Epoch 048 | Train 3.0105 | Val 1.9482
[HyperMedDiff-Risk] Epoch 049 | Train 2.9749 | Val 1.8519
[HyperMedDiff-Risk] Epoch 050 | Train 2.9577 | Val 1.8465
[HyperMedDiff-Risk] Epoch 051 | Train 2.9003 | Val 1.8550
[HyperMedDiff-Risk] Epoch 052 | Train 2.8963 | Val 1.8391
[HyperMedDiff-Risk] Epoch 053 | Train 2.8582 | Val 1.7878
[HyperMedDiff-Risk] Epoch 054 | Train 2.8350 | Val 1.7442
[HyperMedDiff-Risk] Epoch 055 | Train 2.8165 | Val 1.7459
[HyperMedDiff-Risk] Epoch 056 | Train 2.7981 | Val 1.7327
[HyperMedDiff-Risk] Epoch 057 | Train 2.7441 | Val 1.7477
[HyperMedDiff-Risk] Epoch 058 | Train 2.7815 | Val 1.6890
[HyperMedDiff-Risk] Epoch 059 | Train 2.7423 | Val 1.6757
[HyperMedDiff-Risk] Epoch 060 | Train 2.6882 | Val 1.6963
[HyperMedDiff-Risk] Epoch 061 | Train 2.7246 | Val 1.6553
[HyperMedDiff-Risk] Epoch 062 | Train 2.6805 | Val 1.6518
[HyperMedDiff-Risk] Epoch 063 | Train 2.6954 | Val 1.6411
[HyperMedDiff-Risk] Epoch 064 | Train 2.6595 | Val 1.6319
[HyperMedDiff-Risk] Epoch 065 | Train 2.6515 | Val 1.6216
[HyperMedDiff-Risk] Epoch 066 | Train 2.6326 | Val 1.6017
[HyperMedDiff-Risk] Epoch 067 | Train 2.6032 | Val 1.5482
[HyperMedDiff-Risk] Epoch 068 | Train 2.6064 | Val 1.5583
[HyperMedDiff-Risk] Epoch 069 | Train 2.5938 | Val 1.5666
[HyperMedDiff-Risk] Epoch 070 | Train 2.6084 | Val 1.5653
[HyperMedDiff-Risk] Epoch 071 | Train 2.5708 | Val 1.5213
[HyperMedDiff-Risk] Epoch 072 | Train 2.5796 | Val 1.5264
[HyperMedDiff-Risk] Epoch 073 | Train 2.5644 | Val 1.5250
[HyperMedDiff-Risk] Epoch 074 | Train 2.5838 | Val 1.5376
[HyperMedDiff-Risk] Epoch 075 | Train 2.5539 | Val 1.5659
[HyperMedDiff-Risk] Epoch 076 | Train 2.5385 | Val 1.5589
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 16): 1.5213
[HyperMedDiff-Risk] Saved training curve plot to results/plots/16_UnfreezeEmbeddings.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.9504950495049505,
  "f1": 0.9362688296639629,
  "kappa": 0.8958129082529543,
  "auroc": 0.9793002915451895,
  "auprc": 0.9503436738414909
}
[HyperMedDiff-Risk] Code diffusion faithfulness (post-train eval): 0.0023 ± 0.0081
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.3574
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0000
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.4480 std=0.6238
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/16_UnfreezeEmbeddings_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/16_UnfreezeEmbeddings_umap.png
Saved checkpoint to results/checkpoints/16_UnfreezeEmbeddings.pt
[HyperMedDiff-Risk] ===== Experiment 17/18: 17_Pretrain_RadiusOnly =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.0,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.003,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0029 | val=0.0028
[Pretrain] Epoch 02 | train=0.0028 | val=0.0026
[Pretrain] Epoch 03 | train=0.0026 | val=0.0025
[Pretrain] Epoch 04 | train=0.0025 | val=0.0024
[Pretrain] Epoch 05 | train=0.0024 | val=0.0023
[Pretrain] Epoch 06 | train=0.0023 | val=0.0022
[Pretrain] Epoch 07 | train=0.0022 | val=0.0021
[Pretrain] Epoch 08 | train=0.0021 | val=0.0020
[Pretrain] Epoch 09 | train=0.0020 | val=0.0019
[Pretrain] Epoch 10 | train=0.0019 | val=0.0018
[Pretrain] Epoch 11 | train=0.0018 | val=0.0017
[Pretrain] Epoch 12 | train=0.0017 | val=0.0016
[Pretrain] Epoch 13 | train=0.0016 | val=0.0016
[Pretrain] Epoch 14 | train=0.0016 | val=0.0015
[Pretrain] Epoch 15 | train=0.0015 | val=0.0014
[Pretrain] Epoch 16 | train=0.0014 | val=0.0014
[Pretrain] Epoch 17 | train=0.0014 | val=0.0013
[Pretrain] Epoch 18 | train=0.0013 | val=0.0013
[Pretrain] Epoch 19 | train=0.0013 | val=0.0012
[Pretrain] Epoch 20 | train=0.0012 | val=0.0012
[Pretrain] Epoch 21 | train=0.0012 | val=0.0012
[Pretrain] Epoch 22 | train=0.0012 | val=0.0011
[Pretrain] Epoch 23 | train=0.0011 | val=0.0011
[Pretrain] Epoch 24 | train=0.0011 | val=0.0011
[Pretrain] Epoch 25 | train=0.0011 | val=0.0011
[Pretrain] Epoch 26 | train=0.0011 | val=0.0011
[Pretrain] Epoch 27 | train=0.0011 | val=0.0011
[Pretrain] Epoch 28 | train=0.0011 | val=0.0011
[Pretrain] Epoch 29 | train=0.0011 | val=0.0011
[Pretrain] Epoch 30 | train=0.0011 | val=0.0011
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.0110 ± 0.0063
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.4461 | Val 10.9668
[HyperMedDiff-Risk] Epoch 002 | Train 11.3369 | Val 10.7375
[HyperMedDiff-Risk] Epoch 003 | Train 10.9623 | Val 10.1253
[HyperMedDiff-Risk] Epoch 004 | Train 10.2719 | Val 9.2438
[HyperMedDiff-Risk] Epoch 005 | Train 9.5453 | Val 8.5153
[HyperMedDiff-Risk] Epoch 006 | Train 8.9253 | Val 7.7579
[HyperMedDiff-Risk] Epoch 007 | Train 8.3566 | Val 7.2166
[HyperMedDiff-Risk] Epoch 008 | Train 7.9338 | Val 6.7033
[HyperMedDiff-Risk] Epoch 009 | Train 7.5286 | Val 6.4425
[HyperMedDiff-Risk] Epoch 010 | Train 7.1875 | Val 6.0467
[HyperMedDiff-Risk] Epoch 011 | Train 6.9322 | Val 5.7825
[HyperMedDiff-Risk] Epoch 012 | Train 6.6984 | Val 5.4677
[HyperMedDiff-Risk] Epoch 013 | Train 6.4565 | Val 5.2353
[HyperMedDiff-Risk] Epoch 014 | Train 6.2469 | Val 5.0582
[HyperMedDiff-Risk] Epoch 015 | Train 6.0418 | Val 4.7765
[HyperMedDiff-Risk] Epoch 016 | Train 5.8830 | Val 4.5932
[HyperMedDiff-Risk] Epoch 017 | Train 5.6838 | Val 4.4197
[HyperMedDiff-Risk] Epoch 018 | Train 5.5570 | Val 4.3126
[HyperMedDiff-Risk] Epoch 019 | Train 5.3792 | Val 4.1417
[HyperMedDiff-Risk] Epoch 020 | Train 5.3139 | Val 3.9811
[HyperMedDiff-Risk] Epoch 021 | Train 5.1439 | Val 3.8522
[HyperMedDiff-Risk] Epoch 022 | Train 5.0368 | Val 3.8178
[HyperMedDiff-Risk] Epoch 023 | Train 4.9041 | Val 3.6253
[HyperMedDiff-Risk] Epoch 024 | Train 4.7962 | Val 3.4797
[HyperMedDiff-Risk] Epoch 025 | Train 4.7259 | Val 3.4534
[HyperMedDiff-Risk] Epoch 026 | Train 4.6067 | Val 3.4334
[HyperMedDiff-Risk] Epoch 027 | Train 4.5235 | Val 3.2493
[HyperMedDiff-Risk] Epoch 028 | Train 4.4059 | Val 3.2277
[HyperMedDiff-Risk] Epoch 029 | Train 4.3530 | Val 3.0631
[HyperMedDiff-Risk] Epoch 030 | Train 4.2793 | Val 3.0408
[HyperMedDiff-Risk] Epoch 031 | Train 4.2471 | Val 3.0177
[HyperMedDiff-Risk] Epoch 032 | Train 4.1388 | Val 2.9126
[HyperMedDiff-Risk] Epoch 033 | Train 4.1331 | Val 2.8227
[HyperMedDiff-Risk] Epoch 034 | Train 4.0183 | Val 2.9335
[HyperMedDiff-Risk] Epoch 035 | Train 4.0052 | Val 2.8047
[HyperMedDiff-Risk] Epoch 036 | Train 3.9802 | Val 2.6996
[HyperMedDiff-Risk] Epoch 037 | Train 3.8851 | Val 2.7310
[HyperMedDiff-Risk] Epoch 038 | Train 3.8013 | Val 2.7090
[HyperMedDiff-Risk] Epoch 039 | Train 3.8091 | Val 2.5781
[HyperMedDiff-Risk] Epoch 040 | Train 3.7130 | Val 2.6149
[HyperMedDiff-Risk] Epoch 041 | Train 3.7225 | Val 2.4672
[HyperMedDiff-Risk] Epoch 042 | Train 3.6457 | Val 2.4402
[HyperMedDiff-Risk] Epoch 043 | Train 3.6617 | Val 2.4862
[HyperMedDiff-Risk] Epoch 044 | Train 3.6172 | Val 2.4467
[HyperMedDiff-Risk] Epoch 045 | Train 3.5349 | Val 2.3379
[HyperMedDiff-Risk] Epoch 046 | Train 3.5026 | Val 2.3119
[HyperMedDiff-Risk] Epoch 047 | Train 3.4560 | Val 2.2934
[HyperMedDiff-Risk] Epoch 048 | Train 3.4401 | Val 2.2716
[HyperMedDiff-Risk] Epoch 049 | Train 3.4072 | Val 2.2347
[HyperMedDiff-Risk] Epoch 050 | Train 3.3802 | Val 2.2746
[HyperMedDiff-Risk] Epoch 051 | Train 3.3436 | Val 2.1984
[HyperMedDiff-Risk] Epoch 052 | Train 3.3300 | Val 2.1514
[HyperMedDiff-Risk] Epoch 053 | Train 3.3083 | Val 2.1607
[HyperMedDiff-Risk] Epoch 054 | Train 3.2960 | Val 2.0965
[HyperMedDiff-Risk] Epoch 055 | Train 3.2740 | Val 2.0819
[HyperMedDiff-Risk] Epoch 056 | Train 3.2537 | Val 2.1158
[HyperMedDiff-Risk] Epoch 057 | Train 3.2581 | Val 2.1056
[HyperMedDiff-Risk] Epoch 058 | Train 3.2272 | Val 2.0945
[HyperMedDiff-Risk] Epoch 059 | Train 3.2088 | Val 2.1611
[HyperMedDiff-Risk] Epoch 060 | Train 3.1993 | Val 2.0408
[HyperMedDiff-Risk] Epoch 061 | Train 3.1837 | Val 2.0677
[HyperMedDiff-Risk] Epoch 062 | Train 3.1705 | Val 2.0618
[HyperMedDiff-Risk] Epoch 063 | Train 3.1513 | Val 2.0510
[HyperMedDiff-Risk] Epoch 064 | Train 3.1655 | Val 2.0077
[HyperMedDiff-Risk] Epoch 065 | Train 3.1357 | Val 2.0192
[HyperMedDiff-Risk] Epoch 066 | Train 3.1250 | Val 2.0182
[HyperMedDiff-Risk] Epoch 067 | Train 3.1118 | Val 1.9982
[HyperMedDiff-Risk] Epoch 068 | Train 3.1101 | Val 1.9621
[HyperMedDiff-Risk] Epoch 069 | Train 3.0834 | Val 2.0067
[HyperMedDiff-Risk] Epoch 070 | Train 3.1147 | Val 1.9442
[HyperMedDiff-Risk] Epoch 071 | Train 3.0784 | Val 1.9177
[HyperMedDiff-Risk] Epoch 072 | Train 3.0744 | Val 1.9510
[HyperMedDiff-Risk] Epoch 073 | Train 3.0581 | Val 1.9104
[HyperMedDiff-Risk] Epoch 074 | Train 3.0743 | Val 1.9680
[HyperMedDiff-Risk] Epoch 075 | Train 3.0765 | Val 1.9438
[HyperMedDiff-Risk] Epoch 076 | Train 3.0744 | Val 1.9187
[HyperMedDiff-Risk] Epoch 077 | Train 3.0265 | Val 1.9398
[HyperMedDiff-Risk] Epoch 078 | Train 3.0322 | Val 1.9707
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 17): 1.9104
[HyperMedDiff-Risk] Saved training curve plot to results/plots/17_Pretrain_RadiusOnly.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8765117475561655,
  "auprc": 0.8153458518014999
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.0110 ± 0.0063
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.0034
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): -0.0061
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.2829 std=0.0249
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/17_Pretrain_RadiusOnly_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/17_Pretrain_RadiusOnly_umap.png
Saved checkpoint to results/checkpoints/17_Pretrain_RadiusOnly.pt
[HyperMedDiff-Risk] ===== Experiment 18/18: 18_Pretrain_HDDOnly =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8
  ],
  "embed_dim": 128,
  "lambda_hdd": 0.02,
  "dropout": 0.2,
  "train_lr": 0.0001,
  "lambda_radius": 0.0,
  "lambda_s": 1.0,
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100,
  "use_attention": true,
  "pretrain_code_emb": true,
  "freeze_code_emb": true
}
[Pretrain] Epoch 01 | train=0.0209 | val=0.0225
[Pretrain] Epoch 02 | train=0.0212 | val=0.0215
[Pretrain] Epoch 03 | train=0.0221 | val=0.0219
[Pretrain] Epoch 04 | train=0.0206 | val=0.0209
[Pretrain] Epoch 05 | train=0.0204 | val=0.0205
[Pretrain] Epoch 06 | train=0.0211 | val=0.0197
[Pretrain] Epoch 07 | train=0.0201 | val=0.0215
[Pretrain] Epoch 08 | train=0.0193 | val=0.0189
[Pretrain] Epoch 09 | train=0.0202 | val=0.0188
[Pretrain] Epoch 10 | train=0.0195 | val=0.0185
[Pretrain] Epoch 11 | train=0.0180 | val=0.0186
[Pretrain] Epoch 12 | train=0.0187 | val=0.0159
[Pretrain] Epoch 13 | train=0.0182 | val=0.0173
[Pretrain] Epoch 14 | train=0.0190 | val=0.0177
[Pretrain] Epoch 15 | train=0.0179 | val=0.0168
[Pretrain] Epoch 16 | train=0.0172 | val=0.0164
[Pretrain] Epoch 17 | train=0.0174 | val=0.0164
[Pretrain] Epoch 18 | train=0.0166 | val=0.0162
[Pretrain] Epoch 19 | train=0.0163 | val=0.0151
[Pretrain] Epoch 20 | train=0.0168 | val=0.0164
[Pretrain] Epoch 21 | train=0.0160 | val=0.0160
[Pretrain] Epoch 22 | train=0.0158 | val=0.0154
[Pretrain] Epoch 23 | train=0.0153 | val=0.0162
[Pretrain] Epoch 24 | train=0.0156 | val=0.0154
[Pretrain] Epoch 25 | train=0.0160 | val=0.0155
[Pretrain] Epoch 26 | train=0.0142 | val=0.0149
[Pretrain] Epoch 27 | train=0.0138 | val=0.0139
[Pretrain] Epoch 28 | train=0.0141 | val=0.0146
[Pretrain] Epoch 29 | train=0.0151 | val=0.0139
[Pretrain] Epoch 30 | train=0.0129 | val=0.0134
[HyperMedDiff-Risk] Code embedding pretraining enabled.
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-pretrain eval): 0.8088 ± 0.0067
[Check] code_emb.requires_grad = False
[HyperMedDiff-Risk] Epoch 001 | Train 13.8727 | Val 10.9812
[HyperMedDiff-Risk] Epoch 002 | Train 11.3584 | Val 10.8494
[HyperMedDiff-Risk] Epoch 003 | Train 11.1214 | Val 10.4737
[HyperMedDiff-Risk] Epoch 004 | Train 10.5715 | Val 9.6953
[HyperMedDiff-Risk] Epoch 005 | Train 9.8293 | Val 8.7403
[HyperMedDiff-Risk] Epoch 006 | Train 9.1727 | Val 8.1338
[HyperMedDiff-Risk] Epoch 007 | Train 8.6210 | Val 7.5418
[HyperMedDiff-Risk] Epoch 008 | Train 8.1415 | Val 7.0730
[HyperMedDiff-Risk] Epoch 009 | Train 7.7549 | Val 6.6150
[HyperMedDiff-Risk] Epoch 010 | Train 7.4080 | Val 6.2561
[HyperMedDiff-Risk] Epoch 011 | Train 7.1312 | Val 5.8642
[HyperMedDiff-Risk] Epoch 012 | Train 6.8008 | Val 5.7241
[HyperMedDiff-Risk] Epoch 013 | Train 6.5798 | Val 5.2747
[HyperMedDiff-Risk] Epoch 014 | Train 6.2938 | Val 5.1893
[HyperMedDiff-Risk] Epoch 015 | Train 6.1223 | Val 4.8809
[HyperMedDiff-Risk] Epoch 016 | Train 5.9680 | Val 4.7969
[HyperMedDiff-Risk] Epoch 017 | Train 5.7670 | Val 4.4458
[HyperMedDiff-Risk] Epoch 018 | Train 5.6066 | Val 4.4127
[HyperMedDiff-Risk] Epoch 019 | Train 5.4861 | Val 4.2387
[HyperMedDiff-Risk] Epoch 020 | Train 5.2939 | Val 4.0603
[HyperMedDiff-Risk] Epoch 021 | Train 5.1786 | Val 3.9941
[HyperMedDiff-Risk] Epoch 022 | Train 5.0808 | Val 3.8602
[HyperMedDiff-Risk] Epoch 023 | Train 4.9199 | Val 3.6826
[HyperMedDiff-Risk] Epoch 024 | Train 4.8039 | Val 3.5099
[HyperMedDiff-Risk] Epoch 025 | Train 4.7209 | Val 3.5778
[HyperMedDiff-Risk] Epoch 026 | Train 4.5842 | Val 3.3323
[HyperMedDiff-Risk] Epoch 027 | Train 4.5174 | Val 3.2495
[HyperMedDiff-Risk] Epoch 028 | Train 4.4449 | Val 3.1236
[HyperMedDiff-Risk] Epoch 029 | Train 4.3395 | Val 3.0171
[HyperMedDiff-Risk] Epoch 030 | Train 4.2353 | Val 3.0974
[HyperMedDiff-Risk] Epoch 031 | Train 4.2034 | Val 2.9759
[HyperMedDiff-Risk] Epoch 032 | Train 4.1249 | Val 2.8546
[HyperMedDiff-Risk] Epoch 033 | Train 4.0560 | Val 2.8015
[HyperMedDiff-Risk] Epoch 034 | Train 4.0011 | Val 2.8190
[HyperMedDiff-Risk] Epoch 035 | Train 3.9406 | Val 2.7783
[HyperMedDiff-Risk] Epoch 036 | Train 3.8603 | Val 2.7018
[HyperMedDiff-Risk] Epoch 037 | Train 3.8668 | Val 2.6272
[HyperMedDiff-Risk] Epoch 038 | Train 3.8193 | Val 2.6172
[HyperMedDiff-Risk] Epoch 039 | Train 3.7695 | Val 2.6494
[HyperMedDiff-Risk] Epoch 040 | Train 3.7369 | Val 2.6109
[HyperMedDiff-Risk] Epoch 041 | Train 3.6986 | Val 2.5275
[HyperMedDiff-Risk] Epoch 042 | Train 3.6672 | Val 2.4832
[HyperMedDiff-Risk] Epoch 043 | Train 3.6086 | Val 2.4211
[HyperMedDiff-Risk] Epoch 044 | Train 3.5819 | Val 2.3525
[HyperMedDiff-Risk] Epoch 045 | Train 3.5284 | Val 2.3836
[HyperMedDiff-Risk] Epoch 046 | Train 3.5042 | Val 2.3268
[HyperMedDiff-Risk] Epoch 047 | Train 3.4853 | Val 2.3090
[HyperMedDiff-Risk] Epoch 048 | Train 3.4196 | Val 2.3060
[HyperMedDiff-Risk] Epoch 049 | Train 3.4443 | Val 2.2607
[HyperMedDiff-Risk] Epoch 050 | Train 3.3993 | Val 2.2629
[HyperMedDiff-Risk] Epoch 051 | Train 3.3588 | Val 2.1784
[HyperMedDiff-Risk] Epoch 052 | Train 3.3830 | Val 2.1844
[HyperMedDiff-Risk] Epoch 053 | Train 3.3601 | Val 2.2297
[HyperMedDiff-Risk] Epoch 054 | Train 3.3196 | Val 2.1856
[HyperMedDiff-Risk] Epoch 055 | Train 3.2741 | Val 2.2056
[HyperMedDiff-Risk] Epoch 056 | Train 3.2913 | Val 2.1407
[HyperMedDiff-Risk] Epoch 057 | Train 3.2531 | Val 2.1138
[HyperMedDiff-Risk] Epoch 058 | Train 3.2482 | Val 2.1665
[HyperMedDiff-Risk] Epoch 059 | Train 3.2320 | Val 2.0769
[HyperMedDiff-Risk] Epoch 060 | Train 3.2028 | Val 2.0960
[HyperMedDiff-Risk] Epoch 061 | Train 3.2074 | Val 2.0880
[HyperMedDiff-Risk] Epoch 062 | Train 3.2028 | Val 2.1265
[HyperMedDiff-Risk] Epoch 063 | Train 3.1923 | Val 2.0669
[HyperMedDiff-Risk] Epoch 064 | Train 3.1658 | Val 1.9755
[HyperMedDiff-Risk] Epoch 065 | Train 3.1434 | Val 2.0829
[HyperMedDiff-Risk] Epoch 066 | Train 3.1592 | Val 2.0408
[HyperMedDiff-Risk] Epoch 067 | Train 3.1467 | Val 2.0185
[HyperMedDiff-Risk] Epoch 068 | Train 3.1287 | Val 2.0319
[HyperMedDiff-Risk] Epoch 069 | Train 3.1035 | Val 1.9399
[HyperMedDiff-Risk] Epoch 070 | Train 3.0968 | Val 1.9643
[HyperMedDiff-Risk] Epoch 071 | Train 3.0971 | Val 1.9449
[HyperMedDiff-Risk] Epoch 072 | Train 3.0805 | Val 1.9459
[HyperMedDiff-Risk] Epoch 073 | Train 3.0849 | Val 1.9677
[HyperMedDiff-Risk] Epoch 074 | Train 3.0726 | Val 1.9089
[HyperMedDiff-Risk] Epoch 075 | Train 3.0767 | Val 1.9329
[HyperMedDiff-Risk] Epoch 076 | Train 3.0634 | Val 1.9335
[HyperMedDiff-Risk] Epoch 077 | Train 3.0494 | Val 1.8947
[HyperMedDiff-Risk] Epoch 078 | Train 3.0463 | Val 1.9500
[HyperMedDiff-Risk] Epoch 079 | Train 3.0310 | Val 1.9379
[HyperMedDiff-Risk] Epoch 080 | Train 3.0061 | Val 1.9541
[HyperMedDiff-Risk] Epoch 081 | Train 3.0298 | Val 2.0207
[HyperMedDiff-Risk] Epoch 082 | Train 3.0220 | Val 1.8857
[HyperMedDiff-Risk] Epoch 083 | Train 3.0265 | Val 1.9283
[HyperMedDiff-Risk] Epoch 084 | Train 3.0074 | Val 1.8907
[HyperMedDiff-Risk] Epoch 085 | Train 3.0068 | Val 1.9041
[HyperMedDiff-Risk] Epoch 086 | Train 3.0169 | Val 1.8923
[HyperMedDiff-Risk] Epoch 087 | Train 3.0119 | Val 1.8587
[HyperMedDiff-Risk] Epoch 088 | Train 2.9946 | Val 1.8619
[HyperMedDiff-Risk] Epoch 089 | Train 3.0066 | Val 1.8772
[HyperMedDiff-Risk] Epoch 090 | Train 3.0059 | Val 1.9057
[HyperMedDiff-Risk] Epoch 091 | Train 2.9813 | Val 1.8699
[HyperMedDiff-Risk] Epoch 092 | Train 2.9937 | Val 1.8464
[HyperMedDiff-Risk] Epoch 093 | Train 3.0073 | Val 1.8691
[HyperMedDiff-Risk] Epoch 094 | Train 2.9908 | Val 1.8433
[HyperMedDiff-Risk] Epoch 095 | Train 2.9928 | Val 1.8505
[HyperMedDiff-Risk] Epoch 096 | Train 3.0150 | Val 1.8710
[HyperMedDiff-Risk] Epoch 097 | Train 3.0128 | Val 1.8851
[HyperMedDiff-Risk] Epoch 098 | Train 2.9821 | Val 1.8732
[HyperMedDiff-Risk] Epoch 099 | Train 2.9938 | Val 1.8797
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 18): 1.8433
[HyperMedDiff-Risk] Saved training curve plot to results/plots/18_Pretrain_HDDOnly.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8662219173383638,
  "auprc": 0.802212053096634
}
[HyperMedDiff-Risk] Frozen-code diffusion faithfulness (post-train eval): 0.8088 ± 0.0067
[HyperMedDiff-Risk] Diffusion/embedding Spearman rho: 0.7620
[HyperMedDiff-Risk] Tree metric debug summary: kept=4999 | same=1 | missing_code=0 | missing_path=0 | bad_lca=0 | tree_dist_le0=0
[HyperMedDiff-Risk] Warning: >80% of kept pairs have LCA depth 0.
[HyperMedDiff-Risk] Tree/embedding Spearman rho (cms): 0.0121
[HyperMedDiff-Risk] Distortion by depth (depth,count,mean,std):
[HyperMedDiff-Risk] depth=0 count=4999 mean=0.1654 std=0.0504
[HyperMedDiff-Risk] Distortion plot has a single depth bucket.
[HyperMedDiff-Risk] Saved distortion vs depth plot to results/plots/18_Pretrain_HDDOnly_distortion_depth.png
[HyperMedDiff-Risk] Saved diffusion embedding UMAP to results/plots/18_Pretrain_HDDOnly_umap.png
Saved checkpoint to results/checkpoints/18_Pretrain_HDDOnly.pt
[HyperMedDiff-Risk] ==== Ablation Summary ====
Run | Experiment               | ValLoss | AUROC  | AUPRC  | Accuracy | F1     | Diff–Lat ρ | Tree–Lat ρ | Dist μ | Dist σ
----+--------------------------+---------+--------+--------+----------+--------+------------+------------+--------+-------
1   | 01_Baseline              | 1.7283  | 0.8733 | 0.8057 | 0.7534   | 0.7175 | 0.7356     | -0.0156    | 0.1953 | 0.0349
2   | 02_NoDiffusion           | 1.7493  | 0.8737 | 0.8045 | 0.7534   | 0.7175 | 0.8411     | 0.0210     | 0.1961 | 0.0417
3   | 03_LocalDiff             | 1.8890  | 0.8758 | 0.8128 | 0.7534   | 0.7175 | 0.8023     | 0.0181     | 0.2011 | 0.0406
4   | 04_GlobalDiff_Stress     | 1.9065  | 0.8764 | 0.8160 | 0.7534   | 0.7175 | 0.7246     | -0.0011    | 0.1956 | 0.0347
5   | 05_NoHDD                 | 1.8788  | 0.8765 | 0.8160 | 0.7534   | 0.7175 | 0.0093     | 0.0025     | 0.2830 | 0.0248
6   | 06_StrongHDD             | 1.7873  | 0.8712 | 0.7992 | 0.7534   | 0.7175 | 0.8154     | 0.0119     | 0.1731 | 0.0369
7   | 07_HighDropout           | 1.8216  | 0.8739 | 0.8090 | 0.7534   | 0.7175 | 0.7633     | 0.0167     | 0.2035 | 0.0400
8   | 08_SmallDim              | 1.8715  | 0.8624 | 0.7907 | 0.7534   | 0.7175 | 0.5827     | 0.0254     | 0.1648 | 0.0254
9   | 09_NoSynthRisk           | 1.2222  | 0.8844 | 0.8207 | 0.8308   | 0.7729 | 0.7659     | 0.0010     | 0.2038 | 0.0401
10  | 10_GenFocus              | 3.5027  | 0.8711 | 0.8036 | 0.7534   | 0.7175 | 0.7533     | 0.0130     | 0.2018 | 0.0386
11  | 11_NoAttention           | 1.8854  | 0.8453 | 0.7568 | 0.7534   | 0.7175 | 0.7558     | -0.0019    | 0.1978 | 0.0362
12  | 12_NoConsistency         | 1.8218  | 0.8762 | 0.8143 | 0.7534   | 0.7175 | 0.7590     | 0.0003     | 0.2043 | 0.0403
13  | 13_NoFlow                | 0.2783  | 0.9564 | 0.9167 | 0.8821   | 0.8446 | 0.7769     | 0.0116     | 0.2044 | 0.0406
14  | 14_RealOnlyRisk          | 0.5617  | 0.8667 | 0.7949 | 0.7534   | 0.7175 | 0.7622     | 0.0057     | 0.2042 | 0.0401
15  | 15_NoPretrain_RandomInit | 1.7487  | 0.8675 | 0.7933 | 0.7534   | 0.7175 | -0.0042    | 0.0056     | 0.0157 | 0.0015
16  | 16_UnfreezeEmbeddings    | 1.5213  | 0.9793 | 0.9503 | 0.9505   | 0.9363 | 0.3574     | 0.0000     | 0.4480 | 0.6238
17  | 17_Pretrain_RadiusOnly   | 1.9104  | 0.8765 | 0.8153 | 0.7534   | 0.7175 | 0.0034     | -0.0061    | 0.2829 | 0.0249
18  | 18_Pretrain_HDDOnly      | 1.8433  | 0.8662 | 0.8022 | 0.7534   | 0.7175 | 0.7620     | 0.0121     | 0.1654 | 0.0504
