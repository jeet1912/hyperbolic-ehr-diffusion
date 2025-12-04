Using device: mps
[MIMIC] Loading data/mimic_hf_cohort.pkl ...
[MIMIC] Patients: 7397 | Vocab size: 4817
[HyperMedDiff-Risk] Real trajectory stats: {
  "patients": 7397,
  "avg_visits_per_patient": 1.7,
  "avg_codes_per_visit": 424.04,
  "max_visits": 12,
  "max_codes": 9261
}
[HyperMedDiff-Risk] Skipping 01_Baseline (already executed separately).
[HyperMedDiff-Risk] Running 9 ablation configurations.
[HyperMedDiff-Risk] ===== Experiment 1/9: 02_NoDiffusion =====
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
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0035 | val=0.0033
[Pretrain] Epoch 02 | train=0.0033 | val=0.0030
[Pretrain] Epoch 03 | train=0.0031 | val=0.0028
[Pretrain] Epoch 04 | train=0.0029 | val=0.0027
[Pretrain] Epoch 05 | train=0.0027 | val=0.0026
[Pretrain] Epoch 06 | train=0.0026 | val=0.0025
[Pretrain] Epoch 07 | train=0.0025 | val=0.0024
[Pretrain] Epoch 08 | train=0.0024 | val=0.0023
[Pretrain] Epoch 09 | train=0.0023 | val=0.0023
[Pretrain] Epoch 10 | train=0.0023 | val=0.0022
[Pretrain] Epoch 11 | train=0.0022 | val=0.0022
[Pretrain] Epoch 12 | train=0.0022 | val=0.0022
[Pretrain] Epoch 13 | train=0.0022 | val=0.0022
[Pretrain] Epoch 14 | train=0.0022 | val=0.0022
[Pretrain] Epoch 15 | train=0.0022 | val=0.0022
[Pretrain] Epoch 16 | train=0.0022 | val=0.0022
[Pretrain] Epoch 17 | train=0.0022 | val=0.0022
[Pretrain] Epoch 18 | train=0.0022 | val=0.0022
[Pretrain] Epoch 19 | train=0.0022 | val=0.0022
[Pretrain] Epoch 20 | train=0.0022 | val=0.0022
[Pretrain] Epoch 21 | train=0.0022 | val=0.0022
[Pretrain] Epoch 22 | train=0.0022 | val=0.0022
[Pretrain] Epoch 23 | train=0.0022 | val=0.0022
[Pretrain] Epoch 24 | train=0.0022 | val=0.0022
[Pretrain] Epoch 25 | train=0.0022 | val=0.0022
[Pretrain] Epoch 26 | train=0.0022 | val=0.0022
[Pretrain] Epoch 27 | train=0.0022 | val=0.0022
[Pretrain] Epoch 28 | train=0.0021 | val=0.0022
[Pretrain] Epoch 29 | train=0.0021 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0022
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8885
[HyperMedDiff-Risk] Epoch 001 | Train 13.6892 | Val 10.9466
[HyperMedDiff-Risk] Epoch 002 | Train 11.3629 | Val 10.8213
[HyperMedDiff-Risk] Epoch 003 | Train 11.1341 | Val 10.4881
[HyperMedDiff-Risk] Epoch 004 | Train 10.6207 | Val 9.7262
[HyperMedDiff-Risk] Epoch 005 | Train 9.9165 | Val 8.8847
[HyperMedDiff-Risk] Epoch 006 | Train 9.2242 | Val 8.1647
[HyperMedDiff-Risk] Epoch 007 | Train 8.6385 | Val 7.5399
[HyperMedDiff-Risk] Epoch 008 | Train 8.1862 | Val 7.0634
[HyperMedDiff-Risk] Epoch 009 | Train 7.7725 | Val 6.5520
[HyperMedDiff-Risk] Epoch 010 | Train 7.4099 | Val 6.2602
[HyperMedDiff-Risk] Epoch 011 | Train 7.0932 | Val 5.9188
[HyperMedDiff-Risk] Epoch 012 | Train 6.7758 | Val 5.5778
[HyperMedDiff-Risk] Epoch 013 | Train 6.5522 | Val 5.3453
[HyperMedDiff-Risk] Epoch 014 | Train 6.3656 | Val 5.1466
[HyperMedDiff-Risk] Epoch 015 | Train 6.1565 | Val 4.8869
[HyperMedDiff-Risk] Epoch 016 | Train 5.9047 | Val 4.6580
[HyperMedDiff-Risk] Epoch 017 | Train 5.7704 | Val 4.5628
[HyperMedDiff-Risk] Epoch 018 | Train 5.6132 | Val 4.4369
[HyperMedDiff-Risk] Epoch 019 | Train 5.4253 | Val 4.1674
[HyperMedDiff-Risk] Epoch 020 | Train 5.2637 | Val 4.0109
[HyperMedDiff-Risk] Epoch 021 | Train 5.1218 | Val 3.8786
[HyperMedDiff-Risk] Epoch 022 | Train 4.9822 | Val 3.7838
[HyperMedDiff-Risk] Epoch 023 | Train 4.8588 | Val 3.7066
[HyperMedDiff-Risk] Epoch 024 | Train 4.7978 | Val 3.4716
[HyperMedDiff-Risk] Epoch 025 | Train 4.6632 | Val 3.3554
[HyperMedDiff-Risk] Epoch 026 | Train 4.5607 | Val 3.2403
[HyperMedDiff-Risk] Epoch 027 | Train 4.4602 | Val 3.2756
[HyperMedDiff-Risk] Epoch 028 | Train 4.4162 | Val 3.2558
[HyperMedDiff-Risk] Epoch 029 | Train 4.2737 | Val 3.1312
[HyperMedDiff-Risk] Epoch 030 | Train 4.1949 | Val 2.9753
[HyperMedDiff-Risk] Epoch 031 | Train 4.1914 | Val 2.9580
[HyperMedDiff-Risk] Epoch 032 | Train 4.1209 | Val 2.8555
[HyperMedDiff-Risk] Epoch 033 | Train 4.0098 | Val 2.8335
[HyperMedDiff-Risk] Epoch 034 | Train 4.0003 | Val 2.8113
[HyperMedDiff-Risk] Epoch 035 | Train 3.9434 | Val 2.6711
[HyperMedDiff-Risk] Epoch 036 | Train 3.9133 | Val 2.6571
[HyperMedDiff-Risk] Epoch 037 | Train 3.8137 | Val 2.6698
[HyperMedDiff-Risk] Epoch 038 | Train 3.7827 | Val 2.6412
[HyperMedDiff-Risk] Epoch 039 | Train 3.7135 | Val 2.5026
[HyperMedDiff-Risk] Epoch 040 | Train 3.6629 | Val 2.5863
[HyperMedDiff-Risk] Epoch 041 | Train 3.6541 | Val 2.4751
[HyperMedDiff-Risk] Epoch 042 | Train 3.5857 | Val 2.4068
[HyperMedDiff-Risk] Epoch 043 | Train 3.5541 | Val 2.4084
[HyperMedDiff-Risk] Epoch 044 | Train 3.5219 | Val 2.4161
[HyperMedDiff-Risk] Epoch 045 | Train 3.5145 | Val 2.2957
[HyperMedDiff-Risk] Epoch 046 | Train 3.4511 | Val 2.2732
[HyperMedDiff-Risk] Epoch 047 | Train 3.4436 | Val 2.3537
[HyperMedDiff-Risk] Epoch 048 | Train 3.4070 | Val 2.2148
[HyperMedDiff-Risk] Epoch 049 | Train 3.3788 | Val 2.1958
[HyperMedDiff-Risk] Epoch 050 | Train 3.3158 | Val 2.0986
[HyperMedDiff-Risk] Epoch 051 | Train 3.3545 | Val 2.1382
[HyperMedDiff-Risk] Epoch 052 | Train 3.3123 | Val 2.1455
[HyperMedDiff-Risk] Epoch 053 | Train 3.2791 | Val 2.1350
[HyperMedDiff-Risk] Epoch 054 | Train 3.2590 | Val 2.0872
[HyperMedDiff-Risk] Epoch 055 | Train 3.2590 | Val 2.0416
[HyperMedDiff-Risk] Epoch 056 | Train 3.2115 | Val 2.1045
[HyperMedDiff-Risk] Epoch 057 | Train 3.2026 | Val 2.0936
[HyperMedDiff-Risk] Epoch 058 | Train 3.1718 | Val 1.9965
[HyperMedDiff-Risk] Epoch 059 | Train 3.1693 | Val 2.0665
[HyperMedDiff-Risk] Epoch 060 | Train 3.1764 | Val 2.0409
[HyperMedDiff-Risk] Epoch 061 | Train 3.1186 | Val 2.0359
[HyperMedDiff-Risk] Epoch 062 | Train 3.1233 | Val 1.9687
[HyperMedDiff-Risk] Epoch 063 | Train 3.0958 | Val 2.0019
[HyperMedDiff-Risk] Epoch 064 | Train 3.1152 | Val 2.0209
[HyperMedDiff-Risk] Epoch 065 | Train 3.0876 | Val 1.9869
[HyperMedDiff-Risk] Epoch 066 | Train 3.1089 | Val 2.0087
[HyperMedDiff-Risk] Epoch 067 | Train 3.0703 | Val 1.9560
[HyperMedDiff-Risk] Epoch 068 | Train 3.0644 | Val 1.9283
[HyperMedDiff-Risk] Epoch 069 | Train 3.0732 | Val 1.9476
[HyperMedDiff-Risk] Epoch 070 | Train 3.0907 | Val 1.9176
[HyperMedDiff-Risk] Epoch 071 | Train 3.0338 | Val 1.9441
[HyperMedDiff-Risk] Epoch 072 | Train 3.0520 | Val 1.8928
[HyperMedDiff-Risk] Epoch 073 | Train 3.0370 | Val 1.9451
[HyperMedDiff-Risk] Epoch 074 | Train 3.0185 | Val 1.8792
[HyperMedDiff-Risk] Epoch 075 | Train 3.0167 | Val 1.9107
[HyperMedDiff-Risk] Epoch 076 | Train 3.0024 | Val 1.9212
[HyperMedDiff-Risk] Epoch 077 | Train 3.0142 | Val 1.8952
[HyperMedDiff-Risk] Epoch 078 | Train 2.9913 | Val 1.8910
[HyperMedDiff-Risk] Epoch 079 | Train 3.0168 | Val 1.8687
[HyperMedDiff-Risk] Epoch 080 | Train 3.0009 | Val 1.8931
[HyperMedDiff-Risk] Epoch 081 | Train 2.9828 | Val 1.9121
[HyperMedDiff-Risk] Epoch 082 | Train 3.0194 | Val 1.8613
[HyperMedDiff-Risk] Epoch 083 | Train 2.9883 | Val 1.8477
[HyperMedDiff-Risk] Epoch 084 | Train 2.9962 | Val 1.8705
[HyperMedDiff-Risk] Epoch 085 | Train 2.9748 | Val 1.8796
[HyperMedDiff-Risk] Epoch 086 | Train 2.9890 | Val 1.8532
[HyperMedDiff-Risk] Epoch 087 | Train 2.9807 | Val 1.8808
[HyperMedDiff-Risk] Epoch 088 | Train 2.9801 | Val 1.8176
[HyperMedDiff-Risk] Epoch 089 | Train 2.9686 | Val 1.8186
[HyperMedDiff-Risk] Epoch 090 | Train 2.9566 | Val 1.8965
[HyperMedDiff-Risk] Epoch 091 | Train 2.9648 | Val 1.8825
[HyperMedDiff-Risk] Epoch 092 | Train 2.9731 | Val 1.8422
[HyperMedDiff-Risk] Epoch 093 | Train 2.9577 | Val 1.8176
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 1): 1.8176
[HyperMedDiff-Risk] Saved training curve plot to results/plots/02_NoDiffusion.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.868650317269765,
  "auprc": 0.7918767239331813
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8897
Saved checkpoint to results/checkpoints/02_NoDiffusion.pt
[HyperMedDiff-Risk] ===== Experiment 2/9: 03_LocalDiff =====
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
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0036 | val=0.0033
[Pretrain] Epoch 02 | train=0.0033 | val=0.0031
[Pretrain] Epoch 03 | train=0.0031 | val=0.0029
[Pretrain] Epoch 04 | train=0.0029 | val=0.0027
[Pretrain] Epoch 05 | train=0.0027 | val=0.0026
[Pretrain] Epoch 06 | train=0.0026 | val=0.0025
[Pretrain] Epoch 07 | train=0.0025 | val=0.0024
[Pretrain] Epoch 08 | train=0.0024 | val=0.0023
[Pretrain] Epoch 09 | train=0.0023 | val=0.0023
[Pretrain] Epoch 10 | train=0.0023 | val=0.0022
[Pretrain] Epoch 11 | train=0.0022 | val=0.0022
[Pretrain] Epoch 12 | train=0.0022 | val=0.0022
[Pretrain] Epoch 13 | train=0.0022 | val=0.0022
[Pretrain] Epoch 14 | train=0.0022 | val=0.0022
[Pretrain] Epoch 15 | train=0.0022 | val=0.0022
[Pretrain] Epoch 16 | train=0.0022 | val=0.0022
[Pretrain] Epoch 17 | train=0.0022 | val=0.0022
[Pretrain] Epoch 18 | train=0.0022 | val=0.0022
[Pretrain] Epoch 19 | train=0.0022 | val=0.0022
[Pretrain] Epoch 20 | train=0.0022 | val=0.0022
[Pretrain] Epoch 21 | train=0.0022 | val=0.0022
[Pretrain] Epoch 22 | train=0.0022 | val=0.0022
[Pretrain] Epoch 23 | train=0.0021 | val=0.0021
[Pretrain] Epoch 24 | train=0.0022 | val=0.0021
[Pretrain] Epoch 25 | train=0.0021 | val=0.0021
[Pretrain] Epoch 26 | train=0.0021 | val=0.0021
[Pretrain] Epoch 27 | train=0.0021 | val=0.0021
[Pretrain] Epoch 28 | train=0.0021 | val=0.0021
[Pretrain] Epoch 29 | train=0.0021 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0021
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8487
[HyperMedDiff-Risk] Epoch 001 | Train 13.8101 | Val 10.9840
[HyperMedDiff-Risk] Epoch 002 | Train 11.3552 | Val 10.8110
[HyperMedDiff-Risk] Epoch 003 | Train 11.1193 | Val 10.4817
[HyperMedDiff-Risk] Epoch 004 | Train 10.6353 | Val 9.7406
[HyperMedDiff-Risk] Epoch 005 | Train 9.9244 | Val 8.8896
[HyperMedDiff-Risk] Epoch 006 | Train 9.2703 | Val 8.2240
[HyperMedDiff-Risk] Epoch 007 | Train 8.6855 | Val 7.5492
[HyperMedDiff-Risk] Epoch 008 | Train 8.1913 | Val 7.0468
[HyperMedDiff-Risk] Epoch 009 | Train 7.8054 | Val 6.6337
[HyperMedDiff-Risk] Epoch 010 | Train 7.4133 | Val 6.2984
[HyperMedDiff-Risk] Epoch 011 | Train 7.0689 | Val 5.9497
[HyperMedDiff-Risk] Epoch 012 | Train 6.8215 | Val 5.6028
[HyperMedDiff-Risk] Epoch 013 | Train 6.5549 | Val 5.3665
[HyperMedDiff-Risk] Epoch 014 | Train 6.2896 | Val 5.0913
[HyperMedDiff-Risk] Epoch 015 | Train 6.0802 | Val 4.9474
[HyperMedDiff-Risk] Epoch 016 | Train 5.8918 | Val 4.7120
[HyperMedDiff-Risk] Epoch 017 | Train 5.7284 | Val 4.4620
[HyperMedDiff-Risk] Epoch 018 | Train 5.5221 | Val 4.4115
[HyperMedDiff-Risk] Epoch 019 | Train 5.4397 | Val 4.0867
[HyperMedDiff-Risk] Epoch 020 | Train 5.2422 | Val 4.0431
[HyperMedDiff-Risk] Epoch 021 | Train 5.1310 | Val 3.8397
[HyperMedDiff-Risk] Epoch 022 | Train 5.0192 | Val 3.7287
[HyperMedDiff-Risk] Epoch 023 | Train 4.9064 | Val 3.6143
[HyperMedDiff-Risk] Epoch 024 | Train 4.7886 | Val 3.5674
[HyperMedDiff-Risk] Epoch 025 | Train 4.6993 | Val 3.5021
[HyperMedDiff-Risk] Epoch 026 | Train 4.6115 | Val 3.3730
[HyperMedDiff-Risk] Epoch 027 | Train 4.4869 | Val 3.2220
[HyperMedDiff-Risk] Epoch 028 | Train 4.3653 | Val 3.2047
[HyperMedDiff-Risk] Epoch 029 | Train 4.2733 | Val 3.0209
[HyperMedDiff-Risk] Epoch 030 | Train 4.2703 | Val 2.9781
[HyperMedDiff-Risk] Epoch 031 | Train 4.1769 | Val 2.9121
[HyperMedDiff-Risk] Epoch 032 | Train 4.0899 | Val 2.8126
[HyperMedDiff-Risk] Epoch 033 | Train 4.0581 | Val 2.8241
[HyperMedDiff-Risk] Epoch 034 | Train 3.9796 | Val 2.7903
[HyperMedDiff-Risk] Epoch 035 | Train 3.9028 | Val 2.6723
[HyperMedDiff-Risk] Epoch 036 | Train 3.8706 | Val 2.5939
[HyperMedDiff-Risk] Epoch 037 | Train 3.8021 | Val 2.6163
[HyperMedDiff-Risk] Epoch 038 | Train 3.7492 | Val 2.5323
[HyperMedDiff-Risk] Epoch 039 | Train 3.6628 | Val 2.4991
[HyperMedDiff-Risk] Epoch 040 | Train 3.6421 | Val 2.3965
[HyperMedDiff-Risk] Epoch 041 | Train 3.5747 | Val 2.4038
[HyperMedDiff-Risk] Epoch 042 | Train 3.5537 | Val 2.3774
[HyperMedDiff-Risk] Epoch 043 | Train 3.5243 | Val 2.3118
[HyperMedDiff-Risk] Epoch 044 | Train 3.4768 | Val 2.2890
[HyperMedDiff-Risk] Epoch 045 | Train 3.4266 | Val 2.2606
[HyperMedDiff-Risk] Epoch 046 | Train 3.4317 | Val 2.2631
[HyperMedDiff-Risk] Epoch 047 | Train 3.3938 | Val 2.1690
[HyperMedDiff-Risk] Epoch 048 | Train 3.3687 | Val 2.1504
[HyperMedDiff-Risk] Epoch 049 | Train 3.3289 | Val 2.2022
[HyperMedDiff-Risk] Epoch 050 | Train 3.3008 | Val 2.1192
[HyperMedDiff-Risk] Epoch 051 | Train 3.2803 | Val 2.0862
[HyperMedDiff-Risk] Epoch 052 | Train 3.2775 | Val 2.1153
[HyperMedDiff-Risk] Epoch 053 | Train 3.2473 | Val 2.0081
[HyperMedDiff-Risk] Epoch 054 | Train 3.2058 | Val 2.0723
[HyperMedDiff-Risk] Epoch 055 | Train 3.2034 | Val 2.0802
[HyperMedDiff-Risk] Epoch 056 | Train 3.1868 | Val 1.9546
[HyperMedDiff-Risk] Epoch 057 | Train 3.1615 | Val 1.9995
[HyperMedDiff-Risk] Epoch 058 | Train 3.1281 | Val 1.9594
[HyperMedDiff-Risk] Epoch 059 | Train 3.1387 | Val 2.0206
[HyperMedDiff-Risk] Epoch 060 | Train 3.1205 | Val 1.9616
[HyperMedDiff-Risk] Epoch 061 | Train 3.1166 | Val 1.9651
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 2): 1.9546
[HyperMedDiff-Risk] Saved training curve plot to results/plots/03_LocalDiff.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8726462013376779,
  "auprc": 0.8053855917597685
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8533
Saved checkpoint to results/checkpoints/03_LocalDiff.pt
[HyperMedDiff-Risk] ===== Experiment 3/9: 04_GlobalDiff =====
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
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0036 | val=0.0033
[Pretrain] Epoch 02 | train=0.0034 | val=0.0031
[Pretrain] Epoch 03 | train=0.0031 | val=0.0029
[Pretrain] Epoch 04 | train=0.0029 | val=0.0027
[Pretrain] Epoch 05 | train=0.0027 | val=0.0026
[Pretrain] Epoch 06 | train=0.0026 | val=0.0025
[Pretrain] Epoch 07 | train=0.0025 | val=0.0024
[Pretrain] Epoch 08 | train=0.0024 | val=0.0023
[Pretrain] Epoch 09 | train=0.0023 | val=0.0022
[Pretrain] Epoch 10 | train=0.0022 | val=0.0022
[Pretrain] Epoch 11 | train=0.0022 | val=0.0022
[Pretrain] Epoch 12 | train=0.0022 | val=0.0022
[Pretrain] Epoch 13 | train=0.0022 | val=0.0022
[Pretrain] Epoch 14 | train=0.0022 | val=0.0022
[Pretrain] Epoch 15 | train=0.0022 | val=0.0022
[Pretrain] Epoch 16 | train=0.0022 | val=0.0022
[Pretrain] Epoch 17 | train=0.0022 | val=0.0021
[Pretrain] Epoch 18 | train=0.0022 | val=0.0021
[Pretrain] Epoch 19 | train=0.0021 | val=0.0022
[Pretrain] Epoch 20 | train=0.0021 | val=0.0021
[Pretrain] Epoch 21 | train=0.0021 | val=0.0021
[Pretrain] Epoch 22 | train=0.0021 | val=0.0021
[Pretrain] Epoch 23 | train=0.0021 | val=0.0021
[Pretrain] Epoch 24 | train=0.0021 | val=0.0021
[Pretrain] Epoch 25 | train=0.0021 | val=0.0021
[Pretrain] Epoch 26 | train=0.0021 | val=0.0021
[Pretrain] Epoch 27 | train=0.0021 | val=0.0021
[Pretrain] Epoch 28 | train=0.0021 | val=0.0021
[Pretrain] Epoch 29 | train=0.0021 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0021
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8453
[HyperMedDiff-Risk] Epoch 001 | Train 13.6746 | Val 10.9603
[HyperMedDiff-Risk] Epoch 002 | Train 11.3493 | Val 10.7806
[HyperMedDiff-Risk] Epoch 003 | Train 11.0499 | Val 10.3703
[HyperMedDiff-Risk] Epoch 004 | Train 10.4926 | Val 9.5349
[HyperMedDiff-Risk] Epoch 005 | Train 9.7864 | Val 8.8058
[HyperMedDiff-Risk] Epoch 006 | Train 9.1446 | Val 8.0582
[HyperMedDiff-Risk] Epoch 007 | Train 8.6030 | Val 7.4545
[HyperMedDiff-Risk] Epoch 008 | Train 8.0922 | Val 6.9485
[HyperMedDiff-Risk] Epoch 009 | Train 7.7284 | Val 6.5856
[HyperMedDiff-Risk] Epoch 010 | Train 7.3622 | Val 6.2733
[HyperMedDiff-Risk] Epoch 011 | Train 6.9921 | Val 5.9053
[HyperMedDiff-Risk] Epoch 012 | Train 6.7742 | Val 5.5056
[HyperMedDiff-Risk] Epoch 013 | Train 6.5300 | Val 5.3894
[HyperMedDiff-Risk] Epoch 014 | Train 6.3167 | Val 5.0548
[HyperMedDiff-Risk] Epoch 015 | Train 6.1059 | Val 4.9258
[HyperMedDiff-Risk] Epoch 016 | Train 5.9222 | Val 4.6232
[HyperMedDiff-Risk] Epoch 017 | Train 5.7683 | Val 4.5551
[HyperMedDiff-Risk] Epoch 018 | Train 5.6094 | Val 4.3907
[HyperMedDiff-Risk] Epoch 019 | Train 5.3899 | Val 4.1069
[HyperMedDiff-Risk] Epoch 020 | Train 5.2679 | Val 4.0262
[HyperMedDiff-Risk] Epoch 021 | Train 5.1661 | Val 3.9800
[HyperMedDiff-Risk] Epoch 022 | Train 5.0014 | Val 3.7762
[HyperMedDiff-Risk] Epoch 023 | Train 4.8733 | Val 3.7249
[HyperMedDiff-Risk] Epoch 024 | Train 4.7599 | Val 3.4857
[HyperMedDiff-Risk] Epoch 025 | Train 4.6790 | Val 3.4314
[HyperMedDiff-Risk] Epoch 026 | Train 4.5899 | Val 3.4139
[HyperMedDiff-Risk] Epoch 027 | Train 4.4823 | Val 3.2444
[HyperMedDiff-Risk] Epoch 028 | Train 4.3979 | Val 3.1342
[HyperMedDiff-Risk] Epoch 029 | Train 4.2709 | Val 3.0102
[HyperMedDiff-Risk] Epoch 030 | Train 4.2510 | Val 3.0125
[HyperMedDiff-Risk] Epoch 031 | Train 4.1496 | Val 2.9467
[HyperMedDiff-Risk] Epoch 032 | Train 4.0726 | Val 2.8453
[HyperMedDiff-Risk] Epoch 033 | Train 4.0106 | Val 2.7481
[HyperMedDiff-Risk] Epoch 034 | Train 3.9805 | Val 2.7705
[HyperMedDiff-Risk] Epoch 035 | Train 3.8768 | Val 2.6961
[HyperMedDiff-Risk] Epoch 036 | Train 3.8775 | Val 2.6917
[HyperMedDiff-Risk] Epoch 037 | Train 3.8010 | Val 2.6079
[HyperMedDiff-Risk] Epoch 038 | Train 3.7672 | Val 2.5445
[HyperMedDiff-Risk] Epoch 039 | Train 3.6944 | Val 2.5036
[HyperMedDiff-Risk] Epoch 040 | Train 3.6411 | Val 2.4404
[HyperMedDiff-Risk] Epoch 041 | Train 3.6603 | Val 2.4427
[HyperMedDiff-Risk] Epoch 042 | Train 3.5977 | Val 2.3771
[HyperMedDiff-Risk] Epoch 043 | Train 3.5335 | Val 2.3555
[HyperMedDiff-Risk] Epoch 044 | Train 3.5204 | Val 2.3237
[HyperMedDiff-Risk] Epoch 045 | Train 3.5039 | Val 2.3328
[HyperMedDiff-Risk] Epoch 046 | Train 3.4697 | Val 2.2605
[HyperMedDiff-Risk] Epoch 047 | Train 3.4495 | Val 2.2573
[HyperMedDiff-Risk] Epoch 048 | Train 3.3893 | Val 2.1455
[HyperMedDiff-Risk] Epoch 049 | Train 3.3755 | Val 2.2137
[HyperMedDiff-Risk] Epoch 050 | Train 3.3590 | Val 2.1291
[HyperMedDiff-Risk] Epoch 051 | Train 3.3355 | Val 2.1240
[HyperMedDiff-Risk] Epoch 052 | Train 3.2977 | Val 2.1154
[HyperMedDiff-Risk] Epoch 053 | Train 3.2946 | Val 2.1266
[HyperMedDiff-Risk] Epoch 054 | Train 3.2939 | Val 2.1155
[HyperMedDiff-Risk] Epoch 055 | Train 3.2501 | Val 2.1313
[HyperMedDiff-Risk] Epoch 056 | Train 3.2516 | Val 2.1001
[HyperMedDiff-Risk] Epoch 057 | Train 3.2359 | Val 2.0661
[HyperMedDiff-Risk] Epoch 058 | Train 3.2309 | Val 2.1055
[HyperMedDiff-Risk] Epoch 059 | Train 3.1681 | Val 2.0347
[HyperMedDiff-Risk] Epoch 060 | Train 3.1884 | Val 2.0858
[HyperMedDiff-Risk] Epoch 061 | Train 3.1773 | Val 2.0191
[HyperMedDiff-Risk] Epoch 062 | Train 3.1397 | Val 1.9853
[HyperMedDiff-Risk] Epoch 063 | Train 3.1130 | Val 2.0157
[HyperMedDiff-Risk] Epoch 064 | Train 3.1159 | Val 1.9574
[HyperMedDiff-Risk] Epoch 065 | Train 3.1337 | Val 1.9647
[HyperMedDiff-Risk] Epoch 066 | Train 3.1123 | Val 1.9322
[HyperMedDiff-Risk] Epoch 067 | Train 3.1057 | Val 1.9755
[HyperMedDiff-Risk] Epoch 068 | Train 3.0759 | Val 1.9458
[HyperMedDiff-Risk] Epoch 069 | Train 3.0796 | Val 1.9225
[HyperMedDiff-Risk] Epoch 070 | Train 3.0611 | Val 2.0135
[HyperMedDiff-Risk] Epoch 071 | Train 3.0488 | Val 1.9667
[HyperMedDiff-Risk] Epoch 072 | Train 3.0402 | Val 1.9389
[HyperMedDiff-Risk] Epoch 073 | Train 3.0357 | Val 1.9121
[HyperMedDiff-Risk] Epoch 074 | Train 3.0277 | Val 1.8712
[HyperMedDiff-Risk] Epoch 075 | Train 3.0122 | Val 1.8847
[HyperMedDiff-Risk] Epoch 076 | Train 3.0109 | Val 1.8795
[HyperMedDiff-Risk] Epoch 077 | Train 3.0044 | Val 1.9182
[HyperMedDiff-Risk] Epoch 078 | Train 3.0027 | Val 1.8822
[HyperMedDiff-Risk] Epoch 079 | Train 2.9919 | Val 1.9203
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 3): 1.8712
[HyperMedDiff-Risk] Saved training curve plot to results/plots/04_GlobalDiff.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8739598696621507,
  "auprc": 0.8058157329291071
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8380
Saved checkpoint to results/checkpoints/04_GlobalDiff.pt
[HyperMedDiff-Risk] ===== Experiment 4/9: 05_NoHDD =====
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
  "train_epochs": 100
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
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.0367
[HyperMedDiff-Risk] Epoch 001 | Train 13.4293 | Val 10.9678
[HyperMedDiff-Risk] Epoch 002 | Train 11.3600 | Val 10.7856
[HyperMedDiff-Risk] Epoch 003 | Train 11.0929 | Val 10.4244
[HyperMedDiff-Risk] Epoch 004 | Train 10.5993 | Val 9.6279
[HyperMedDiff-Risk] Epoch 005 | Train 9.8791 | Val 8.8211
[HyperMedDiff-Risk] Epoch 006 | Train 9.2050 | Val 8.1379
[HyperMedDiff-Risk] Epoch 007 | Train 8.6718 | Val 7.5700
[HyperMedDiff-Risk] Epoch 008 | Train 8.2039 | Val 7.1302
[HyperMedDiff-Risk] Epoch 009 | Train 7.7487 | Val 6.6069
[HyperMedDiff-Risk] Epoch 010 | Train 7.3923 | Val 6.2576
[HyperMedDiff-Risk] Epoch 011 | Train 7.0910 | Val 5.9769
[HyperMedDiff-Risk] Epoch 012 | Train 6.8536 | Val 5.6177
[HyperMedDiff-Risk] Epoch 013 | Train 6.6358 | Val 5.4564
[HyperMedDiff-Risk] Epoch 014 | Train 6.3732 | Val 5.2020
[HyperMedDiff-Risk] Epoch 015 | Train 6.1722 | Val 4.8739
[HyperMedDiff-Risk] Epoch 016 | Train 5.9924 | Val 4.6765
[HyperMedDiff-Risk] Epoch 017 | Train 5.8236 | Val 4.5980
[HyperMedDiff-Risk] Epoch 018 | Train 5.6699 | Val 4.4063
[HyperMedDiff-Risk] Epoch 019 | Train 5.4645 | Val 4.1999
[HyperMedDiff-Risk] Epoch 020 | Train 5.3619 | Val 4.1558
[HyperMedDiff-Risk] Epoch 021 | Train 5.1861 | Val 4.0227
[HyperMedDiff-Risk] Epoch 022 | Train 5.0955 | Val 3.8067
[HyperMedDiff-Risk] Epoch 023 | Train 4.9238 | Val 3.6729
[HyperMedDiff-Risk] Epoch 024 | Train 4.8441 | Val 3.5641
[HyperMedDiff-Risk] Epoch 025 | Train 4.7032 | Val 3.5096
[HyperMedDiff-Risk] Epoch 026 | Train 4.6325 | Val 3.4235
[HyperMedDiff-Risk] Epoch 027 | Train 4.5079 | Val 3.3690
[HyperMedDiff-Risk] Epoch 028 | Train 4.4588 | Val 3.2414
[HyperMedDiff-Risk] Epoch 029 | Train 4.3968 | Val 3.0896
[HyperMedDiff-Risk] Epoch 030 | Train 4.3153 | Val 3.0366
[HyperMedDiff-Risk] Epoch 031 | Train 4.1808 | Val 2.9841
[HyperMedDiff-Risk] Epoch 032 | Train 4.1205 | Val 2.8751
[HyperMedDiff-Risk] Epoch 033 | Train 4.0524 | Val 2.8646
[HyperMedDiff-Risk] Epoch 034 | Train 4.0166 | Val 2.7717
[HyperMedDiff-Risk] Epoch 035 | Train 3.9455 | Val 2.6889
[HyperMedDiff-Risk] Epoch 036 | Train 3.8859 | Val 2.6953
[HyperMedDiff-Risk] Epoch 037 | Train 3.8557 | Val 2.6072
[HyperMedDiff-Risk] Epoch 038 | Train 3.8283 | Val 2.5756
[HyperMedDiff-Risk] Epoch 039 | Train 3.7570 | Val 2.5432
[HyperMedDiff-Risk] Epoch 040 | Train 3.7107 | Val 2.5143
[HyperMedDiff-Risk] Epoch 041 | Train 3.6630 | Val 2.3759
[HyperMedDiff-Risk] Epoch 042 | Train 3.6549 | Val 2.4579
[HyperMedDiff-Risk] Epoch 043 | Train 3.5977 | Val 2.4149
[HyperMedDiff-Risk] Epoch 044 | Train 3.5917 | Val 2.4124
[HyperMedDiff-Risk] Epoch 045 | Train 3.5512 | Val 2.3776
[HyperMedDiff-Risk] Epoch 046 | Train 3.4652 | Val 2.2933
[HyperMedDiff-Risk] Epoch 047 | Train 3.4730 | Val 2.3091
[HyperMedDiff-Risk] Epoch 048 | Train 3.4071 | Val 2.2302
[HyperMedDiff-Risk] Epoch 049 | Train 3.3660 | Val 2.1732
[HyperMedDiff-Risk] Epoch 050 | Train 3.3468 | Val 2.1720
[HyperMedDiff-Risk] Epoch 051 | Train 3.3507 | Val 2.1415
[HyperMedDiff-Risk] Epoch 052 | Train 3.3194 | Val 2.1790
[HyperMedDiff-Risk] Epoch 053 | Train 3.3005 | Val 2.0942
[HyperMedDiff-Risk] Epoch 054 | Train 3.2888 | Val 2.1179
[HyperMedDiff-Risk] Epoch 055 | Train 3.2611 | Val 2.0683
[HyperMedDiff-Risk] Epoch 056 | Train 3.2185 | Val 2.0718
[HyperMedDiff-Risk] Epoch 057 | Train 3.1963 | Val 2.0293
[HyperMedDiff-Risk] Epoch 058 | Train 3.1893 | Val 2.0409
[HyperMedDiff-Risk] Epoch 059 | Train 3.1455 | Val 1.9935
[HyperMedDiff-Risk] Epoch 060 | Train 3.1368 | Val 2.0478
[HyperMedDiff-Risk] Epoch 061 | Train 3.1193 | Val 1.9574
[HyperMedDiff-Risk] Epoch 062 | Train 3.1087 | Val 1.9758
[HyperMedDiff-Risk] Epoch 063 | Train 3.1478 | Val 1.9454
[HyperMedDiff-Risk] Epoch 064 | Train 3.0877 | Val 1.9203
[HyperMedDiff-Risk] Epoch 065 | Train 3.1007 | Val 1.9774
[HyperMedDiff-Risk] Epoch 066 | Train 3.0969 | Val 1.9101
[HyperMedDiff-Risk] Epoch 067 | Train 3.0774 | Val 1.9119
[HyperMedDiff-Risk] Epoch 068 | Train 3.0369 | Val 1.9200
[HyperMedDiff-Risk] Epoch 069 | Train 3.0898 | Val 1.8728
[HyperMedDiff-Risk] Epoch 070 | Train 3.0570 | Val 1.8894
[HyperMedDiff-Risk] Epoch 071 | Train 3.0596 | Val 1.8717
[HyperMedDiff-Risk] Epoch 072 | Train 3.0118 | Val 1.8684
[HyperMedDiff-Risk] Epoch 073 | Train 3.0257 | Val 1.8411
[HyperMedDiff-Risk] Epoch 074 | Train 3.0209 | Val 1.8350
[HyperMedDiff-Risk] Epoch 075 | Train 3.0048 | Val 1.8893
[HyperMedDiff-Risk] Epoch 076 | Train 2.9803 | Val 1.8572
[HyperMedDiff-Risk] Epoch 077 | Train 2.9908 | Val 1.8627
[HyperMedDiff-Risk] Epoch 078 | Train 2.9930 | Val 1.8942
[HyperMedDiff-Risk] Epoch 079 | Train 2.9721 | Val 1.8446
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 4): 1.8350
[HyperMedDiff-Risk] Saved training curve plot to results/plots/05_NoHDD.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8806036700394444,
  "auprc": 0.8291334402105711
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: -0.0021
Saved checkpoint to results/checkpoints/05_NoHDD.pt
[HyperMedDiff-Risk] ===== Experiment 5/9: 06_StrongHDD =====
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
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0068 | val=0.0058
[Pretrain] Epoch 02 | train=0.0055 | val=0.0049
[Pretrain] Epoch 03 | train=0.0049 | val=0.0043
[Pretrain] Epoch 04 | train=0.0045 | val=0.0040
[Pretrain] Epoch 05 | train=0.0039 | val=0.0036
[Pretrain] Epoch 06 | train=0.0036 | val=0.0033
[Pretrain] Epoch 07 | train=0.0032 | val=0.0031
[Pretrain] Epoch 08 | train=0.0031 | val=0.0029
[Pretrain] Epoch 09 | train=0.0030 | val=0.0029
[Pretrain] Epoch 10 | train=0.0029 | val=0.0028
[Pretrain] Epoch 11 | train=0.0028 | val=0.0028
[Pretrain] Epoch 12 | train=0.0028 | val=0.0028
[Pretrain] Epoch 13 | train=0.0028 | val=0.0028
[Pretrain] Epoch 14 | train=0.0027 | val=0.0027
[Pretrain] Epoch 15 | train=0.0028 | val=0.0027
[Pretrain] Epoch 16 | train=0.0027 | val=0.0027
[Pretrain] Epoch 17 | train=0.0027 | val=0.0027
[Pretrain] Epoch 18 | train=0.0026 | val=0.0026
[Pretrain] Epoch 19 | train=0.0026 | val=0.0026
[Pretrain] Epoch 20 | train=0.0026 | val=0.0026
[Pretrain] Epoch 21 | train=0.0026 | val=0.0026
[Pretrain] Epoch 22 | train=0.0026 | val=0.0026
[Pretrain] Epoch 23 | train=0.0025 | val=0.0025
[Pretrain] Epoch 24 | train=0.0025 | val=0.0024
[Pretrain] Epoch 25 | train=0.0025 | val=0.0024
[Pretrain] Epoch 26 | train=0.0024 | val=0.0024
[Pretrain] Epoch 27 | train=0.0024 | val=0.0024
[Pretrain] Epoch 28 | train=0.0024 | val=0.0024
[Pretrain] Epoch 29 | train=0.0023 | val=0.0024
[Pretrain] Epoch 30 | train=0.0024 | val=0.0023
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.9024
[HyperMedDiff-Risk] Epoch 001 | Train 13.2920 | Val 10.9663
[HyperMedDiff-Risk] Epoch 002 | Train 11.3567 | Val 10.7832
[HyperMedDiff-Risk] Epoch 003 | Train 11.0288 | Val 10.2717
[HyperMedDiff-Risk] Epoch 004 | Train 10.4040 | Val 9.4371
[HyperMedDiff-Risk] Epoch 005 | Train 9.6831 | Val 8.6449
[HyperMedDiff-Risk] Epoch 006 | Train 9.0479 | Val 7.9936
[HyperMedDiff-Risk] Epoch 007 | Train 8.5003 | Val 7.3442
[HyperMedDiff-Risk] Epoch 008 | Train 8.0292 | Val 6.9535
[HyperMedDiff-Risk] Epoch 009 | Train 7.6808 | Val 6.4776
[HyperMedDiff-Risk] Epoch 010 | Train 7.3199 | Val 6.1772
[HyperMedDiff-Risk] Epoch 011 | Train 7.0805 | Val 5.8555
[HyperMedDiff-Risk] Epoch 012 | Train 6.7439 | Val 5.5462
[HyperMedDiff-Risk] Epoch 013 | Train 6.5005 | Val 5.2498
[HyperMedDiff-Risk] Epoch 014 | Train 6.3214 | Val 5.0356
[HyperMedDiff-Risk] Epoch 015 | Train 6.0909 | Val 4.8217
[HyperMedDiff-Risk] Epoch 016 | Train 5.9130 | Val 4.6516
[HyperMedDiff-Risk] Epoch 017 | Train 5.7290 | Val 4.4045
[HyperMedDiff-Risk] Epoch 018 | Train 5.6025 | Val 4.4022
[HyperMedDiff-Risk] Epoch 019 | Train 5.3984 | Val 4.3303
[HyperMedDiff-Risk] Epoch 020 | Train 5.3033 | Val 4.0308
[HyperMedDiff-Risk] Epoch 021 | Train 5.1214 | Val 3.9015
[HyperMedDiff-Risk] Epoch 022 | Train 5.0204 | Val 3.8158
[HyperMedDiff-Risk] Epoch 023 | Train 4.8746 | Val 3.6663
[HyperMedDiff-Risk] Epoch 024 | Train 4.7782 | Val 3.5369
[HyperMedDiff-Risk] Epoch 025 | Train 4.6842 | Val 3.4153
[HyperMedDiff-Risk] Epoch 026 | Train 4.5723 | Val 3.3849
[HyperMedDiff-Risk] Epoch 027 | Train 4.5053 | Val 3.1657
[HyperMedDiff-Risk] Epoch 028 | Train 4.4211 | Val 3.1453
[HyperMedDiff-Risk] Epoch 029 | Train 4.3308 | Val 3.1048
[HyperMedDiff-Risk] Epoch 030 | Train 4.2522 | Val 2.9992
[HyperMedDiff-Risk] Epoch 031 | Train 4.1506 | Val 2.9314
[HyperMedDiff-Risk] Epoch 032 | Train 4.0714 | Val 2.8610
[HyperMedDiff-Risk] Epoch 033 | Train 4.0117 | Val 2.7566
[HyperMedDiff-Risk] Epoch 034 | Train 3.9650 | Val 2.6978
[HyperMedDiff-Risk] Epoch 035 | Train 3.9124 | Val 2.6077
[HyperMedDiff-Risk] Epoch 036 | Train 3.8656 | Val 2.5304
[HyperMedDiff-Risk] Epoch 037 | Train 3.7864 | Val 2.6163
[HyperMedDiff-Risk] Epoch 038 | Train 3.7410 | Val 2.4485
[HyperMedDiff-Risk] Epoch 039 | Train 3.6826 | Val 2.4886
[HyperMedDiff-Risk] Epoch 040 | Train 3.6569 | Val 2.3617
[HyperMedDiff-Risk] Epoch 041 | Train 3.6079 | Val 2.4960
[HyperMedDiff-Risk] Epoch 042 | Train 3.5854 | Val 2.3867
[HyperMedDiff-Risk] Epoch 043 | Train 3.5482 | Val 2.3008
[HyperMedDiff-Risk] Epoch 044 | Train 3.5058 | Val 2.3461
[HyperMedDiff-Risk] Epoch 045 | Train 3.4780 | Val 2.3356
[HyperMedDiff-Risk] Epoch 046 | Train 3.4594 | Val 2.2379
[HyperMedDiff-Risk] Epoch 047 | Train 3.4453 | Val 2.2431
[HyperMedDiff-Risk] Epoch 048 | Train 3.3869 | Val 2.1779
[HyperMedDiff-Risk] Epoch 049 | Train 3.3785 | Val 2.1836
[HyperMedDiff-Risk] Epoch 050 | Train 3.3526 | Val 2.1393
[HyperMedDiff-Risk] Epoch 051 | Train 3.3334 | Val 2.1619
[HyperMedDiff-Risk] Epoch 052 | Train 3.2859 | Val 2.1006
[HyperMedDiff-Risk] Epoch 053 | Train 3.2575 | Val 2.1716
[HyperMedDiff-Risk] Epoch 054 | Train 3.2816 | Val 2.1856
[HyperMedDiff-Risk] Epoch 055 | Train 3.2625 | Val 2.1098
[HyperMedDiff-Risk] Epoch 056 | Train 3.2160 | Val 2.1166
[HyperMedDiff-Risk] Epoch 057 | Train 3.2260 | Val 2.0738
[HyperMedDiff-Risk] Epoch 058 | Train 3.2111 | Val 2.0545
[HyperMedDiff-Risk] Epoch 059 | Train 3.1955 | Val 2.0561
[HyperMedDiff-Risk] Epoch 060 | Train 3.2003 | Val 2.0567
[HyperMedDiff-Risk] Epoch 061 | Train 3.1700 | Val 2.0474
[HyperMedDiff-Risk] Epoch 062 | Train 3.1602 | Val 2.0334
[HyperMedDiff-Risk] Epoch 063 | Train 3.1360 | Val 2.0754
[HyperMedDiff-Risk] Epoch 064 | Train 3.1270 | Val 2.0737
[HyperMedDiff-Risk] Epoch 065 | Train 3.1442 | Val 1.9892
[HyperMedDiff-Risk] Epoch 066 | Train 3.1277 | Val 2.0126
[HyperMedDiff-Risk] Epoch 067 | Train 3.1070 | Val 2.0191
[HyperMedDiff-Risk] Epoch 068 | Train 3.0724 | Val 1.9831
[HyperMedDiff-Risk] Epoch 069 | Train 3.0973 | Val 1.9236
[HyperMedDiff-Risk] Epoch 070 | Train 3.0850 | Val 1.9888
[HyperMedDiff-Risk] Epoch 071 | Train 3.0711 | Val 1.9894
[HyperMedDiff-Risk] Epoch 072 | Train 3.0475 | Val 1.9752
[HyperMedDiff-Risk] Epoch 073 | Train 3.0577 | Val 1.9096
[HyperMedDiff-Risk] Epoch 074 | Train 3.0242 | Val 1.9350
[HyperMedDiff-Risk] Epoch 075 | Train 3.0506 | Val 1.9354
[HyperMedDiff-Risk] Epoch 076 | Train 3.0173 | Val 1.9005
[HyperMedDiff-Risk] Epoch 077 | Train 3.0207 | Val 1.9652
[HyperMedDiff-Risk] Epoch 078 | Train 3.0262 | Val 1.9120
[HyperMedDiff-Risk] Epoch 079 | Train 3.0268 | Val 1.9203
[HyperMedDiff-Risk] Epoch 080 | Train 3.0308 | Val 1.8682
[HyperMedDiff-Risk] Epoch 081 | Train 3.0169 | Val 1.8973
[HyperMedDiff-Risk] Epoch 082 | Train 3.0314 | Val 1.9410
[HyperMedDiff-Risk] Epoch 083 | Train 2.9910 | Val 1.8918
[HyperMedDiff-Risk] Epoch 084 | Train 2.9986 | Val 1.8784
[HyperMedDiff-Risk] Epoch 085 | Train 2.9885 | Val 1.8790
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 5): 1.8682
[HyperMedDiff-Risk] Saved training curve plot to results/plots/06_StrongHDD.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.869590121762991,
  "auprc": 0.8021507886728527
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.9071
Saved checkpoint to results/checkpoints/06_StrongHDD.pt
[HyperMedDiff-Risk] ===== Experiment 6/9: 07_HighDropout =====
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
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0036 | val=0.0033
[Pretrain] Epoch 02 | train=0.0033 | val=0.0031
[Pretrain] Epoch 03 | train=0.0031 | val=0.0029
[Pretrain] Epoch 04 | train=0.0029 | val=0.0027
[Pretrain] Epoch 05 | train=0.0027 | val=0.0026
[Pretrain] Epoch 06 | train=0.0026 | val=0.0025
[Pretrain] Epoch 07 | train=0.0025 | val=0.0024
[Pretrain] Epoch 08 | train=0.0024 | val=0.0023
[Pretrain] Epoch 09 | train=0.0023 | val=0.0023
[Pretrain] Epoch 10 | train=0.0023 | val=0.0022
[Pretrain] Epoch 11 | train=0.0022 | val=0.0022
[Pretrain] Epoch 12 | train=0.0022 | val=0.0022
[Pretrain] Epoch 13 | train=0.0022 | val=0.0022
[Pretrain] Epoch 14 | train=0.0022 | val=0.0022
[Pretrain] Epoch 15 | train=0.0022 | val=0.0022
[Pretrain] Epoch 16 | train=0.0022 | val=0.0022
[Pretrain] Epoch 17 | train=0.0022 | val=0.0022
[Pretrain] Epoch 18 | train=0.0022 | val=0.0022
[Pretrain] Epoch 19 | train=0.0022 | val=0.0021
[Pretrain] Epoch 20 | train=0.0022 | val=0.0021
[Pretrain] Epoch 21 | train=0.0021 | val=0.0021
[Pretrain] Epoch 22 | train=0.0021 | val=0.0021
[Pretrain] Epoch 23 | train=0.0021 | val=0.0021
[Pretrain] Epoch 24 | train=0.0021 | val=0.0021
[Pretrain] Epoch 25 | train=0.0021 | val=0.0021
[Pretrain] Epoch 26 | train=0.0021 | val=0.0021
[Pretrain] Epoch 27 | train=0.0021 | val=0.0021
[Pretrain] Epoch 28 | train=0.0021 | val=0.0021
[Pretrain] Epoch 29 | train=0.0021 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0021
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8014
[HyperMedDiff-Risk] Epoch 001 | Train 13.5403 | Val 10.9692
[HyperMedDiff-Risk] Epoch 002 | Train 11.3951 | Val 10.8371
[HyperMedDiff-Risk] Epoch 003 | Train 11.1471 | Val 10.4857
[HyperMedDiff-Risk] Epoch 004 | Train 10.6481 | Val 9.7306
[HyperMedDiff-Risk] Epoch 005 | Train 9.8842 | Val 8.8042
[HyperMedDiff-Risk] Epoch 006 | Train 9.2298 | Val 8.1924
[HyperMedDiff-Risk] Epoch 007 | Train 8.7204 | Val 7.5268
[HyperMedDiff-Risk] Epoch 008 | Train 8.2415 | Val 7.0500
[HyperMedDiff-Risk] Epoch 009 | Train 7.7959 | Val 6.6767
[HyperMedDiff-Risk] Epoch 010 | Train 7.4653 | Val 6.3768
[HyperMedDiff-Risk] Epoch 011 | Train 7.1609 | Val 6.0046
[HyperMedDiff-Risk] Epoch 012 | Train 6.8998 | Val 5.7698
[HyperMedDiff-Risk] Epoch 013 | Train 6.6563 | Val 5.5429
[HyperMedDiff-Risk] Epoch 014 | Train 6.4156 | Val 5.1156
[HyperMedDiff-Risk] Epoch 015 | Train 6.2339 | Val 5.0499
[HyperMedDiff-Risk] Epoch 016 | Train 6.0585 | Val 4.8070
[HyperMedDiff-Risk] Epoch 017 | Train 5.8986 | Val 4.7098
[HyperMedDiff-Risk] Epoch 018 | Train 5.7383 | Val 4.5735
[HyperMedDiff-Risk] Epoch 019 | Train 5.6049 | Val 4.3593
[HyperMedDiff-Risk] Epoch 020 | Train 5.4487 | Val 4.2554
[HyperMedDiff-Risk] Epoch 021 | Train 5.3319 | Val 4.1226
[HyperMedDiff-Risk] Epoch 022 | Train 5.2047 | Val 4.0367
[HyperMedDiff-Risk] Epoch 023 | Train 5.0684 | Val 3.8529
[HyperMedDiff-Risk] Epoch 024 | Train 4.9658 | Val 3.7240
[HyperMedDiff-Risk] Epoch 025 | Train 4.8636 | Val 3.6643
[HyperMedDiff-Risk] Epoch 026 | Train 4.7222 | Val 3.5698
[HyperMedDiff-Risk] Epoch 027 | Train 4.6714 | Val 3.4575
[HyperMedDiff-Risk] Epoch 028 | Train 4.5691 | Val 3.2913
[HyperMedDiff-Risk] Epoch 029 | Train 4.4923 | Val 3.2668
[HyperMedDiff-Risk] Epoch 030 | Train 4.3830 | Val 3.2415
[HyperMedDiff-Risk] Epoch 031 | Train 4.3127 | Val 3.1037
[HyperMedDiff-Risk] Epoch 032 | Train 4.2498 | Val 2.9739
[HyperMedDiff-Risk] Epoch 033 | Train 4.1699 | Val 2.9133
[HyperMedDiff-Risk] Epoch 034 | Train 4.1102 | Val 2.9256
[HyperMedDiff-Risk] Epoch 035 | Train 4.0572 | Val 2.8348
[HyperMedDiff-Risk] Epoch 036 | Train 3.9648 | Val 2.7245
[HyperMedDiff-Risk] Epoch 037 | Train 3.9548 | Val 2.7163
[HyperMedDiff-Risk] Epoch 038 | Train 3.8536 | Val 2.6956
[HyperMedDiff-Risk] Epoch 039 | Train 3.8155 | Val 2.6592
[HyperMedDiff-Risk] Epoch 040 | Train 3.7558 | Val 2.5455
[HyperMedDiff-Risk] Epoch 041 | Train 3.7261 | Val 2.5147
[HyperMedDiff-Risk] Epoch 042 | Train 3.6633 | Val 2.4728
[HyperMedDiff-Risk] Epoch 043 | Train 3.6395 | Val 2.3864
[HyperMedDiff-Risk] Epoch 044 | Train 3.5861 | Val 2.3873
[HyperMedDiff-Risk] Epoch 045 | Train 3.5234 | Val 2.3791
[HyperMedDiff-Risk] Epoch 046 | Train 3.4835 | Val 2.3510
[HyperMedDiff-Risk] Epoch 047 | Train 3.4674 | Val 2.3040
[HyperMedDiff-Risk] Epoch 048 | Train 3.4525 | Val 2.2267
[HyperMedDiff-Risk] Epoch 049 | Train 3.3939 | Val 2.2145
[HyperMedDiff-Risk] Epoch 050 | Train 3.3834 | Val 2.1899
[HyperMedDiff-Risk] Epoch 051 | Train 3.3614 | Val 2.1346
[HyperMedDiff-Risk] Epoch 052 | Train 3.3093 | Val 2.1525
[HyperMedDiff-Risk] Epoch 053 | Train 3.3134 | Val 2.0922
[HyperMedDiff-Risk] Epoch 054 | Train 3.2853 | Val 2.0743
[HyperMedDiff-Risk] Epoch 055 | Train 3.2500 | Val 2.0760
[HyperMedDiff-Risk] Epoch 056 | Train 3.2240 | Val 2.0923
[HyperMedDiff-Risk] Epoch 057 | Train 3.2035 | Val 1.9883
[HyperMedDiff-Risk] Epoch 058 | Train 3.1783 | Val 2.0047
[HyperMedDiff-Risk] Epoch 059 | Train 3.1691 | Val 2.0226
[HyperMedDiff-Risk] Epoch 060 | Train 3.1535 | Val 1.9979
[HyperMedDiff-Risk] Epoch 061 | Train 3.1568 | Val 1.9348
[HyperMedDiff-Risk] Epoch 062 | Train 3.1326 | Val 1.9309
[HyperMedDiff-Risk] Epoch 063 | Train 3.1225 | Val 1.9328
[HyperMedDiff-Risk] Epoch 064 | Train 3.0810 | Val 1.9303
[HyperMedDiff-Risk] Epoch 065 | Train 3.0905 | Val 1.9417
[HyperMedDiff-Risk] Epoch 066 | Train 3.0744 | Val 1.9371
[HyperMedDiff-Risk] Epoch 067 | Train 3.0704 | Val 1.9186
[HyperMedDiff-Risk] Epoch 068 | Train 3.0617 | Val 1.8489
[HyperMedDiff-Risk] Epoch 069 | Train 3.0341 | Val 1.8515
[HyperMedDiff-Risk] Epoch 070 | Train 3.0401 | Val 1.8009
[HyperMedDiff-Risk] Epoch 071 | Train 3.0343 | Val 1.8945
[HyperMedDiff-Risk] Epoch 072 | Train 2.9971 | Val 1.8375
[HyperMedDiff-Risk] Epoch 073 | Train 3.0395 | Val 1.8724
[HyperMedDiff-Risk] Epoch 074 | Train 3.0063 | Val 1.8265
[HyperMedDiff-Risk] Epoch 075 | Train 3.0083 | Val 1.8831
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 6): 1.8009
[HyperMedDiff-Risk] Saved training curve plot to results/plots/07_HighDropout.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.873267021094152,
  "auprc": 0.8062831575059137
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8127
Saved checkpoint to results/checkpoints/07_HighDropout.pt
[HyperMedDiff-Risk] ===== Experiment 7/9: 08_SmallDim =====
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
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0038 | val=0.0035
[Pretrain] Epoch 02 | train=0.0035 | val=0.0033
[Pretrain] Epoch 03 | train=0.0033 | val=0.0032
[Pretrain] Epoch 04 | train=0.0032 | val=0.0030
[Pretrain] Epoch 05 | train=0.0030 | val=0.0029
[Pretrain] Epoch 06 | train=0.0029 | val=0.0028
[Pretrain] Epoch 07 | train=0.0028 | val=0.0026
[Pretrain] Epoch 08 | train=0.0027 | val=0.0026
[Pretrain] Epoch 09 | train=0.0026 | val=0.0025
[Pretrain] Epoch 10 | train=0.0025 | val=0.0024
[Pretrain] Epoch 11 | train=0.0024 | val=0.0024
[Pretrain] Epoch 12 | train=0.0024 | val=0.0023
[Pretrain] Epoch 13 | train=0.0023 | val=0.0023
[Pretrain] Epoch 14 | train=0.0023 | val=0.0023
[Pretrain] Epoch 15 | train=0.0022 | val=0.0022
[Pretrain] Epoch 16 | train=0.0022 | val=0.0022
[Pretrain] Epoch 17 | train=0.0022 | val=0.0022
[Pretrain] Epoch 18 | train=0.0022 | val=0.0022
[Pretrain] Epoch 19 | train=0.0022 | val=0.0022
[Pretrain] Epoch 20 | train=0.0022 | val=0.0022
[Pretrain] Epoch 21 | train=0.0022 | val=0.0022
[Pretrain] Epoch 22 | train=0.0022 | val=0.0022
[Pretrain] Epoch 23 | train=0.0022 | val=0.0022
[Pretrain] Epoch 24 | train=0.0022 | val=0.0021
[Pretrain] Epoch 25 | train=0.0021 | val=0.0021
[Pretrain] Epoch 26 | train=0.0021 | val=0.0021
[Pretrain] Epoch 27 | train=0.0021 | val=0.0021
[Pretrain] Epoch 28 | train=0.0021 | val=0.0021
[Pretrain] Epoch 29 | train=0.0021 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0021
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.7495
[HyperMedDiff-Risk] Epoch 001 | Train 12.5729 | Val 10.8255
[HyperMedDiff-Risk] Epoch 002 | Train 10.8969 | Val 10.1268
[HyperMedDiff-Risk] Epoch 003 | Train 10.1124 | Val 9.1007
[HyperMedDiff-Risk] Epoch 004 | Train 9.2748 | Val 8.2768
[HyperMedDiff-Risk] Epoch 005 | Train 8.6396 | Val 7.6350
[HyperMedDiff-Risk] Epoch 006 | Train 8.1006 | Val 6.9620
[HyperMedDiff-Risk] Epoch 007 | Train 7.6344 | Val 6.4907
[HyperMedDiff-Risk] Epoch 008 | Train 7.2803 | Val 6.1157
[HyperMedDiff-Risk] Epoch 009 | Train 6.9114 | Val 5.7869
[HyperMedDiff-Risk] Epoch 010 | Train 6.6155 | Val 5.4266
[HyperMedDiff-Risk] Epoch 011 | Train 6.2941 | Val 5.0410
[HyperMedDiff-Risk] Epoch 012 | Train 6.0687 | Val 4.8067
[HyperMedDiff-Risk] Epoch 013 | Train 5.8605 | Val 4.7500
[HyperMedDiff-Risk] Epoch 014 | Train 5.6717 | Val 4.4632
[HyperMedDiff-Risk] Epoch 015 | Train 5.5117 | Val 4.2448
[HyperMedDiff-Risk] Epoch 016 | Train 5.3276 | Val 4.0944
[HyperMedDiff-Risk] Epoch 017 | Train 5.2144 | Val 4.0015
[HyperMedDiff-Risk] Epoch 018 | Train 5.1104 | Val 3.8857
[HyperMedDiff-Risk] Epoch 019 | Train 4.9364 | Val 3.6300
[HyperMedDiff-Risk] Epoch 020 | Train 4.7629 | Val 3.5994
[HyperMedDiff-Risk] Epoch 021 | Train 4.6756 | Val 3.5059
[HyperMedDiff-Risk] Epoch 022 | Train 4.6022 | Val 3.4084
[HyperMedDiff-Risk] Epoch 023 | Train 4.5134 | Val 3.3329
[HyperMedDiff-Risk] Epoch 024 | Train 4.3867 | Val 3.2723
[HyperMedDiff-Risk] Epoch 025 | Train 4.3185 | Val 3.0576
[HyperMedDiff-Risk] Epoch 026 | Train 4.2245 | Val 3.0832
[HyperMedDiff-Risk] Epoch 027 | Train 4.1543 | Val 2.9959
[HyperMedDiff-Risk] Epoch 028 | Train 4.1038 | Val 2.9091
[HyperMedDiff-Risk] Epoch 029 | Train 4.0111 | Val 2.8768
[HyperMedDiff-Risk] Epoch 030 | Train 3.9864 | Val 2.7777
[HyperMedDiff-Risk] Epoch 031 | Train 3.9123 | Val 2.7181
[HyperMedDiff-Risk] Epoch 032 | Train 3.9189 | Val 2.7279
[HyperMedDiff-Risk] Epoch 033 | Train 3.8579 | Val 2.5586
[HyperMedDiff-Risk] Epoch 034 | Train 3.7646 | Val 2.6397
[HyperMedDiff-Risk] Epoch 035 | Train 3.7192 | Val 2.5807
[HyperMedDiff-Risk] Epoch 036 | Train 3.6750 | Val 2.5087
[HyperMedDiff-Risk] Epoch 037 | Train 3.6102 | Val 2.4900
[HyperMedDiff-Risk] Epoch 038 | Train 3.6066 | Val 2.4543
[HyperMedDiff-Risk] Epoch 039 | Train 3.6032 | Val 2.4455
[HyperMedDiff-Risk] Epoch 040 | Train 3.5413 | Val 2.3934
[HyperMedDiff-Risk] Epoch 041 | Train 3.4855 | Val 2.4130
[HyperMedDiff-Risk] Epoch 042 | Train 3.4420 | Val 2.3323
[HyperMedDiff-Risk] Epoch 043 | Train 3.4691 | Val 2.2515
[HyperMedDiff-Risk] Epoch 044 | Train 3.4256 | Val 2.2751
[HyperMedDiff-Risk] Epoch 045 | Train 3.3853 | Val 2.2869
[HyperMedDiff-Risk] Epoch 046 | Train 3.3543 | Val 2.2115
[HyperMedDiff-Risk] Epoch 047 | Train 3.3079 | Val 2.2274
[HyperMedDiff-Risk] Epoch 048 | Train 3.3320 | Val 2.2441
[HyperMedDiff-Risk] Epoch 049 | Train 3.2798 | Val 2.2182
[HyperMedDiff-Risk] Epoch 050 | Train 3.2656 | Val 2.1815
[HyperMedDiff-Risk] Epoch 051 | Train 3.2378 | Val 2.2023
[HyperMedDiff-Risk] Epoch 052 | Train 3.2762 | Val 2.0531
[HyperMedDiff-Risk] Epoch 053 | Train 3.2372 | Val 2.1211
[HyperMedDiff-Risk] Epoch 054 | Train 3.1801 | Val 2.1007
[HyperMedDiff-Risk] Epoch 055 | Train 3.1639 | Val 2.1045
[HyperMedDiff-Risk] Epoch 056 | Train 3.1345 | Val 2.1152
[HyperMedDiff-Risk] Epoch 057 | Train 3.1314 | Val 2.1008
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 7): 2.0531
[HyperMedDiff-Risk] Saved training curve plot to results/plots/08_SmallDim.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8686091579488939,
  "auprc": 0.7927600294303611
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.7374
Saved checkpoint to results/checkpoints/08_SmallDim.pt
[HyperMedDiff-Risk] ===== Experiment 8/9: 09_DiscrimOnly =====
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
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0036 | val=0.0033
[Pretrain] Epoch 02 | train=0.0033 | val=0.0031
[Pretrain] Epoch 03 | train=0.0031 | val=0.0029
[Pretrain] Epoch 04 | train=0.0029 | val=0.0027
[Pretrain] Epoch 05 | train=0.0027 | val=0.0026
[Pretrain] Epoch 06 | train=0.0026 | val=0.0025
[Pretrain] Epoch 07 | train=0.0025 | val=0.0024
[Pretrain] Epoch 08 | train=0.0024 | val=0.0023
[Pretrain] Epoch 09 | train=0.0023 | val=0.0023
[Pretrain] Epoch 10 | train=0.0023 | val=0.0022
[Pretrain] Epoch 11 | train=0.0022 | val=0.0022
[Pretrain] Epoch 12 | train=0.0022 | val=0.0022
[Pretrain] Epoch 13 | train=0.0022 | val=0.0022
[Pretrain] Epoch 14 | train=0.0022 | val=0.0022
[Pretrain] Epoch 15 | train=0.0022 | val=0.0022
[Pretrain] Epoch 16 | train=0.0022 | val=0.0022
[Pretrain] Epoch 17 | train=0.0022 | val=0.0021
[Pretrain] Epoch 18 | train=0.0022 | val=0.0022
[Pretrain] Epoch 19 | train=0.0022 | val=0.0021
[Pretrain] Epoch 20 | train=0.0021 | val=0.0021
[Pretrain] Epoch 21 | train=0.0021 | val=0.0021
[Pretrain] Epoch 22 | train=0.0021 | val=0.0021
[Pretrain] Epoch 23 | train=0.0021 | val=0.0021
[Pretrain] Epoch 24 | train=0.0021 | val=0.0021
[Pretrain] Epoch 25 | train=0.0021 | val=0.0021
[Pretrain] Epoch 26 | train=0.0021 | val=0.0021
[Pretrain] Epoch 27 | train=0.0021 | val=0.0021
[Pretrain] Epoch 28 | train=0.0021 | val=0.0021
[Pretrain] Epoch 29 | train=0.0021 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0021
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8251
[HyperMedDiff-Risk] Epoch 001 | Train 12.9754 | Val 10.2740
[HyperMedDiff-Risk] Epoch 002 | Train 10.7095 | Val 10.1799
[HyperMedDiff-Risk] Epoch 003 | Train 10.4625 | Val 9.7376
[HyperMedDiff-Risk] Epoch 004 | Train 9.9368 | Val 8.9952
[HyperMedDiff-Risk] Epoch 005 | Train 9.2609 | Val 8.2340
[HyperMedDiff-Risk] Epoch 006 | Train 8.6224 | Val 7.5295
[HyperMedDiff-Risk] Epoch 007 | Train 8.1015 | Val 7.0484
[HyperMedDiff-Risk] Epoch 008 | Train 7.6225 | Val 6.5219
[HyperMedDiff-Risk] Epoch 009 | Train 7.2376 | Val 6.0523
[HyperMedDiff-Risk] Epoch 010 | Train 6.8677 | Val 5.6741
[HyperMedDiff-Risk] Epoch 011 | Train 6.6025 | Val 5.4397
[HyperMedDiff-Risk] Epoch 012 | Train 6.2668 | Val 5.1494
[HyperMedDiff-Risk] Epoch 013 | Train 6.0430 | Val 4.9138
[HyperMedDiff-Risk] Epoch 014 | Train 5.8328 | Val 4.5593
[HyperMedDiff-Risk] Epoch 015 | Train 5.6240 | Val 4.3899
[HyperMedDiff-Risk] Epoch 016 | Train 5.4229 | Val 4.2050
[HyperMedDiff-Risk] Epoch 017 | Train 5.2516 | Val 3.9548
[HyperMedDiff-Risk] Epoch 018 | Train 5.0653 | Val 3.7698
[HyperMedDiff-Risk] Epoch 019 | Train 4.9419 | Val 3.5729
[HyperMedDiff-Risk] Epoch 020 | Train 4.7801 | Val 3.5208
[HyperMedDiff-Risk] Epoch 021 | Train 4.6623 | Val 3.3678
[HyperMedDiff-Risk] Epoch 022 | Train 4.5099 | Val 3.2107
[HyperMedDiff-Risk] Epoch 023 | Train 4.3632 | Val 3.1760
[HyperMedDiff-Risk] Epoch 024 | Train 4.2622 | Val 3.0303
[HyperMedDiff-Risk] Epoch 025 | Train 4.1590 | Val 2.9139
[HyperMedDiff-Risk] Epoch 026 | Train 4.0716 | Val 2.8340
[HyperMedDiff-Risk] Epoch 027 | Train 4.0003 | Val 2.7778
[HyperMedDiff-Risk] Epoch 028 | Train 3.9245 | Val 2.6968
[HyperMedDiff-Risk] Epoch 029 | Train 3.8308 | Val 2.6214
[HyperMedDiff-Risk] Epoch 030 | Train 3.7727 | Val 2.5624
[HyperMedDiff-Risk] Epoch 031 | Train 3.7153 | Val 2.5854
[HyperMedDiff-Risk] Epoch 032 | Train 3.6354 | Val 2.3878
[HyperMedDiff-Risk] Epoch 033 | Train 3.5682 | Val 2.2722
[HyperMedDiff-Risk] Epoch 034 | Train 3.4769 | Val 2.2608
[HyperMedDiff-Risk] Epoch 035 | Train 3.4235 | Val 2.2162
[HyperMedDiff-Risk] Epoch 036 | Train 3.4002 | Val 2.1688
[HyperMedDiff-Risk] Epoch 037 | Train 3.3007 | Val 2.1486
[HyperMedDiff-Risk] Epoch 038 | Train 3.2610 | Val 2.1128
[HyperMedDiff-Risk] Epoch 039 | Train 3.2373 | Val 2.0273
[HyperMedDiff-Risk] Epoch 040 | Train 3.1549 | Val 1.9660
[HyperMedDiff-Risk] Epoch 041 | Train 3.1567 | Val 1.9881
[HyperMedDiff-Risk] Epoch 042 | Train 3.1132 | Val 1.9422
[HyperMedDiff-Risk] Epoch 043 | Train 3.1006 | Val 1.8784
[HyperMedDiff-Risk] Epoch 044 | Train 3.0284 | Val 1.7784
[HyperMedDiff-Risk] Epoch 045 | Train 2.9895 | Val 1.8072
[HyperMedDiff-Risk] Epoch 046 | Train 2.9501 | Val 1.6871
[HyperMedDiff-Risk] Epoch 047 | Train 2.9340 | Val 1.6672
[HyperMedDiff-Risk] Epoch 048 | Train 2.8967 | Val 1.7604
[HyperMedDiff-Risk] Epoch 049 | Train 2.8387 | Val 1.6462
[HyperMedDiff-Risk] Epoch 050 | Train 2.7873 | Val 1.6200
[HyperMedDiff-Risk] Epoch 051 | Train 2.7914 | Val 1.5541
[HyperMedDiff-Risk] Epoch 052 | Train 2.7466 | Val 1.6641
[HyperMedDiff-Risk] Epoch 053 | Train 2.7319 | Val 1.5251
[HyperMedDiff-Risk] Epoch 054 | Train 2.7145 | Val 1.4919
[HyperMedDiff-Risk] Epoch 055 | Train 2.7001 | Val 1.5139
[HyperMedDiff-Risk] Epoch 056 | Train 2.6330 | Val 1.5151
[HyperMedDiff-Risk] Epoch 057 | Train 2.6566 | Val 1.4379
[HyperMedDiff-Risk] Epoch 058 | Train 2.6265 | Val 1.4458
[HyperMedDiff-Risk] Epoch 059 | Train 2.6270 | Val 1.4160
[HyperMedDiff-Risk] Epoch 060 | Train 2.5845 | Val 1.4468
[HyperMedDiff-Risk] Epoch 061 | Train 2.6028 | Val 1.3948
[HyperMedDiff-Risk] Epoch 062 | Train 2.5774 | Val 1.3688
[HyperMedDiff-Risk] Epoch 063 | Train 2.5636 | Val 1.4079
[HyperMedDiff-Risk] Epoch 064 | Train 2.5386 | Val 1.3491
[HyperMedDiff-Risk] Epoch 065 | Train 2.5265 | Val 1.3565
[HyperMedDiff-Risk] Epoch 066 | Train 2.5163 | Val 1.3632
[HyperMedDiff-Risk] Epoch 067 | Train 2.5189 | Val 1.3219
[HyperMedDiff-Risk] Epoch 068 | Train 2.5365 | Val 1.2758
[HyperMedDiff-Risk] Epoch 069 | Train 2.5029 | Val 1.2891
[HyperMedDiff-Risk] Epoch 070 | Train 2.4880 | Val 1.3043
[HyperMedDiff-Risk] Epoch 071 | Train 2.4671 | Val 1.3391
[HyperMedDiff-Risk] Epoch 072 | Train 2.4670 | Val 1.3583
[HyperMedDiff-Risk] Epoch 073 | Train 2.4694 | Val 1.2602
[HyperMedDiff-Risk] Epoch 074 | Train 2.4527 | Val 1.3318
[HyperMedDiff-Risk] Epoch 075 | Train 2.4489 | Val 1.3190
[HyperMedDiff-Risk] Epoch 076 | Train 2.4380 | Val 1.2503
[HyperMedDiff-Risk] Epoch 077 | Train 2.4421 | Val 1.2609
[HyperMedDiff-Risk] Epoch 078 | Train 2.4248 | Val 1.2479
[HyperMedDiff-Risk] Epoch 079 | Train 2.4134 | Val 1.2770
[HyperMedDiff-Risk] Epoch 080 | Train 2.4133 | Val 1.2799
[HyperMedDiff-Risk] Epoch 081 | Train 2.4124 | Val 1.2568
[HyperMedDiff-Risk] Epoch 082 | Train 2.4031 | Val 1.2226
[HyperMedDiff-Risk] Epoch 083 | Train 2.3891 | Val 1.2460
[HyperMedDiff-Risk] Epoch 084 | Train 2.3959 | Val 1.2057
[HyperMedDiff-Risk] Epoch 085 | Train 2.3816 | Val 1.2453
[HyperMedDiff-Risk] Epoch 086 | Train 2.3749 | Val 1.2456
[HyperMedDiff-Risk] Epoch 087 | Train 2.3872 | Val 1.2480
[HyperMedDiff-Risk] Epoch 088 | Train 2.3740 | Val 1.2185
[HyperMedDiff-Risk] Epoch 089 | Train 2.3665 | Val 1.2104
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 8): 1.2057
[HyperMedDiff-Risk] Saved training curve plot to results/plots/09_DiscrimOnly.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8757983193277312,
  "auprc": 0.8125455307097418
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8261
Saved checkpoint to results/checkpoints/09_DiscrimOnly.pt
[HyperMedDiff-Risk] ===== Experiment 9/9: 10_GenFocus =====
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
  "lambda_d": 1.0,
  "lambda_consistency": 0.1,
  "train_epochs": 100
}
[Pretrain] Epoch 01 | train=0.0036 | val=0.0033
[Pretrain] Epoch 02 | train=0.0033 | val=0.0031
[Pretrain] Epoch 03 | train=0.0031 | val=0.0029
[Pretrain] Epoch 04 | train=0.0029 | val=0.0027
[Pretrain] Epoch 05 | train=0.0027 | val=0.0026
[Pretrain] Epoch 06 | train=0.0026 | val=0.0025
[Pretrain] Epoch 07 | train=0.0025 | val=0.0024
[Pretrain] Epoch 08 | train=0.0024 | val=0.0023
[Pretrain] Epoch 09 | train=0.0023 | val=0.0023
[Pretrain] Epoch 10 | train=0.0023 | val=0.0022
[Pretrain] Epoch 11 | train=0.0022 | val=0.0022
[Pretrain] Epoch 12 | train=0.0022 | val=0.0022
[Pretrain] Epoch 13 | train=0.0022 | val=0.0022
[Pretrain] Epoch 14 | train=0.0022 | val=0.0022
[Pretrain] Epoch 15 | train=0.0022 | val=0.0022
[Pretrain] Epoch 16 | train=0.0022 | val=0.0022
[Pretrain] Epoch 17 | train=0.0022 | val=0.0021
[Pretrain] Epoch 18 | train=0.0022 | val=0.0022
[Pretrain] Epoch 19 | train=0.0022 | val=0.0022
[Pretrain] Epoch 20 | train=0.0022 | val=0.0021
[Pretrain] Epoch 21 | train=0.0021 | val=0.0021
[Pretrain] Epoch 22 | train=0.0021 | val=0.0021
[Pretrain] Epoch 23 | train=0.0021 | val=0.0021
[Pretrain] Epoch 24 | train=0.0021 | val=0.0021
[Pretrain] Epoch 25 | train=0.0021 | val=0.0021
[Pretrain] Epoch 26 | train=0.0021 | val=0.0021
[Pretrain] Epoch 27 | train=0.0021 | val=0.0021
[Pretrain] Epoch 28 | train=0.0021 | val=0.0021
[Pretrain] Epoch 29 | train=0.0021 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0021
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8365
[HyperMedDiff-Risk] Epoch 001 | Train 14.1277 | Val 11.6100
[HyperMedDiff-Risk] Epoch 002 | Train 11.9696 | Val 11.3965
[HyperMedDiff-Risk] Epoch 003 | Train 11.6715 | Val 10.9470
[HyperMedDiff-Risk] Epoch 004 | Train 11.0937 | Val 10.1070
[HyperMedDiff-Risk] Epoch 005 | Train 10.3647 | Val 9.3295
[HyperMedDiff-Risk] Epoch 006 | Train 9.6889 | Val 8.5427
[HyperMedDiff-Risk] Epoch 007 | Train 9.1176 | Val 8.0897
[HyperMedDiff-Risk] Epoch 008 | Train 8.6591 | Val 7.6516
[HyperMedDiff-Risk] Epoch 009 | Train 8.2624 | Val 7.2196
[HyperMedDiff-Risk] Epoch 010 | Train 7.9087 | Val 6.7114
[HyperMedDiff-Risk] Epoch 011 | Train 7.6105 | Val 6.4359
[HyperMedDiff-Risk] Epoch 012 | Train 7.3595 | Val 6.1164
[HyperMedDiff-Risk] Epoch 013 | Train 7.1082 | Val 5.8994
[HyperMedDiff-Risk] Epoch 014 | Train 6.8596 | Val 5.6638
[HyperMedDiff-Risk] Epoch 015 | Train 6.6566 | Val 5.4731
[HyperMedDiff-Risk] Epoch 016 | Train 6.4452 | Val 5.1876
[HyperMedDiff-Risk] Epoch 017 | Train 6.2865 | Val 5.0082
[HyperMedDiff-Risk] Epoch 018 | Train 6.0813 | Val 4.8728
[HyperMedDiff-Risk] Epoch 019 | Train 5.9753 | Val 4.6528
[HyperMedDiff-Risk] Epoch 020 | Train 5.8439 | Val 4.4944
[HyperMedDiff-Risk] Epoch 021 | Train 5.6665 | Val 4.4688
[HyperMedDiff-Risk] Epoch 022 | Train 5.5776 | Val 4.2937
[HyperMedDiff-Risk] Epoch 023 | Train 5.4249 | Val 4.2073
[HyperMedDiff-Risk] Epoch 024 | Train 5.3217 | Val 4.1346
[HyperMedDiff-Risk] Epoch 025 | Train 5.2007 | Val 3.9595
[HyperMedDiff-Risk] Epoch 026 | Train 5.1555 | Val 3.9879
[HyperMedDiff-Risk] Epoch 027 | Train 5.0295 | Val 3.7783
[HyperMedDiff-Risk] Epoch 028 | Train 4.9933 | Val 3.7070
[HyperMedDiff-Risk] Epoch 029 | Train 4.8631 | Val 3.6751
[HyperMedDiff-Risk] Epoch 030 | Train 4.7774 | Val 3.4407
[HyperMedDiff-Risk] Epoch 031 | Train 4.7403 | Val 3.5449
[HyperMedDiff-Risk] Epoch 032 | Train 4.6778 | Val 3.4860
[HyperMedDiff-Risk] Epoch 033 | Train 4.5771 | Val 3.3612
[HyperMedDiff-Risk] Epoch 034 | Train 4.5305 | Val 3.2900
[HyperMedDiff-Risk] Epoch 035 | Train 4.4545 | Val 3.2447
[HyperMedDiff-Risk] Epoch 036 | Train 4.4105 | Val 3.2332
[HyperMedDiff-Risk] Epoch 037 | Train 4.3865 | Val 3.1744
[HyperMedDiff-Risk] Epoch 038 | Train 4.2865 | Val 3.1885
[HyperMedDiff-Risk] Epoch 039 | Train 4.2991 | Val 3.0617
[HyperMedDiff-Risk] Epoch 040 | Train 4.2181 | Val 3.0588
[HyperMedDiff-Risk] Epoch 041 | Train 4.1805 | Val 2.9750
[HyperMedDiff-Risk] Epoch 042 | Train 4.1623 | Val 2.9074
[HyperMedDiff-Risk] Epoch 043 | Train 4.1191 | Val 2.9486
[HyperMedDiff-Risk] Epoch 044 | Train 4.0713 | Val 2.8691
[HyperMedDiff-Risk] Epoch 045 | Train 3.9955 | Val 2.8429
[HyperMedDiff-Risk] Epoch 046 | Train 3.9772 | Val 2.7326
[HyperMedDiff-Risk] Epoch 047 | Train 3.9592 | Val 2.7765
[HyperMedDiff-Risk] Epoch 048 | Train 3.9505 | Val 2.8181
[HyperMedDiff-Risk] Epoch 049 | Train 3.9192 | Val 2.6815
[HyperMedDiff-Risk] Epoch 050 | Train 3.8806 | Val 2.7122
[HyperMedDiff-Risk] Epoch 051 | Train 3.8514 | Val 2.7251
[HyperMedDiff-Risk] Epoch 052 | Train 3.8477 | Val 2.7398
[HyperMedDiff-Risk] Epoch 053 | Train 3.8039 | Val 2.6452
[HyperMedDiff-Risk] Epoch 054 | Train 3.7910 | Val 2.6706
[HyperMedDiff-Risk] Epoch 055 | Train 3.7699 | Val 2.6589
[HyperMedDiff-Risk] Epoch 056 | Train 3.7892 | Val 2.5677
[HyperMedDiff-Risk] Epoch 057 | Train 3.7329 | Val 2.5975
[HyperMedDiff-Risk] Epoch 058 | Train 3.7396 | Val 2.5743
[HyperMedDiff-Risk] Epoch 059 | Train 3.7598 | Val 2.5494
[HyperMedDiff-Risk] Epoch 060 | Train 3.7023 | Val 2.5646
[HyperMedDiff-Risk] Epoch 061 | Train 3.6808 | Val 2.5844
[HyperMedDiff-Risk] Epoch 062 | Train 3.7036 | Val 2.6026
[HyperMedDiff-Risk] Epoch 063 | Train 3.6865 | Val 2.5445
[HyperMedDiff-Risk] Epoch 064 | Train 3.6567 | Val 2.5334
[HyperMedDiff-Risk] Epoch 065 | Train 3.6691 | Val 2.5125
[HyperMedDiff-Risk] Epoch 066 | Train 3.6398 | Val 2.5256
[HyperMedDiff-Risk] Epoch 067 | Train 3.6496 | Val 2.4995
[HyperMedDiff-Risk] Epoch 068 | Train 3.6207 | Val 2.4665
[HyperMedDiff-Risk] Epoch 069 | Train 3.6173 | Val 2.4874
[HyperMedDiff-Risk] Epoch 070 | Train 3.5995 | Val 2.5250
[HyperMedDiff-Risk] Epoch 071 | Train 3.6227 | Val 2.4380
[HyperMedDiff-Risk] Epoch 072 | Train 3.5853 | Val 2.5331
[HyperMedDiff-Risk] Epoch 073 | Train 3.5910 | Val 2.4804
[HyperMedDiff-Risk] Epoch 074 | Train 3.5826 | Val 2.4451
[HyperMedDiff-Risk] Epoch 075 | Train 3.5949 | Val 2.4960
[HyperMedDiff-Risk] Epoch 076 | Train 3.6048 | Val 2.5185
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 9): 2.4380
[HyperMedDiff-Risk] Saved training curve plot to results/plots/10_GenFocus.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8764362888012348,
  "auprc": 0.8115396498658187
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8419
Saved checkpoint to results/checkpoints/10_GenFocus.pt
[HyperMedDiff-Risk] ==== Ablation Summary ====
Run | Experiment     | ValLoss | AUROC  | AUPRC  | Accuracy | F1     | Kappa   | Correlation |
----+----------------+---------+--------+--------+----------+--------+---------+-------------+
0   | MedDiffusion   | NA      | NA     | 0.7064 | NA       | 0.6679 | 0.4526  | NA          |    
1   | Base           | 1.9489  | 0.8711 | 0.7991 | 0.7533   | 0.7175 | 0.50455 | 0.8385      |
2   | 02_NoDiffusion | 1.8176  | 0.8687 | 0.7919 | 0.7534   | 0.7175 | 0.50455 | 0.8897      |
3   | 03_LocalDiff   | 1.9546  | 0.8726 | 0.8054 | 0.7534   | 0.7175 | 0.50455 | 0.8533      |
4   | 04_GlobalDiff  | 1.8712  | 0.8740 | 0.8058 | 0.7534   | 0.7175 | 0.50455 | 0.8380      |
5   | 05_NoHDD       | 1.8350  | 0.8806 | 0.8291 | 0.7534   | 0.7175 | 0.50455 | -0.0021     | 
6   | 06_StrongHDD   | 1.8682  | 0.8696 | 0.8022 | 0.7534   | 0.7175 | 0.50455 | 0.9071      |
7   | 07_HighDropout | 1.8009  | 0.8733 | 0.8063 | 0.7534   | 0.7175 | 0.50455 | 0.8127      |
8   | 08_SmallDim    | 2.0531  | 0.8686 | 0.7928 | 0.7534   | 0.7175 | 0.50455 | 0.7374      |
9   | 09_DiscrimOnly | 1.2057  | 0.8758 | 0.8125 | 0.7534   | 0.7175 | 0.50455 | 0.8261      |
10  | 10_GenFocus    | 2.4380  | 0.8764 | 0.8115 | 0.7534   | 0.7175 | 0.50455 | 0.8419      |