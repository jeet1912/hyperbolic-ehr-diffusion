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
[HyperMedDiff-Risk] Running 6 HDD sweep configurations.
[HyperMedDiff-Risk] ===== Experiment 1/6: HDD_Sweep_0.0 =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8,
    16
  ],
  "dropout": 0.5,
  "train_lr": 0.0001,
  "train_epochs": 100,
  "lambda_s": 0.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.0,
  "lambda_hdd": 0.0,
  "lambda_radius": 0.003,
  "embed_dim": 128
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
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.0173
[HyperMedDiff-Risk] Epoch 001 | Train 0.5998 | Val 0.5620
[HyperMedDiff-Risk] Epoch 002 | Train 0.5552 | Val 0.5650
[HyperMedDiff-Risk] Epoch 003 | Train 0.5549 | Val 0.5615
[HyperMedDiff-Risk] Epoch 004 | Train 0.5532 | Val 0.5610
[HyperMedDiff-Risk] Epoch 005 | Train 0.5543 | Val 0.5570
[HyperMedDiff-Risk] Epoch 006 | Train 0.5251 | Val 0.4848
[HyperMedDiff-Risk] Epoch 007 | Train 0.4818 | Val 0.4640
[HyperMedDiff-Risk] Epoch 008 | Train 0.4716 | Val 0.4666
[HyperMedDiff-Risk] Epoch 009 | Train 0.4572 | Val 0.4306
[HyperMedDiff-Risk] Epoch 010 | Train 0.4429 | Val 0.4735
[HyperMedDiff-Risk] Epoch 011 | Train 0.4377 | Val 0.5022
[HyperMedDiff-Risk] Epoch 012 | Train 0.4282 | Val 0.4497
[HyperMedDiff-Risk] Epoch 013 | Train 0.4205 | Val 0.4071
[HyperMedDiff-Risk] Epoch 014 | Train 0.4143 | Val 0.4101
[HyperMedDiff-Risk] Epoch 015 | Train 0.4064 | Val 0.4290
[HyperMedDiff-Risk] Epoch 016 | Train 0.3987 | Val 0.4104
[HyperMedDiff-Risk] Epoch 017 | Train 0.3941 | Val 0.4350
[HyperMedDiff-Risk] Epoch 018 | Train 0.3863 | Val 0.3830
[HyperMedDiff-Risk] Epoch 019 | Train 0.3857 | Val 0.4984
[HyperMedDiff-Risk] Epoch 020 | Train 0.3823 | Val 0.4017
[HyperMedDiff-Risk] Epoch 021 | Train 0.3642 | Val 0.3822
[HyperMedDiff-Risk] Epoch 022 | Train 0.3560 | Val 0.3485
[HyperMedDiff-Risk] Epoch 023 | Train 0.3560 | Val 0.3486
[HyperMedDiff-Risk] Epoch 024 | Train 0.3407 | Val 0.3399
[HyperMedDiff-Risk] Epoch 025 | Train 0.3395 | Val 0.3692
[HyperMedDiff-Risk] Epoch 026 | Train 0.3368 | Val 0.3588
[HyperMedDiff-Risk] Epoch 027 | Train 0.3331 | Val 0.4085
[HyperMedDiff-Risk] Epoch 028 | Train 0.3335 | Val 0.3966
[HyperMedDiff-Risk] Epoch 029 | Train 0.3267 | Val 0.3727
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 1): 0.3399
[HyperMedDiff-Risk] Saved training curve plot to results/plots/HDD_Sweep_0.0.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.8442844284428442,
  "f1": 0.7897934386391251,
  "kappa": 0.6663449370110042,
  "auroc": 0.933472817698508,
  "auprc": 0.8688987557941306
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.0043
Saved checkpoint to results/checkpoints/HDD_Sweep_0.0.pt
[HyperMedDiff-Risk] ===== Experiment 2/6: HDD_Sweep_0.001 =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8,
    16
  ],
  "dropout": 0.5,
  "train_lr": 0.0001,
  "train_epochs": 100,
  "lambda_s": 0.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.0,
  "lambda_hdd": 0.001,
  "lambda_radius": 0.003,
  "embed_dim": 128
}
[Pretrain] Epoch 01 | train=0.0029 | val=0.0028
[Pretrain] Epoch 02 | train=0.0028 | val=0.0027
[Pretrain] Epoch 03 | train=0.0027 | val=0.0025
[Pretrain] Epoch 04 | train=0.0025 | val=0.0024
[Pretrain] Epoch 05 | train=0.0024 | val=0.0023
[Pretrain] Epoch 06 | train=0.0023 | val=0.0022
[Pretrain] Epoch 07 | train=0.0022 | val=0.0021
[Pretrain] Epoch 08 | train=0.0021 | val=0.0020
[Pretrain] Epoch 09 | train=0.0020 | val=0.0019
[Pretrain] Epoch 10 | train=0.0019 | val=0.0018
[Pretrain] Epoch 11 | train=0.0018 | val=0.0018
[Pretrain] Epoch 12 | train=0.0018 | val=0.0017
[Pretrain] Epoch 13 | train=0.0017 | val=0.0016
[Pretrain] Epoch 14 | train=0.0016 | val=0.0016
[Pretrain] Epoch 15 | train=0.0016 | val=0.0015
[Pretrain] Epoch 16 | train=0.0015 | val=0.0015
[Pretrain] Epoch 17 | train=0.0015 | val=0.0014
[Pretrain] Epoch 18 | train=0.0014 | val=0.0014
[Pretrain] Epoch 19 | train=0.0014 | val=0.0014
[Pretrain] Epoch 20 | train=0.0014 | val=0.0014
[Pretrain] Epoch 21 | train=0.0014 | val=0.0013
[Pretrain] Epoch 22 | train=0.0013 | val=0.0013
[Pretrain] Epoch 23 | train=0.0013 | val=0.0013
[Pretrain] Epoch 24 | train=0.0013 | val=0.0013
[Pretrain] Epoch 25 | train=0.0013 | val=0.0013
[Pretrain] Epoch 26 | train=0.0013 | val=0.0013
[Pretrain] Epoch 27 | train=0.0013 | val=0.0013
[Pretrain] Epoch 28 | train=0.0013 | val=0.0013
[Pretrain] Epoch 29 | train=0.0013 | val=0.0013
[Pretrain] Epoch 30 | train=0.0013 | val=0.0014
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.3269
[HyperMedDiff-Risk] Epoch 001 | Train 0.5931 | Val 0.5611
[HyperMedDiff-Risk] Epoch 002 | Train 0.5549 | Val 0.5623
[HyperMedDiff-Risk] Epoch 003 | Train 0.5547 | Val 0.5619
[HyperMedDiff-Risk] Epoch 004 | Train 0.5547 | Val 0.5628
[HyperMedDiff-Risk] Epoch 005 | Train 0.5527 | Val 0.5612
[HyperMedDiff-Risk] Epoch 006 | Train 0.5420 | Val 0.4953
[HyperMedDiff-Risk] Epoch 007 | Train 0.4964 | Val 0.4627
[HyperMedDiff-Risk] Epoch 008 | Train 0.4714 | Val 0.4505
[HyperMedDiff-Risk] Epoch 009 | Train 0.4567 | Val 0.4296
[HyperMedDiff-Risk] Epoch 010 | Train 0.4466 | Val 0.4294
[HyperMedDiff-Risk] Epoch 011 | Train 0.4346 | Val 0.4261
[HyperMedDiff-Risk] Epoch 012 | Train 0.4359 | Val 0.4201
[HyperMedDiff-Risk] Epoch 013 | Train 0.4193 | Val 0.4148
[HyperMedDiff-Risk] Epoch 014 | Train 0.4158 | Val 0.4268
[HyperMedDiff-Risk] Epoch 015 | Train 0.4196 | Val 0.3989
[HyperMedDiff-Risk] Epoch 016 | Train 0.4058 | Val 0.4465
[HyperMedDiff-Risk] Epoch 017 | Train 0.3981 | Val 0.3910
[HyperMedDiff-Risk] Epoch 018 | Train 0.3922 | Val 0.3914
[HyperMedDiff-Risk] Epoch 019 | Train 0.3884 | Val 0.3869
[HyperMedDiff-Risk] Epoch 020 | Train 0.3789 | Val 0.3967
[HyperMedDiff-Risk] Epoch 021 | Train 0.3771 | Val 0.3978
[HyperMedDiff-Risk] Epoch 022 | Train 0.3695 | Val 0.4252
[HyperMedDiff-Risk] Epoch 023 | Train 0.3641 | Val 0.4536
[HyperMedDiff-Risk] Epoch 024 | Train 0.3592 | Val 0.4124
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 2): 0.3869
[HyperMedDiff-Risk] Saved training curve plot to results/plots/HDD_Sweep_0.001.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.8370837083708371,
  "f1": 0.7717528373266078,
  "kappa": 0.646105490274065,
  "auroc": 0.9099742754244555,
  "auprc": 0.8481089079590309
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.3181
Saved checkpoint to results/checkpoints/HDD_Sweep_0.001.pt
[HyperMedDiff-Risk] ===== Experiment 3/6: HDD_Sweep_0.005 =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8,
    16
  ],
  "dropout": 0.5,
  "train_lr": 0.0001,
  "train_epochs": 100,
  "lambda_s": 0.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.0,
  "lambda_hdd": 0.005,
  "lambda_radius": 0.003,
  "embed_dim": 128
}
[Pretrain] Epoch 01 | train=0.0031 | val=0.0029
[Pretrain] Epoch 02 | train=0.0029 | val=0.0028
[Pretrain] Epoch 03 | train=0.0028 | val=0.0026
[Pretrain] Epoch 04 | train=0.0026 | val=0.0025
[Pretrain] Epoch 05 | train=0.0025 | val=0.0024
[Pretrain] Epoch 06 | train=0.0024 | val=0.0023
[Pretrain] Epoch 07 | train=0.0023 | val=0.0022
[Pretrain] Epoch 08 | train=0.0022 | val=0.0021
[Pretrain] Epoch 09 | train=0.0021 | val=0.0020
[Pretrain] Epoch 10 | train=0.0020 | val=0.0020
[Pretrain] Epoch 11 | train=0.0020 | val=0.0019
[Pretrain] Epoch 12 | train=0.0019 | val=0.0019
[Pretrain] Epoch 13 | train=0.0019 | val=0.0018
[Pretrain] Epoch 14 | train=0.0018 | val=0.0018
[Pretrain] Epoch 15 | train=0.0018 | val=0.0018
[Pretrain] Epoch 16 | train=0.0018 | val=0.0018
[Pretrain] Epoch 17 | train=0.0018 | val=0.0018
[Pretrain] Epoch 18 | train=0.0018 | val=0.0018
[Pretrain] Epoch 19 | train=0.0018 | val=0.0018
[Pretrain] Epoch 20 | train=0.0018 | val=0.0018
[Pretrain] Epoch 21 | train=0.0018 | val=0.0018
[Pretrain] Epoch 22 | train=0.0018 | val=0.0018
[Pretrain] Epoch 23 | train=0.0018 | val=0.0018
[Pretrain] Epoch 24 | train=0.0018 | val=0.0018
[Pretrain] Epoch 25 | train=0.0018 | val=0.0018
[Pretrain] Epoch 26 | train=0.0018 | val=0.0018
[Pretrain] Epoch 27 | train=0.0018 | val=0.0018
[Pretrain] Epoch 28 | train=0.0018 | val=0.0018
[Pretrain] Epoch 29 | train=0.0018 | val=0.0018
[Pretrain] Epoch 30 | train=0.0018 | val=0.0018
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.5554
[HyperMedDiff-Risk] Epoch 001 | Train 0.5948 | Val 0.5615
[HyperMedDiff-Risk] Epoch 002 | Train 0.5554 | Val 0.5632
[HyperMedDiff-Risk] Epoch 003 | Train 0.5540 | Val 0.5689
[HyperMedDiff-Risk] Epoch 004 | Train 0.5540 | Val 0.5635
[HyperMedDiff-Risk] Epoch 005 | Train 0.5536 | Val 0.5630
[HyperMedDiff-Risk] Epoch 006 | Train 0.5543 | Val 0.5645
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 3): 0.5615
[HyperMedDiff-Risk] Saved training curve plot to results/plots/HDD_Sweep_0.005.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8594066197907735,
  "auprc": 0.7812454529398739
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.5537
Saved checkpoint to results/checkpoints/HDD_Sweep_0.005.pt
[HyperMedDiff-Risk] ===== Experiment 4/6: HDD_Sweep_0.01 =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8,
    16
  ],
  "dropout": 0.5,
  "train_lr": 0.0001,
  "train_epochs": 100,
  "lambda_s": 0.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.0,
  "lambda_hdd": 0.01,
  "lambda_radius": 0.003,
  "embed_dim": 128
}
[Pretrain] Epoch 01 | train=0.0032 | val=0.0031
[Pretrain] Epoch 02 | train=0.0031 | val=0.0029
[Pretrain] Epoch 03 | train=0.0029 | val=0.0027
[Pretrain] Epoch 04 | train=0.0027 | val=0.0026
[Pretrain] Epoch 05 | train=0.0026 | val=0.0024
[Pretrain] Epoch 06 | train=0.0025 | val=0.0023
[Pretrain] Epoch 07 | train=0.0023 | val=0.0022
[Pretrain] Epoch 08 | train=0.0023 | val=0.0022
[Pretrain] Epoch 09 | train=0.0022 | val=0.0021
[Pretrain] Epoch 10 | train=0.0021 | val=0.0021
[Pretrain] Epoch 11 | train=0.0021 | val=0.0020
[Pretrain] Epoch 12 | train=0.0020 | val=0.0020
[Pretrain] Epoch 13 | train=0.0020 | val=0.0020
[Pretrain] Epoch 14 | train=0.0020 | val=0.0020
[Pretrain] Epoch 15 | train=0.0020 | val=0.0020
[Pretrain] Epoch 16 | train=0.0020 | val=0.0020
[Pretrain] Epoch 17 | train=0.0020 | val=0.0020
[Pretrain] Epoch 18 | train=0.0020 | val=0.0020
[Pretrain] Epoch 19 | train=0.0020 | val=0.0020
[Pretrain] Epoch 20 | train=0.0020 | val=0.0020
[Pretrain] Epoch 21 | train=0.0020 | val=0.0020
[Pretrain] Epoch 22 | train=0.0020 | val=0.0020
[Pretrain] Epoch 23 | train=0.0020 | val=0.0020
[Pretrain] Epoch 24 | train=0.0020 | val=0.0020
[Pretrain] Epoch 25 | train=0.0020 | val=0.0020
[Pretrain] Epoch 26 | train=0.0020 | val=0.0020
[Pretrain] Epoch 27 | train=0.0020 | val=0.0020
[Pretrain] Epoch 28 | train=0.0020 | val=0.0020
[Pretrain] Epoch 29 | train=0.0020 | val=0.0020
[Pretrain] Epoch 30 | train=0.0020 | val=0.0020
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.6804
[HyperMedDiff-Risk] Epoch 001 | Train 0.6055 | Val 0.5623
[HyperMedDiff-Risk] Epoch 002 | Train 0.5551 | Val 0.5642
[HyperMedDiff-Risk] Epoch 003 | Train 0.5549 | Val 0.5624
[HyperMedDiff-Risk] Epoch 004 | Train 0.5538 | Val 0.5622
[HyperMedDiff-Risk] Epoch 005 | Train 0.5544 | Val 0.5629
[HyperMedDiff-Risk] Epoch 006 | Train 0.5536 | Val 0.5601
[HyperMedDiff-Risk] Epoch 007 | Train 0.5508 | Val 0.5429
[HyperMedDiff-Risk] Epoch 008 | Train 0.5209 | Val 0.4801
[HyperMedDiff-Risk] Epoch 009 | Train 0.4911 | Val 0.4831
[HyperMedDiff-Risk] Epoch 010 | Train 0.4812 | Val 0.4522
[HyperMedDiff-Risk] Epoch 011 | Train 0.4729 | Val 0.4370
[HyperMedDiff-Risk] Epoch 012 | Train 0.4545 | Val 0.4482
[HyperMedDiff-Risk] Epoch 013 | Train 0.4533 | Val 0.4283
[HyperMedDiff-Risk] Epoch 014 | Train 0.4473 | Val 0.4605
[HyperMedDiff-Risk] Epoch 015 | Train 0.4388 | Val 0.4550
[HyperMedDiff-Risk] Epoch 016 | Train 0.4281 | Val 0.4134
[HyperMedDiff-Risk] Epoch 017 | Train 0.4234 | Val 0.4819
[HyperMedDiff-Risk] Epoch 018 | Train 0.4326 | Val 0.4295
[HyperMedDiff-Risk] Epoch 019 | Train 0.4194 | Val 0.4214
[HyperMedDiff-Risk] Epoch 020 | Train 0.4100 | Val 0.4721
[HyperMedDiff-Risk] Epoch 021 | Train 0.4080 | Val 0.4144
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 4): 0.4134
[HyperMedDiff-Risk] Saved training curve plot to results/plots/HDD_Sweep_0.01.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.8298829882988299,
  "f1": 0.7652173913043478,
  "kappa": 0.6324894330145007,
  "auroc": 0.9034539530097753,
  "auprc": 0.826135744428983
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.6683
Saved checkpoint to results/checkpoints/HDD_Sweep_0.01.pt
[HyperMedDiff-Risk] ===== Experiment 5/6: HDD_Sweep_0.025 =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8,
    16
  ],
  "dropout": 0.5,
  "train_lr": 0.0001,
  "train_epochs": 100,
  "lambda_s": 0.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.0,
  "lambda_hdd": 0.025,
  "lambda_radius": 0.003,
  "embed_dim": 128
}
[Pretrain] Epoch 01 | train=0.0038 | val=0.0035
[Pretrain] Epoch 02 | train=0.0035 | val=0.0032
[Pretrain] Epoch 03 | train=0.0032 | val=0.0030
[Pretrain] Epoch 04 | train=0.0030 | val=0.0028
[Pretrain] Epoch 05 | train=0.0028 | val=0.0027
[Pretrain] Epoch 06 | train=0.0027 | val=0.0025
[Pretrain] Epoch 07 | train=0.0025 | val=0.0024
[Pretrain] Epoch 08 | train=0.0024 | val=0.0024
[Pretrain] Epoch 09 | train=0.0024 | val=0.0023
[Pretrain] Epoch 10 | train=0.0023 | val=0.0023
[Pretrain] Epoch 11 | train=0.0023 | val=0.0022
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
[Pretrain] Epoch 26 | train=0.0022 | val=0.0021
[Pretrain] Epoch 27 | train=0.0021 | val=0.0022
[Pretrain] Epoch 28 | train=0.0021 | val=0.0022
[Pretrain] Epoch 29 | train=0.0022 | val=0.0021
[Pretrain] Epoch 30 | train=0.0021 | val=0.0021
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8546
[HyperMedDiff-Risk] Epoch 001 | Train 0.6034 | Val 0.5639
[HyperMedDiff-Risk] Epoch 002 | Train 0.5557 | Val 0.5628
[HyperMedDiff-Risk] Epoch 003 | Train 0.5544 | Val 0.5651
[HyperMedDiff-Risk] Epoch 004 | Train 0.5546 | Val 0.5654
[HyperMedDiff-Risk] Epoch 005 | Train 0.5543 | Val 0.5703
[HyperMedDiff-Risk] Epoch 006 | Train 0.5545 | Val 0.5626
[HyperMedDiff-Risk] Epoch 007 | Train 0.5539 | Val 0.5625
[HyperMedDiff-Risk] Epoch 008 | Train 0.5540 | Val 0.5624
[HyperMedDiff-Risk] Epoch 009 | Train 0.5533 | Val 0.5621
[HyperMedDiff-Risk] Epoch 010 | Train 0.5524 | Val 0.5620
[HyperMedDiff-Risk] Epoch 011 | Train 0.5445 | Val 0.5096
[HyperMedDiff-Risk] Epoch 012 | Train 0.5150 | Val 0.5007
[HyperMedDiff-Risk] Epoch 013 | Train 0.4884 | Val 0.4682
[HyperMedDiff-Risk] Epoch 014 | Train 0.4792 | Val 0.4619
[HyperMedDiff-Risk] Epoch 015 | Train 0.4603 | Val 0.4456
[HyperMedDiff-Risk] Epoch 016 | Train 0.4470 | Val 0.4384
[HyperMedDiff-Risk] Epoch 017 | Train 0.4454 | Val 0.4934
[HyperMedDiff-Risk] Epoch 018 | Train 0.4344 | Val 0.4207
[HyperMedDiff-Risk] Epoch 019 | Train 0.4288 | Val 0.4263
[HyperMedDiff-Risk] Epoch 020 | Train 0.4233 | Val 0.4117
[HyperMedDiff-Risk] Epoch 021 | Train 0.4209 | Val 0.4066
[HyperMedDiff-Risk] Epoch 022 | Train 0.4208 | Val 0.4310
[HyperMedDiff-Risk] Epoch 023 | Train 0.4133 | Val 0.4088
[HyperMedDiff-Risk] Epoch 024 | Train 0.4050 | Val 0.4218
[HyperMedDiff-Risk] Epoch 025 | Train 0.4023 | Val 0.4231
[HyperMedDiff-Risk] Epoch 026 | Train 0.3946 | Val 0.4506
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 5): 0.4066
[HyperMedDiff-Risk] Saved training curve plot to results/plots/HDD_Sweep_0.025.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.8388838883888389,
  "f1": 0.779284833538841,
  "kappa": 0.652885825445393,
  "auroc": 0.9074326873606585,
  "auprc": 0.839514301553864
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8514
Saved checkpoint to results/checkpoints/HDD_Sweep_0.025.pt
[HyperMedDiff-Risk] ===== Experiment 6/6: HDD_Sweep_0.05 =====
{
  "diffusion_steps": [
    1,
    2,
    4,
    8,
    16
  ],
  "dropout": 0.5,
  "train_lr": 0.0001,
  "train_epochs": 100,
  "lambda_s": 0.0,
  "lambda_d": 0.0,
  "lambda_consistency": 0.0,
  "lambda_hdd": 0.05,
  "lambda_radius": 0.003,
  "embed_dim": 128
}
[Pretrain] Epoch 01 | train=0.0048 | val=0.0042
[Pretrain] Epoch 02 | train=0.0043 | val=0.0038
[Pretrain] Epoch 03 | train=0.0038 | val=0.0035
[Pretrain] Epoch 04 | train=0.0035 | val=0.0032
[Pretrain] Epoch 05 | train=0.0032 | val=0.0030
[Pretrain] Epoch 06 | train=0.0030 | val=0.0029
[Pretrain] Epoch 07 | train=0.0028 | val=0.0027
[Pretrain] Epoch 08 | train=0.0027 | val=0.0026
[Pretrain] Epoch 09 | train=0.0026 | val=0.0025
[Pretrain] Epoch 10 | train=0.0025 | val=0.0025
[Pretrain] Epoch 11 | train=0.0025 | val=0.0025
[Pretrain] Epoch 12 | train=0.0024 | val=0.0024
[Pretrain] Epoch 13 | train=0.0024 | val=0.0024
[Pretrain] Epoch 14 | train=0.0024 | val=0.0025
[Pretrain] Epoch 15 | train=0.0024 | val=0.0024
[Pretrain] Epoch 16 | train=0.0025 | val=0.0024
[Pretrain] Epoch 17 | train=0.0024 | val=0.0024
[Pretrain] Epoch 18 | train=0.0024 | val=0.0024
[Pretrain] Epoch 19 | train=0.0024 | val=0.0024
[Pretrain] Epoch 20 | train=0.0024 | val=0.0023
[Pretrain] Epoch 21 | train=0.0024 | val=0.0023
[Pretrain] Epoch 22 | train=0.0023 | val=0.0023
[Pretrain] Epoch 23 | train=0.0023 | val=0.0023
[Pretrain] Epoch 24 | train=0.0023 | val=0.0023
[Pretrain] Epoch 25 | train=0.0023 | val=0.0023
[Pretrain] Epoch 26 | train=0.0023 | val=0.0023
[Pretrain] Epoch 27 | train=0.0023 | val=0.0023
[Pretrain] Epoch 28 | train=0.0022 | val=0.0023
[Pretrain] Epoch 29 | train=0.0022 | val=0.0022
[Pretrain] Epoch 30 | train=0.0022 | val=0.0022
[HyperMedDiff-Risk] Diffusion/embedding correlation after pretraining: 0.8852
[HyperMedDiff-Risk] Epoch 001 | Train 0.5967 | Val 0.5609
[HyperMedDiff-Risk] Epoch 002 | Train 0.5558 | Val 0.5621
[HyperMedDiff-Risk] Epoch 003 | Train 0.5540 | Val 0.5643
[HyperMedDiff-Risk] Epoch 004 | Train 0.5546 | Val 0.5645
[HyperMedDiff-Risk] Epoch 005 | Train 0.5548 | Val 0.5632
[HyperMedDiff-Risk] Epoch 006 | Train 0.5544 | Val 0.5638
[HyperMedDiff-Risk] Early stopping.
[HyperMedDiff-Risk] Best validation total loss (run 6): 0.5609
[HyperMedDiff-Risk] Saved training curve plot to results/plots/HDD_Sweep_0.05.png
[HyperMedDiff-Risk] Test risk metrics (MedDiffusion-style):
{
  "accuracy": 0.7533753375337534,
  "f1": 0.7175257731958763,
  "kappa": 0.5045506331174116,
  "auroc": 0.8678237009089349,
  "auprc": 0.7931639278832308
}
[HyperMedDiff-Risk] Diffusion/embedding correlation after training: 0.8908
Saved checkpoint to results/checkpoints/HDD_Sweep_0.05.pt
[HyperMedDiff-Risk] ==== Ablation Summary ====
Run | Experiment      | ValLoss | AUROC  | AUPRC  | Accuracy | F1     | Kappa   | Correlation |
----+-----------------+---------+--------+--------+----------+--------+---------+-------------+
1   | HDD_Sweep_0.0   | 0.3399  | 0.9335 | 0.8689 | 0.8443   | 0.7898 | 0.6663  | 0.0043      |
2   | HDD_Sweep_0.001 | 0.3869  | 0.9100 | 0.8481 | 0.8371   | 0.7718 | 0.6461  | 0.3181      |
3   | HDD_Sweep_0.005 | 0.5615  | 0.8594 | 0.7812 | 0.7534   | 0.7175 | 0.5045  | 0.5537      |
4   | HDD_Sweep_0.01  | 0.4134  | 0.9035 | 0.8261 | 0.8299   | 0.7652 | 0.6324  | 0.6683      |
5   | HDD_Sweep_0.025 | 0.4066  | 0.9074 | 0.8395 | 0.8389   | 0.7793 | 0.6528  | 0.8514      |   
6   | HDD_Sweep_0.05  | 0.5609  | 0.8678 | 0.7932 | 0.7534   | 0.7175 | 0.50455 | 0.8908      |
[HyperMedDiff-Risk] Saved corr vs. AUPRC plot to results/plots/corr_vs_auprc.png
