Using device: mps

hg_ddpm_depth7 | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

=== Pretraining hyperbolic code embeddings (HDD-style) ===
[Pretrain-HDD] Epoch   1 | train=0.095129 | val=0.089797 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   2 | train=0.088499 | val=0.087420 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   3 | train=0.088495 | val=0.085106 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   4 | train=0.085312 | val=0.084299 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   5 | train=0.082613 | val=0.083037 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   6 | train=0.082289 | val=0.083395 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   7 | train=0.083101 | val=0.080321 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   8 | train=0.078694 | val=0.079667 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   9 | train=0.078451 | val=0.077943 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  10 | train=0.079223 | val=0.078763 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  11 | train=0.077978 | val=0.078674 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  12 | train=0.077940 | val=0.077720 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  13 | train=0.077950 | val=0.076721 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  14 | train=0.079143 | val=0.076254 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  15 | train=0.076393 | val=0.077022 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  16 | train=0.075985 | val=0.076011 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  17 | train=0.075025 | val=0.075840 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  18 | train=0.073368 | val=0.074009 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  19 | train=0.075350 | val=0.073297 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  20 | train=0.075493 | val=0.074095 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  21 | train=0.072655 | val=0.075889 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  22 | train=0.074477 | val=0.073998 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  23 | train=0.074805 | val=0.073697 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  24 | train=0.073796 | val=0.071621 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  25 | train=0.073171 | val=0.072451 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  26 | train=0.073376 | val=0.072502 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  27 | train=0.072737 | val=0.071480 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  28 | train=0.072904 | val=0.071370 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  29 | train=0.072060 | val=0.070833 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  30 | train=0.073498 | val=0.073656 | rad=0.003 pair=0.01 hdd=0.02
Saved pretraining checkpoint to results/checkpoints/hg_ddpm_pretrain_rad0.003_pair0.01_hdd0.02_val0.0708.pt

Training Hyperbolic Graph DDPM (Global) | depth=7 | lambda_recon=1000
[HG-DDPM] Epoch   1 | Train 61.668604 | Val 58.837242 | lambda_recon=1000
[HG-DDPM] Epoch   2 | Train 57.297646 | Val 54.797513 | lambda_recon=1000
[HG-DDPM] Epoch   3 | Train 54.561541 | Val 51.730292 | lambda_recon=1000
[HG-DDPM] Epoch   4 | Train 52.632792 | Val 49.843790 | lambda_recon=1000
[HG-DDPM] Epoch   5 | Train 51.014793 | Val 48.351627 | lambda_recon=1000
[HG-DDPM] Epoch   6 | Train 49.864169 | Val 47.199192 | lambda_recon=1000
[HG-DDPM] Epoch   7 | Train 48.590137 | Val 45.114584 | lambda_recon=1000
[HG-DDPM] Epoch   8 | Train 47.389112 | Val 44.385150 | lambda_recon=1000
[HG-DDPM] Epoch   9 | Train 46.458494 | Val 43.794604 | lambda_recon=1000
[HG-DDPM] Epoch  10 | Train 45.713391 | Val 43.177084 | lambda_recon=1000
[HG-DDPM] Epoch  11 | Train 45.133430 | Val 42.997875 | lambda_recon=1000
[HG-DDPM] Epoch  12 | Train 44.655887 | Val 41.963714 | lambda_recon=1000
[HG-DDPM] Epoch  13 | Train 44.240484 | Val 41.264827 | lambda_recon=1000
[HG-DDPM] Epoch  14 | Train 43.656335 | Val 40.843750 | lambda_recon=1000
[HG-DDPM] Epoch  15 | Train 43.354998 | Val 40.548796 | lambda_recon=1000
[HG-DDPM] Epoch  16 | Train 43.016015 | Val 40.949823 | lambda_recon=1000
[HG-DDPM] Epoch  17 | Train 42.971306 | Val 40.791926 | lambda_recon=1000
[HG-DDPM] Epoch  18 | Train 42.627593 | Val 41.006386 | lambda_recon=1000
[HG-DDPM] Early stopping at epoch 18 after 3 epochs without improvement.
[HG-DDPM] depth=7 | lambda_recon=1000 | pretrain_val=0.070833 | best_val=40.548796
Test Recall@4: 0.0102
Tree/embedding correlation: 0.7698

[HG-DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C131d2', 'C232d3', 'C244d1', 'C320d3']
  Visit 2: ['C102d3', 'C120d4', 'C244d1']
  Visit 3: ['C021d3', 'C021d4', 'C312d4', 'C313d3']
  Visit 4: ['C102d3', 'C224d3', 'C320d3', 'C343d4']
  Visit 5: ['C021d3', 'C021d4', 'C312d4', 'C344d4']
  Visit 6: ['C030d3', 'C224d3', 'C424d3', 'C440d4']

[HG-DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C000d3', 'C001d4', 'C243d4', 'C320d3']
  Visit 2: ['C012d3', 'C030d3', 'C102d3', 'C110d2']
  Visit 3: ['C232d3', 'C244d1', 'C344d4', 'C402d1']
  Visit 4: ['C102d3', 'C202d4', 'C212d1', 'C301d4']
  Visit 5: ['C130d4', 'C223d2', 'C232d1', 'C313d3']
  Visit 6: ['C012d3', 'C014d3', 'C024d2', 'C300d4']

[HG-DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C224d3', 'C243d3', 'C320d3', 'C440d4']
  Visit 2: ['C022d3', 'C114d4', 'C341d4', 'C413d4']
  Visit 3: ['C114d4', 'C130d4', 'C301d4', 'C442d4']
  Visit 4: ['C030d3', 'C102d3', 'C424d3', 'C440d4']
  Visit 5: ['C114d4', 'C130d4', 'C341d4', 'C442d4']
  Visit 6: ['C021d3', 'C021d4', 'C232d3', 'C401d3']
Synthetic stats (N=1000): {'mean_depth': 6.395033860045147, 'std_depth': 0.7178999957654014, 'mean_tree_dist': 12.202759300320276, 'std_tree_dist': 2.3155338994870953, 'mean_root_purity': 0.5112361111111111, 'std_root_purity': 0.15166483451783075}
Saved HG-DDPM model checkpoint to results/checkpoints/hg_ddpm_global_lrecon1000_depth7_best40.5488.pt

Training Hyperbolic Graph DDPM (Global) | depth=7 | lambda_recon=2000
[HG-DDPM] Epoch   1 | Train 91.014313 | Val 88.125670 | lambda_recon=2000
[HG-DDPM] Epoch   2 | Train 86.215123 | Val 83.451202 | lambda_recon=2000
[HG-DDPM] Epoch   3 | Train 83.736214 | Val 81.133332 | lambda_recon=2000
[HG-DDPM] Epoch   4 | Train 81.831934 | Val 79.062650 | lambda_recon=2000
[HG-DDPM] Epoch   5 | Train 80.227746 | Val 77.587092 | lambda_recon=2000
[HG-DDPM] Epoch   6 | Train 79.038002 | Val 76.735802 | lambda_recon=2000
[HG-DDPM] Epoch   7 | Train 77.722985 | Val 74.633863 | lambda_recon=2000
[HG-DDPM] Epoch   8 | Train 76.473225 | Val 73.320577 | lambda_recon=2000
[HG-DDPM] Epoch   9 | Train 75.685162 | Val 72.929742 | lambda_recon=2000
[HG-DDPM] Epoch  10 | Train 74.730631 | Val 72.246454 | lambda_recon=2000
[HG-DDPM] Epoch  11 | Train 73.985469 | Val 71.419572 | lambda_recon=2000
[HG-DDPM] Epoch  12 | Train 73.344867 | Val 71.196210 | lambda_recon=2000
[HG-DDPM] Epoch  13 | Train 72.680391 | Val 69.963864 | lambda_recon=2000
[HG-DDPM] Epoch  14 | Train 72.369212 | Val 69.474232 | lambda_recon=2000
[HG-DDPM] Epoch  15 | Train 71.739042 | Val 68.287046 | lambda_recon=2000
[HG-DDPM] Epoch  16 | Train 71.242673 | Val 68.600283 | lambda_recon=2000
[HG-DDPM] Epoch  17 | Train 71.022088 | Val 67.947100 | lambda_recon=2000
[HG-DDPM] Epoch  18 | Train 70.413219 | Val 67.279412 | lambda_recon=2000
[HG-DDPM] Epoch  19 | Train 70.049604 | Val 67.073703 | lambda_recon=2000
[HG-DDPM] Epoch  20 | Train 69.728169 | Val 67.377854 | lambda_recon=2000
[HG-DDPM] Epoch  21 | Train 69.629588 | Val 66.928219 | lambda_recon=2000
[HG-DDPM] Epoch  22 | Train 69.374987 | Val 66.745580 | lambda_recon=2000
[HG-DDPM] Epoch  23 | Train 69.245893 | Val 66.904453 | lambda_recon=2000
[HG-DDPM] Epoch  24 | Train 69.004660 | Val 66.801467 | lambda_recon=2000
[HG-DDPM] Epoch  25 | Train 68.771620 | Val 66.574476 | lambda_recon=2000
[HG-DDPM] Epoch  26 | Train 68.667266 | Val 66.340784 | lambda_recon=2000
[HG-DDPM] Epoch  27 | Train 68.682651 | Val 66.685439 | lambda_recon=2000
[HG-DDPM] Epoch  28 | Train 68.407396 | Val 66.461364 | lambda_recon=2000
[HG-DDPM] Epoch  29 | Train 68.340340 | Val 66.190543 | lambda_recon=2000
[HG-DDPM] Epoch  30 | Train 68.238126 | Val 66.319209 | lambda_recon=2000
[HG-DDPM] Epoch  31 | Train 68.196463 | Val 65.847590 | lambda_recon=2000
[HG-DDPM] Epoch  32 | Train 67.956554 | Val 65.547956 | lambda_recon=2000
[HG-DDPM] Epoch  33 | Train 67.876891 | Val 66.113216 | lambda_recon=2000
[HG-DDPM] Epoch  34 | Train 67.732264 | Val 66.105476 | lambda_recon=2000
[HG-DDPM] Epoch  35 | Train 67.911678 | Val 65.738356 | lambda_recon=2000
[HG-DDPM] Early stopping at epoch 35 after 3 epochs without improvement.
[HG-DDPM] depth=7 | lambda_recon=2000 | pretrain_val=0.070833 | best_val=65.547956
Test Recall@4: 0.0103
Tree/embedding correlation: 0.7607

[HG-DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C003d4', 'C010d3', 'C021d3', 'C123d4']
  Visit 2: ['C003d4', 'C010d3', 'C123d4', 'C131d2']
  Visit 3: ['C122d3', 'C222d4', 'C320d4', 'C324d3']
  Visit 4: ['C124d4', 'C140d4', 'C204d4', 'C313d3']
  Visit 5: ['C012d3', 'C140d4', 'C312d4', 'C402d2']
  Visit 6: ['C012d3', 'C302d4', 'C303d3']

[HG-DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C122d3', 'C312d4', 'C313d3', 'C412d3']
  Visit 2: ['C010d3', 'C302d4', 'C444d4']
  Visit 3: ['C044d4', 'C302d4', 'C444d4']
  Visit 4: ['C130d4', 'C140d3', 'C322d4', 'C334d3']
  Visit 5: ['C124d4', 'C140d4', 'C220d2', 'C312d4']
  Visit 6: ['C012d3', 'C124d4', 'C243d3', 'C312d4']

[HG-DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C140d3', 'C302d4', 'C314d4', 'C334d3']
  Visit 2: ['C124d4', 'C140d4', 'C242d3', 'C313d4']
  Visit 3: ['C003d4', 'C122d3', 'C222d4', 'C320d4']
  Visit 4: ['C010d3', 'C104d2', 'C122d3', 'C131d2']
  Visit 5: ['C003d4', 'C010d3', 'C123d4', 'C211d4']
  Visit 6: ['C302d4', 'C412d3', 'C443d4']
Synthetic stats (N=1000): {'mean_depth': 6.556034482758621, 'std_depth': 0.5983647477642123, 'mean_tree_dist': 12.345932378561937, 'std_tree_dist': 2.595481554506383, 'mean_root_purity': 0.5163194444444444, 'std_root_purity': 0.15443631011252776}
Saved HG-DDPM model checkpoint to results/checkpoints/hg_ddpm_global_lrecon2000_depth7_best65.5480.pt
