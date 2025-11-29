Using device: mps

rectified_depth2 | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Pretraining hyperbolic graph embeddings (Rectified) ===
[Pretrain] Epoch   1 | train=0.084093 | val=0.078877 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   2 | train=0.079546 | val=0.074398 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   3 | train=0.073537 | val=0.076239 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   4 | train=0.072836 | val=0.072066 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   5 | train=0.075046 | val=0.073842 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   6 | train=0.074024 | val=0.070136 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   7 | train=0.071365 | val=0.070757 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   8 | train=0.071310 | val=0.069640 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   9 | train=0.070457 | val=0.070048 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  10 | train=0.069082 | val=0.068836 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  11 | train=0.067958 | val=0.068881 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  12 | train=0.068186 | val=0.066937 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  13 | train=0.067376 | val=0.066519 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  14 | train=0.068133 | val=0.066853 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  15 | train=0.066365 | val=0.066997 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  16 | train=0.065443 | val=0.064393 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  17 | train=0.066035 | val=0.065164 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  18 | train=0.064132 | val=0.063383 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  19 | train=0.064113 | val=0.063474 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  20 | train=0.063599 | val=0.063059 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  21 | train=0.064106 | val=0.063653 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  22 | train=0.062403 | val=0.062692 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  23 | train=0.062826 | val=0.062377 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  24 | train=0.062214 | val=0.061855 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  25 | train=0.062276 | val=0.061540 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  26 | train=0.062021 | val=0.061072 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  27 | train=0.061561 | val=0.061713 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  28 | train=0.060566 | val=0.061577 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  29 | train=0.061031 | val=0.060362 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  30 | train=0.060343 | val=0.061162 | rad=0.003 pair=0.01 hdd=0.02
Saved pretraining checkpoint to results/checkpoints/hyperbolic_rectified_pretrain_rad0.003_pair0.01_val0.0604.pt
[Rectified] Epoch   1 | Train 105.745667 | Val 72.556434 | lambda_recon=1
[Rectified] Epoch   2 | Train 68.068950 | Val 50.021494 | lambda_recon=1
[Rectified] Epoch   3 | Train 53.472990 | Val 38.056656 | lambda_recon=1
[Rectified] Epoch   4 | Train 44.446606 | Val 29.524684 | lambda_recon=1
[Rectified] Epoch   5 | Train 38.067071 | Val 24.007946 | lambda_recon=1
[Rectified] Epoch   6 | Train 33.470038 | Val 20.264759 | lambda_recon=1
[Rectified] Epoch   7 | Train 29.810830 | Val 16.246056 | lambda_recon=1
[Rectified] Epoch   8 | Train 26.987510 | Val 14.930276 | lambda_recon=1
[Rectified] Epoch   9 | Train 24.906571 | Val 12.710737 | lambda_recon=1
[Rectified] Epoch  10 | Train 23.379227 | Val 11.584026 | lambda_recon=1
[Rectified] Epoch  11 | Train 21.511659 | Val 10.753931 | lambda_recon=1
[Rectified] Epoch  12 | Train 20.417930 | Val 10.400625 | lambda_recon=1
[Rectified] Epoch  13 | Train 19.074775 | Val 8.778056 | lambda_recon=1
[Rectified] Epoch  14 | Train 18.171749 | Val 8.822393 | lambda_recon=1
[Rectified] Epoch  15 | Train 17.437555 | Val 8.176777 | lambda_recon=1
[Rectified] Epoch  16 | Train 16.682784 | Val 8.059823 | lambda_recon=1
[Rectified] Epoch  17 | Train 15.942759 | Val 7.511813 | lambda_recon=1
[Rectified] Epoch  18 | Train 15.249171 | Val 7.021710 | lambda_recon=1
[Rectified] Epoch  19 | Train 14.754943 | Val 6.707394 | lambda_recon=1
[Rectified] Epoch  20 | Train 14.100952 | Val 6.408986 | lambda_recon=1
[Rectified] Epoch  21 | Train 13.674279 | Val 6.429701 | lambda_recon=1
[Rectified] Epoch  22 | Train 13.304951 | Val 5.929953 | lambda_recon=1
[Rectified] Epoch  23 | Train 12.965351 | Val 5.877863 | lambda_recon=1
[Rectified] Epoch  24 | Train 12.571392 | Val 5.863462 | lambda_recon=1
[Rectified] Epoch  25 | Train 12.348345 | Val 5.923701 | lambda_recon=1
[Rectified] Epoch  26 | Train 12.079127 | Val 6.134069 | lambda_recon=1
[Rectified] Epoch  27 | Train 11.806584 | Val 5.770550 | lambda_recon=1
[Rectified] Epoch  28 | Train 11.536620 | Val 5.471941 | lambda_recon=1
[Rectified] Epoch  29 | Train 11.311452 | Val 5.581121 | lambda_recon=1
[Rectified] Epoch  30 | Train 11.207745 | Val 5.646412 | lambda_recon=1
[Rectified] Epoch  31 | Train 11.184579 | Val 5.456217 | lambda_recon=1
[Rectified] Epoch  32 | Train 10.878260 | Val 5.443507 | lambda_recon=1
[Rectified] Epoch  33 | Train 10.871982 | Val 4.979096 | lambda_recon=1
[Rectified] Epoch  34 | Train 10.711800 | Val 5.197510 | lambda_recon=1
[Rectified] Epoch  35 | Train 10.573320 | Val 5.262654 | lambda_recon=1
[Rectified] Epoch  36 | Train 10.417017 | Val 5.580744 | lambda_recon=1
[Rectified] Early stopping.
[Summary Rectified] depth=2 | lambda_recon=1 | pretrain_val=0.060362 | best_val=4.979096
Test Recall@4: 0.0687
Tree-Embedding Correlation: 0.8885

[Rectified] Sample trajectory 1:
  Visit 1: ['C202', 'C340', 'C401', 'C402']
  Visit 2: ['C143', 'C144', 'C331', 'C424']
  Visit 3: ['C202', 'C203', 'C204', 'C310']
  Visit 4: ['C020', 'C022', 'C023', 'C024']
  Visit 5: ['C031', 'C033', 'C034', 'C333']
  Visit 6: ['C140', 'C200', 'C203', 'C204']

[Rectified] Sample trajectory 2:
  Visit 1: ['C023', 'C024', 'C113', 'C301']
  Visit 2: ['C100', 'C101', 'C143', 'C333']
  Visit 3: ['C022', 'C023', 'C113', 'C301']
  Visit 4: ['C202', 'C204', 'C310', 'C313']
  Visit 5: ['C020', 'C024', 'C121', 'C301']
  Visit 6: ['C204', 'C214', 'C222', 'C413']

[Rectified] Sample trajectory 3:
  Visit 1: ['C020', 'C222', 'C310', 'C313']
  Visit 2: ['C100', 'C101', 'C143', 'C333']
  Visit 3: ['C004', 'C144', 'C424']
  Visit 4: ['C224', 'C301', 'C302', 'C304']
  Visit 5: ['C031', 'C101', 'C204', 'C333']
  Visit 6: ['C020', 'C021', 'C024', 'C340']
Synthetic stats (N=1000): {'mean_depth': 1.9973043551512089, 'std_depth': 0.05184957422814923, 'mean_tree_dist': 2.3292252074603566, 'std_tree_dist': 0.7645176731174653, 'mean_root_purity': 0.5949166666666666, 'std_root_purity': 0.20001178784706108}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth2_lrecon1_best4.9791.pt
[Rectified] Epoch   1 | Train 106.015654 | Val 72.578873 | lambda_recon=10
[Rectified] Epoch   2 | Train 68.227357 | Val 50.958015 | lambda_recon=10
[Rectified] Epoch   3 | Train 53.254041 | Val 37.589709 | lambda_recon=10
[Rectified] Epoch   4 | Train 44.264008 | Val 30.242156 | lambda_recon=10
[Rectified] Epoch   5 | Train 38.398370 | Val 24.311053 | lambda_recon=10
[Rectified] Epoch   6 | Train 34.243889 | Val 21.433225 | lambda_recon=10
[Rectified] Epoch   7 | Train 31.005274 | Val 18.222356 | lambda_recon=10
[Rectified] Epoch   8 | Train 27.968292 | Val 15.487239 | lambda_recon=10
[Rectified] Epoch   9 | Train 25.533803 | Val 13.481339 | lambda_recon=10
[Rectified] Epoch  10 | Train 23.599125 | Val 12.560016 | lambda_recon=10
[Rectified] Epoch  11 | Train 22.134561 | Val 11.645760 | lambda_recon=10
[Rectified] Epoch  12 | Train 20.545830 | Val 10.119263 | lambda_recon=10
[Rectified] Epoch  13 | Train 19.562668 | Val 9.876222 | lambda_recon=10
[Rectified] Epoch  14 | Train 18.840728 | Val 9.627436 | lambda_recon=10
[Rectified] Epoch  15 | Train 18.009831 | Val 9.255868 | lambda_recon=10
[Rectified] Epoch  16 | Train 17.420189 | Val 8.985465 | lambda_recon=10
[Rectified] Epoch  17 | Train 16.482437 | Val 8.104139 | lambda_recon=10
[Rectified] Epoch  18 | Train 15.995151 | Val 7.957185 | lambda_recon=10
[Rectified] Epoch  19 | Train 15.613553 | Val 7.869619 | lambda_recon=10
[Rectified] Epoch  20 | Train 14.766253 | Val 6.756423 | lambda_recon=10
[Rectified] Epoch  21 | Train 14.169178 | Val 6.840323 | lambda_recon=10
[Rectified] Epoch  22 | Train 13.722561 | Val 6.602555 | lambda_recon=10
[Rectified] Epoch  23 | Train 13.440895 | Val 6.620983 | lambda_recon=10
[Rectified] Epoch  24 | Train 13.226409 | Val 6.332044 | lambda_recon=10
[Rectified] Epoch  25 | Train 12.922452 | Val 6.235639 | lambda_recon=10
[Rectified] Epoch  26 | Train 12.675726 | Val 6.003517 | lambda_recon=10
[Rectified] Epoch  27 | Train 12.551268 | Val 6.048344 | lambda_recon=10
[Rectified] Epoch  28 | Train 12.235101 | Val 5.877350 | lambda_recon=10
[Rectified] Epoch  29 | Train 12.054465 | Val 5.862024 | lambda_recon=10
[Rectified] Epoch  30 | Train 11.840045 | Val 5.982505 | lambda_recon=10
[Rectified] Epoch  31 | Train 11.718425 | Val 5.897341 | lambda_recon=10
[Rectified] Epoch  32 | Train 11.584952 | Val 5.999276 | lambda_recon=10
[Rectified] Early stopping.
[Summary Rectified] depth=2 | lambda_recon=10 | pretrain_val=0.060362 | best_val=5.862024
Test Recall@4: 0.1300
Tree-Embedding Correlation: 0.8988

[Rectified] Sample trajectory 1:
  Visit 1: ['C213', 'C214', 'C330', 'C411']
  Visit 2: ['C211', 'C212', 'C314', 'C411']
  Visit 3: ['C033', 'C211', 'C213', 'C214']
  Visit 4: ['C212', 'C213', 'C214', 'C310']
  Visit 5: ['C110', 'C111', 'C113', 'C114']
  Visit 6: ['C110', 'C111', 'C112', 'C114']

[Rectified] Sample trajectory 2:
  Visit 1: ['C134', 'C342', 'C402', 'C421']
  Visit 2: ['C033', 'C211', 'C310', 'C311']
  Visit 3: ['C121', 'C123', 'C213', 'C214']
  Visit 4: ['C121', 'C122', 'C340', 'C344']
  Visit 5: ['C100', 'C101', 'C143', 'C240']
  Visit 6: ['C141', 'C230', 'C232', 'C233']

[Rectified] Sample trajectory 3:
  Visit 1: ['C020', 'C021', 'C211', 'C411']
  Visit 2: ['C211', 'C214', 'C310', 'C313']
  Visit 3: ['C103', 'C104', 'C333', 'C334']
  Visit 4: ['C141', 'C233', 'C333', 'C334']
  Visit 5: ['C031', 'C212', 'C213', 'C214']
  Visit 6: ['C232', 'C320', 'C342', 'C402']
Synthetic stats (N=1000): {'mean_depth': 1.9982455389618725, 'std_depth': 0.04184952693392276, 'mean_tree_dist': 2.295000853096741, 'std_tree_dist': 0.7230270031339298, 'mean_root_purity': 0.5939305555555555, 'std_root_purity': 0.18145493469624155}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth2_lrecon10_best5.8620.pt
[Rectified] Epoch   1 | Train 105.134939 | Val 71.854024 | lambda_recon=100
[Rectified] Epoch   2 | Train 67.892935 | Val 50.370697 | lambda_recon=100
[Rectified] Epoch   3 | Train 53.579181 | Val 38.016668 | lambda_recon=100
[Rectified] Epoch   4 | Train 45.046558 | Val 31.210738 | lambda_recon=100
[Rectified] Epoch   5 | Train 39.324661 | Val 25.082749 | lambda_recon=100
[Rectified] Epoch   6 | Train 34.803258 | Val 21.230902 | lambda_recon=100
[Rectified] Epoch   7 | Train 31.696949 | Val 19.027666 | lambda_recon=100
[Rectified] Epoch   8 | Train 29.020236 | Val 16.061780 | lambda_recon=100
[Rectified] Epoch   9 | Train 26.793562 | Val 14.363433 | lambda_recon=100
[Rectified] Epoch  10 | Train 25.128322 | Val 12.630773 | lambda_recon=100
[Rectified] Epoch  11 | Train 23.475993 | Val 12.122998 | lambda_recon=100
[Rectified] Epoch  12 | Train 22.232159 | Val 11.014923 | lambda_recon=100
[Rectified] Epoch  13 | Train 21.024758 | Val 10.724592 | lambda_recon=100
[Rectified] Epoch  14 | Train 19.546018 | Val 10.262363 | lambda_recon=100
[Rectified] Epoch  15 | Train 18.675430 | Val 9.480295 | lambda_recon=100
[Rectified] Epoch  16 | Train 17.994966 | Val 8.995604 | lambda_recon=100
[Rectified] Epoch  17 | Train 17.002939 | Val 8.592660 | lambda_recon=100
[Rectified] Epoch  18 | Train 16.414052 | Val 8.603900 | lambda_recon=100
[Rectified] Epoch  19 | Train 15.739449 | Val 8.365088 | lambda_recon=100
[Rectified] Epoch  20 | Train 15.396297 | Val 8.034916 | lambda_recon=100
[Rectified] Epoch  21 | Train 14.853009 | Val 8.043380 | lambda_recon=100
[Rectified] Epoch  22 | Train 14.239643 | Val 7.587374 | lambda_recon=100
[Rectified] Epoch  23 | Train 13.742453 | Val 7.780583 | lambda_recon=100
[Rectified] Epoch  24 | Train 13.405380 | Val 6.735509 | lambda_recon=100
[Rectified] Epoch  25 | Train 13.164978 | Val 6.716828 | lambda_recon=100
[Rectified] Epoch  26 | Train 12.609491 | Val 6.221212 | lambda_recon=100
[Rectified] Epoch  27 | Train 12.638983 | Val 6.366018 | lambda_recon=100
[Rectified] Epoch  28 | Train 12.067015 | Val 6.205883 | lambda_recon=100
[Rectified] Epoch  29 | Train 12.204999 | Val 6.097410 | lambda_recon=100
[Rectified] Epoch  30 | Train 11.759952 | Val 5.846763 | lambda_recon=100
[Rectified] Epoch  31 | Train 11.650740 | Val 6.037598 | lambda_recon=100
[Rectified] Epoch  32 | Train 11.505714 | Val 5.660475 | lambda_recon=100
[Rectified] Epoch  33 | Train 11.418360 | Val 5.854285 | lambda_recon=100
[Rectified] Epoch  34 | Train 11.267867 | Val 5.827658 | lambda_recon=100
[Rectified] Epoch  35 | Train 11.258859 | Val 5.670661 | lambda_recon=100
[Rectified] Early stopping.
[Summary Rectified] depth=2 | lambda_recon=100 | pretrain_val=0.060362 | best_val=5.660475
Test Recall@4: 0.1306
Tree-Embedding Correlation: 0.8885

[Rectified] Sample trajectory 1:
  Visit 1: ['C033', 'C212', 'C341', 'C343']
  Visit 2: ['C013', 'C130', 'C131']
  Visit 3: ['C112', 'C120', 'C122', 'C332']
  Visit 4: ['C001', 'C002', 'C104', 'C143']
  Visit 5: ['C130', 'C131', 'C143', 'C302']
  Visit 6: ['C020', 'C022', 'C122', 'C332']

[Rectified] Sample trajectory 2:
  Visit 1: ['C022', 'C122', 'C213', 'C214']
  Visit 2: ['C030', 'C031', 'C033', 'C221']
  Visit 3: ['C212', 'C214', 'C222', 'C300']
  Visit 4: ['C143', 'C221', 'C311', 'C400']
  Visit 5: ['C211', 'C212', 'C214', 'C443']
  Visit 6: ['C120', 'C241', 'C320', 'C402']

[Rectified] Sample trajectory 3:
  Visit 1: ['C122', 'C213', 'C214', 'C332']
  Visit 2: ['C211', 'C212', 'C214', 'C444']
  Visit 3: ['C001', 'C102', 'C104', 'C143']
  Visit 4: ['C013', 'C131', 'C302', 'C313']
  Visit 5: ['C211', 'C212', 'C214', 'C300']
  Visit 6: ['C213', 'C214', 'C332', 'C433']
Synthetic stats (N=1000): {'mean_depth': 1.995416316232128, 'std_depth': 0.06754756554449838, 'mean_tree_dist': 2.370142053068882, 'std_tree_dist': 0.8016723716414305, 'mean_root_purity': 0.5770138888888888, 'std_root_purity': 0.18345480771504855}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth2_lrecon100_best5.6605.pt
[Rectified] Epoch   1 | Train 111.456340 | Val 77.956904 | lambda_recon=1000
[Rectified] Epoch   2 | Train 73.990640 | Val 56.474129 | lambda_recon=1000
[Rectified] Epoch   3 | Train 58.982936 | Val 43.613915 | lambda_recon=1000
[Rectified] Epoch   4 | Train 49.694458 | Val 35.089073 | lambda_recon=1000
[Rectified] Epoch   5 | Train 43.633499 | Val 30.432915 | lambda_recon=1000
[Rectified] Epoch   6 | Train 39.602425 | Val 25.961699 | lambda_recon=1000
[Rectified] Epoch   7 | Train 36.510356 | Val 23.590851 | lambda_recon=1000
[Rectified] Epoch   8 | Train 33.951877 | Val 21.486292 | lambda_recon=1000
[Rectified] Epoch   9 | Train 31.934157 | Val 19.244893 | lambda_recon=1000
[Rectified] Epoch  10 | Train 29.601286 | Val 17.375096 | lambda_recon=1000
[Rectified] Epoch  11 | Train 27.775361 | Val 16.058345 | lambda_recon=1000
[Rectified] Epoch  12 | Train 26.601402 | Val 15.531767 | lambda_recon=1000
[Rectified] Epoch  13 | Train 25.166353 | Val 14.302734 | lambda_recon=1000
[Rectified] Epoch  14 | Train 24.273297 | Val 14.650708 | lambda_recon=1000
[Rectified] Epoch  15 | Train 23.659781 | Val 14.864531 | lambda_recon=1000
[Rectified] Epoch  16 | Train 22.912300 | Val 14.092674 | lambda_recon=1000
[Rectified] Epoch  17 | Train 22.333671 | Val 13.723713 | lambda_recon=1000
[Rectified] Epoch  18 | Train 21.830215 | Val 13.389891 | lambda_recon=1000
[Rectified] Epoch  19 | Train 21.268022 | Val 13.124761 | lambda_recon=1000
[Rectified] Epoch  20 | Train 20.396603 | Val 12.420680 | lambda_recon=1000
[Rectified] Epoch  21 | Train 20.051675 | Val 12.030861 | lambda_recon=1000
[Rectified] Epoch  22 | Train 19.471692 | Val 12.103196 | lambda_recon=1000
[Rectified] Epoch  23 | Train 19.003113 | Val 11.919111 | lambda_recon=1000
[Rectified] Epoch  24 | Train 18.754090 | Val 11.867563 | lambda_recon=1000
[Rectified] Epoch  25 | Train 18.677249 | Val 11.499819 | lambda_recon=1000
[Rectified] Epoch  26 | Train 18.406572 | Val 11.496641 | lambda_recon=1000
[Rectified] Epoch  27 | Train 17.939202 | Val 11.450952 | lambda_recon=1000
[Rectified] Epoch  28 | Train 17.774914 | Val 11.086572 | lambda_recon=1000
[Rectified] Epoch  29 | Train 17.504183 | Val 11.186950 | lambda_recon=1000
[Rectified] Epoch  30 | Train 17.475326 | Val 10.870327 | lambda_recon=1000
[Rectified] Epoch  31 | Train 17.037004 | Val 11.173109 | lambda_recon=1000
[Rectified] Epoch  32 | Train 17.094300 | Val 10.793651 | lambda_recon=1000
[Rectified] Epoch  33 | Train 16.845255 | Val 11.002339 | lambda_recon=1000
[Rectified] Epoch  34 | Train 16.781879 | Val 10.687870 | lambda_recon=1000
[Rectified] Epoch  35 | Train 16.514707 | Val 10.941385 | lambda_recon=1000
[Rectified] Epoch  36 | Train 16.444943 | Val 10.980395 | lambda_recon=1000
[Rectified] Epoch  37 | Train 16.404511 | Val 10.796809 | lambda_recon=1000
[Rectified] Early stopping.
[Summary Rectified] depth=2 | lambda_recon=1000 | pretrain_val=0.060362 | best_val=10.687870
Test Recall@4: 0.0836
Tree-Embedding Correlation: 0.8785

[Rectified] Sample trajectory 1:
  Visit 1: ['C230', 'C233', 'C400', 'C402']
  Visit 2: ['C213', 'C214', 'C222', 'C413']
  Visit 3: ['C110', 'C111', 'C112', 'C113']
  Visit 4: ['C002', 'C141', 'C210', 'C221']
  Visit 5: ['C022', 'C122', 'C213', 'C214']
  Visit 6: ['C124', 'C200', 'C213', 'C214']

[Rectified] Sample trajectory 2:
  Visit 1: ['C213', 'C214', 'C310', 'C313']
  Visit 2: ['C140', 'C141', 'C210', 'C221']
  Visit 3: ['C213', 'C214', 'C222', 'C413']
  Visit 4: ['C020', 'C021', 'C022', 'C330']
  Visit 5: ['C201', 'C214', 'C222', 'C413']
  Visit 6: ['C020', 'C021', 'C022', 'C233']

[Rectified] Sample trajectory 3:
  Visit 1: ['C033', 'C230', 'C232', 'C234']
  Visit 2: ['C021', 'C230', 'C233', 'C334']
  Visit 3: ['C111', 'C114', 'C230', 'C232']
  Visit 4: ['C002', 'C141', 'C221', 'C230']
  Visit 5: ['C020', 'C023', 'C111', 'C233']
  Visit 6: ['C141', 'C221', 'C230', 'C232']
Synthetic stats (N=1000): {'mean_depth': 1.99890996293874, 'std_depth': 0.03299771023063817, 'mean_tree_dist': 2.4866115377042726, 'std_tree_dist': 0.8670012664441326, 'mean_root_purity': 0.5854444444444444, 'std_root_purity': 0.15420851283926829}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth2_lrecon1000_best10.6879.pt
[Rectified] Epoch   1 | Train 119.467516 | Val 83.588692 | lambda_recon=2000
[Rectified] Epoch   2 | Train 79.074479 | Val 61.209655 | lambda_recon=2000
[Rectified] Epoch   3 | Train 64.608518 | Val 48.956384 | lambda_recon=2000
[Rectified] Epoch   4 | Train 55.996582 | Val 41.616541 | lambda_recon=2000
[Rectified] Epoch   5 | Train 50.176938 | Val 36.228749 | lambda_recon=2000
[Rectified] Epoch   6 | Train 45.810540 | Val 32.669140 | lambda_recon=2000
[Rectified] Epoch   7 | Train 42.656404 | Val 29.321177 | lambda_recon=2000
[Rectified] Epoch   8 | Train 39.399324 | Val 27.314326 | lambda_recon=2000
[Rectified] Epoch   9 | Train 37.418049 | Val 25.706142 | lambda_recon=2000
[Rectified] Epoch  10 | Train 35.705847 | Val 24.526785 | lambda_recon=2000
[Rectified] Epoch  11 | Train 34.152134 | Val 23.770519 | lambda_recon=2000
[Rectified] Epoch  12 | Train 32.535572 | Val 22.363471 | lambda_recon=2000
[Rectified] Epoch  13 | Train 31.220632 | Val 21.402547 | lambda_recon=2000
[Rectified] Epoch  14 | Train 30.259618 | Val 21.219905 | lambda_recon=2000
[Rectified] Epoch  15 | Train 29.368520 | Val 20.662997 | lambda_recon=2000
[Rectified] Epoch  16 | Train 28.376429 | Val 20.114937 | lambda_recon=2000
[Rectified] Epoch  17 | Train 27.729612 | Val 20.179069 | lambda_recon=2000
[Rectified] Epoch  18 | Train 27.178298 | Val 19.504821 | lambda_recon=2000
[Rectified] Epoch  19 | Train 26.563641 | Val 19.394920 | lambda_recon=2000
[Rectified] Epoch  20 | Train 26.095244 | Val 19.129777 | lambda_recon=2000
[Rectified] Epoch  21 | Train 25.881195 | Val 18.924791 | lambda_recon=2000
[Rectified] Epoch  22 | Train 25.671297 | Val 18.759922 | lambda_recon=2000
[Rectified] Epoch  23 | Train 25.238816 | Val 18.282922 | lambda_recon=2000
[Rectified] Epoch  24 | Train 24.847136 | Val 18.307020 | lambda_recon=2000
[Rectified] Epoch  25 | Train 24.709298 | Val 18.764985 | lambda_recon=2000
[Rectified] Epoch  26 | Train 24.322594 | Val 18.279206 | lambda_recon=2000
[Rectified] Epoch  27 | Train 24.112200 | Val 18.215680 | lambda_recon=2000
[Rectified] Epoch  28 | Train 23.775210 | Val 18.001935 | lambda_recon=2000
[Rectified] Epoch  29 | Train 23.702071 | Val 18.118487 | lambda_recon=2000
[Rectified] Epoch  30 | Train 23.628033 | Val 17.719934 | lambda_recon=2000
[Rectified] Epoch  31 | Train 23.486300 | Val 18.082549 | lambda_recon=2000
[Rectified] Epoch  32 | Train 23.478439 | Val 17.825262 | lambda_recon=2000
[Rectified] Epoch  33 | Train 23.258051 | Val 17.930179 | lambda_recon=2000
[Rectified] Early stopping.
[Summary Rectified] depth=2 | lambda_recon=2000 | pretrain_val=0.060362 | best_val=17.719934
Test Recall@4: 0.1186
Tree-Embedding Correlation: 0.8779

[Rectified] Sample trajectory 1:
  Visit 1: ['C141', 'C143', 'C420', 'C424']
  Visit 2: ['C213', 'C214', 'C340', 'C343']
  Visit 3: ['C141', 'C143', 'C331', 'C333']
  Visit 4: ['C030', 'C214', 'C323', 'C443']
  Visit 5: ['C141', 'C143', 'C333']
  Visit 6: ['C143', 'C230', 'C232']

[Rectified] Sample trajectory 2:
  Visit 1: ['C002', 'C141', 'C143', 'C333']
  Visit 2: ['C000', 'C001', 'C002', 'C104']
  Visit 3: ['C213', 'C214', 'C441', 'C444']
  Visit 4: ['C004', 'C321', 'C322', 'C323']
  Visit 5: ['C211', 'C213', 'C214', 'C444']
  Visit 6: ['C121', 'C122', 'C124', 'C214']

[Rectified] Sample trajectory 3:
  Visit 1: ['C213', 'C214', 'C443', 'C444']
  Visit 2: ['C130', 'C323', 'C441', 'C443']
  Visit 3: ['C410', 'C411', 'C412', 'C413']
  Visit 4: ['C230', 'C232', 'C322']
  Visit 5: ['C210', 'C242', 'C411', 'C412']
  Visit 6: ['C141', 'C143', 'C333']
Synthetic stats (N=1000): {'mean_depth': 1.9984860936891735, 'std_depth': 0.03887948557412549, 'mean_tree_dist': 2.2846211074503135, 'std_tree_dist': 0.7038893116547202, 'mean_root_purity': 0.566625, 'std_root_purity': 0.16105769648552107}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth2_lrecon2000_best17.7199.pt

rectified_depth7 | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

=== Pretraining hyperbolic graph embeddings (Rectified) ===
[Pretrain] Epoch   1 | train=0.096093 | val=0.089530 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   2 | train=0.090683 | val=0.083832 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   3 | train=0.086023 | val=0.085344 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   4 | train=0.084676 | val=0.083375 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   5 | train=0.083323 | val=0.081763 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   6 | train=0.083870 | val=0.082698 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   7 | train=0.082230 | val=0.080247 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   8 | train=0.080632 | val=0.077968 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   9 | train=0.082691 | val=0.078693 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  10 | train=0.078577 | val=0.077227 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  11 | train=0.078493 | val=0.079079 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  12 | train=0.078594 | val=0.078814 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  13 | train=0.077080 | val=0.078625 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  14 | train=0.077022 | val=0.075915 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  15 | train=0.078044 | val=0.074687 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  16 | train=0.075129 | val=0.079277 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  17 | train=0.076354 | val=0.075094 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  18 | train=0.073401 | val=0.076079 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  19 | train=0.076065 | val=0.073450 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  20 | train=0.074343 | val=0.074664 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  21 | train=0.075458 | val=0.073327 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  22 | train=0.073706 | val=0.073420 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  23 | train=0.074487 | val=0.072481 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  24 | train=0.072677 | val=0.073242 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  25 | train=0.073061 | val=0.074697 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  26 | train=0.073046 | val=0.074316 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  27 | train=0.073107 | val=0.071676 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  28 | train=0.071820 | val=0.071607 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  29 | train=0.071310 | val=0.071620 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  30 | train=0.071501 | val=0.071130 | rad=0.003 pair=0.01 hdd=0.02
Saved pretraining checkpoint to results/checkpoints/hyperbolic_rectified_pretrain_rad0.003_pair0.01_val0.0711.pt
[Rectified] Epoch   1 | Train 105.167837 | Val 71.095020 | lambda_recon=1
[Rectified] Epoch   2 | Train 66.885295 | Val 49.100818 | lambda_recon=1
[Rectified] Epoch   3 | Train 52.099410 | Val 36.661546 | lambda_recon=1
[Rectified] Epoch   4 | Train 43.311164 | Val 28.912230 | lambda_recon=1
[Rectified] Epoch   5 | Train 37.771009 | Val 24.175712 | lambda_recon=1
[Rectified] Epoch   6 | Train 33.380089 | Val 20.077923 | lambda_recon=1
[Rectified] Epoch   7 | Train 30.037006 | Val 17.525043 | lambda_recon=1
[Rectified] Epoch   8 | Train 27.217708 | Val 14.717440 | lambda_recon=1
[Rectified] Epoch   9 | Train 25.261164 | Val 13.572108 | lambda_recon=1
[Rectified] Epoch  10 | Train 23.547461 | Val 12.598760 | lambda_recon=1
[Rectified] Epoch  11 | Train 22.282119 | Val 11.632511 | lambda_recon=1
[Rectified] Epoch  12 | Train 20.753615 | Val 10.353028 | lambda_recon=1
[Rectified] Epoch  13 | Train 19.462005 | Val 9.463589 | lambda_recon=1
[Rectified] Epoch  14 | Train 18.364572 | Val 8.755782 | lambda_recon=1
[Rectified] Epoch  15 | Train 17.570545 | Val 8.353191 | lambda_recon=1
[Rectified] Epoch  16 | Train 16.795891 | Val 7.944429 | lambda_recon=1
[Rectified] Epoch  17 | Train 15.815396 | Val 7.566419 | lambda_recon=1
[Rectified] Epoch  18 | Train 15.279157 | Val 7.515429 | lambda_recon=1
[Rectified] Epoch  19 | Train 14.687685 | Val 7.117675 | lambda_recon=1
[Rectified] Epoch  20 | Train 14.426411 | Val 7.056161 | lambda_recon=1
[Rectified] Epoch  21 | Train 13.969402 | Val 6.526279 | lambda_recon=1
[Rectified] Epoch  22 | Train 13.636591 | Val 6.504399 | lambda_recon=1
[Rectified] Epoch  23 | Train 13.401023 | Val 6.588310 | lambda_recon=1
[Rectified] Epoch  24 | Train 13.009538 | Val 6.538959 | lambda_recon=1
[Rectified] Epoch  25 | Train 12.648407 | Val 6.598312 | lambda_recon=1
[Rectified] Early stopping.
[Summary Rectified] depth=7 | lambda_recon=1 | pretrain_val=0.071130 | best_val=6.504399
Test Recall@4: 0.0124
Tree-Embedding Correlation: 0.7743

[Rectified] Sample trajectory 1:
  Visit 1: ['C130d3', 'C134d3', 'C440d3', 'C444d3']
  Visit 2: ['C030d4', 'C140d3', 'C413d4', 'C440d2']
  Visit 3: ['C044d4', 'C134d3', 'C434d4', 'C440d3']
  Visit 4: ['C133d4', 'C233d3', 'C300d4', 'C331d4']
  Visit 5: ['C040d4', 'C134d3', 'C322d4', 'C440d3']
  Visit 6: ['C211d3', 'C424d4', 'C440d3']

[Rectified] Sample trajectory 2:
  Visit 1: ['C023d4', 'C220d2', 'C344d4', 'C402d3']
  Visit 2: ['C040d4', 'C044d4', 'C134d3', 'C440d3']
  Visit 3: ['C014d3', 'C044d4', 'C440d3', 'C444d3']
  Visit 4: ['C133d4', 'C233d3', 'C300d4', 'C344d4']
  Visit 5: ['C004d4', 'C044d4', 'C122d2', 'C213d4']
  Visit 6: ['C213d3', 'C223d2', 'C320d2', 'C440d2']

[Rectified] Sample trajectory 3:
  Visit 1: ['C004d4', 'C040d4', 'C044d4', 'C440d3']
  Visit 2: ['C220d2', 'C322d4', 'C420d3', 'C421d4']
  Visit 3: ['C223d2', 'C224d3', 'C323d4', 'C444d3']
  Visit 4: ['C213d3', 'C220d2', 'C324d2', 'C440d2']
  Visit 5: ['C023d4', 'C220d2', 'C344d4', 'C402d3']
  Visit 6: ['C213d3', 'C413d4', 'C440d2', 'C440d3']
Synthetic stats (N=1000): {'mean_depth': 6.155641532444594, 'std_depth': 0.7927234013418895, 'mean_tree_dist': 11.397131552917903, 'std_tree_dist': 2.2124363778834693, 'mean_root_purity': 0.46501388888888884, 'std_root_purity': 0.13154811748750408}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth7_lrecon1_best6.5044.pt
[Rectified] Epoch   1 | Train 105.337963 | Val 71.914505 | lambda_recon=10
[Rectified] Epoch   2 | Train 68.464694 | Val 50.711131 | lambda_recon=10
[Rectified] Epoch   3 | Train 53.661141 | Val 37.880091 | lambda_recon=10
[Rectified] Epoch   4 | Train 44.234715 | Val 30.106812 | lambda_recon=10
[Rectified] Epoch   5 | Train 38.274891 | Val 24.744239 | lambda_recon=10
[Rectified] Epoch   6 | Train 34.183983 | Val 20.971124 | lambda_recon=10
[Rectified] Epoch   7 | Train 30.868281 | Val 18.066221 | lambda_recon=10
[Rectified] Epoch   8 | Train 27.936635 | Val 15.287214 | lambda_recon=10
[Rectified] Epoch   9 | Train 25.939523 | Val 14.031025 | lambda_recon=10
[Rectified] Epoch  10 | Train 24.413548 | Val 12.149229 | lambda_recon=10
[Rectified] Epoch  11 | Train 22.715491 | Val 11.219609 | lambda_recon=10
[Rectified] Epoch  12 | Train 21.579944 | Val 10.816326 | lambda_recon=10
[Rectified] Epoch  13 | Train 20.732470 | Val 10.882959 | lambda_recon=10
[Rectified] Epoch  14 | Train 19.583528 | Val 9.659934 | lambda_recon=10
[Rectified] Epoch  15 | Train 18.247936 | Val 9.421536 | lambda_recon=10
[Rectified] Epoch  16 | Train 17.584170 | Val 9.127146 | lambda_recon=10
[Rectified] Epoch  17 | Train 16.882802 | Val 9.022928 | lambda_recon=10
[Rectified] Epoch  18 | Train 16.072053 | Val 8.357067 | lambda_recon=10
[Rectified] Epoch  19 | Train 15.287074 | Val 7.715872 | lambda_recon=10
[Rectified] Epoch  20 | Train 14.623259 | Val 7.639099 | lambda_recon=10
[Rectified] Epoch  21 | Train 14.220099 | Val 7.223292 | lambda_recon=10
[Rectified] Epoch  22 | Train 13.710331 | Val 6.575020 | lambda_recon=10
[Rectified] Epoch  23 | Train 13.200943 | Val 6.469057 | lambda_recon=10
[Rectified] Epoch  24 | Train 12.893333 | Val 6.330051 | lambda_recon=10
[Rectified] Epoch  25 | Train 12.570322 | Val 6.488499 | lambda_recon=10
[Rectified] Epoch  26 | Train 12.321040 | Val 5.792259 | lambda_recon=10
[Rectified] Epoch  27 | Train 12.047728 | Val 6.022312 | lambda_recon=10
[Rectified] Epoch  28 | Train 11.938647 | Val 6.130314 | lambda_recon=10
[Rectified] Epoch  29 | Train 11.865742 | Val 5.701834 | lambda_recon=10
[Rectified] Epoch  30 | Train 11.678612 | Val 5.943431 | lambda_recon=10
[Rectified] Epoch  31 | Train 11.338083 | Val 5.742870 | lambda_recon=10
[Rectified] Epoch  32 | Train 11.301121 | Val 5.533176 | lambda_recon=10
[Rectified] Epoch  33 | Train 11.081094 | Val 5.724255 | lambda_recon=10
[Rectified] Epoch  34 | Train 11.074797 | Val 6.279044 | lambda_recon=10
[Rectified] Epoch  35 | Train 10.844461 | Val 5.752343 | lambda_recon=10
[Rectified] Early stopping.
[Summary Rectified] depth=7 | lambda_recon=10 | pretrain_val=0.071130 | best_val=5.533176
Test Recall@4: 0.0482
Tree-Embedding Correlation: 0.7444

[Rectified] Sample trajectory 1:
  Visit 1: ['C030d4', 'C034d3', 'C043d4', 'C322d4']
  Visit 2: ['C231d3', 'C233d3', 'C434d3']
  Visit 3: ['C012d3', 'C140d4', 'C310d4']
  Visit 4: ['C030d3', 'C041d4', 'C332d3', 'C332d4']
  Visit 5: ['C030d4', 'C034d3', 'C103d3', 'C244d4']
  Visit 6: ['C030d4', 'C034d3', 'C043d4', 'C322d4']

[Rectified] Sample trajectory 2:
  Visit 1: ['C044d3', 'C114d4', 'C323d4', 'C344d4']
  Visit 2: ['C032d3', 'C114d4', 'C233d3', 'C323d4']
  Visit 3: ['C003d2', 'C034d3', 'C111d3', 'C322d4']
  Visit 4: ['C032d3', 'C231d3', 'C233d3', 'C434d3']
  Visit 5: ['C012d2', 'C221d4', 'C300d3', 'C322d4']
  Visit 6: ['C013d3', 'C343d4', 'C433d3']

[Rectified] Sample trajectory 3:
  Visit 1: ['C032d3', 'C114d4', 'C233d3', 'C323d4']
  Visit 2: ['C003d2', 'C030d4', 'C111d3', 'C440d2']
  Visit 3: ['C030d4', 'C034d3', 'C210d4', 'C221d4']
  Visit 4: ['C140d4', 'C203d4', 'C341d4', 'C423d4']
  Visit 5: ['C102d4', 'C341d4', 'C401d1', 'C444d2']
  Visit 6: ['C041d4', 'C111d3', 'C144d4', 'C302d4']
Synthetic stats (N=1000): {'mean_depth': 6.50110662674131, 'std_depth': 0.655685339454897, 'mean_tree_dist': 12.53534563653425, 'std_tree_dist': 1.916888700631137, 'mean_root_purity': 0.48966666666666664, 'std_root_purity': 0.15698159835541942}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth7_lrecon10_best5.5332.pt
[Rectified] Epoch   1 | Train 105.300690 | Val 72.542655 | lambda_recon=100
[Rectified] Epoch   2 | Train 67.974087 | Val 50.323639 | lambda_recon=100
[Rectified] Epoch   3 | Train 53.184450 | Val 37.688522 | lambda_recon=100
[Rectified] Epoch   4 | Train 44.098573 | Val 29.589799 | lambda_recon=100
[Rectified] Epoch   5 | Train 38.018526 | Val 24.785683 | lambda_recon=100
[Rectified] Epoch   6 | Train 33.531105 | Val 20.077841 | lambda_recon=100
[Rectified] Epoch   7 | Train 30.518101 | Val 18.089203 | lambda_recon=100
[Rectified] Epoch   8 | Train 27.923918 | Val 15.578127 | lambda_recon=100
[Rectified] Epoch   9 | Train 25.633274 | Val 14.405238 | lambda_recon=100
[Rectified] Epoch  10 | Train 23.949985 | Val 13.160812 | lambda_recon=100
[Rectified] Epoch  11 | Train 22.295008 | Val 11.639923 | lambda_recon=100
[Rectified] Epoch  12 | Train 21.097359 | Val 11.042129 | lambda_recon=100
[Rectified] Epoch  13 | Train 19.895710 | Val 10.673351 | lambda_recon=100
[Rectified] Epoch  14 | Train 18.807789 | Val 9.349986 | lambda_recon=100
[Rectified] Epoch  15 | Train 17.887455 | Val 8.648290 | lambda_recon=100
[Rectified] Epoch  16 | Train 16.562928 | Val 8.032764 | lambda_recon=100
[Rectified] Epoch  17 | Train 15.839547 | Val 7.028173 | lambda_recon=100
[Rectified] Epoch  18 | Train 15.025742 | Val 7.009807 | lambda_recon=100
[Rectified] Epoch  19 | Train 14.645602 | Val 6.497141 | lambda_recon=100
[Rectified] Epoch  20 | Train 14.218096 | Val 6.626715 | lambda_recon=100
[Rectified] Epoch  21 | Train 13.625016 | Val 6.610401 | lambda_recon=100
[Rectified] Epoch  22 | Train 13.359888 | Val 6.413035 | lambda_recon=100
[Rectified] Epoch  23 | Train 12.998953 | Val 6.275693 | lambda_recon=100
[Rectified] Epoch  24 | Train 12.723390 | Val 6.529189 | lambda_recon=100
[Rectified] Epoch  25 | Train 12.438323 | Val 6.340118 | lambda_recon=100
[Rectified] Epoch  26 | Train 12.183079 | Val 5.700214 | lambda_recon=100
[Rectified] Epoch  27 | Train 11.942278 | Val 5.930077 | lambda_recon=100
[Rectified] Epoch  28 | Train 11.825062 | Val 6.072892 | lambda_recon=100
[Rectified] Epoch  29 | Train 11.652828 | Val 5.979662 | lambda_recon=100
[Rectified] Early stopping.
[Summary Rectified] depth=7 | lambda_recon=100 | pretrain_val=0.071130 | best_val=5.700214
Test Recall@4: 0.0414
Tree-Embedding Correlation: 0.7869

[Rectified] Sample trajectory 1:
  Visit 1: ['C030d4', 'C111d4', 'C210d4', 'C440d2']
  Visit 2: ['C023d4', 'C030d3', 'C101d4', 'C400d3']
  Visit 3: ['C002d4', 'C014d4', 'C204d3', 'C440d2']
  Visit 4: ['C034d3', 'C121d4', 'C224d4', 'C322d4']
  Visit 5: ['C224d3', 'C240d4', 'C343d4', 'C403d3']
  Visit 6: ['C034d3', 'C111d4', 'C300d3', 'C441d4']

[Rectified] Sample trajectory 2:
  Visit 1: ['C021d4', 'C121d4', 'C140d4', 'C442d3']
  Visit 2: ['C000d4', 'C043d4', 'C144d4', 'C300d3']
  Visit 3: ['C030d2', 'C323d4', 'C343d4', 'C404d4']
  Visit 4: ['C111d4', 'C242d4', 'C440d2', 'C441d4']
  Visit 5: ['C021d4', 'C131d4', 'C242d3', 'C312d4']
  Visit 6: ['C034d3', 'C111d4', 'C300d3', 'C441d4']

[Rectified] Sample trajectory 3:
  Visit 1: ['C231d3', 'C311d4', 'C323d4', 'C343d4']
  Visit 2: ['C030d4', 'C111d4', 'C301d3', 'C440d2']
  Visit 3: ['C323d4', 'C403d3', 'C433d3', 'C433d4']
  Visit 4: ['C000d4', 'C030d4', 'C111d4', 'C440d2']
  Visit 5: ['C021d4', 'C030d3', 'C101d4', 'C312d4']
  Visit 6: ['C233d3', 'C314d4', 'C323d4', 'C343d4']
Synthetic stats (N=1000): {'mean_depth': 6.508627566228641, 'std_depth': 0.683604450020868, 'mean_tree_dist': 12.483581584292486, 'std_tree_dist': 2.2039467050773536, 'mean_root_purity': 0.45798611111111115, 'std_root_purity': 0.1452245062463845}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth7_lrecon100_best5.7002.pt
[Rectified] Epoch   1 | Train 106.470755 | Val 73.512900 | lambda_recon=1000
[Rectified] Epoch   2 | Train 69.647805 | Val 52.359393 | lambda_recon=1000
[Rectified] Epoch   3 | Train 55.038103 | Val 39.705553 | lambda_recon=1000
[Rectified] Epoch   4 | Train 46.291760 | Val 31.871761 | lambda_recon=1000
[Rectified] Epoch   5 | Train 40.572041 | Val 26.840851 | lambda_recon=1000
[Rectified] Epoch   6 | Train 36.108645 | Val 22.577048 | lambda_recon=1000
[Rectified] Epoch   7 | Train 32.765034 | Val 19.668824 | lambda_recon=1000
[Rectified] Epoch   8 | Train 29.870328 | Val 16.832535 | lambda_recon=1000
[Rectified] Epoch   9 | Train 26.934179 | Val 14.767931 | lambda_recon=1000
[Rectified] Epoch  10 | Train 25.027292 | Val 13.755972 | lambda_recon=1000
[Rectified] Epoch  11 | Train 23.454002 | Val 12.215410 | lambda_recon=1000
[Rectified] Epoch  12 | Train 21.671100 | Val 11.248295 | lambda_recon=1000
[Rectified] Epoch  13 | Train 20.395647 | Val 10.497603 | lambda_recon=1000
[Rectified] Epoch  14 | Train 19.197049 | Val 9.736184 | lambda_recon=1000
[Rectified] Epoch  15 | Train 18.230679 | Val 8.532179 | lambda_recon=1000
[Rectified] Epoch  16 | Train 17.557606 | Val 9.350929 | lambda_recon=1000
[Rectified] Epoch  17 | Train 16.986886 | Val 9.080818 | lambda_recon=1000
[Rectified] Epoch  18 | Train 16.440900 | Val 8.468695 | lambda_recon=1000
[Rectified] Epoch  19 | Train 16.015531 | Val 8.600022 | lambda_recon=1000
[Rectified] Epoch  20 | Train 15.777937 | Val 7.997411 | lambda_recon=1000
[Rectified] Epoch  21 | Train 15.376532 | Val 7.472453 | lambda_recon=1000
[Rectified] Epoch  22 | Train 14.797477 | Val 7.927120 | lambda_recon=1000
[Rectified] Epoch  23 | Train 14.663745 | Val 7.898843 | lambda_recon=1000
[Rectified] Epoch  24 | Train 14.540552 | Val 7.719712 | lambda_recon=1000
[Rectified] Early stopping.
[Summary Rectified] depth=7 | lambda_recon=1000 | pretrain_val=0.071130 | best_val=7.472453
Test Recall@4: 0.0443
Tree-Embedding Correlation: 0.7690

[Rectified] Sample trajectory 1:
  Visit 1: ['C012d3', 'C032d3', 'C131d3', 'C233d3']
  Visit 2: ['C014d4', 'C030d4', 'C113d3', 'C340d4']
  Visit 3: ['C010d4', 'C032d3', 'C314d4', 'C343d4']
  Visit 4: ['C124d3', 'C213d3', 'C311d4', 'C341d4']
  Visit 5: ['C032d3', 'C044d3', 'C131d3', 'C323d4']
  Visit 6: ['C032d3', 'C131d4', 'C211d3', 'C233d3']

[Rectified] Sample trajectory 2:
  Visit 1: ['C113d3', 'C114d4', 'C400d3', 'C402d4']
  Visit 2: ['C013d4', 'C143d3', 'C144d4', 'C420d4']
  Visit 3: ['C012d4', 'C032d3', 'C231d3', 'C343d4']
  Visit 4: ['C034d3', 'C100d3', 'C421d2', 'C424d3']
  Visit 5: ['C010d4', 'C134d2', 'C343d4', 'C402d4']
  Visit 6: ['C044d3', 'C131d3', 'C301d4', 'C440d3']

[Rectified] Sample trajectory 3:
  Visit 1: ['C010d4', 'C112d4', 'C114d2', 'C134d2']
  Visit 2: ['C023d3', 'C032d3', 'C423d4', 'C432d4']
  Visit 3: ['C030d4', 'C124d3', 'C144d4', 'C430d3']
  Visit 4: ['C030d4', 'C124d3', 'C144d4', 'C340d4']
  Visit 5: ['C032d3', 'C233d3', 'C323d4', 'C434d3']
  Visit 6: ['C014d4', 'C030d4', 'C144d4', 'C210d4']
Synthetic stats (N=1000): {'mean_depth': 6.443621434013375, 'std_depth': 0.6614707424632841, 'mean_tree_dist': 12.162623143655345, 'std_tree_dist': 2.0415525690248404, 'mean_root_purity': 0.4810694444444445, 'std_root_purity': 0.1547824482574825}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth7_lrecon1000_best7.4725.pt
[Rectified] Epoch   1 | Train 108.165439 | Val 74.937837 | lambda_recon=2000
[Rectified] Epoch   2 | Train 71.155771 | Val 53.667351 | lambda_recon=2000
[Rectified] Epoch   3 | Train 57.033117 | Val 41.627535 | lambda_recon=2000
[Rectified] Epoch   4 | Train 48.236541 | Val 34.731324 | lambda_recon=2000
[Rectified] Epoch   5 | Train 42.447166 | Val 28.544918 | lambda_recon=2000
[Rectified] Epoch   6 | Train 37.751852 | Val 24.363124 | lambda_recon=2000
[Rectified] Epoch   7 | Train 34.506614 | Val 21.761572 | lambda_recon=2000
[Rectified] Epoch   8 | Train 31.551624 | Val 18.729470 | lambda_recon=2000
[Rectified] Epoch   9 | Train 29.462706 | Val 16.940076 | lambda_recon=2000
[Rectified] Epoch  10 | Train 27.617072 | Val 15.941770 | lambda_recon=2000
[Rectified] Epoch  11 | Train 26.135141 | Val 14.412528 | lambda_recon=2000
[Rectified] Epoch  12 | Train 24.345150 | Val 13.327272 | lambda_recon=2000
[Rectified] Epoch  13 | Train 23.018472 | Val 12.993403 | lambda_recon=2000
[Rectified] Epoch  14 | Train 21.972668 | Val 12.793567 | lambda_recon=2000
[Rectified] Epoch  15 | Train 21.160260 | Val 11.781109 | lambda_recon=2000
[Rectified] Epoch  16 | Train 19.863326 | Val 11.013278 | lambda_recon=2000
[Rectified] Epoch  17 | Train 19.196933 | Val 11.368900 | lambda_recon=2000
[Rectified] Epoch  18 | Train 18.703255 | Val 10.529555 | lambda_recon=2000
[Rectified] Epoch  19 | Train 18.123515 | Val 10.677900 | lambda_recon=2000
[Rectified] Epoch  20 | Train 17.801402 | Val 10.153805 | lambda_recon=2000
[Rectified] Epoch  21 | Train 17.403956 | Val 10.149036 | lambda_recon=2000
[Rectified] Epoch  22 | Train 17.020084 | Val 10.197034 | lambda_recon=2000
[Rectified] Epoch  23 | Train 16.895691 | Val 9.653163 | lambda_recon=2000
[Rectified] Epoch  24 | Train 16.373478 | Val 9.457251 | lambda_recon=2000
[Rectified] Epoch  25 | Train 16.291252 | Val 9.719806 | lambda_recon=2000
[Rectified] Epoch  26 | Train 16.134305 | Val 9.690361 | lambda_recon=2000
[Rectified] Epoch  27 | Train 15.808881 | Val 9.705969 | lambda_recon=2000
[Rectified] Early stopping.
[Summary Rectified] depth=7 | lambda_recon=2000 | pretrain_val=0.071130 | best_val=9.457251
Test Recall@4: 0.0275
Tree-Embedding Correlation: 0.7755

[Rectified] Sample trajectory 1:
  Visit 1: ['C041d4', 'C130d4', 'C332d4', 'C412d4']
  Visit 2: ['C114d3', 'C231d3', 'C241d3', 'C411d3']
  Visit 3: ['C023d3', 'C113d3', 'C124d3', 'C300d3']
  Visit 4: ['C001d4', 'C014d4', 'C030d4', 'C111d4']
  Visit 5: ['C032d2', 'C114d4', 'C122d4', 'C131d4']
  Visit 6: ['C032d2', 'C114d4', 'C403d3', 'C433d3']

[Rectified] Sample trajectory 2:
  Visit 1: ['C020d3', 'C023d3', 'C231d3', 'C301d3']
  Visit 2: ['C113d3', 'C401d4', 'C411d4', 'C434d4']
  Visit 3: ['C012d3', 'C024d3', 'C312d4', 'C433d3']
  Visit 4: ['C023d3', 'C113d3', 'C124d3', 'C300d3']
  Visit 5: ['C131d4', 'C231d3', 'C311d4', 'C402d4']
  Visit 6: ['C001d3', 'C041d4', 'C100d3', 'C122d3']

[Rectified] Sample trajectory 3:
  Visit 1: ['C032d2', 'C114d4', 'C122d4', 'C343d4']
  Visit 2: ['C032d2', 'C131d4', 'C343d4', 'C433d3']
  Visit 3: ['C312d4', 'C321d4', 'C422d3', 'C440d3']
  Visit 4: ['C001d4', 'C041d4', 'C112d1', 'C344d3']
  Visit 5: ['C020d4', 'C343d4', 'C344d4', 'C401d1']
  Visit 6: ['C114d4', 'C122d4', 'C323d4', 'C343d4']
Synthetic stats (N=1000): {'mean_depth': 6.534708018533205, 'std_depth': 0.6643093682184736, 'mean_tree_dist': 12.75095680699836, 'std_tree_dist': 2.2401672428374106, 'mean_root_purity': 0.4950694444444444, 'std_root_purity': 0.13855235269286623}
Saved rectified model checkpoint to results/checkpoints/graph_rectified1_depth7_lrecon2000_best9.4573.pt
