Using device: mps

hyperbolic_graph_ddpm_depth2 | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Pretraining hyperbolic graph embeddings (DDPM) ===
[Pretrain-DDPM] Epoch   1 | train=0.084093 | val=0.078877 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   2 | train=0.079546 | val=0.074398 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   3 | train=0.073537 | val=0.076239 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   4 | train=0.072836 | val=0.072066 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   5 | train=0.075046 | val=0.073842 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   6 | train=0.074024 | val=0.070136 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   7 | train=0.071365 | val=0.070757 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   8 | train=0.071310 | val=0.069640 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   9 | train=0.070457 | val=0.070048 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  10 | train=0.069082 | val=0.068836 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  11 | train=0.067958 | val=0.068881 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  12 | train=0.068186 | val=0.066937 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  13 | train=0.067376 | val=0.066519 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  14 | train=0.068133 | val=0.066853 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  15 | train=0.066365 | val=0.066997 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  16 | train=0.065443 | val=0.064393 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  17 | train=0.066035 | val=0.065164 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  18 | train=0.064132 | val=0.063383 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  19 | train=0.064113 | val=0.063474 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  20 | train=0.063599 | val=0.063059 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  21 | train=0.064106 | val=0.063653 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  22 | train=0.062403 | val=0.062692 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  23 | train=0.062826 | val=0.062377 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  24 | train=0.062214 | val=0.061855 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  25 | train=0.062276 | val=0.061540 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  26 | train=0.062021 | val=0.061072 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  27 | train=0.061561 | val=0.061713 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  28 | train=0.060566 | val=0.061577 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  29 | train=0.061031 | val=0.060362 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  30 | train=0.060343 | val=0.061162 | rad=0.003 pair=0.01 hdd=0.02
Saved pretraining checkpoint to results/checkpoints/hyperbolic_ddpm_pretrain_rad0.003_pair0.01_hdd0.02_val0.0604.pt

Training hyperbolic Graph DDPM | depth=2 | lambda_recon=1
[DDPM] Epoch   1 | Train 128.417133 | Val 109.631373 | lambda_recon=1
[DDPM] Epoch   2 | Train 84.537871 | Val 60.236247 | lambda_recon=1
[DDPM] Epoch   3 | Train 57.997521 | Val 42.435056 | lambda_recon=1
[DDPM] Epoch   4 | Train 45.428334 | Val 32.273454 | lambda_recon=1
[DDPM] Epoch   5 | Train 37.446061 | Val 26.405640 | lambda_recon=1
[DDPM] Epoch   6 | Train 32.046448 | Val 21.752204 | lambda_recon=1
[DDPM] Epoch   7 | Train 27.450419 | Val 18.181297 | lambda_recon=1
[DDPM] Epoch   8 | Train 23.961162 | Val 16.042857 | lambda_recon=1
[DDPM] Epoch   9 | Train 21.257976 | Val 14.126526 | lambda_recon=1
[DDPM] Epoch  10 | Train 19.145118 | Val 12.944209 | lambda_recon=1
[DDPM] Epoch  11 | Train 17.489379 | Val 11.615315 | lambda_recon=1
[DDPM] Epoch  12 | Train 15.995757 | Val 10.109221 | lambda_recon=1
[DDPM] Epoch  13 | Train 14.591566 | Val 9.147835 | lambda_recon=1
[DDPM] Epoch  14 | Train 13.419169 | Val 8.866928 | lambda_recon=1
[DDPM] Epoch  15 | Train 12.874738 | Val 8.213317 | lambda_recon=1
[DDPM] Epoch  16 | Train 11.553654 | Val 7.474783 | lambda_recon=1
[DDPM] Epoch  17 | Train 10.432121 | Val 7.138080 | lambda_recon=1
[DDPM] Epoch  18 | Train 9.956242 | Val 6.543865 | lambda_recon=1
[DDPM] Epoch  19 | Train 9.327968 | Val 6.574953 | lambda_recon=1
[DDPM] Epoch  20 | Train 8.952101 | Val 6.017322 | lambda_recon=1
[DDPM] Epoch  21 | Train 8.053905 | Val 5.563220 | lambda_recon=1
[DDPM] Epoch  22 | Train 7.315465 | Val 4.142764 | lambda_recon=1
[DDPM] Epoch  23 | Train 6.534601 | Val 3.741582 | lambda_recon=1
[DDPM] Epoch  24 | Train 6.267102 | Val 4.264244 | lambda_recon=1
[DDPM] Epoch  25 | Train 5.832402 | Val 4.076166 | lambda_recon=1
[DDPM] Epoch  26 | Train 5.744477 | Val 3.982062 | lambda_recon=1
[DDPM] Early stopping at epoch 26 after 3 epochs without improvement.
[DDPM] depth=2 | lambda_recon=1 | pretrain_val=0.060362 | best_val=3.741582
Test Recall@4: 0.0301
Tree/embedding correlation: 0.8937

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C023', 'C024', 'C300', 'C301']
  Visit 2: ['C023', 'C024', 'C300', 'C301']
  Visit 3: ['C002', 'C004', 'C321', 'C422']
  Visit 4: ['C023', 'C024', 'C300', 'C301']
  Visit 5: ['C002', 'C004', 'C321', 'C422']
  Visit 6: ['C023', 'C024', 'C300', 'C301']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C002', 'C004', 'C321', 'C422']
  Visit 2: ['C002', 'C004', 'C321', 'C422']
  Visit 3: ['C002', 'C004', 'C321', 'C422']
  Visit 4: ['C002', 'C004', 'C321', 'C422']
  Visit 5: ['C023', 'C024', 'C300', 'C301']
  Visit 6: ['C023', 'C024', 'C300', 'C301']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C023', 'C024', 'C300', 'C301']
  Visit 2: ['C002', 'C004', 'C321', 'C422']
  Visit 3: ['C023', 'C024', 'C300', 'C301']
  Visit 4: ['C023', 'C024', 'C300', 'C301']
  Visit 5: ['C002', 'C004', 'C321', 'C422']
  Visit 6: ['C002', 'C004', 'C321', 'C422']
Synthetic stats (N=1000): {'mean_depth': 2.0, 'std_depth': 0.0, 'mean_tree_dist': 2.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon1_depth2_best3.7416.pt

Training hyperbolic Graph DDPM | depth=2 | lambda_recon=10
[DDPM] Epoch   1 | Train 128.682427 | Val 107.743228 | lambda_recon=10
[DDPM] Epoch   2 | Train 84.554949 | Val 60.568513 | lambda_recon=10
[DDPM] Epoch   3 | Train 58.677114 | Val 42.934270 | lambda_recon=10
[DDPM] Epoch   4 | Train 46.006951 | Val 32.735202 | lambda_recon=10
[DDPM] Epoch   5 | Train 37.584570 | Val 26.727301 | lambda_recon=10
[DDPM] Epoch   6 | Train 32.484273 | Val 22.604518 | lambda_recon=10
[DDPM] Epoch   7 | Train 28.142064 | Val 19.163986 | lambda_recon=10
[DDPM] Epoch   8 | Train 24.518598 | Val 16.501043 | lambda_recon=10
[DDPM] Epoch   9 | Train 21.824422 | Val 14.037420 | lambda_recon=10
[DDPM] Epoch  10 | Train 19.427743 | Val 12.167916 | lambda_recon=10
[DDPM] Epoch  11 | Train 17.405714 | Val 11.715883 | lambda_recon=10
[DDPM] Epoch  12 | Train 15.922373 | Val 10.842062 | lambda_recon=10
[DDPM] Epoch  13 | Train 14.971446 | Val 9.317802 | lambda_recon=10
[DDPM] Epoch  14 | Train 13.438187 | Val 8.457601 | lambda_recon=10
[DDPM] Epoch  15 | Train 12.213442 | Val 8.072507 | lambda_recon=10
[DDPM] Epoch  16 | Train 11.469335 | Val 7.705931 | lambda_recon=10
[DDPM] Epoch  17 | Train 10.644602 | Val 6.578131 | lambda_recon=10
[DDPM] Epoch  18 | Train 9.299433 | Val 5.414137 | lambda_recon=10
[DDPM] Epoch  19 | Train 8.759622 | Val 5.191114 | lambda_recon=10
[DDPM] Epoch  20 | Train 8.321768 | Val 5.409919 | lambda_recon=10
[DDPM] Epoch  21 | Train 7.953090 | Val 5.113461 | lambda_recon=10
[DDPM] Epoch  22 | Train 7.668974 | Val 4.973403 | lambda_recon=10
[DDPM] Epoch  23 | Train 7.282235 | Val 5.256885 | lambda_recon=10
[DDPM] Epoch  24 | Train 7.256519 | Val 5.422055 | lambda_recon=10
[DDPM] Epoch  25 | Train 7.069981 | Val 4.824625 | lambda_recon=10
[DDPM] Epoch  26 | Train 6.498969 | Val 4.906836 | lambda_recon=10
[DDPM] Epoch  27 | Train 6.525376 | Val 4.626843 | lambda_recon=10
[DDPM] Epoch  28 | Train 6.378501 | Val 4.655222 | lambda_recon=10
[DDPM] Epoch  29 | Train 6.067787 | Val 4.694618 | lambda_recon=10
[DDPM] Epoch  30 | Train 6.220391 | Val 4.484182 | lambda_recon=10
[DDPM] Epoch  31 | Train 6.056029 | Val 4.760323 | lambda_recon=10
[DDPM] Epoch  32 | Train 5.921602 | Val 4.781623 | lambda_recon=10
[DDPM] Epoch  33 | Train 5.628486 | Val 4.585995 | lambda_recon=10
[DDPM] Early stopping at epoch 33 after 3 epochs without improvement.
[DDPM] depth=2 | lambda_recon=10 | pretrain_val=0.060362 | best_val=4.484182
Test Recall@4: 0.0347
Tree/embedding correlation: 0.8863

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C141', 'C230', 'C234']
  Visit 2: ['C020', 'C022', 'C023', 'C024']
  Visit 3: ['C020', 'C022', 'C023', 'C024']
  Visit 4: ['C141', 'C230', 'C234']
  Visit 5: ['C020', 'C022', 'C023', 'C024']
  Visit 6: ['C141', 'C230', 'C234']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C020', 'C022', 'C023', 'C024']
  Visit 2: ['C141', 'C230', 'C234']
  Visit 3: ['C141', 'C230', 'C234']
  Visit 4: ['C020', 'C022', 'C023', 'C024']
  Visit 5: ['C141', 'C230', 'C234']
  Visit 6: ['C020', 'C022', 'C023', 'C024']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C141', 'C230', 'C234']
  Visit 2: ['C020', 'C022', 'C023', 'C024']
  Visit 3: ['C020', 'C022', 'C023', 'C024']
  Visit 4: ['C141', 'C230', 'C234']
  Visit 5: ['C020', 'C022', 'C023', 'C024']
  Visit 6: ['C020', 'C022', 'C023', 'C024']
Synthetic stats (N=1000): {'mean_depth': 2.0, 'std_depth': 0.0, 'mean_tree_dist': 2.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.8362777777777777, 'std_root_purity': 0.16664065537764597}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon10_depth2_best4.4842.pt

Training hyperbolic Graph DDPM | depth=2 | lambda_recon=100
[DDPM] Epoch   1 | Train 139.878418 | Val 121.881091 | lambda_recon=100
[DDPM] Epoch   2 | Train 95.646023 | Val 70.561084 | lambda_recon=100
[DDPM] Epoch   3 | Train 68.468305 | Val 52.955193 | lambda_recon=100
[DDPM] Epoch   4 | Train 55.512160 | Val 42.351152 | lambda_recon=100
[DDPM] Epoch   5 | Train 47.368598 | Val 35.974498 | lambda_recon=100
[DDPM] Epoch   6 | Train 41.794837 | Val 32.157976 | lambda_recon=100
[DDPM] Epoch   7 | Train 37.650206 | Val 28.805497 | lambda_recon=100
[DDPM] Epoch   8 | Train 34.536231 | Val 26.620345 | lambda_recon=100
[DDPM] Epoch   9 | Train 31.608378 | Val 24.286843 | lambda_recon=100
[DDPM] Epoch  10 | Train 29.390519 | Val 22.684246 | lambda_recon=100
[DDPM] Epoch  11 | Train 27.222336 | Val 21.047421 | lambda_recon=100
[DDPM] Epoch  12 | Train 25.733673 | Val 20.360889 | lambda_recon=100
[DDPM] Epoch  13 | Train 24.109785 | Val 18.703426 | lambda_recon=100
[DDPM] Epoch  14 | Train 22.796660 | Val 17.454606 | lambda_recon=100
[DDPM] Epoch  15 | Train 21.686793 | Val 17.207675 | lambda_recon=100
[DDPM] Epoch  16 | Train 20.968301 | Val 16.965389 | lambda_recon=100
[DDPM] Epoch  17 | Train 20.209008 | Val 15.916278 | lambda_recon=100
[DDPM] Epoch  18 | Train 19.213585 | Val 15.677462 | lambda_recon=100
[DDPM] Epoch  19 | Train 18.876649 | Val 15.626544 | lambda_recon=100
[DDPM] Epoch  20 | Train 18.499957 | Val 15.739042 | lambda_recon=100
[DDPM] Epoch  21 | Train 18.088884 | Val 15.124214 | lambda_recon=100
[DDPM] Epoch  22 | Train 17.068868 | Val 14.637583 | lambda_recon=100
[DDPM] Epoch  23 | Train 16.769394 | Val 14.663878 | lambda_recon=100
[DDPM] Epoch  24 | Train 16.638719 | Val 14.769758 | lambda_recon=100
[DDPM] Epoch  25 | Train 16.251319 | Val 14.376228 | lambda_recon=100
[DDPM] Epoch  26 | Train 16.147563 | Val 14.257965 | lambda_recon=100
[DDPM] Epoch  27 | Train 16.010947 | Val 14.902287 | lambda_recon=100
[DDPM] Epoch  28 | Train 15.856251 | Val 14.505844 | lambda_recon=100
[DDPM] Epoch  29 | Train 15.640532 | Val 14.348575 | lambda_recon=100
[DDPM] Early stopping at epoch 29 after 3 epochs without improvement.
[DDPM] depth=2 | lambda_recon=100 | pretrain_val=0.060362 | best_val=14.257965
Test Recall@4: 0.0396
Tree/embedding correlation: 0.9034

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C143', 'C321', 'C322']
  Visit 2: ['C143', 'C321', 'C322']
  Visit 3: ['C143', 'C321', 'C322']
  Visit 4: ['C020', 'C022', 'C023', 'C024']
  Visit 5: ['C020', 'C022', 'C023', 'C024']
  Visit 6: ['C020', 'C022', 'C023', 'C024']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C020', 'C022', 'C023', 'C024']
  Visit 2: ['C143', 'C321', 'C322']
  Visit 3: ['C020', 'C022', 'C023', 'C024']
  Visit 4: ['C020', 'C022', 'C023', 'C024']
  Visit 5: ['C020', 'C022', 'C023', 'C024']
  Visit 6: ['C020', 'C022', 'C023', 'C024']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C143', 'C321', 'C322']
  Visit 2: ['C143', 'C321', 'C322']
  Visit 3: ['C143', 'C321', 'C322']
  Visit 4: ['C020', 'C022', 'C023', 'C024']
  Visit 5: ['C020', 'C022', 'C023', 'C024']
  Visit 6: ['C020', 'C022', 'C023', 'C024']
Synthetic stats (N=1000): {'mean_depth': 2.0, 'std_depth': 0.0, 'mean_tree_dist': 2.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.8071944444444443, 'std_root_purity': 0.15839063047095828}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon100_depth2_best14.2580.pt

Training hyperbolic Graph DDPM | depth=2 | lambda_recon=1000
[DDPM] Epoch   1 | Train 241.621042 | Val 214.838614 | lambda_recon=1000
[DDPM] Epoch   2 | Train 189.643658 | Val 165.640896 | lambda_recon=1000
[DDPM] Epoch   3 | Train 163.968419 | Val 148.705843 | lambda_recon=1000
[DDPM] Epoch   4 | Train 151.572520 | Val 138.872771 | lambda_recon=1000
[DDPM] Epoch   5 | Train 143.815128 | Val 132.911287 | lambda_recon=1000
[DDPM] Epoch   6 | Train 138.090139 | Val 127.785383 | lambda_recon=1000
[DDPM] Epoch   7 | Train 133.489159 | Val 124.737679 | lambda_recon=1000
[DDPM] Epoch   8 | Train 130.060331 | Val 121.992569 | lambda_recon=1000
[DDPM] Epoch   9 | Train 126.991597 | Val 119.220397 | lambda_recon=1000
[DDPM] Epoch  10 | Train 124.451600 | Val 117.726739 | lambda_recon=1000
[DDPM] Epoch  11 | Train 122.564036 | Val 116.778719 | lambda_recon=1000
[DDPM] Epoch  12 | Train 120.730914 | Val 115.172424 | lambda_recon=1000
[DDPM] Epoch  13 | Train 119.598537 | Val 114.820957 | lambda_recon=1000
[DDPM] Epoch  14 | Train 118.556009 | Val 113.644207 | lambda_recon=1000
[DDPM] Epoch  15 | Train 117.404833 | Val 113.244369 | lambda_recon=1000
[DDPM] Epoch  16 | Train 116.655352 | Val 112.856815 | lambda_recon=1000
[DDPM] Epoch  17 | Train 115.224881 | Val 110.750387 | lambda_recon=1000
[DDPM] Epoch  18 | Train 114.157057 | Val 110.981439 | lambda_recon=1000
[DDPM] Epoch  19 | Train 113.625285 | Val 110.514920 | lambda_recon=1000
[DDPM] Epoch  20 | Train 112.998474 | Val 110.366479 | lambda_recon=1000
[DDPM] Epoch  21 | Train 112.499320 | Val 110.120038 | lambda_recon=1000
[DDPM] Epoch  22 | Train 112.150466 | Val 110.176948 | lambda_recon=1000
[DDPM] Epoch  23 | Train 111.803478 | Val 109.958224 | lambda_recon=1000
[DDPM] Epoch  24 | Train 111.492558 | Val 109.986039 | lambda_recon=1000
[DDPM] Epoch  25 | Train 111.390402 | Val 109.703662 | lambda_recon=1000
[DDPM] Epoch  26 | Train 111.032816 | Val 109.746633 | lambda_recon=1000
[DDPM] Epoch  27 | Train 110.836993 | Val 109.208993 | lambda_recon=1000
[DDPM] Epoch  28 | Train 110.673837 | Val 109.601348 | lambda_recon=1000
[DDPM] Epoch  29 | Train 110.494713 | Val 109.346369 | lambda_recon=1000
[DDPM] Epoch  30 | Train 110.094762 | Val 109.045407 | lambda_recon=1000
[DDPM] Epoch  31 | Train 110.108726 | Val 108.913720 | lambda_recon=1000
[DDPM] Epoch  32 | Train 109.661418 | Val 108.753135 | lambda_recon=1000
[DDPM] Epoch  33 | Train 109.721538 | Val 109.006385 | lambda_recon=1000
[DDPM] Epoch  34 | Train 109.539727 | Val 108.711224 | lambda_recon=1000
[DDPM] Epoch  35 | Train 109.530525 | Val 108.556818 | lambda_recon=1000
[DDPM] Epoch  36 | Train 109.215294 | Val 108.549115 | lambda_recon=1000
[DDPM] Epoch  37 | Train 108.993358 | Val 108.515408 | lambda_recon=1000
[DDPM] Epoch  38 | Train 109.050150 | Val 108.275577 | lambda_recon=1000
[DDPM] Epoch  39 | Train 108.962522 | Val 108.390205 | lambda_recon=1000
[DDPM] Epoch  40 | Train 108.798524 | Val 108.634694 | lambda_recon=1000
[DDPM] Epoch  41 | Train 108.791746 | Val 108.759817 | lambda_recon=1000
[DDPM] Early stopping at epoch 41 after 3 epochs without improvement.
[DDPM] depth=2 | lambda_recon=1000 | pretrain_val=0.060362 | best_val=108.275577
Test Recall@4: 0.0413
Tree/embedding correlation: 0.8743

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C141', 'C143', 'C234']
  Visit 2: ['C020', 'C022', 'C023', 'C024']
  Visit 3: ['C141', 'C143', 'C234']
  Visit 4: ['C020', 'C022', 'C023', 'C024']
  Visit 5: ['C141', 'C143', 'C234']
  Visit 6: ['C020', 'C022', 'C023', 'C024']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C020', 'C022', 'C023', 'C024']
  Visit 2: ['C020', 'C022', 'C023', 'C024']
  Visit 3: ['C141', 'C143', 'C234']
  Visit 4: ['C020', 'C022', 'C023', 'C024']
  Visit 5: ['C141', 'C143', 'C234']
  Visit 6: ['C141', 'C143', 'C234']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C141', 'C143', 'C234']
  Visit 2: ['C020', 'C022', 'C023', 'C024']
  Visit 3: ['C141', 'C143', 'C234']
  Visit 4: ['C141', 'C143', 'C234']
  Visit 5: ['C020', 'C022', 'C023', 'C024']
  Visit 6: ['C020', 'C022', 'C023', 'C024']
Synthetic stats (N=1000): {'mean_depth': 2.0, 'std_depth': 0.0, 'mean_tree_dist': 2.0046256143394046, 'std_tree_dist': 0.09607201658543449, 'mean_root_purity': 0.8303194444444444, 'std_root_purity': 0.1664622073437855}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon1000_depth2_best108.2756.pt

Training hyperbolic Graph DDPM | depth=2 | lambda_recon=2000
[DDPM] Epoch   1 | Train 404.765890 | Val 325.228033 | lambda_recon=2000
[DDPM] Epoch   2 | Train 299.433395 | Val 275.387639 | lambda_recon=2000
[DDPM] Epoch   3 | Train 273.501219 | Val 258.734122 | lambda_recon=2000
[DDPM] Epoch   4 | Train 260.614368 | Val 248.230549 | lambda_recon=2000
[DDPM] Epoch   5 | Train 252.470363 | Val 241.745365 | lambda_recon=2000
[DDPM] Epoch   6 | Train 246.204625 | Val 236.477819 | lambda_recon=2000
[DDPM] Epoch   7 | Train 241.281129 | Val 232.450301 | lambda_recon=2000
[DDPM] Epoch   8 | Train 237.949449 | Val 230.300695 | lambda_recon=2000
[DDPM] Epoch   9 | Train 235.323279 | Val 228.123256 | lambda_recon=2000
[DDPM] Epoch  10 | Train 232.706942 | Val 226.014831 | lambda_recon=2000
[DDPM] Epoch  11 | Train 230.831623 | Val 225.113667 | lambda_recon=2000
[DDPM] Epoch  12 | Train 229.114283 | Val 223.719155 | lambda_recon=2000
[DDPM] Epoch  13 | Train 227.526742 | Val 222.440808 | lambda_recon=2000
[DDPM] Epoch  14 | Train 226.376384 | Val 221.929803 | lambda_recon=2000
[DDPM] Epoch  15 | Train 225.293838 | Val 221.102403 | lambda_recon=2000
[DDPM] Epoch  16 | Train 223.783973 | Val 220.061196 | lambda_recon=2000
[DDPM] Epoch  17 | Train 222.836158 | Val 219.276502 | lambda_recon=2000
[DDPM] Epoch  18 | Train 221.767747 | Val 218.507203 | lambda_recon=2000
[DDPM] Epoch  19 | Train 221.221468 | Val 218.491307 | lambda_recon=2000
[DDPM] Epoch  20 | Train 220.478522 | Val 218.393412 | lambda_recon=2000
[DDPM] Epoch  21 | Train 219.955595 | Val 217.520934 | lambda_recon=2000
[DDPM] Epoch  22 | Train 219.734591 | Val 217.735606 | lambda_recon=2000
[DDPM] Epoch  23 | Train 219.389592 | Val 217.475814 | lambda_recon=2000
[DDPM] Epoch  24 | Train 218.848379 | Val 216.956242 | lambda_recon=2000
[DDPM] Epoch  25 | Train 218.435428 | Val 216.927761 | lambda_recon=2000
[DDPM] Epoch  26 | Train 217.925031 | Val 216.893307 | lambda_recon=2000
[DDPM] Epoch  27 | Train 217.882331 | Val 216.665587 | lambda_recon=2000
[DDPM] Epoch  28 | Train 217.604277 | Val 216.553162 | lambda_recon=2000
[DDPM] Epoch  29 | Train 217.229557 | Val 216.465696 | lambda_recon=2000
[DDPM] Epoch  30 | Train 216.969426 | Val 215.881860 | lambda_recon=2000
[DDPM] Epoch  31 | Train 216.617738 | Val 215.825286 | lambda_recon=2000
[DDPM] Epoch  32 | Train 216.492216 | Val 215.821429 | lambda_recon=2000
[DDPM] Epoch  33 | Train 216.235018 | Val 215.652979 | lambda_recon=2000
[DDPM] Epoch  34 | Train 215.998454 | Val 215.463390 | lambda_recon=2000
[DDPM] Epoch  35 | Train 215.857508 | Val 215.550478 | lambda_recon=2000
[DDPM] Epoch  36 | Train 215.819400 | Val 215.453823 | lambda_recon=2000
[DDPM] Epoch  37 | Train 215.420825 | Val 215.222615 | lambda_recon=2000
[DDPM] Epoch  38 | Train 215.312262 | Val 214.709765 | lambda_recon=2000
[DDPM] Epoch  39 | Train 215.016487 | Val 214.752176 | lambda_recon=2000
[DDPM] Epoch  40 | Train 214.973756 | Val 214.488891 | lambda_recon=2000
[DDPM] Epoch  41 | Train 214.922813 | Val 214.751649 | lambda_recon=2000
[DDPM] Epoch  42 | Train 214.505138 | Val 214.513861 | lambda_recon=2000
[DDPM] Epoch  43 | Train 214.365923 | Val 214.506407 | lambda_recon=2000
[DDPM] Early stopping at epoch 43 after 3 epochs without improvement.
[DDPM] depth=2 | lambda_recon=2000 | pretrain_val=0.060362 | best_val=214.488891
Test Recall@4: 0.0355
Tree/embedding correlation: 0.8929
