Using device: mps

hyperbolic_graph_ddpm_depth7 | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

=== Pretraining hyperbolic graph embeddings (DDPM) ===
[Pretrain-DDPM] Epoch   1 | train=0.095129 | val=0.089797 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   2 | train=0.088499 | val=0.087420 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   3 | train=0.088495 | val=0.085106 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   4 | train=0.085312 | val=0.084299 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   5 | train=0.082613 | val=0.083037 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   6 | train=0.082289 | val=0.083395 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   7 | train=0.083101 | val=0.080321 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   8 | train=0.078694 | val=0.079667 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch   9 | train=0.078451 | val=0.077943 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  10 | train=0.079223 | val=0.078763 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  11 | train=0.077978 | val=0.078674 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  12 | train=0.077940 | val=0.077720 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  13 | train=0.077950 | val=0.076721 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  14 | train=0.079143 | val=0.076254 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  15 | train=0.076393 | val=0.077022 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  16 | train=0.075985 | val=0.076011 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  17 | train=0.075025 | val=0.075840 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  18 | train=0.073368 | val=0.074009 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  19 | train=0.075350 | val=0.073297 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  20 | train=0.075493 | val=0.074095 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  21 | train=0.072655 | val=0.075889 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  22 | train=0.074477 | val=0.073998 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  23 | train=0.074805 | val=0.073697 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  24 | train=0.073796 | val=0.071621 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  25 | train=0.073171 | val=0.072451 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  26 | train=0.073376 | val=0.072502 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  27 | train=0.072737 | val=0.071480 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  28 | train=0.072904 | val=0.071370 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  29 | train=0.072060 | val=0.070833 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-DDPM] Epoch  30 | train=0.073498 | val=0.073656 | rad=0.003 pair=0.01 hdd=0.02
Saved pretraining checkpoint to results/checkpoints/hyperbolic_ddpm_pretrain_rad0.003_pair0.01_hdd0.02_val0.0708.pt

Training hyperbolic Graph DDPM | depth=7 | lambda_recon=1
[DDPM] Epoch   1 | Train 126.142330 | Val 100.433351 | lambda_recon=1
[DDPM] Epoch   2 | Train 80.638286 | Val 58.302701 | lambda_recon=1
[DDPM] Epoch   3 | Train 56.798558 | Val 41.769869 | lambda_recon=1
[DDPM] Epoch   4 | Train 45.196298 | Val 32.295954 | lambda_recon=1
[DDPM] Epoch   5 | Train 37.072365 | Val 25.841504 | lambda_recon=1
[DDPM] Epoch   6 | Train 31.274498 | Val 21.319384 | lambda_recon=1
[DDPM] Epoch   7 | Train 27.293273 | Val 18.137719 | lambda_recon=1
[DDPM] Epoch   8 | Train 23.778486 | Val 16.189528 | lambda_recon=1
[DDPM] Epoch   9 | Train 21.249440 | Val 13.918392 | lambda_recon=1
[DDPM] Epoch  10 | Train 18.880259 | Val 12.230839 | lambda_recon=1
[DDPM] Epoch  11 | Train 16.645512 | Val 10.080022 | lambda_recon=1
[DDPM] Epoch  12 | Train 14.917350 | Val 8.887671 | lambda_recon=1
[DDPM] Epoch  13 | Train 13.606710 | Val 8.351470 | lambda_recon=1
[DDPM] Epoch  14 | Train 12.689580 | Val 7.972131 | lambda_recon=1
[DDPM] Epoch  15 | Train 11.418775 | Val 6.031929 | lambda_recon=1
[DDPM] Epoch  16 | Train 10.026060 | Val 5.886280 | lambda_recon=1
[DDPM] Epoch  17 | Train 9.094154 | Val 5.415902 | lambda_recon=1
[DDPM] Epoch  18 | Train 8.399130 | Val 4.984755 | lambda_recon=1
[DDPM] Epoch  19 | Train 7.763206 | Val 4.700356 | lambda_recon=1
[DDPM] Epoch  20 | Train 7.465439 | Val 4.678882 | lambda_recon=1
[DDPM] Epoch  21 | Train 7.146250 | Val 5.040465 | lambda_recon=1
[DDPM] Epoch  22 | Train 6.915933 | Val 4.273777 | lambda_recon=1
[DDPM] Epoch  23 | Train 6.569556 | Val 3.886520 | lambda_recon=1
[DDPM] Epoch  24 | Train 6.386048 | Val 4.492647 | lambda_recon=1
[DDPM] Epoch  25 | Train 5.993605 | Val 4.351614 | lambda_recon=1
[DDPM] Epoch  26 | Train 5.881007 | Val 4.273106 | lambda_recon=1
[DDPM] Early stopping at epoch 26 after 3 epochs without improvement.
[DDPM] depth=7 | lambda_recon=1 | pretrain_val=0.070833 | best_val=3.886520
Test Recall@4: 0.0118
Tree/embedding correlation: 0.7737

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C101d2', 'C312d3', 'C324d3', 'C344d4']
  Visit 2: ['C101d2', 'C312d3', 'C324d3', 'C344d4']
  Visit 3: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 4: ['C101d2', 'C312d3', 'C324d3', 'C344d4']
  Visit 5: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 6: ['C101d2', 'C312d3', 'C324d3', 'C344d4']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 2: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 3: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 4: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 5: ['C002d3', 'C312d3', 'C324d3', 'C344d4']
  Visit 6: ['C002d3', 'C312d3', 'C324d3', 'C344d4']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C002d3', 'C101d2', 'C324d3', 'C344d4']
  Visit 2: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 3: ['C002d3', 'C101d2', 'C324d3', 'C344d4']
  Visit 4: ['C002d3', 'C101d2', 'C324d3', 'C344d4']
  Visit 5: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
  Visit 6: ['C104d4', 'C300d4', 'C330d3', 'C402d4']
Synthetic stats (N=1000): {'mean_depth': 6.399666666666667, 'std_depth': 0.6656574861660379, 'mean_tree_dist': 12.79708853238265, 'std_tree_dist': 0.40216713431945567, 'mean_root_purity': 0.585375, 'std_root_purity': 0.11855319217549563}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon1_depth7_best3.8865.pt

Training hyperbolic Graph DDPM | depth=7 | lambda_recon=10
[DDPM] Epoch   1 | Train 126.549260 | Val 99.716709 | lambda_recon=10
[DDPM] Epoch   2 | Train 79.513983 | Val 57.089870 | lambda_recon=10
[DDPM] Epoch   3 | Train 55.977745 | Val 40.581229 | lambda_recon=10
[DDPM] Epoch   4 | Train 43.796171 | Val 31.130575 | lambda_recon=10
[DDPM] Epoch   5 | Train 35.589158 | Val 24.344810 | lambda_recon=10
[DDPM] Epoch   6 | Train 29.984664 | Val 20.438713 | lambda_recon=10
[DDPM] Epoch   7 | Train 25.682680 | Val 16.671633 | lambda_recon=10
[DDPM] Epoch   8 | Train 22.157962 | Val 14.254757 | lambda_recon=10
[DDPM] Epoch   9 | Train 19.347713 | Val 11.870028 | lambda_recon=10
[DDPM] Epoch  10 | Train 17.126735 | Val 9.926889 | lambda_recon=10
[DDPM] Epoch  11 | Train 14.972773 | Val 9.222761 | lambda_recon=10
[DDPM] Epoch  12 | Train 13.944140 | Val 9.239079 | lambda_recon=10
[DDPM] Epoch  13 | Train 13.064940 | Val 7.450928 | lambda_recon=10
[DDPM] Epoch  14 | Train 11.457881 | Val 6.560122 | lambda_recon=10
[DDPM] Epoch  15 | Train 10.439060 | Val 6.561409 | lambda_recon=10
[DDPM] Epoch  16 | Train 9.869334 | Val 6.248216 | lambda_recon=10
[DDPM] Epoch  17 | Train 9.335877 | Val 5.933104 | lambda_recon=10
[DDPM] Epoch  18 | Train 8.575792 | Val 5.036192 | lambda_recon=10
[DDPM] Epoch  19 | Train 8.008899 | Val 4.730481 | lambda_recon=10
[DDPM] Epoch  20 | Train 7.689260 | Val 4.935821 | lambda_recon=10
[DDPM] Epoch  21 | Train 7.391325 | Val 4.557722 | lambda_recon=10
[DDPM] Epoch  22 | Train 7.065895 | Val 4.528205 | lambda_recon=10
[DDPM] Epoch  23 | Train 6.666713 | Val 4.638825 | lambda_recon=10
[DDPM] Epoch  24 | Train 6.667741 | Val 4.718010 | lambda_recon=10
[DDPM] Epoch  25 | Train 6.487214 | Val 4.557513 | lambda_recon=10
[DDPM] Early stopping at epoch 25 after 3 epochs without improvement.
[DDPM] depth=7 | lambda_recon=10 | pretrain_val=0.070833 | best_val=4.528205
Test Recall@4: 0.0117
Tree/embedding correlation: 0.7733

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C034d4', 'C101d2', 'C303d3', 'C441d4']
  Visit 2: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 3: ['C034d4', 'C101d2', 'C303d3', 'C441d4']
  Visit 4: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 5: ['C034d4', 'C101d2', 'C303d3', 'C441d4']
  Visit 6: ['C034d4', 'C101d2', 'C303d3', 'C441d4']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 2: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 3: ['C034d4', 'C101d2', 'C303d3', 'C441d4']
  Visit 4: ['C034d4', 'C101d2', 'C303d3', 'C441d4']
  Visit 5: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 6: ['C000d1', 'C004d3', 'C144d1', 'C302d4']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 2: ['C034d4', 'C101d2', 'C303d3', 'C441d4']
  Visit 3: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 4: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
  Visit 5: ['C034d4', 'C101d2', 'C303d3', 'C441d4']
  Visit 6: ['C000d1', 'C004d3', 'C144d1', 'C302d4']
Synthetic stats (N=1000): {'mean_depth': 5.750333333333334, 'std_depth': 1.1988188168174354, 'mean_tree_dist': 8.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.3749166666666667, 'std_root_purity': 0.12499997222221913}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon10_depth7_best4.5282.pt

Training hyperbolic Graph DDPM | depth=7 | lambda_recon=100
[DDPM] Epoch   1 | Train 133.020111 | Val 111.785338 | lambda_recon=100
[DDPM] Epoch   2 | Train 86.684455 | Val 62.779056 | lambda_recon=100
[DDPM] Epoch   3 | Train 61.248687 | Val 45.520277 | lambda_recon=100
[DDPM] Epoch   4 | Train 48.153202 | Val 34.929246 | lambda_recon=100
[DDPM] Epoch   5 | Train 39.901397 | Val 28.314550 | lambda_recon=100
[DDPM] Epoch   6 | Train 34.034719 | Val 23.413772 | lambda_recon=100
[DDPM] Epoch   7 | Train 29.271799 | Val 20.229123 | lambda_recon=100
[DDPM] Epoch   8 | Train 26.013772 | Val 17.995626 | lambda_recon=100
[DDPM] Epoch   9 | Train 23.490839 | Val 16.030475 | lambda_recon=100
[DDPM] Epoch  10 | Train 21.586326 | Val 14.484869 | lambda_recon=100
[DDPM] Epoch  11 | Train 19.392855 | Val 13.546587 | lambda_recon=100
[DDPM] Epoch  12 | Train 17.951909 | Val 12.027144 | lambda_recon=100
[DDPM] Epoch  13 | Train 16.609651 | Val 11.202242 | lambda_recon=100
[DDPM] Epoch  14 | Train 15.004060 | Val 9.528084 | lambda_recon=100
[DDPM] Epoch  15 | Train 13.521259 | Val 9.010218 | lambda_recon=100
[DDPM] Epoch  16 | Train 12.886634 | Val 8.844913 | lambda_recon=100
[DDPM] Epoch  17 | Train 12.263831 | Val 8.576244 | lambda_recon=100
[DDPM] Epoch  18 | Train 11.813385 | Val 8.383435 | lambda_recon=100
[DDPM] Epoch  19 | Train 11.423680 | Val 8.323224 | lambda_recon=100
[DDPM] Epoch  20 | Train 10.636180 | Val 7.800498 | lambda_recon=100
[DDPM] Epoch  21 | Train 10.052266 | Val 7.179776 | lambda_recon=100
[DDPM] Epoch  22 | Train 9.827233 | Val 7.128044 | lambda_recon=100
[DDPM] Epoch  23 | Train 9.574111 | Val 7.316075 | lambda_recon=100
[DDPM] Epoch  24 | Train 9.235466 | Val 6.980483 | lambda_recon=100
[DDPM] Epoch  25 | Train 9.183461 | Val 6.758064 | lambda_recon=100
[DDPM] Epoch  26 | Train 8.774022 | Val 6.640567 | lambda_recon=100
[DDPM] Epoch  27 | Train 8.668881 | Val 6.910891 | lambda_recon=100
[DDPM] Epoch  28 | Train 8.480662 | Val 6.962730 | lambda_recon=100
[DDPM] Epoch  29 | Train 8.385856 | Val 6.822244 | lambda_recon=100
[DDPM] Early stopping at epoch 29 after 3 epochs without improvement.
[DDPM] depth=7 | lambda_recon=100 | pretrain_val=0.070833 | best_val=6.640567
Test Recall@4: 0.0157
Tree/embedding correlation: 0.7503

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 2: ['C013d3', 'C043d4', 'C133d4', 'C442d4']
  Visit 3: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 4: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 5: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 6: ['C013d3', 'C043d4', 'C133d4', 'C442d4']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C013d3', 'C043d4', 'C133d4', 'C442d4']
  Visit 2: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 3: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 4: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 5: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 6: ['C002d3', 'C010d4', 'C034d4', 'C434d3']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 2: ['C013d3', 'C043d4', 'C133d4', 'C442d4']
  Visit 3: ['C013d3', 'C043d4', 'C133d4', 'C442d4']
  Visit 4: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 5: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
  Visit 6: ['C002d3', 'C010d4', 'C034d4', 'C434d3']
Synthetic stats (N=1000): {'mean_depth': 6.6265833333333335, 'std_depth': 0.4837113392533011, 'mean_tree_dist': 13.24840657497484, 'std_tree_dist': 0.43208882013321004, 'mean_root_purity': 0.6234166666666666, 'std_root_purity': 0.12498997181996466}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon100_depth7_best6.6406.pt

Training hyperbolic Graph DDPM | depth=7 | lambda_recon=1000
[DDPM] Epoch   1 | Train 245.250137 | Val 140.391819 | lambda_recon=1000
[DDPM] Epoch   2 | Train 112.894655 | Val 88.165603 | lambda_recon=1000
[DDPM] Epoch   3 | Train 86.629855 | Val 71.304253 | lambda_recon=1000
[DDPM] Epoch   4 | Train 74.304745 | Val 61.408278 | lambda_recon=1000
[DDPM] Epoch   5 | Train 65.950700 | Val 54.511617 | lambda_recon=1000
[DDPM] Epoch   6 | Train 60.221837 | Val 50.097097 | lambda_recon=1000
[DDPM] Epoch   7 | Train 56.199826 | Val 47.501329 | lambda_recon=1000
[DDPM] Epoch   8 | Train 52.995169 | Val 45.078986 | lambda_recon=1000
[DDPM] Epoch   9 | Train 50.269746 | Val 42.525672 | lambda_recon=1000
[DDPM] Epoch  10 | Train 47.686934 | Val 40.076218 | lambda_recon=1000
[DDPM] Epoch  11 | Train 45.696767 | Val 39.606415 | lambda_recon=1000
[DDPM] Epoch  12 | Train 44.549041 | Val 39.119619 | lambda_recon=1000
[DDPM] Epoch  13 | Train 43.465139 | Val 38.222564 | lambda_recon=1000
[DDPM] Epoch  14 | Train 41.805087 | Val 36.885105 | lambda_recon=1000
[DDPM] Epoch  15 | Train 41.028057 | Val 36.563063 | lambda_recon=1000
[DDPM] Epoch  16 | Train 39.915909 | Val 35.787619 | lambda_recon=1000
[DDPM] Epoch  17 | Train 38.647141 | Val 34.176156 | lambda_recon=1000
[DDPM] Epoch  18 | Train 37.884219 | Val 34.333917 | lambda_recon=1000
[DDPM] Epoch  19 | Train 37.531915 | Val 34.664718 | lambda_recon=1000
[DDPM] Epoch  20 | Train 36.569673 | Val 33.251578 | lambda_recon=1000
[DDPM] Epoch  21 | Train 36.007300 | Val 33.462233 | lambda_recon=1000
[DDPM] Epoch  22 | Train 35.876362 | Val 33.281658 | lambda_recon=1000
[DDPM] Epoch  23 | Train 35.417851 | Val 32.987126 | lambda_recon=1000
[DDPM] Epoch  24 | Train 35.166318 | Val 33.139312 | lambda_recon=1000
[DDPM] Epoch  25 | Train 34.841865 | Val 32.457027 | lambda_recon=1000
[DDPM] Epoch  26 | Train 34.821832 | Val 33.020019 | lambda_recon=1000
[DDPM] Epoch  27 | Train 34.665014 | Val 33.017948 | lambda_recon=1000
[DDPM] Epoch  28 | Train 34.350855 | Val 32.851273 | lambda_recon=1000
[DDPM] Early stopping at epoch 28 after 3 epochs without improvement.
[DDPM] depth=7 | lambda_recon=1000 | pretrain_val=0.070833 | best_val=32.457027
Test Recall@4: 0.0117
Tree/embedding correlation: 0.7897

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C003d3', 'C142d4', 'C233d4', 'C434d3']
  Visit 2: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 3: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 4: ['C003d3', 'C142d4', 'C233d4', 'C434d3']
  Visit 5: ['C003d3', 'C142d4', 'C233d4', 'C434d3']
  Visit 6: ['C003d3', 'C142d4', 'C233d4', 'C434d3']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 2: ['C003d3', 'C142d4', 'C233d4', 'C434d3']
  Visit 3: ['C003d3', 'C142d4', 'C233d4', 'C434d3']
  Visit 4: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 5: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 6: ['C001d4', 'C013d4', 'C023d2', 'C424d4']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 2: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 3: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 4: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 5: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
  Visit 6: ['C001d4', 'C013d4', 'C023d2', 'C424d4']
Synthetic stats (N=1000): {'mean_depth': 6.5, 'std_depth': 0.7076369125476709, 'mean_tree_dist': 12.666666666666666, 'std_tree_dist': 0.9428090415820634, 'mean_root_purity': 0.50075, 'std_root_purity': 0.24999887499746878}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon1000_depth7_best32.4570.pt

Training hyperbolic Graph DDPM | depth=7 | lambda_recon=2000
[DDPM] Epoch   1 | Train 291.930328 | Val 171.656712 | lambda_recon=2000
[DDPM] Epoch   2 | Train 143.465106 | Val 118.285950 | lambda_recon=2000
[DDPM] Epoch   3 | Train 116.512762 | Val 101.440834 | lambda_recon=2000
[DDPM] Epoch   4 | Train 103.755673 | Val 90.501214 | lambda_recon=2000
[DDPM] Epoch   5 | Train 95.434522 | Val 83.975444 | lambda_recon=2000
[DDPM] Epoch   6 | Train 89.558831 | Val 79.519002 | lambda_recon=2000
[DDPM] Epoch   7 | Train 85.068091 | Val 76.519112 | lambda_recon=2000
[DDPM] Epoch   8 | Train 81.785725 | Val 73.208489 | lambda_recon=2000
[DDPM] Epoch   9 | Train 78.387877 | Val 71.208220 | lambda_recon=2000
[DDPM] Epoch  10 | Train 75.782225 | Val 68.983562 | lambda_recon=2000
[DDPM] Epoch  11 | Train 74.255160 | Val 68.309407 | lambda_recon=2000
[DDPM] Epoch  12 | Train 73.246968 | Val 67.686462 | lambda_recon=2000
[DDPM] Epoch  13 | Train 71.824073 | Val 65.766220 | lambda_recon=2000
[DDPM] Epoch  14 | Train 70.166239 | Val 64.864655 | lambda_recon=2000
[DDPM] Epoch  15 | Train 68.801633 | Val 64.603091 | lambda_recon=2000
[DDPM] Epoch  16 | Train 68.304706 | Val 63.833161 | lambda_recon=2000
[DDPM] Epoch  17 | Train 67.274013 | Val 63.397425 | lambda_recon=2000
[DDPM] Epoch  18 | Train 66.628460 | Val 63.007048 | lambda_recon=2000
[DDPM] Epoch  19 | Train 66.153725 | Val 63.312118 | lambda_recon=2000
[DDPM] Epoch  20 | Train 65.717382 | Val 62.758729 | lambda_recon=2000
[DDPM] Epoch  21 | Train 65.253085 | Val 62.503113 | lambda_recon=2000
[DDPM] Epoch  22 | Train 64.470409 | Val 61.910889 | lambda_recon=2000
[DDPM] Epoch  23 | Train 64.109096 | Val 61.539605 | lambda_recon=2000
[DDPM] Epoch  24 | Train 63.973052 | Val 62.106748 | lambda_recon=2000
[DDPM] Epoch  25 | Train 63.688028 | Val 61.832153 | lambda_recon=2000
[DDPM] Epoch  26 | Train 63.608786 | Val 62.164554 | lambda_recon=2000
[DDPM] Early stopping at epoch 26 after 3 epochs without improvement.
[DDPM] depth=7 | lambda_recon=2000 | pretrain_val=0.070833 | best_val=61.539605
Test Recall@4: 0.0120
Tree/embedding correlation: 0.7598

[DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C004d4', 'C343d4', 'C430d2']
  Visit 2: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 3: ['C004d4', 'C343d4', 'C430d2']
  Visit 4: ['C004d4', 'C343d4', 'C430d2']
  Visit 5: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 6: ['C004d4', 'C343d4', 'C430d2']

[DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C004d4', 'C343d4', 'C430d2']
  Visit 2: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 3: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 4: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 5: ['C004d4', 'C343d4', 'C430d2']
  Visit 6: ['C004d4', 'C343d4', 'C430d2']

[DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C004d4', 'C343d4', 'C430d2']
  Visit 2: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 3: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 4: ['C004d4', 'C343d4', 'C430d2']
  Visit 5: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
  Visit 6: ['C043d4', 'C230d3', 'C331d2', 'C420d3']
Synthetic stats (N=1000): {'mean_depth': 6.142911567217487, 'std_depth': 0.8330327933410975, 'mean_tree_dist': nan, 'std_tree_dist': nan, 'mean_root_purity': 0.29168055555555555, 'std_root_purity': 0.041666664351851776}
Saved DDPM model checkpoint to results/checkpoints/hyperbolic_ddpm_lrecon2000_depth7_best61.5396.pt
