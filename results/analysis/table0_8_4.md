Using device: mps

hg_ddpm_depth2 | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Pretraining hyperbolic code embeddings (HDD-style) ===
[Pretrain-HDD] Epoch   1 | train=0.084093 | val=0.078877 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   2 | train=0.079546 | val=0.074398 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   3 | train=0.073537 | val=0.076239 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   4 | train=0.072836 | val=0.072066 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   5 | train=0.075046 | val=0.073842 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   6 | train=0.074024 | val=0.070136 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   7 | train=0.071365 | val=0.070757 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   8 | train=0.071310 | val=0.069640 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch   9 | train=0.070457 | val=0.070048 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  10 | train=0.069082 | val=0.068836 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  11 | train=0.067958 | val=0.068881 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  12 | train=0.068186 | val=0.066937 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  13 | train=0.067376 | val=0.066519 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  14 | train=0.068133 | val=0.066853 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  15 | train=0.066365 | val=0.066997 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  16 | train=0.065443 | val=0.064393 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  17 | train=0.066035 | val=0.065164 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  18 | train=0.064132 | val=0.063383 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  19 | train=0.064113 | val=0.063474 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  20 | train=0.063599 | val=0.063059 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  21 | train=0.064106 | val=0.063653 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  22 | train=0.062403 | val=0.062692 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  23 | train=0.062826 | val=0.062377 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  24 | train=0.062214 | val=0.061855 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  25 | train=0.062276 | val=0.061540 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  26 | train=0.062021 | val=0.061072 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  27 | train=0.061561 | val=0.061713 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  28 | train=0.060566 | val=0.061577 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  29 | train=0.061031 | val=0.060362 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain-HDD] Epoch  30 | train=0.060343 | val=0.061162 | rad=0.003 pair=0.01 hdd=0.02
Saved pretraining checkpoint to results/checkpoints/hg_ddpm_pretrain_rad0.003_pair0.01_hdd0.02_val0.0604.pt

Training Hyperbolic Graph DDPM (Global) | depth=2 | lambda_recon=1
[HG-DDPM] Epoch   1 | Train 32.850808 | Val 31.882700 | lambda_recon=1
[HG-DDPM] Epoch   2 | Train 30.116532 | Val 27.551431 | lambda_recon=1
[HG-DDPM] Epoch   3 | Train 27.076631 | Val 24.480555 | lambda_recon=1
[HG-DDPM] Epoch   4 | Train 25.206111 | Val 22.602769 | lambda_recon=1
[HG-DDPM] Epoch   5 | Train 23.583358 | Val 21.045337 | lambda_recon=1
[HG-DDPM] Epoch   6 | Train 22.054133 | Val 19.075944 | lambda_recon=1
[HG-DDPM] Epoch   7 | Train 20.910024 | Val 17.291470 | lambda_recon=1
[HG-DDPM] Epoch   8 | Train 19.255493 | Val 15.988523 | lambda_recon=1
[HG-DDPM] Epoch   9 | Train 18.415083 | Val 15.702885 | lambda_recon=1
[HG-DDPM] Epoch  10 | Train 17.519219 | Val 14.626360 | lambda_recon=1
[HG-DDPM] Epoch  11 | Train 16.831182 | Val 14.301956 | lambda_recon=1
[HG-DDPM] Epoch  12 | Train 16.471987 | Val 14.462579 | lambda_recon=1
[HG-DDPM] Epoch  13 | Train 15.904405 | Val 12.751652 | lambda_recon=1
[HG-DDPM] Epoch  14 | Train 15.500808 | Val 13.368561 | lambda_recon=1
[HG-DDPM] Epoch  15 | Train 15.421399 | Val 12.137725 | lambda_recon=1
[HG-DDPM] Epoch  16 | Train 15.139360 | Val 12.680753 | lambda_recon=1
[HG-DDPM] Epoch  17 | Train 14.905828 | Val 12.402995 | lambda_recon=1
[HG-DDPM] Epoch  18 | Train 14.777147 | Val 13.042965 | lambda_recon=1
[HG-DDPM] Early stopping at epoch 18 after 3 epochs without improvement.
[HG-DDPM] depth=2 | lambda_recon=1 | pretrain_val=0.060362 | best_val=12.137725
Test Recall@4: 0.0224
Tree/embedding correlation: 0.8964

[HG-DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C204', 'C332', 'C334', 'C433']
  Visit 2: ['C144', 'C204', 'C221', 'C410']
  Visit 3: ['C300', 'C301', 'C302', 'C304']
  Visit 4: ['C103', 'C204', 'C332', 'C431']
  Visit 5: ['C043', 'C202', 'C204', 'C433']
  Visit 6: ['C034', 'C144', 'C232', 'C432']

[HG-DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C143', 'C300', 'C302', 'C304']
  Visit 2: ['C330', 'C332', 'C334', 'C433']
  Visit 3: ['C101', 'C204', 'C334', 'C433']
  Visit 4: ['C114', 'C204', 'C332', 'C431']
  Visit 5: ['C041', 'C220', 'C301', 'C302']
  Visit 6: ['C202', 'C204', 'C332', 'C433']

[HG-DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C204', 'C240', 'C334', 'C414']
  Visit 2: ['C114', 'C204', 'C431', 'C433']
  Visit 3: ['C220', 'C223', 'C301', 'C302']
  Visit 4: ['C023', 'C344', 'C440', 'C443']
  Visit 5: ['C234', 'C324', 'C342', 'C344']
  Visit 6: ['C043', 'C202', 'C204', 'C433']
Synthetic stats (N=1000): {'mean_depth': 1.9832455920447898, 'std_depth': 0.12834990365902324, 'mean_tree_dist': 2.3556565414972495, 'std_tree_dist': 0.879778777121785, 'mean_root_purity': 0.513986111111111, 'std_root_purity': 0.151129971632424}
Saved HG-DDPM model checkpoint to results/checkpoints/hg_ddpm_global_lrecon1_depth2_best12.1377.pt

Training Hyperbolic Graph DDPM (Global) | depth=2 | lambda_recon=10
[HG-DDPM] Epoch   1 | Train 33.568346 | Val 31.281315 | lambda_recon=10
[HG-DDPM] Epoch   2 | Train 29.886811 | Val 27.501704 | lambda_recon=10
[HG-DDPM] Epoch   3 | Train 27.685520 | Val 25.362838 | lambda_recon=10
[HG-DDPM] Epoch   4 | Train 25.991594 | Val 23.317815 | lambda_recon=10
[HG-DDPM] Epoch   5 | Train 24.393459 | Val 21.600356 | lambda_recon=10
[HG-DDPM] Epoch   6 | Train 23.048281 | Val 20.291308 | lambda_recon=10
[HG-DDPM] Epoch   7 | Train 21.717002 | Val 18.810124 | lambda_recon=10
[HG-DDPM] Epoch   8 | Train 20.312395 | Val 17.661021 | lambda_recon=10
[HG-DDPM] Epoch   9 | Train 19.374790 | Val 17.127252 | lambda_recon=10
[HG-DDPM] Epoch  10 | Train 18.409178 | Val 15.418956 | lambda_recon=10
[HG-DDPM] Epoch  11 | Train 17.898702 | Val 15.818742 | lambda_recon=10
[HG-DDPM] Epoch  12 | Train 17.279367 | Val 15.421944 | lambda_recon=10
[HG-DDPM] Epoch  13 | Train 16.620469 | Val 13.905133 | lambda_recon=10
[HG-DDPM] Epoch  14 | Train 16.222516 | Val 14.322138 | lambda_recon=10
[HG-DDPM] Epoch  15 | Train 15.621087 | Val 12.703353 | lambda_recon=10
[HG-DDPM] Epoch  16 | Train 15.283890 | Val 12.020042 | lambda_recon=10
[HG-DDPM] Epoch  17 | Train 14.882396 | Val 11.956168 | lambda_recon=10
[HG-DDPM] Epoch  18 | Train 14.489571 | Val 11.529890 | lambda_recon=10
[HG-DDPM] Epoch  19 | Train 14.183036 | Val 11.503240 | lambda_recon=10
[HG-DDPM] Epoch  20 | Train 14.186194 | Val 11.244327 | lambda_recon=10
[HG-DDPM] Epoch  21 | Train 13.736818 | Val 11.298311 | lambda_recon=10
[HG-DDPM] Epoch  22 | Train 13.696159 | Val 11.352299 | lambda_recon=10
[HG-DDPM] Epoch  23 | Train 13.541226 | Val 10.489334 | lambda_recon=10
[HG-DDPM] Epoch  24 | Train 12.982720 | Val 10.486713 | lambda_recon=10
[HG-DDPM] Epoch  25 | Train 12.893003 | Val 9.778084 | lambda_recon=10
[HG-DDPM] Epoch  26 | Train 12.545362 | Val 10.484063 | lambda_recon=10
[HG-DDPM] Epoch  27 | Train 12.670464 | Val 10.842094 | lambda_recon=10
[HG-DDPM] Epoch  28 | Train 12.451979 | Val 9.703702 | lambda_recon=10
[HG-DDPM] Epoch  29 | Train 12.378627 | Val 9.527645 | lambda_recon=10
[HG-DDPM] Epoch  30 | Train 12.154016 | Val 10.171057 | lambda_recon=10
[HG-DDPM] Epoch  31 | Train 12.104431 | Val 10.058017 | lambda_recon=10
[HG-DDPM] Epoch  32 | Train 12.063239 | Val 10.060704 | lambda_recon=10
[HG-DDPM] Early stopping at epoch 32 after 3 epochs without improvement.
[HG-DDPM] depth=2 | lambda_recon=10 | pretrain_val=0.060362 | best_val=9.527645
Test Recall@4: 0.0210
Tree/embedding correlation: 0.8880

[HG-DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C121', 'C331', 'C402', 'C404']
  Visit 2: ['C021', 'C122', 'C404', 'C432']
  Visit 3: ['C121', 'C122', 'C123', 'C404']
  Visit 4: ['C113', 'C140', 'C213', 'C214']
  Visit 5: ['C211', 'C212', 'C441', 'C444']
  Visit 6: ['C040', 'C113', 'C214', 'C222']

[HG-DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C040', 'C121', 'C140', 'C402']
  Visit 2: ['C211', 'C212', 'C441', 'C444']
  Visit 3: ['C131', 'C211', 'C311', 'C314']
  Visit 4: ['C021', 'C121', 'C122', 'C342']
  Visit 5: ['C124', 'C314', 'C400', 'C404']
  Visit 6: ['C043', 'C222', 'C224', 'C404']

[HG-DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C211', 'C311', 'C313', 'C314']
  Visit 2: ['C214', 'C34', 'C343', 'C440']
  Visit 3: ['C001', 'C212', 'C440', 'C444']
  Visit 4: ['C121', 'C122', 'C402', 'C404']
  Visit 5: ['C040', 'C113', 'C121', 'C214']
  Visit 6: ['C131', 'C211', 'C212', 'C441']
Synthetic stats (N=1000): {'mean_depth': 1.9875376380060221, 'std_depth': 0.11093715125019651, 'mean_tree_dist': 2.555064782096584, 'std_tree_dist': 0.9223966249347354, 'mean_root_purity': 0.5600277777777778, 'std_root_purity': 0.1572283052415684}
Saved HG-DDPM model checkpoint to results/checkpoints/hg_ddpm_global_lrecon10_depth2_best9.5276.pt

Training Hyperbolic Graph DDPM (Global) | depth=2 | lambda_recon=100
[HG-DDPM] Epoch   1 | Train 44.783616 | Val 40.403531 | lambda_recon=100
[HG-DDPM] Epoch   2 | Train 39.385545 | Val 36.627749 | lambda_recon=100
[HG-DDPM] Epoch   3 | Train 37.072973 | Val 34.435679 | lambda_recon=100
[HG-DDPM] Epoch   4 | Train 35.144116 | Val 32.704692 | lambda_recon=100
[HG-DDPM] Epoch   5 | Train 33.438630 | Val 31.210845 | lambda_recon=100
[HG-DDPM] Epoch   6 | Train 32.045560 | Val 29.476162 | lambda_recon=100
[HG-DDPM] Epoch   7 | Train 30.758223 | Val 28.039479 | lambda_recon=100
[HG-DDPM] Epoch   8 | Train 29.702096 | Val 27.179845 | lambda_recon=100
[HG-DDPM] Epoch   9 | Train 28.552811 | Val 25.891312 | lambda_recon=100
[HG-DDPM] Epoch  10 | Train 27.934223 | Val 24.685704 | lambda_recon=100
[HG-DDPM] Epoch  11 | Train 27.213122 | Val 25.021527 | lambda_recon=100
[HG-DDPM] Epoch  12 | Train 26.722975 | Val 23.370483 | lambda_recon=100
[HG-DDPM] Epoch  13 | Train 26.249195 | Val 23.315029 | lambda_recon=100
[HG-DDPM] Epoch  14 | Train 25.639980 | Val 22.578664 | lambda_recon=100
[HG-DDPM] Epoch  15 | Train 25.002197 | Val 22.368790 | lambda_recon=100
[HG-DDPM] Epoch  16 | Train 24.950794 | Val 22.477561 | lambda_recon=100
[HG-DDPM] Epoch  17 | Train 24.609882 | Val 21.647868 | lambda_recon=100
[HG-DDPM] Epoch  18 | Train 24.198976 | Val 22.967663 | lambda_recon=100
[HG-DDPM] Epoch  19 | Train 24.197543 | Val 22.172205 | lambda_recon=100
[HG-DDPM] Epoch  20 | Train 24.060464 | Val 21.386675 | lambda_recon=100
[HG-DDPM] Epoch  21 | Train 23.784420 | Val 21.220355 | lambda_recon=100
[HG-DDPM] Epoch  22 | Train 23.389330 | Val 22.086707 | lambda_recon=100
[HG-DDPM] Epoch  23 | Train 23.213416 | Val 20.708183 | lambda_recon=100
[HG-DDPM] Epoch  24 | Train 23.077836 | Val 20.618088 | lambda_recon=100
[HG-DDPM] Epoch  25 | Train 22.826230 | Val 20.166898 | lambda_recon=100
[HG-DDPM] Epoch  26 | Train 22.555855 | Val 19.661664 | lambda_recon=100
[HG-DDPM] Epoch  27 | Train 22.438397 | Val 20.324806 | lambda_recon=100
[HG-DDPM] Epoch  28 | Train 22.370324 | Val 20.465736 | lambda_recon=100
[HG-DDPM] Epoch  29 | Train 22.280604 | Val 20.881491 | lambda_recon=100
[HG-DDPM] Early stopping at epoch 29 after 3 epochs without improvement.
[HG-DDPM] depth=2 | lambda_recon=100 | pretrain_val=0.060362 | best_val=19.661664
Test Recall@4: 0.0223
Tree/embedding correlation: 0.8978

[HG-DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C230', 'C231', 'C232', 'C420']
  Visit 2: ['C021', 'C314', 'C422', 'C423']
  Visit 3: ['C313', 'C341', 'C344', 'C442']
  Visit 4: ['C031', 'C213', 'C314', 'C344']
  Visit 5: ['C031', 'C314', 'C344', 'C422']
  Visit 6: ['C211', 'C314', 'C342', 'C344']

[HG-DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C23', 'C230', 'C231', 'C233']
  Visit 2: ['C031', 'C213', 'C344', 'C442']
  Visit 3: ['C020', 'C021', 'C022', 'C440']
  Visit 4: ['C140', 'C312', 'C314', 'C432']
  Visit 5: ['C031', 'C213', 'C314', 'C344']
  Visit 6: ['C020', 'C021', 'C023', 'C423']

[HG-DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C141', 'C142', 'C200', 'C420']
  Visit 2: ['C031', 'C213', 'C344', 'C440']
  Visit 3: ['C021', 'C124', 'C312', 'C314']
  Visit 4: ['C44', 'C440', 'C442', 'C443']
  Visit 5: ['C001', 'C003', 'C103']
  Visit 6: ['C003', 'C230', 'C233', 'C420']
Synthetic stats (N=1000): {'mean_depth': 1.9872539596305738, 'std_depth': 0.11217655202548871, 'mean_tree_dist': 2.30239340305712, 'std_tree_dist': 0.8177500280080978, 'mean_root_purity': 0.5596944444444445, 'std_root_purity': 0.15749485343345251}
Saved HG-DDPM model checkpoint to results/checkpoints/hg_ddpm_global_lrecon100_depth2_best19.6617.pt

Training Hyperbolic Graph DDPM (Global) | depth=2 | lambda_recon=1000
[HG-DDPM] Epoch   1 | Train 145.977771 | Val 136.972501 | lambda_recon=1000
[HG-DDPM] Epoch   2 | Train 135.677237 | Val 133.220989 | lambda_recon=1000
[HG-DDPM] Epoch   3 | Train 133.265023 | Val 130.763348 | lambda_recon=1000
[HG-DDPM] Epoch   4 | Train 131.397985 | Val 129.099295 | lambda_recon=1000
[HG-DDPM] Epoch   5 | Train 129.743222 | Val 127.233983 | lambda_recon=1000
[HG-DDPM] Epoch   6 | Train 128.296847 | Val 126.276576 | lambda_recon=1000
[HG-DDPM] Epoch   7 | Train 127.108966 | Val 126.064231 | lambda_recon=1000
[HG-DDPM] Epoch   8 | Train 126.106116 | Val 123.540379 | lambda_recon=1000
[HG-DDPM] Epoch   9 | Train 125.085925 | Val 122.507258 | lambda_recon=1000
[HG-DDPM] Epoch  10 | Train 124.425570 | Val 121.752671 | lambda_recon=1000
[HG-DDPM] Epoch  11 | Train 123.475778 | Val 120.566522 | lambda_recon=1000
[HG-DDPM] Epoch  12 | Train 123.060290 | Val 120.899027 | lambda_recon=1000
[HG-DDPM] Epoch  13 | Train 122.401347 | Val 119.469383 | lambda_recon=1000
[HG-DDPM] Epoch  14 | Train 121.914291 | Val 119.741346 | lambda_recon=1000
[HG-DDPM] Epoch  15 | Train 121.533958 | Val 120.030964 | lambda_recon=1000
[HG-DDPM] Epoch  16 | Train 120.944125 | Val 119.021096 | lambda_recon=1000
[HG-DDPM] Epoch  17 | Train 120.545616 | Val 117.822570 | lambda_recon=1000
[HG-DDPM] Epoch  18 | Train 120.376115 | Val 117.299907 | lambda_recon=1000
[HG-DDPM] Epoch  19 | Train 119.839124 | Val 117.772869 | lambda_recon=1000
[HG-DDPM] Epoch  20 | Train 119.506073 | Val 117.403458 | lambda_recon=1000
[HG-DDPM] Epoch  21 | Train 119.312421 | Val 116.830418 | lambda_recon=1000
[HG-DDPM] Epoch  22 | Train 119.060090 | Val 117.053017 | lambda_recon=1000
[HG-DDPM] Epoch  23 | Train 118.902019 | Val 116.913741 | lambda_recon=1000
[HG-DDPM] Epoch  24 | Train 118.545011 | Val 116.302886 | lambda_recon=1000
[HG-DDPM] Epoch  25 | Train 118.128599 | Val 116.360956 | lambda_recon=1000
[HG-DDPM] Epoch  26 | Train 118.412739 | Val 116.478146 | lambda_recon=1000
[HG-DDPM] Epoch  27 | Train 118.010314 | Val 116.086340 | lambda_recon=1000
[HG-DDPM] Epoch  28 | Train 117.808712 | Val 115.789174 | lambda_recon=1000
[HG-DDPM] Epoch  29 | Train 117.669433 | Val 116.126262 | lambda_recon=1000
[HG-DDPM] Epoch  30 | Train 117.705914 | Val 115.819632 | lambda_recon=1000
[HG-DDPM] Epoch  31 | Train 117.422852 | Val 116.044104 | lambda_recon=1000
[HG-DDPM] Early stopping at epoch 31 after 3 epochs without improvement.
[HG-DDPM] depth=2 | lambda_recon=1000 | pretrain_val=0.060362 | best_val=115.789174
Test Recall@4: 0.0236
Tree/embedding correlation: 0.8863

[HG-DDPM] Sample hyperbolic trajectory 1:
  Visit 1: ['C002', 'C110', 'C112', 'C434']
  Visit 2: ['C011', 'C013', 'C014', 'C140']
  Visit 3: ['C000', 'C003', 'C122', 'C340']
  Visit 4: ['C102', 'C110', 'C112', 'C314']
  Visit 5: ['C013', 'C203', 'C302', 'C434']
  Visit 6: ['C102', 'C111', 'C113', 'C314']

[HG-DDPM] Sample hyperbolic trajectory 2:
  Visit 1: ['C110', 'C431', 'C432', 'C434']
  Visit 2: ['C110', 'C112', 'C314', 'C434']
  Visit 3: ['C113', 'C311', 'C312', 'C314']
  Visit 4: ['C110', 'C314', 'C431', 'C434']
  Visit 5: ['C031', 'C314', 'C431', 'C434']
  Visit 6: ['C000', 'C003', 'C122', 'C343']

[HG-DDPM] Sample hyperbolic trajectory 3:
  Visit 1: ['C122', 'C340', 'C411', 'C414']
  Visit 2: ['C013', 'C014', 'C202', 'C322']
  Visit 3: ['C011', 'C012', 'C101', 'C334']
  Visit 4: ['C122', 'C340', 'C411', 'C414']
  Visit 5: ['C011', 'C012', 'C013', 'C014']
  Visit 6: ['C012', 'C101', 'C102', 'C312']
Synthetic stats (N=1000): {'mean_depth': 1.9981655965980154, 'std_depth': 0.04279063409372755, 'mean_tree_dist': 2.26528442317916, 'std_tree_dist': 0.6919538604900634, 'mean_root_purity': 0.5722777777777778, 'std_root_purity': 0.20201644750299022}
Saved HG-DDPM model checkpoint to results/checkpoints/hg_ddpm_global_lrecon1000_depth2_best115.7892.pt

Training Hyperbolic Graph DDPM (Global) | depth=2 | lambda_recon=2000
[HG-DDPM] Epoch   1 | Train 247.709686 | Val 243.751883 | lambda_recon=2000
[HG-DDPM] Epoch   2 | Train 242.299561 | Val 239.838800 | lambda_recon=2000
[HG-DDPM] Epoch   3 | Train 239.654514 | Val 237.745792 | lambda_recon=2000
[HG-DDPM] Epoch   4 | Train 237.975579 | Val 235.912988 | lambda_recon=2000
[HG-DDPM] Epoch   5 | Train 236.470868 | Val 234.767502 | lambda_recon=2000
[HG-DDPM] Epoch   6 | Train 235.432254 | Val 233.127239 | lambda_recon=2000
[HG-DDPM] Epoch   7 | Train 234.500889 | Val 232.752451 | lambda_recon=2000
[HG-DDPM] Epoch   8 | Train 233.725894 | Val 231.920644 | lambda_recon=2000
[HG-DDPM] Epoch   9 | Train 233.020721 | Val 231.008203 | lambda_recon=2000
[HG-DDPM] Epoch  10 | Train 232.237092 | Val 230.510060 | lambda_recon=2000
[HG-DDPM] Epoch  11 | Train 231.531052 | Val 229.689949 | lambda_recon=2000
[HG-DDPM] Epoch  12 | Train 231.192377 | Val 228.648311 | lambda_recon=2000
[HG-DDPM] Epoch  13 | Train 230.467706 | Val 228.527491 | lambda_recon=2000
[HG-DDPM] Epoch  14 | Train 230.101367 | Val 228.954063 | lambda_recon=2000
[HG-DDPM] Epoch  15 | Train 229.698525 | Val 227.571450 | lambda_recon=2000
[HG-DDPM] Epoch  16 | Train 229.409053 | Val 227.034001 | lambda_recon=2000
[HG-DDPM] Epoch  17 | Train 228.988179 | Val 226.824886 | lambda_recon=2000
[HG-DDPM] Epoch  18 | Train 228.526238 | Val 227.328240 | lambda_recon=2000
[HG-DDPM] Epoch  19 | Train 228.581395 | Val 226.615808 | lambda_recon=2000
[HG-DDPM] Epoch  20 | Train 228.163206 | Val 227.086872 | lambda_recon=2000
[HG-DDPM] Epoch  21 | Train 227.843872 | Val 225.304461 | lambda_recon=2000
[HG-DDPM] Epoch  22 | Train 227.718512 | Val 225.537403 | lambda_recon=2000
[HG-DDPM] Epoch  23 | Train 227.307390 | Val 226.042540 | lambda_recon=2000
[HG-DDPM] Epoch  24 | Train 227.311219 | Val 226.149649 | lambda_recon=2000
[HG-DDPM] Early stopping at epoch 24 after 3 epochs without improvement.
[HG-DDPM] depth=2 | lambda_recon=2000 | pretrain_val=0.060362 | best_val=225.304461
Test Recall@4: 0.0340
Tree/embedding correlation: 0.8852
