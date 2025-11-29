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
Saved pretraining checkpoint to results/checkpoints/hyperbolic_rectified2_pretrain_rad0.003_pair0.01_hdd0.02_val0.0604.pt
[Rectified2] Epoch   1 | Train 11.010214 | Val 8.483109 | lambda_recon=1
[Rectified2] Epoch   2 | Train 7.339168 | Val 5.065808 | lambda_recon=1
[Rectified2] Epoch   3 | Train 5.334157 | Val 3.514894 | lambda_recon=1
[Rectified2] Epoch   4 | Train 4.222287 | Val 2.542696 | lambda_recon=1
[Rectified2] Epoch   5 | Train 3.486737 | Val 1.967171 | lambda_recon=1
[Rectified2] Epoch   6 | Train 2.997804 | Val 1.627593 | lambda_recon=1
[Rectified2] Epoch   7 | Train 2.646305 | Val 1.266132 | lambda_recon=1
[Rectified2] Epoch   8 | Train 2.403881 | Val 1.149312 | lambda_recon=1
[Rectified2] Epoch   9 | Train 2.219909 | Val 0.995384 | lambda_recon=1
[Rectified2] Epoch  10 | Train 2.105449 | Val 0.904139 | lambda_recon=1
[Rectified2] Epoch  11 | Train 2.003046 | Val 0.835452 | lambda_recon=1
[Rectified2] Epoch  12 | Train 1.939726 | Val 0.821431 | lambda_recon=1
[Rectified2] Epoch  13 | Train 1.870916 | Val 0.697094 | lambda_recon=1
[Rectified2] Epoch  14 | Train 1.836679 | Val 0.706360 | lambda_recon=1
[Rectified2] Epoch  15 | Train 1.794769 | Val 0.646661 | lambda_recon=1
[Rectified2] Epoch  16 | Train 1.772587 | Val 0.672135 | lambda_recon=1
[Rectified2] Epoch  17 | Train 1.750831 | Val 0.647791 | lambda_recon=1
[Rectified2] Epoch  18 | Train 1.724647 | Val 0.604177 | lambda_recon=1
[Rectified2] Epoch  19 | Train 1.712568 | Val 0.598493 | lambda_recon=1
[Rectified2] Epoch  20 | Train 1.694386 | Val 0.596713 | lambda_recon=1
[Rectified2] Epoch  21 | Train 1.689719 | Val 0.615311 | lambda_recon=1
[Rectified2] Epoch  22 | Train 1.682807 | Val 0.606163 | lambda_recon=1
[Rectified2] Epoch  23 | Train 1.675440 | Val 0.562152 | lambda_recon=1
[Rectified2] Epoch  24 | Train 1.654409 | Val 0.582389 | lambda_recon=1
[Rectified2] Epoch  25 | Train 1.648974 | Val 0.595242 | lambda_recon=1
[Rectified2] Epoch  26 | Train 1.625449 | Val 0.609615 | lambda_recon=1
[Rectified] Early stopping.
[Summary Rectified2] depth=2 | lambda_recon=1 | pretrain_val=0.060362 | best_val=0.562152
Test Recall@4: 0.1175
Tree-Embedding Correlation: 0.8805

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: ['C002', 'C342', 'C420', 'C424']
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: ['C040', 'C130', 'C141', 'C342']
  Visit 4: ['C102', 'C211', 'C422', 'C423']
  Visit 5: []
  Visit 6: ['C002', 'C041', 'C342', 'C343']

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []
Synthetic stats (N=1000): {'mean_depth': 1.9839483700148932, 'std_depth': 0.1256740830876755, 'mean_tree_dist': 2.616636528028933, 'std_tree_dist': 1.0083081132528877, 'mean_root_purity': 0.5324521813883516, 'std_root_purity': 0.16597733263999778}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth2_lrecon1_best0.5622.pt
[Rectified2] Epoch   1 | Train 11.093191 | Val 8.204793 | lambda_recon=10
[Rectified2] Epoch   2 | Train 7.178832 | Val 4.956927 | lambda_recon=10
[Rectified2] Epoch   3 | Train 5.286853 | Val 3.512350 | lambda_recon=10
[Rectified2] Epoch   4 | Train 4.257926 | Val 2.683633 | lambda_recon=10
[Rectified2] Epoch   5 | Train 3.584620 | Val 2.098241 | lambda_recon=10
[Rectified2] Epoch   6 | Train 3.151066 | Val 1.844136 | lambda_recon=10
[Rectified2] Epoch   7 | Train 2.837099 | Val 1.483223 | lambda_recon=10
[Rectified2] Epoch   8 | Train 2.586241 | Val 1.278677 | lambda_recon=10
[Rectified2] Epoch   9 | Train 2.396753 | Val 1.212936 | lambda_recon=10
[Rectified2] Epoch  10 | Train 2.246942 | Val 1.027637 | lambda_recon=10
[Rectified2] Epoch  11 | Train 2.156142 | Val 1.023302 | lambda_recon=10
[Rectified2] Epoch  12 | Train 2.060073 | Val 0.949758 | lambda_recon=10
[Rectified2] Epoch  13 | Train 1.998338 | Val 0.849015 | lambda_recon=10
[Rectified2] Epoch  14 | Train 1.944902 | Val 0.850554 | lambda_recon=10
[Rectified2] Epoch  15 | Train 1.897962 | Val 0.746572 | lambda_recon=10
[Rectified2] Epoch  16 | Train 1.873968 | Val 0.790701 | lambda_recon=10
[Rectified2] Epoch  17 | Train 1.860668 | Val 0.758174 | lambda_recon=10
[Rectified2] Epoch  18 | Train 1.836854 | Val 0.744518 | lambda_recon=10
[Rectified2] Epoch  19 | Train 1.795438 | Val 0.675654 | lambda_recon=10
[Rectified2] Epoch  20 | Train 1.781816 | Val 0.742499 | lambda_recon=10
[Rectified2] Epoch  21 | Train 1.775542 | Val 0.720115 | lambda_recon=10
[Rectified2] Epoch  22 | Train 1.741082 | Val 0.662328 | lambda_recon=10
[Rectified2] Epoch  23 | Train 1.745097 | Val 0.660007 | lambda_recon=10
[Rectified2] Epoch  24 | Train 1.740617 | Val 0.669234 | lambda_recon=10
[Rectified2] Epoch  25 | Train 1.721222 | Val 0.660411 | lambda_recon=10
[Rectified2] Epoch  26 | Train 1.721200 | Val 0.687626 | lambda_recon=10
[Rectified] Early stopping.
[Summary Rectified2] depth=2 | lambda_recon=10 | pretrain_val=0.060362 | best_val=0.660007
Test Recall@4: 0.1336
Tree-Embedding Correlation: 0.8847

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: ['C211', 'C212', 'C324', 'C344']
  Visit 2: []
  Visit 3: []
  Visit 4: ['C211', 'C212', 'C214', 'C324']
  Visit 5: []
  Visit 6: ['C041', 'C411', 'C412', 'C414']

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: ['C024', 'C210', 'C211', 'C444']
  Visit 6: ['C21', 'C211', 'C213', 'C214']
Synthetic stats (N=1000): {'mean_depth': 1.9798161830960535, 'std_depth': 0.14062869706832462, 'mean_tree_dist': 2.2285475094181666, 'std_tree_dist': 0.7979707047567156, 'mean_root_purity': 0.5689393939393939, 'std_root_purity': 0.17549929277894785}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth2_lrecon10_best0.6600.pt
[Rectified2] Epoch   1 | Train 11.693453 | Val 8.756420 | lambda_recon=100
[Rectified2] Epoch   2 | Train 7.729507 | Val 5.526466 | lambda_recon=100
[Rectified2] Epoch   3 | Train 5.887027 | Val 4.144538 | lambda_recon=100
[Rectified2] Epoch   4 | Train 4.838527 | Val 3.232200 | lambda_recon=100
[Rectified2] Epoch   5 | Train 4.160530 | Val 2.628844 | lambda_recon=100
[Rectified2] Epoch   6 | Train 3.654534 | Val 2.253476 | lambda_recon=100
[Rectified2] Epoch   7 | Train 3.332276 | Val 1.975345 | lambda_recon=100
[Rectified2] Epoch   8 | Train 3.096721 | Val 1.833979 | lambda_recon=100
[Rectified2] Epoch   9 | Train 2.906186 | Val 1.674238 | lambda_recon=100
[Rectified2] Epoch  10 | Train 2.756232 | Val 1.528955 | lambda_recon=100
[Rectified2] Epoch  11 | Train 2.650272 | Val 1.509215 | lambda_recon=100
[Rectified2] Epoch  12 | Train 2.568483 | Val 1.401598 | lambda_recon=100
[Rectified2] Epoch  13 | Train 2.513323 | Val 1.356266 | lambda_recon=100
[Rectified2] Epoch  14 | Train 2.464046 | Val 1.306152 | lambda_recon=100
[Rectified2] Epoch  15 | Train 2.415252 | Val 1.284838 | lambda_recon=100
[Rectified2] Epoch  16 | Train 2.391858 | Val 1.321021 | lambda_recon=100
[Rectified2] Epoch  17 | Train 2.377335 | Val 1.264652 | lambda_recon=100
[Rectified2] Epoch  18 | Train 2.361143 | Val 1.300653 | lambda_recon=100
[Rectified2] Epoch  19 | Train 2.365831 | Val 1.266100 | lambda_recon=100
[Rectified2] Epoch  20 | Train 2.343412 | Val 1.290039 | lambda_recon=100
[Rectified] Early stopping.
[Summary Rectified2] depth=2 | lambda_recon=100 | pretrain_val=0.060362 | best_val=1.264652
Test Recall@4: 0.1077
Tree-Embedding Correlation: 0.8767

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: ['C030', 'C034', 'C120', 'C330']
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: ['C030', 'C342', 'C343', 'C420']
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: ['C033', 'C104', 'C230', 'C232']
  Visit 2: ['C230', 'C232', 'C320']
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []
Synthetic stats (N=1000): {'mean_depth': 1.9737034331628927, 'std_depth': 0.16001580362479492, 'mean_tree_dist': 2.4849767245027508, 'std_tree_dist': 1.0002574094685424, 'mean_root_purity': 0.583276759447839, 'std_root_purity': 0.18875222168104142}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth2_lrecon100_best1.2647.pt
[Rectified2] Epoch   1 | Train 17.293804 | Val 14.536016 | lambda_recon=1000
[Rectified2] Epoch   2 | Train 13.537618 | Val 11.239821 | lambda_recon=1000
[Rectified2] Epoch   3 | Train 11.679422 | Val 9.814519 | lambda_recon=1000
[Rectified2] Epoch   4 | Train 10.574942 | Val 8.870958 | lambda_recon=1000
[Rectified2] Epoch   5 | Train 9.879821 | Val 8.354044 | lambda_recon=1000
[Rectified2] Epoch   6 | Train 9.385814 | Val 7.944115 | lambda_recon=1000
[Rectified2] Epoch   7 | Train 9.035405 | Val 7.626176 | lambda_recon=1000
[Rectified2] Epoch   8 | Train 8.769813 | Val 7.490275 | lambda_recon=1000
[Rectified2] Epoch   9 | Train 8.614548 | Val 7.378653 | lambda_recon=1000
[Rectified2] Epoch  10 | Train 8.492535 | Val 7.275343 | lambda_recon=1000
[Rectified2] Epoch  11 | Train 8.417933 | Val 7.272573 | lambda_recon=1000
[Rectified2] Epoch  12 | Train 8.336467 | Val 7.140536 | lambda_recon=1000
[Rectified2] Epoch  13 | Train 8.255619 | Val 7.107032 | lambda_recon=1000
[Rectified2] Epoch  14 | Train 8.216601 | Val 7.027874 | lambda_recon=1000
[Rectified2] Epoch  15 | Train 8.183396 | Val 7.058581 | lambda_recon=1000
[Rectified2] Epoch  16 | Train 8.160828 | Val 7.057253 | lambda_recon=1000
[Rectified2] Epoch  17 | Train 8.145941 | Val 7.000512 | lambda_recon=1000
[Rectified2] Epoch  18 | Train 8.116861 | Val 7.009328 | lambda_recon=1000
[Rectified2] Epoch  19 | Train 8.099947 | Val 6.993497 | lambda_recon=1000
[Rectified2] Epoch  20 | Train 8.095328 | Val 6.997138 | lambda_recon=1000
[Rectified2] Epoch  21 | Train 8.086973 | Val 7.018612 | lambda_recon=1000
[Rectified2] Epoch  22 | Train 8.076323 | Val 6.981856 | lambda_recon=1000
[Rectified2] Epoch  23 | Train 8.068259 | Val 6.989680 | lambda_recon=1000
[Rectified2] Epoch  24 | Train 8.054400 | Val 6.935212 | lambda_recon=1000
[Rectified2] Epoch  25 | Train 8.042196 | Val 6.944922 | lambda_recon=1000
[Rectified2] Epoch  26 | Train 8.051647 | Val 6.955503 | lambda_recon=1000
[Rectified2] Epoch  27 | Train 8.045519 | Val 6.952379 | lambda_recon=1000
[Rectified] Early stopping.
[Summary Rectified2] depth=2 | lambda_recon=1000 | pretrain_val=0.060362 | best_val=6.935212
Test Recall@4: 0.1357
Tree-Embedding Correlation: 0.8935

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: ['C202', 'C240', 'C241', 'C242']
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: ['C124', 'C202', 'C204', 'C310']
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: ['C201', 'C202', 'C204', 'C414']
Synthetic stats (N=1000): {'mean_depth': 1.9855977789345827, 'std_depth': 0.1191419199685827, 'mean_tree_dist': 2.485466377440347, 'std_tree_dist': 0.9526223934878459, 'mean_root_purity': 0.5484180790960451, 'std_root_purity': 0.16956614176767398}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth2_lrecon1000_best6.9352.pt
[Rectified2] Epoch   1 | Train 23.971447 | Val 20.418957 | lambda_recon=2000
[Rectified2] Epoch   2 | Train 19.563372 | Val 17.250988 | lambda_recon=2000
[Rectified2] Epoch   3 | Train 17.739535 | Val 15.876029 | lambda_recon=2000
[Rectified2] Epoch   4 | Train 16.760447 | Val 15.121411 | lambda_recon=2000
[Rectified2] Epoch   5 | Train 16.087759 | Val 14.577107 | lambda_recon=2000
[Rectified2] Epoch   6 | Train 15.642318 | Val 14.231246 | lambda_recon=2000
[Rectified2] Epoch   7 | Train 15.354064 | Val 13.962309 | lambda_recon=2000
[Rectified2] Epoch   8 | Train 15.126201 | Val 13.843311 | lambda_recon=2000
[Rectified2] Epoch   9 | Train 14.961206 | Val 13.660907 | lambda_recon=2000
[Rectified2] Epoch  10 | Train 14.849445 | Val 13.627011 | lambda_recon=2000
[Rectified2] Epoch  11 | Train 14.765903 | Val 13.596129 | lambda_recon=2000
[Rectified2] Epoch  12 | Train 14.680276 | Val 13.484669 | lambda_recon=2000
[Rectified2] Epoch  13 | Train 14.613083 | Val 13.455240 | lambda_recon=2000
[Rectified2] Epoch  14 | Train 14.594495 | Val 13.444100 | lambda_recon=2000
[Rectified2] Epoch  15 | Train 14.531101 | Val 13.363021 | lambda_recon=2000
[Rectified2] Epoch  16 | Train 14.514639 | Val 13.312105 | lambda_recon=2000
[Rectified2] Epoch  17 | Train 14.476764 | Val 13.271819 | lambda_recon=2000
[Rectified2] Epoch  18 | Train 14.442566 | Val 13.266017 | lambda_recon=2000
[Rectified2] Epoch  19 | Train 14.424983 | Val 13.297889 | lambda_recon=2000
[Rectified2] Epoch  20 | Train 14.424902 | Val 13.231951 | lambda_recon=2000
[Rectified2] Epoch  21 | Train 14.395749 | Val 13.284793 | lambda_recon=2000
[Rectified2] Epoch  22 | Train 14.397949 | Val 13.243967 | lambda_recon=2000
[Rectified2] Epoch  23 | Train 14.368012 | Val 13.270173 | lambda_recon=2000
[Rectified] Early stopping.
[Summary Rectified2] depth=2 | lambda_recon=2000 | pretrain_val=0.060362 | best_val=13.231951
Test Recall@4: 0.1019
Tree-Embedding Correlation: 0.8887

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: ['C030']
  Visit 4: []
  Visit 5: []
  Visit 6: ['C030', 'C233', 'C303', 'C322']

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: []
  Visit 2: []
  Visit 3: ['C023', 'C233', 'C301', 'C332']
  Visit 4: []
  Visit 5: []
  Visit 6: []
Synthetic stats (N=1000): {'mean_depth': 1.9863731656184487, 'std_depth': 0.1159359468253446, 'mean_tree_dist': 2.414383561643836, 'std_tree_dist': 0.9025450867975423, 'mean_root_purity': 0.5608260184559982, 'std_root_purity': 0.18235787093793127}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth2_lrecon2000_best13.2320.pt
[Depth 2] lambda_recon=1 | best_val=0.562152 | test_recall=0.1175 | corr=0.8805
[Depth 2] lambda_recon=10 | best_val=0.660007 | test_recall=0.1336 | corr=0.8847
[Depth 2] lambda_recon=100 | best_val=1.264652 | test_recall=0.1077 | corr=0.8767
[Depth 2] lambda_recon=1000 | best_val=6.935212 | test_recall=0.1357 | corr=0.8935
[Depth 2] lambda_recon=2000 | best_val=13.231951 | test_recall=0.1019 | corr=0.8887

rectified_depth7 | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

=== Pretraining hyperbolic graph embeddings (Rectified) ===
[Pretrain] Epoch   1 | train=0.095069 | val=0.090477 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   2 | train=0.091813 | val=0.088902 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   3 | train=0.087662 | val=0.084522 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   4 | train=0.086497 | val=0.084543 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   5 | train=0.084661 | val=0.081639 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   6 | train=0.082975 | val=0.081420 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   7 | train=0.080454 | val=0.080568 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   8 | train=0.080187 | val=0.079693 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch   9 | train=0.079140 | val=0.080395 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  10 | train=0.079211 | val=0.077424 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  11 | train=0.078340 | val=0.077078 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  12 | train=0.077373 | val=0.077230 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  13 | train=0.076149 | val=0.075864 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  14 | train=0.075968 | val=0.074769 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  15 | train=0.075146 | val=0.078178 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  16 | train=0.074985 | val=0.076310 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  17 | train=0.077149 | val=0.075030 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  18 | train=0.078939 | val=0.075144 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  19 | train=0.074306 | val=0.075525 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  20 | train=0.074301 | val=0.073499 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  21 | train=0.073459 | val=0.073875 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  22 | train=0.074290 | val=0.073085 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  23 | train=0.075479 | val=0.073858 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  24 | train=0.074229 | val=0.072835 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  25 | train=0.073432 | val=0.073580 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  26 | train=0.073085 | val=0.072849 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  27 | train=0.072401 | val=0.072421 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  28 | train=0.072548 | val=0.073398 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  29 | train=0.073348 | val=0.072271 | rad=0.003 pair=0.01 hdd=0.02
[Pretrain] Epoch  30 | train=0.071455 | val=0.071553 | rad=0.003 pair=0.01 hdd=0.02
Saved pretraining checkpoint to results/checkpoints/hyperbolic_rectified2_pretrain_rad0.003_pair0.01_hdd0.02_val0.0716.pt
[Rectified2] Epoch   1 | Train 10.824264 | Val 7.793879 | lambda_recon=1
[Rectified2] Epoch   2 | Train 6.854862 | Val 4.600181 | lambda_recon=1
[Rectified2] Epoch   3 | Train 5.027723 | Val 3.295943 | lambda_recon=1
[Rectified2] Epoch   4 | Train 4.009873 | Val 2.389118 | lambda_recon=1
[Rectified2] Epoch   5 | Train 3.342751 | Val 1.808503 | lambda_recon=1
[Rectified2] Epoch   6 | Train 2.912547 | Val 1.551886 | lambda_recon=1
[Rectified2] Epoch   7 | Train 2.596114 | Val 1.255178 | lambda_recon=1
[Rectified2] Epoch   8 | Train 2.362349 | Val 1.075556 | lambda_recon=1
[Rectified2] Epoch   9 | Train 2.197829 | Val 0.918530 | lambda_recon=1
[Rectified2] Epoch  10 | Train 2.048267 | Val 0.842959 | lambda_recon=1
[Rectified2] Epoch  11 | Train 1.960187 | Val 0.748739 | lambda_recon=1
[Rectified2] Epoch  12 | Train 1.891918 | Val 0.767025 | lambda_recon=1
[Rectified2] Epoch  13 | Train 1.867704 | Val 0.728979 | lambda_recon=1
[Rectified2] Epoch  14 | Train 1.812092 | Val 0.690050 | lambda_recon=1
[Rectified2] Epoch  15 | Train 1.775485 | Val 0.672836 | lambda_recon=1
[Rectified2] Epoch  16 | Train 1.765537 | Val 0.702425 | lambda_recon=1
[Rectified2] Epoch  17 | Train 1.750251 | Val 0.669185 | lambda_recon=1
[Rectified2] Epoch  18 | Train 1.761119 | Val 0.666147 | lambda_recon=1
[Rectified2] Epoch  19 | Train 1.721343 | Val 0.646848 | lambda_recon=1
[Rectified2] Epoch  20 | Train 1.696834 | Val 0.627995 | lambda_recon=1
[Rectified2] Epoch  21 | Train 1.705749 | Val 0.619285 | lambda_recon=1
[Rectified2] Epoch  22 | Train 1.694095 | Val 0.572205 | lambda_recon=1
[Rectified2] Epoch  23 | Train 1.678997 | Val 0.616757 | lambda_recon=1
[Rectified2] Epoch  24 | Train 1.663062 | Val 0.597565 | lambda_recon=1
[Rectified2] Epoch  25 | Train 1.661866 | Val 0.624707 | lambda_recon=1
[Rectified] Early stopping.
[Summary Rectified2] depth=7 | lambda_recon=1 | pretrain_val=0.071553 | best_val=0.572205
Test Recall@4: 0.0306
Tree-Embedding Correlation: 0.7413

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: ['C032d3', 'C241d3', 'C313d4', 'C402d2']
  Visit 4: ['C012d4', 'C123d4', 'C201d4', 'C404d2']
  Visit 5: ['C201d4', 'C221d4', 'C402d2', 'C423d4']
  Visit 6: []

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: []
  Visit 2: ['C244d4', 'C320d3', 'C404d3', 'C422d3']
  Visit 3: ['C020d4', 'C201d4', 'C233d1', 'C423d4']
  Visit 4: ['C010d1', 'C400d4', 'C402d2', 'C404d3']
  Visit 5: []
  Visit 6: ['C301d4', 'C321d3', 'C403d3', 'C404d2']
Synthetic stats (N=1000): {'mean_depth': 6.283285642147787, 'std_depth': 0.9952107385455388, 'mean_tree_dist': 11.979510244877561, 'std_tree_dist': 2.2346326696877434, 'mean_root_purity': 0.5209412780656304, 'std_root_purity': 0.1700924026878844}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth7_lrecon1_best0.5722.pt
[Rectified2] Epoch   1 | Train 10.963962 | Val 8.129722 | lambda_recon=10
[Rectified2] Epoch   2 | Train 7.113587 | Val 4.805132 | lambda_recon=10
[Rectified2] Epoch   3 | Train 5.188175 | Val 3.492071 | lambda_recon=10
[Rectified2] Epoch   4 | Train 4.124546 | Val 2.515365 | lambda_recon=10
[Rectified2] Epoch   5 | Train 3.449696 | Val 1.974687 | lambda_recon=10
[Rectified2] Epoch   6 | Train 2.957900 | Val 1.534498 | lambda_recon=10
[Rectified2] Epoch   7 | Train 2.644915 | Val 1.304418 | lambda_recon=10
[Rectified2] Epoch   8 | Train 2.411828 | Val 1.122887 | lambda_recon=10
[Rectified2] Epoch   9 | Train 2.245264 | Val 0.973401 | lambda_recon=10
[Rectified2] Epoch  10 | Train 2.125239 | Val 0.908239 | lambda_recon=10
[Rectified2] Epoch  11 | Train 2.015446 | Val 0.858465 | lambda_recon=10
[Rectified2] Epoch  12 | Train 1.960404 | Val 0.807525 | lambda_recon=10
[Rectified2] Epoch  13 | Train 1.898114 | Val 0.745220 | lambda_recon=10
[Rectified2] Epoch  14 | Train 1.848255 | Val 0.714311 | lambda_recon=10
[Rectified2] Epoch  15 | Train 1.815746 | Val 0.697923 | lambda_recon=10
[Rectified2] Epoch  16 | Train 1.786725 | Val 0.656262 | lambda_recon=10
[Rectified2] Epoch  17 | Train 1.765395 | Val 0.686592 | lambda_recon=10
[Rectified2] Epoch  18 | Train 1.757234 | Val 0.668650 | lambda_recon=10
[Rectified2] Epoch  19 | Train 1.750226 | Val 0.624817 | lambda_recon=10
[Rectified2] Epoch  20 | Train 1.735996 | Val 0.690055 | lambda_recon=10
[Rectified2] Epoch  21 | Train 1.715221 | Val 0.669155 | lambda_recon=10
[Rectified2] Epoch  22 | Train 1.701866 | Val 0.614501 | lambda_recon=10
[Rectified2] Epoch  23 | Train 1.701724 | Val 0.591611 | lambda_recon=10
[Rectified2] Epoch  24 | Train 1.670776 | Val 0.548235 | lambda_recon=10
[Rectified2] Epoch  25 | Train 1.657728 | Val 0.599205 | lambda_recon=10
[Rectified2] Epoch  26 | Train 1.641583 | Val 0.662937 | lambda_recon=10
[Rectified2] Epoch  27 | Train 1.631560 | Val 0.579062 | lambda_recon=10
[Rectified] Early stopping.
[Summary Rectified2] depth=7 | lambda_recon=10 | pretrain_val=0.071553 | best_val=0.548235
Test Recall@4: 0.0213
Tree-Embedding Correlation: 0.7508

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: ['C030d4', 'C123d3', 'C142d3', 'C341d3']
  Visit 6: []

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []
Synthetic stats (N=1000): {'mean_depth': 6.223664959765911, 'std_depth': 0.7972449077783269, 'mean_tree_dist': 11.832786885245902, 'std_tree_dist': 2.146267924505799, 'mean_root_purity': 0.46813725490196073, 'std_root_purity': 0.14957006178816037}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth7_lrecon10_best0.5482.pt
[Rectified2] Epoch   1 | Train 11.186593 | Val 8.261639 | lambda_recon=100
[Rectified2] Epoch   2 | Train 7.195861 | Val 5.013491 | lambda_recon=100
[Rectified2] Epoch   3 | Train 5.334876 | Val 3.514724 | lambda_recon=100
[Rectified2] Epoch   4 | Train 4.305306 | Val 2.742418 | lambda_recon=100
[Rectified2] Epoch   5 | Train 3.599083 | Val 2.144391 | lambda_recon=100
[Rectified2] Epoch   6 | Train 3.130973 | Val 1.727136 | lambda_recon=100
[Rectified2] Epoch   7 | Train 2.811656 | Val 1.460253 | lambda_recon=100
[Rectified2] Epoch   8 | Train 2.576583 | Val 1.278715 | lambda_recon=100
[Rectified2] Epoch   9 | Train 2.392934 | Val 1.196789 | lambda_recon=100
[Rectified2] Epoch  10 | Train 2.278340 | Val 1.101850 | lambda_recon=100
[Rectified2] Epoch  11 | Train 2.195332 | Val 1.001285 | lambda_recon=100
[Rectified2] Epoch  12 | Train 2.123525 | Val 1.021975 | lambda_recon=100
[Rectified2] Epoch  13 | Train 2.086022 | Val 0.953581 | lambda_recon=100
[Rectified2] Epoch  14 | Train 2.036272 | Val 0.885605 | lambda_recon=100
[Rectified2] Epoch  15 | Train 1.999668 | Val 0.897873 | lambda_recon=100
[Rectified2] Epoch  16 | Train 1.973016 | Val 0.853528 | lambda_recon=100
[Rectified2] Epoch  17 | Train 1.949415 | Val 0.842626 | lambda_recon=100
[Rectified2] Epoch  18 | Train 1.929326 | Val 0.862423 | lambda_recon=100
[Rectified2] Epoch  19 | Train 1.914107 | Val 0.850726 | lambda_recon=100
[Rectified2] Epoch  20 | Train 1.901152 | Val 0.781256 | lambda_recon=100
[Rectified2] Epoch  21 | Train 1.910149 | Val 0.797493 | lambda_recon=100
[Rectified2] Epoch  22 | Train 1.870574 | Val 0.831278 | lambda_recon=100
[Rectified2] Epoch  23 | Train 1.869208 | Val 0.814963 | lambda_recon=100
[Rectified] Early stopping.
[Summary Rectified2] depth=7 | lambda_recon=100 | pretrain_val=0.071553 | best_val=0.781256
Test Recall@4: 0.0320
Tree-Embedding Correlation: 0.7248

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: []
  Visit 3: ['C221d3', 'C232d4', 'C242d4', 'C311d4']
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: ['C230d4', 'C244d2', 'C413d3', 'C434d3']
  Visit 2: []
  Visit 3: ['C100d4', 'C202d4']
  Visit 4: []
  Visit 5: []
  Visit 6: ['C014d4', 'C100d4', 'C120d3', 'C331d3']
Synthetic stats (N=1000): {'mean_depth': 6.336462699077955, 'std_depth': 1.1880742525889543, 'mean_tree_dist': 12.261800219538967, 'std_tree_dist': 1.873045456978486, 'mean_root_purity': 0.5016181229773463, 'std_root_purity': 0.15225468174924475}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth7_lrecon100_best0.7813.pt
[Rectified2] Epoch   1 | Train 12.651691 | Val 9.635859 | lambda_recon=1000
[Rectified2] Epoch   2 | Train 8.724379 | Val 6.528510 | lambda_recon=1000
[Rectified2] Epoch   3 | Train 6.915839 | Val 5.145035 | lambda_recon=1000
[Rectified2] Epoch   4 | Train 5.916284 | Val 4.275442 | lambda_recon=1000
[Rectified2] Epoch   5 | Train 5.253707 | Val 3.803931 | lambda_recon=1000
[Rectified2] Epoch   6 | Train 4.782750 | Val 3.406055 | lambda_recon=1000
[Rectified2] Epoch   7 | Train 4.473085 | Val 3.137251 | lambda_recon=1000
[Rectified2] Epoch   8 | Train 4.264191 | Val 2.994533 | lambda_recon=1000
[Rectified2] Epoch   9 | Train 4.085048 | Val 2.883726 | lambda_recon=1000
[Rectified2] Epoch  10 | Train 3.952334 | Val 2.735972 | lambda_recon=1000
[Rectified2] Epoch  11 | Train 3.854344 | Val 2.678923 | lambda_recon=1000
[Rectified2] Epoch  12 | Train 3.793737 | Val 2.619355 | lambda_recon=1000
[Rectified2] Epoch  13 | Train 3.716479 | Val 2.626657 | lambda_recon=1000
[Rectified2] Epoch  14 | Train 3.676378 | Val 2.612858 | lambda_recon=1000
[Rectified2] Epoch  15 | Train 3.658057 | Val 2.555698 | lambda_recon=1000
[Rectified2] Epoch  16 | Train 3.646515 | Val 2.565917 | lambda_recon=1000
[Rectified2] Epoch  17 | Train 3.619155 | Val 2.556831 | lambda_recon=1000
[Rectified2] Epoch  18 | Train 3.612659 | Val 2.540690 | lambda_recon=1000
[Rectified2] Epoch  19 | Train 3.560956 | Val 2.525920 | lambda_recon=1000
[Rectified2] Epoch  20 | Train 3.555022 | Val 2.513182 | lambda_recon=1000
[Rectified2] Epoch  21 | Train 3.527564 | Val 2.517377 | lambda_recon=1000
[Rectified2] Epoch  22 | Train 3.518781 | Val 2.470027 | lambda_recon=1000
[Rectified2] Epoch  23 | Train 3.499616 | Val 2.437450 | lambda_recon=1000
[Rectified2] Epoch  24 | Train 3.499767 | Val 2.432548 | lambda_recon=1000
[Rectified2] Epoch  25 | Train 3.475816 | Val 2.434005 | lambda_recon=1000
[Rectified2] Epoch  26 | Train 3.460570 | Val 2.448164 | lambda_recon=1000
[Rectified2] Epoch  27 | Train 3.464939 | Val 2.428555 | lambda_recon=1000
[Rectified2] Epoch  28 | Train 3.450489 | Val 2.436580 | lambda_recon=1000
[Rectified2] Epoch  29 | Train 3.448875 | Val 2.452123 | lambda_recon=1000
[Rectified2] Epoch  30 | Train 3.437312 | Val 2.392798 | lambda_recon=1000
[Rectified2] Epoch  31 | Train 3.426361 | Val 2.411514 | lambda_recon=1000
[Rectified2] Epoch  32 | Train 3.404441 | Val 2.414050 | lambda_recon=1000
[Rectified2] Epoch  33 | Train 3.413993 | Val 2.456005 | lambda_recon=1000
[Rectified] Early stopping.
[Summary Rectified2] depth=7 | lambda_recon=1000 | pretrain_val=0.071553 | best_val=2.392798
Test Recall@4: 0.0479
Tree-Embedding Correlation: 0.7419

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: ['C003d4', 'C041d3', 'C214d3', 'C430d1']
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: ['C010d4', 'C223d3', 'C412d4', 'C442d3']
  Visit 5: []
  Visit 6: []

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: []
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []
Synthetic stats (N=1000): {'mean_depth': 6.1306634990666895, 'std_depth': 0.7899535807362865, 'mean_tree_dist': 11.642487046632125, 'std_tree_dist': 2.3394915596351766, 'mean_root_purity': 0.47581699346405226, 'std_root_purity': 0.15552568780101536}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth7_lrecon1000_best2.3928.pt
[Rectified2] Epoch   1 | Train 15.052772 | Val 12.061582 | lambda_recon=2000
[Rectified2] Epoch   2 | Train 10.765529 | Val 8.555683 | lambda_recon=2000
[Rectified2] Epoch   3 | Train 8.819457 | Val 7.081427 | lambda_recon=2000
[Rectified2] Epoch   4 | Train 7.717033 | Val 6.135884 | lambda_recon=2000
[Rectified2] Epoch   5 | Train 7.030365 | Val 5.555507 | lambda_recon=2000
[Rectified2] Epoch   6 | Train 6.582458 | Val 5.220250 | lambda_recon=2000
[Rectified2] Epoch   7 | Train 6.257887 | Val 4.911116 | lambda_recon=2000
[Rectified2] Epoch   8 | Train 6.037751 | Val 4.835864 | lambda_recon=2000
[Rectified2] Epoch   9 | Train 5.851951 | Val 4.681754 | lambda_recon=2000
[Rectified2] Epoch  10 | Train 5.745426 | Val 4.577305 | lambda_recon=2000
[Rectified2] Epoch  11 | Train 5.634146 | Val 4.468191 | lambda_recon=2000
[Rectified2] Epoch  12 | Train 5.532251 | Val 4.392106 | lambda_recon=2000
[Rectified2] Epoch  13 | Train 5.483748 | Val 4.376003 | lambda_recon=2000
[Rectified2] Epoch  14 | Train 5.451357 | Val 4.346877 | lambda_recon=2000
[Rectified2] Epoch  15 | Train 5.419411 | Val 4.314018 | lambda_recon=2000
[Rectified2] Epoch  16 | Train 5.402921 | Val 4.356498 | lambda_recon=2000
[Rectified2] Epoch  17 | Train 5.370280 | Val 4.280576 | lambda_recon=2000
[Rectified2] Epoch  18 | Train 5.372886 | Val 4.299408 | lambda_recon=2000
[Rectified2] Epoch  19 | Train 5.338804 | Val 4.269233 | lambda_recon=2000
[Rectified2] Epoch  20 | Train 5.325673 | Val 4.316064 | lambda_recon=2000
[Rectified2] Epoch  21 | Train 5.336150 | Val 4.311011 | lambda_recon=2000
[Rectified2] Epoch  22 | Train 5.309327 | Val 4.274678 | lambda_recon=2000
[Rectified] Early stopping.
[Summary Rectified2] depth=7 | lambda_recon=2000 | pretrain_val=0.071553 | best_val=4.269233
Test Recall@4: 0.0275
Tree-Embedding Correlation: 0.7355

[Rectified-Hyp] Sample trajectory 1:
  Visit 1: []
  Visit 2: []
  Visit 3: ['C014d4', 'C142d4', 'C403d2', 'C432d4']
  Visit 4: ['C034d4', 'C403d3', 'C420d4']
  Visit 5: ['C011d4', 'C013d4', 'C103d2', 'C142d4']
  Visit 6: ['C120d4', 'C211d3', 'C330d2', 'C342d4']

[Rectified-Hyp] Sample trajectory 2:
  Visit 1: ['C142d4', 'C241d3']
  Visit 2: []
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: ['C142d4', 'C403d3', 'C420d4']

[Rectified-Hyp] Sample trajectory 3:
  Visit 1: ['C200d4']
  Visit 2: ['C011d4', 'C032d3', 'C142d4', 'C211d3']
  Visit 3: []
  Visit 4: []
  Visit 5: []
  Visit 6: []
Synthetic stats (N=1000): {'mean_depth': 6.342504913346436, 'std_depth': 0.771154296905873, 'mean_tree_dist': 12.321215409658166, 'std_tree_dist': 2.1131632096619213, 'mean_root_purity': 0.5221258616855682, 'std_root_purity': 0.1736678129350323}
Saved rectified model checkpoint to results/checkpoints/graph_rectified2_depth7_lrecon2000_best4.2692.pt
[Depth 7] lambda_recon=1 | best_val=0.572205 | test_recall=0.0306 | corr=0.7413
[Depth 7] lambda_recon=10 | best_val=0.548235 | test_recall=0.0213 | corr=0.7508
[Depth 7] lambda_recon=100 | best_val=0.781256 | test_recall=0.0320 | corr=0.7248
[Depth 7] lambda_recon=1000 | best_val=2.392798 | test_recall=0.0479 | corr=0.7419
[Depth 7] lambda_recon=2000 | best_val=4.269233 | test_recall=0.0275 | corr=0.7355
