Using device: mps

depth2_final | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

--- Running hyperbolic ---

Training HYPERBOLIC rectified | Depth 2 | lambda_recon=2000.0
Epoch   1 | Train 35.93400 | Val 26.91032 | lambda_recon=2000.0
Epoch   2 | Train 26.43561 | Val 21.91786 | lambda_recon=2000.0
Epoch   3 | Train 22.87851 | Val 18.90509 | lambda_recon=2000.0
Epoch   4 | Train 20.65067 | Val 16.71894 | lambda_recon=2000.0
Epoch   5 | Train 18.45320 | Val 15.11304 | lambda_recon=2000.0
Epoch   6 | Train 17.24818 | Val 14.15702 | lambda_recon=2000.0
Epoch   7 | Train 16.00480 | Val 12.48881 | lambda_recon=2000.0
Epoch   8 | Train 15.11096 | Val 11.80946 | lambda_recon=2000.0
Epoch   9 | Train 14.47593 | Val 11.57178 | lambda_recon=2000.0
Epoch  10 | Train 14.08944 | Val 11.02502 | lambda_recon=2000.0
Epoch  11 | Train 13.81104 | Val 11.05768 | lambda_recon=2000.0
Epoch  12 | Train 13.54385 | Val 10.89625 | lambda_recon=2000.0
Epoch  13 | Train 13.33847 | Val 10.79668 | lambda_recon=2000.0
Epoch  14 | Train 13.26398 | Val 10.41753 | lambda_recon=2000.0
Epoch  15 | Train 13.09934 | Val 10.35032 | lambda_recon=2000.0
Epoch  16 | Train 12.95478 | Val 10.29768 | lambda_recon=2000.0
Epoch  17 | Train 12.81363 | Val 10.25280 | lambda_recon=2000.0
Epoch  18 | Train 12.63880 | Val 9.95761 | lambda_recon=2000.0
Epoch  19 | Train 12.48881 | Val 9.84464 | lambda_recon=2000.0
Epoch  20 | Train 12.35595 | Val 9.57359 | lambda_recon=2000.0
Epoch  21 | Train 12.15051 | Val 9.63024 | lambda_recon=2000.0
Epoch  22 | Train 11.99568 | Val 9.41113 | lambda_recon=2000.0
Epoch  23 | Train 11.98820 | Val 9.35395 | lambda_recon=2000.0
Epoch  24 | Train 11.85744 | Val 9.33325 | lambda_recon=2000.0
Epoch  25 | Train 11.87287 | Val 9.13755 | lambda_recon=2000.0
Epoch  26 | Train 11.76398 | Val 9.15992 | lambda_recon=2000.0
Epoch  27 | Train 11.65323 | Val 9.06577 | lambda_recon=2000.0
Epoch  28 | Train 11.63794 | Val 9.14308 | lambda_recon=2000.0
Epoch  29 | Train 11.60235 | Val 9.01450 | lambda_recon=2000.0
Epoch  30 | Train 11.53285 | Val 8.97759 | lambda_recon=2000.0
Epoch  31 | Train 11.49574 | Val 8.93823 | lambda_recon=2000.0
Epoch  32 | Train 11.51547 | Val 8.94672 | lambda_recon=2000.0
Epoch  33 | Train 11.45154 | Val 9.04013 | lambda_recon=2000.0
Epoch  34 | Train 11.42428 | Val 8.98893 | lambda_recon=2000.0
Epoch  35 | Train 11.38297 | Val 9.03452 | lambda_recon=2000.0
Epoch  36 | Train 11.36932 | Val 8.95348 | lambda_recon=2000.0
Early stopping triggered.
Best validation loss (lambda_recon=2000.0): 8.938228
Test Recall@4 (lambda_recon=2000.0): 0.5743

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C011', 'C120', 'C123', 'C221']
  Visit 2: ['C044', 'C100', 'C3', 'C314']
  Visit 3: ['C044', 'C10', 'C101', 'C102']
  Visit 4: ['C120', 'C123', 'C423', 'C424']
  Visit 5: ['C044', 'C10', 'C101', 'C102']
  Visit 6: ['C104', 'C122', 'C244', 'C434']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C04', 'C044', 'C101', 'C102']
  Visit 2: ['C104', 'C203', 'C231', 'C421']
  Visit 3: ['C011', 'C120', 'C123', 'C221']
  Visit 4: ['C044', 'C102', 'C112', 'C314']
  Visit 5: ['C011', 'C123', 'C421', 'C424']
  Visit 6: ['C011', 'C12', 'C123', 'C220']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C013', 'C42', 'C420', 'C424']
  Visit 2: ['C013', 'C2', 'C322', 'C420']
  Visit 3: ['C044', 'C10', 'C101', 'C102']
  Visit 4: ['C011', 'C12', 'C123', 'C424']
  Visit 5: ['C123', 'C42', 'C421', 'C424']
  Visit 6: ['C013', 'C32', 'C322', 'C324']
Tree-Embedding Correlation (lambda_recon=2000.0): 0.6162
Synthetic (hyperbolic, lambda_recon=2000.0) stats (N=1000): {'mean_depth': 1.815347421949898, 'std_depth': 0.4378791082336013, 'mean_tree_dist': 1.8659919028340082, 'std_tree_dist': 1.0043313726093148, 'mean_root_purity': 0.6015972222222221, 'std_root_purity': 0.17482397176415784}
[Summary] depth2_final | hyperbolic | lambda_recon=2000.0: best_val=8.938228, test_recall=0.5743, corr=0.6162

depth7_final | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

--- Running hyperbolic ---

Training HYPERBOLIC rectified | Depth 7 | lambda_recon=2000.0
Epoch   1 | Train 26.48497 | Val 18.24960 | lambda_recon=2000.0
Epoch   2 | Train 17.83418 | Val 13.63047 | lambda_recon=2000.0
Epoch   3 | Train 14.88944 | Val 11.58137 | lambda_recon=2000.0
Epoch   4 | Train 13.50753 | Val 10.50229 | lambda_recon=2000.0
Epoch   5 | Train 12.26280 | Val 8.95516 | lambda_recon=2000.0
Epoch   6 | Train 11.26404 | Val 8.06386 | lambda_recon=2000.0
Epoch   7 | Train 10.71778 | Val 7.77180 | lambda_recon=2000.0
Epoch   8 | Train 10.39005 | Val 7.44071 | lambda_recon=2000.0
Epoch   9 | Train 9.93259 | Val 6.81039 | lambda_recon=2000.0
Epoch  10 | Train 9.46437 | Val 6.44184 | lambda_recon=2000.0
Epoch  11 | Train 9.20185 | Val 6.42384 | lambda_recon=2000.0
Epoch  12 | Train 9.02297 | Val 6.23010 | lambda_recon=2000.0
Epoch  13 | Train 8.88162 | Val 6.18626 | lambda_recon=2000.0
Epoch  14 | Train 8.82218 | Val 6.14749 | lambda_recon=2000.0
Epoch  15 | Train 8.67916 | Val 6.02722 | lambda_recon=2000.0
Epoch  16 | Train 8.61529 | Val 5.93877 | lambda_recon=2000.0
Epoch  17 | Train 8.49744 | Val 5.90972 | lambda_recon=2000.0
Epoch  18 | Train 8.41584 | Val 5.97258 | lambda_recon=2000.0
Epoch  19 | Train 8.36954 | Val 5.70403 | lambda_recon=2000.0
Epoch  20 | Train 8.16837 | Val 5.34885 | lambda_recon=2000.0
Epoch  21 | Train 8.02409 | Val 5.23536 | lambda_recon=2000.0
Epoch  22 | Train 7.86733 | Val 5.08369 | lambda_recon=2000.0
Epoch  23 | Train 7.79926 | Val 5.08411 | lambda_recon=2000.0
Epoch  24 | Train 7.67448 | Val 5.04088 | lambda_recon=2000.0
Epoch  25 | Train 7.57390 | Val 4.92218 | lambda_recon=2000.0
Epoch  26 | Train 7.49940 | Val 5.14891 | lambda_recon=2000.0
Epoch  27 | Train 7.43895 | Val 4.89296 | lambda_recon=2000.0
Epoch  28 | Train 7.34054 | Val 4.86624 | lambda_recon=2000.0
Epoch  29 | Train 7.36107 | Val 4.76202 | lambda_recon=2000.0
Epoch  30 | Train 7.26509 | Val 4.69313 | lambda_recon=2000.0
Epoch  31 | Train 7.21374 | Val 4.70492 | lambda_recon=2000.0
Epoch  32 | Train 7.17311 | Val 4.67975 | lambda_recon=2000.0
Epoch  33 | Train 7.11796 | Val 4.62195 | lambda_recon=2000.0
Epoch  34 | Train 7.02372 | Val 4.56454 | lambda_recon=2000.0
Epoch  35 | Train 6.96368 | Val 4.40130 | lambda_recon=2000.0
Epoch  36 | Train 6.92641 | Val 4.38446 | lambda_recon=2000.0
Epoch  37 | Train 6.88990 | Val 4.37315 | lambda_recon=2000.0
Epoch  38 | Train 6.84247 | Val 4.29394 | lambda_recon=2000.0
Epoch  39 | Train 6.79244 | Val 4.27085 | lambda_recon=2000.0
Epoch  40 | Train 6.74142 | Val 4.40561 | lambda_recon=2000.0
Epoch  41 | Train 6.76905 | Val 4.36765 | lambda_recon=2000.0
Epoch  42 | Train 6.72291 | Val 4.30410 | lambda_recon=2000.0
Epoch  43 | Train 6.69263 | Val 4.18536 | lambda_recon=2000.0
Epoch  44 | Train 6.68627 | Val 4.11924 | lambda_recon=2000.0
Epoch  45 | Train 6.71917 | Val 4.25116 | lambda_recon=2000.0
Epoch  46 | Train 6.74207 | Val 4.23951 | lambda_recon=2000.0
Epoch  47 | Train 6.77710 | Val 4.28699 | lambda_recon=2000.0
Epoch  48 | Train 6.83878 | Val 4.36717 | lambda_recon=2000.0
Epoch  49 | Train 7.02061 | Val 4.59737 | lambda_recon=2000.0
Early stopping triggered.
Best validation loss (lambda_recon=2000.0): 4.119239
Test Recall@4 (lambda_recon=2000.0): 0.4277

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C311d3', 'C311d4', 'C402d3', 'C402d4']
  Visit 2: ['C402d3', 'C402d4', 'C442d3', 'C442d4']
  Visit 3: ['C010d3', 'C010d4', 'C244d3', 'C244d4']
  Visit 4: ['C402d3', 'C402d4', 'C442d3', 'C442d4']
  Visit 5: ['C042d3', 'C042d4', 'C402d3', 'C402d4']
  Visit 6: ['C143d3', 'C143d4', 'C302d3', 'C302d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C013d4', 'C042d3', 'C042d4', 'C402d4']
  Visit 2: ['C013d4', 'C042d3', 'C042d4', 'C402d4']
  Visit 3: ['C143d3', 'C143d4', 'C302d3', 'C302d4']
  Visit 4: ['C143d3', 'C143d4', 'C243d0', 'C302d3']
  Visit 5: ['C010d3', 'C244d4', 'C343d3', 'C343d4']
  Visit 6: ['C042d3', 'C042d4', 'C402d3', 'C402d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C010d3', 'C402d3', 'C402d4', 'C442d3']
  Visit 2: ['C143d3', 'C143d4', 'C302d3', 'C302d4']
  Visit 3: ['C041d3', 'C042d3', 'C042d4', 'C234d4']
  Visit 4: ['C001d4', 'C003d3', 'C314d3', 'C444d2']
  Visit 5: ['C043d3', 'C043d4', 'C143d3', 'C143d4']
  Visit 6: ['C041d3', 'C041d4', 'C042d4', 'C104d4']
Tree-Embedding Correlation (lambda_recon=2000.0): 0.0973
Synthetic (hyperbolic, lambda_recon=2000.0) stats (N=1000): {'mean_depth': 6.250875, 'std_depth': 1.0172774454600213, 'mean_tree_dist': 4.618560949476634, 'std_tree_dist': 5.321639475311989, 'mean_root_purity': 0.5625416666666667, 'std_root_purity': 0.1450395001274557}
[Summary] depth7_final | hyperbolic | lambda_recon=2000.0: best_val=4.119239, test_recall=0.4277, corr=0.0973


Using device: mps

hyperbolic_ddpm_depth7 | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

Training hyperbolic DDPM | lambda_recon=500
Epoch   1 | Train 97.557543 | Val 34.453503 | lambda_recon=500 | lambda_pair=0.01
Epoch   2 | Train 30.563920 | Val 25.997929 | lambda_recon=500 | lambda_pair=0.01
Epoch   3 | Train 26.305883 | Val 22.971406 | lambda_recon=500 | lambda_pair=0.01
Epoch   4 | Train 24.048435 | Val 21.523698 | lambda_recon=500 | lambda_pair=0.01
Epoch   5 | Train 22.984067 | Val 20.498504 | lambda_recon=500 | lambda_pair=0.01
Epoch   6 | Train 21.950762 | Val 20.117552 | lambda_recon=500 | lambda_pair=0.01
Epoch   7 | Train 21.531864 | Val 19.866097 | lambda_recon=500 | lambda_pair=0.01
Epoch   8 | Train 20.843664 | Val 19.024115 | lambda_recon=500 | lambda_pair=0.01
Epoch   9 | Train 20.174478 | Val 18.227795 | lambda_recon=500 | lambda_pair=0.01
Epoch  10 | Train 19.343157 | Val 17.842103 | lambda_recon=500 | lambda_pair=0.01
Epoch  11 | Train 19.069674 | Val 17.712096 | lambda_recon=500 | lambda_pair=0.01
Epoch  12 | Train 18.839719 | Val 17.619236 | lambda_recon=500 | lambda_pair=0.01
Epoch  13 | Train 18.315740 | Val 16.726358 | lambda_recon=500 | lambda_pair=0.01
Epoch  14 | Train 17.818719 | Val 16.595808 | lambda_recon=500 | lambda_pair=0.01
Epoch  15 | Train 17.649633 | Val 16.610942 | lambda_recon=500 | lambda_pair=0.01
Epoch  16 | Train 17.560386 | Val 16.588306 | lambda_recon=500 | lambda_pair=0.01
Epoch  17 | Train 17.451591 | Val 16.545613 | lambda_recon=500 | lambda_pair=0.01
Epoch  18 | Train 17.398724 | Val 16.524720 | lambda_recon=500 | lambda_pair=0.01
Epoch  19 | Train 17.348162 | Val 16.512965 | lambda_recon=500 | lambda_pair=0.01
Epoch  20 | Train 17.276535 | Val 16.471351 | lambda_recon=500 | lambda_pair=0.01
Epoch  21 | Train 17.208459 | Val 16.521862 | lambda_recon=500 | lambda_pair=0.01
Epoch  22 | Train 17.160286 | Val 16.515749 | lambda_recon=500 | lambda_pair=0.01
Epoch  23 | Train 17.123188 | Val 16.520894 | lambda_recon=500 | lambda_pair=0.01
Epoch  24 | Train 17.063306 | Val 16.444642 | lambda_recon=500 | lambda_pair=0.01
Epoch  25 | Train 16.960790 | Val 15.980995 | lambda_recon=500 | lambda_pair=0.01
Epoch  26 | Train 16.428791 | Val 15.798514 | lambda_recon=500 | lambda_pair=0.01
Epoch  27 | Train 16.268360 | Val 15.666840 | lambda_recon=500 | lambda_pair=0.01
Epoch  28 | Train 16.202412 | Val 15.692155 | lambda_recon=500 | lambda_pair=0.01
Epoch  29 | Train 16.155677 | Val 15.620040 | lambda_recon=500 | lambda_pair=0.01
Epoch  30 | Train 16.118603 | Val 15.638631 | lambda_recon=500 | lambda_pair=0.01
Epoch  31 | Train 16.053028 | Val 15.636053 | lambda_recon=500 | lambda_pair=0.01
Epoch  32 | Train 16.047897 | Val 15.621720 | lambda_recon=500 | lambda_pair=0.01
Epoch  33 | Train 15.987230 | Val 15.635725 | lambda_recon=500 | lambda_pair=0.01
Epoch  34 | Train 15.936149 | Val 15.581834 | lambda_recon=500 | lambda_pair=0.01
Epoch  35 | Train 15.904376 | Val 15.519096 | lambda_recon=500 | lambda_pair=0.01
Epoch  36 | Train 15.874176 | Val 15.540523 | lambda_recon=500 | lambda_pair=0.01
Epoch  37 | Train 15.865997 | Val 15.556458 | lambda_recon=500 | lambda_pair=0.01
Epoch  38 | Train 15.847354 | Val 15.569513 | lambda_recon=500 | lambda_pair=0.01
Epoch  39 | Train 15.828001 | Val 15.561683 | lambda_recon=500 | lambda_pair=0.01
Epoch  40 | Train 15.800570 | Val 15.551645 | lambda_recon=500 | lambda_pair=0.01
Epoch  41 | Train 15.791753 | Val 15.510556 | lambda_recon=500 | lambda_pair=0.01
Epoch  42 | Train 15.795891 | Val 15.516504 | lambda_recon=500 | lambda_pair=0.01
Epoch  43 | Train 15.761975 | Val 15.520716 | lambda_recon=500 | lambda_pair=0.01
Epoch  44 | Train 15.766080 | Val 15.543693 | lambda_recon=500 | lambda_pair=0.01
Epoch  45 | Train 15.760716 | Val 15.530975 | lambda_recon=500 | lambda_pair=0.01
Epoch  46 | Train 15.749053 | Val 15.521648 | lambda_recon=500 | lambda_pair=0.01
Epoch  47 | Train 15.750366 | Val 15.511832 | lambda_recon=500 | lambda_pair=0.01
Epoch  48 | Train 15.747772 | Val 15.522687 | lambda_recon=500 | lambda_pair=0.01
Epoch  49 | Train 15.735498 | Val 15.544676 | lambda_recon=500 | lambda_pair=0.01
Epoch  50 | Train 15.731382 | Val 15.533704 | lambda_recon=500 | lambda_pair=0.01
Best validation loss: 15.510556
Test Recall@4: 0.0099

Sample hyperbolic trajectory 1:
  Visit 1: ['C032d2', 'C230d2', 'C321d2', 'C330d2']
  Visit 2: ['C032d2', 'C230d2', 'C321d2', 'C330d2']
  Visit 3: ['C102d4', 'C301d3', 'C404d3', 'C413d4']
  Visit 4: ['C032d2', 'C230d2', 'C321d2', 'C330d2']
  Visit 5: ['C032d2', 'C230d2', 'C321d2', 'C330d2']
  Visit 6: ['C032d2', 'C230d2', 'C321d2', 'C330d2']

Sample hyperbolic trajectory 2:
  Visit 1: ['C023d2', 'C032d2', 'C230d2', 'C330d2']
  Visit 2: ['C102d4', 'C301d3', 'C404d3', 'C413d4']
  Visit 3: ['C023d2', 'C032d2', 'C230d2', 'C330d2']
  Visit 4: ['C102d4', 'C301d3', 'C404d3', 'C413d4']
  Visit 5: ['C023d2', 'C032d2', 'C230d2', 'C330d2']
  Visit 6: ['C102d4', 'C301d3', 'C404d3', 'C413d4']

Sample hyperbolic trajectory 3:
  Visit 1: ['C023d2', 'C032d2', 'C230d2', 'C330d2']
  Visit 2: ['C102d4', 'C301d3', 'C404d3', 'C413d4']
  Visit 3: ['C102d4', 'C301d3', 'C404d3', 'C413d4']
  Visit 4: ['C023d2', 'C032d2', 'C230d2', 'C330d2']
  Visit 5: ['C102d4', 'C301d3', 'C404d3', 'C413d4']
  Visit 6: ['C102d4', 'C301d3', 'C404d3', 'C413d4']
Tree/embedding correlation: 0.3903
Synthetic stats (N=1000): {'mean_depth': 5.74475, 'std_depth': 0.8286117531751526, 'mean_tree_dist': 11.4895, 'std_tree_dist': 1.4999632495498012, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

Training hyperbolic DDPM | lambda_recon=1000
Epoch   1 | Train 77.218435 | Val 44.536961 | lambda_recon=1000 | lambda_pair=0.01
Epoch   2 | Train 43.190884 | Val 38.996759 | lambda_recon=1000 | lambda_pair=0.01
Epoch   3 | Train 39.492430 | Val 36.357251 | lambda_recon=1000 | lambda_pair=0.01
Epoch   4 | Train 37.526086 | Val 34.944880 | lambda_recon=1000 | lambda_pair=0.01
Epoch   5 | Train 36.176362 | Val 34.156219 | lambda_recon=1000 | lambda_pair=0.01
Epoch   6 | Train 35.273268 | Val 33.074064 | lambda_recon=1000 | lambda_pair=0.01
Epoch   7 | Train 34.488216 | Val 32.798240 | lambda_recon=1000 | lambda_pair=0.01
Epoch   8 | Train 33.930097 | Val 32.022586 | lambda_recon=1000 | lambda_pair=0.01
Epoch   9 | Train 32.966268 | Val 30.946732 | lambda_recon=1000 | lambda_pair=0.01
Epoch  10 | Train 32.122240 | Val 30.716712 | lambda_recon=1000 | lambda_pair=0.01
Epoch  11 | Train 31.881093 | Val 30.596543 | lambda_recon=1000 | lambda_pair=0.01
Epoch  12 | Train 31.719690 | Val 30.524381 | lambda_recon=1000 | lambda_pair=0.01
Epoch  13 | Train 31.574192 | Val 30.428553 | lambda_recon=1000 | lambda_pair=0.01
Epoch  14 | Train 31.457447 | Val 30.400783 | lambda_recon=1000 | lambda_pair=0.01
Epoch  15 | Train 31.336409 | Val 30.339717 | lambda_recon=1000 | lambda_pair=0.01
Epoch  16 | Train 31.197377 | Val 30.295681 | lambda_recon=1000 | lambda_pair=0.01
Epoch  17 | Train 31.103211 | Val 30.218369 | lambda_recon=1000 | lambda_pair=0.01
Epoch  18 | Train 30.975687 | Val 30.133483 | lambda_recon=1000 | lambda_pair=0.01
Epoch  19 | Train 30.847979 | Val 29.939262 | lambda_recon=1000 | lambda_pair=0.01
Epoch  20 | Train 30.397261 | Val 29.473941 | lambda_recon=1000 | lambda_pair=0.01
Epoch  21 | Train 30.147953 | Val 29.434766 | lambda_recon=1000 | lambda_pair=0.01
Epoch  22 | Train 30.067731 | Val 29.394667 | lambda_recon=1000 | lambda_pair=0.01
Epoch  23 | Train 29.994177 | Val 29.397206 | lambda_recon=1000 | lambda_pair=0.01
Epoch  24 | Train 29.955442 | Val 29.314483 | lambda_recon=1000 | lambda_pair=0.01
Epoch  25 | Train 29.868294 | Val 29.378472 | lambda_recon=1000 | lambda_pair=0.01
Epoch  26 | Train 29.825546 | Val 29.317667 | lambda_recon=1000 | lambda_pair=0.01
Epoch  27 | Train 29.789348 | Val 29.320812 | lambda_recon=1000 | lambda_pair=0.01
Epoch  28 | Train 29.747510 | Val 29.312615 | lambda_recon=1000 | lambda_pair=0.01
Epoch  29 | Train 29.694572 | Val 29.252228 | lambda_recon=1000 | lambda_pair=0.01
Epoch  30 | Train 29.668460 | Val 29.249617 | lambda_recon=1000 | lambda_pair=0.01
Epoch  31 | Train 29.641507 | Val 29.265031 | lambda_recon=1000 | lambda_pair=0.01
Epoch  32 | Train 29.620160 | Val 29.225259 | lambda_recon=1000 | lambda_pair=0.01
Epoch  33 | Train 29.594037 | Val 29.219820 | lambda_recon=1000 | lambda_pair=0.01
Epoch  34 | Train 29.574808 | Val 29.236177 | lambda_recon=1000 | lambda_pair=0.01
Epoch  35 | Train 29.572697 | Val 29.249701 | lambda_recon=1000 | lambda_pair=0.01
Epoch  36 | Train 29.563856 | Val 29.211856 | lambda_recon=1000 | lambda_pair=0.01
Epoch  37 | Train 29.541222 | Val 29.195463 | lambda_recon=1000 | lambda_pair=0.01
Epoch  38 | Train 29.501992 | Val 29.168502 | lambda_recon=1000 | lambda_pair=0.01
Epoch  39 | Train 29.481503 | Val 29.160735 | lambda_recon=1000 | lambda_pair=0.01
Epoch  40 | Train 29.471961 | Val 29.155186 | lambda_recon=1000 | lambda_pair=0.01
Epoch  41 | Train 29.425878 | Val 29.157214 | lambda_recon=1000 | lambda_pair=0.01
Epoch  42 | Train 29.396785 | Val 29.116863 | lambda_recon=1000 | lambda_pair=0.01
Epoch  43 | Train 29.365030 | Val 29.112644 | lambda_recon=1000 | lambda_pair=0.01
Epoch  44 | Train 29.262991 | Val 28.974101 | lambda_recon=1000 | lambda_pair=0.01
Epoch  45 | Train 29.146532 | Val 28.838702 | lambda_recon=1000 | lambda_pair=0.01
Epoch  46 | Train 28.999592 | Val 28.677349 | lambda_recon=1000 | lambda_pair=0.01
Epoch  47 | Train 28.827141 | Val 28.534983 | lambda_recon=1000 | lambda_pair=0.01
Epoch  48 | Train 28.692694 | Val 28.490585 | lambda_recon=1000 | lambda_pair=0.01
Epoch  49 | Train 28.585183 | Val 28.392442 | lambda_recon=1000 | lambda_pair=0.01
Epoch  50 | Train 28.516568 | Val 28.236598 | lambda_recon=1000 | lambda_pair=0.01
Best validation loss: 28.236598
Test Recall@4: 0.0158

Sample hyperbolic trajectory 1:
  Visit 1: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 2: ['C020d3', 'C020d4', 'C424d3', 'C424d4']
  Visit 3: ['C020d3', 'C020d4', 'C424d3', 'C424d4']
  Visit 4: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 5: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 6: ['C104d3', 'C104d4', 'C242d4', 'C414d4']

Sample hyperbolic trajectory 2:
  Visit 1: ['C020d3', 'C020d4', 'C424d3', 'C424d4']
  Visit 2: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 3: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 4: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 5: ['C020d3', 'C020d4', 'C424d3', 'C424d4']
  Visit 6: ['C104d3', 'C104d4', 'C242d4', 'C414d4']

Sample hyperbolic trajectory 3:
  Visit 1: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 2: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 3: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 4: ['C104d3', 'C104d4', 'C242d4', 'C414d4']
  Visit 5: ['C020d3', 'C020d4', 'C424d3', 'C424d4']
  Visit 6: ['C020d3', 'C020d4', 'C424d3', 'C424d4']
Tree/embedding correlation: 0.4967
Synthetic stats (N=1000): {'mean_depth': 6.6265, 'std_depth': 0.48373313924104894, 'mean_tree_dist': 1.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}


Using device: mps

hyperbolic_ddpm_depth2 | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

Training hyperbolic DDPM | lambda_recon=2000
Epoch   1 | Train 393.421847 | Val 230.691135 | lambda_recon=2000 | lambda_pair=0.01
Epoch   2 | Train 227.926853 | Val 223.668001 | lambda_recon=2000 | lambda_pair=0.01
Epoch   3 | Train 223.222413 | Val 220.163410 | lambda_recon=2000 | lambda_pair=0.01
Epoch   4 | Train 220.908431 | Val 218.832928 | lambda_recon=2000 | lambda_pair=0.01
Epoch   5 | Train 219.899297 | Val 218.276997 | lambda_recon=2000 | lambda_pair=0.01
Epoch   6 | Train 219.237110 | Val 217.773713 | lambda_recon=2000 | lambda_pair=0.01
Epoch   7 | Train 218.236889 | Val 216.835373 | lambda_recon=2000 | lambda_pair=0.01
Epoch   8 | Train 217.299846 | Val 215.775239 | lambda_recon=2000 | lambda_pair=0.01
Epoch   9 | Train 216.416807 | Val 214.823641 | lambda_recon=2000 | lambda_pair=0.01
Epoch  10 | Train 215.664676 | Val 214.584281 | lambda_recon=2000 | lambda_pair=0.01
Epoch  11 | Train 215.358308 | Val 214.317140 | lambda_recon=2000 | lambda_pair=0.01
Epoch  12 | Train 214.846620 | Val 213.377709 | lambda_recon=2000 | lambda_pair=0.01
Epoch  13 | Train 213.882520 | Val 212.831287 | lambda_recon=2000 | lambda_pair=0.01
Epoch  14 | Train 213.216549 | Val 211.997755 | lambda_recon=2000 | lambda_pair=0.01
Epoch  15 | Train 211.869438 | Val 210.519282 | lambda_recon=2000 | lambda_pair=0.01
Epoch  16 | Train 210.731650 | Val 209.705421 | lambda_recon=2000 | lambda_pair=0.01
Epoch  17 | Train 209.943618 | Val 209.000328 | lambda_recon=2000 | lambda_pair=0.01
Epoch  18 | Train 209.206663 | Val 208.228140 | lambda_recon=2000 | lambda_pair=0.01
Epoch  19 | Train 208.242394 | Val 207.073945 | lambda_recon=2000 | lambda_pair=0.01
Epoch  20 | Train 206.793618 | Val 205.336588 | lambda_recon=2000 | lambda_pair=0.01
Epoch  21 | Train 204.686855 | Val 203.043377 | lambda_recon=2000 | lambda_pair=0.01
Epoch  22 | Train 202.185309 | Val 200.528035 | lambda_recon=2000 | lambda_pair=0.01
Epoch  23 | Train 199.447891 | Val 197.820188 | lambda_recon=2000 | lambda_pair=0.01
Epoch  24 | Train 196.862620 | Val 195.321532 | lambda_recon=2000 | lambda_pair=0.01
Epoch  25 | Train 194.540230 | Val 192.836630 | lambda_recon=2000 | lambda_pair=0.01
Epoch  26 | Train 191.707186 | Val 189.557696 | lambda_recon=2000 | lambda_pair=0.01
Epoch  27 | Train 187.472736 | Val 184.431060 | lambda_recon=2000 | lambda_pair=0.01
Epoch  28 | Train 181.851085 | Val 178.756917 | lambda_recon=2000 | lambda_pair=0.01
Epoch  29 | Train 176.855984 | Val 174.360830 | lambda_recon=2000 | lambda_pair=0.01
Epoch  30 | Train 173.240611 | Val 171.508987 | lambda_recon=2000 | lambda_pair=0.01
Epoch  31 | Train 170.670858 | Val 169.318948 | lambda_recon=2000 | lambda_pair=0.01
Epoch  32 | Train 168.487217 | Val 167.127665 | lambda_recon=2000 | lambda_pair=0.01
Epoch  33 | Train 166.163040 | Val 165.012543 | lambda_recon=2000 | lambda_pair=0.01
Epoch  34 | Train 164.318990 | Val 163.202912 | lambda_recon=2000 | lambda_pair=0.01
Epoch  35 | Train 162.693726 | Val 161.780326 | lambda_recon=2000 | lambda_pair=0.01
Epoch  36 | Train 161.103118 | Val 160.171180 | lambda_recon=2000 | lambda_pair=0.01
Epoch  37 | Train 159.398429 | Val 158.587450 | lambda_recon=2000 | lambda_pair=0.01
Epoch  38 | Train 157.859562 | Val 157.338952 | lambda_recon=2000 | lambda_pair=0.01
Epoch  39 | Train 156.620629 | Val 156.293051 | lambda_recon=2000 | lambda_pair=0.01
Epoch  40 | Train 155.497087 | Val 155.304683 | lambda_recon=2000 | lambda_pair=0.01
Best validation loss: 155.304683
Test Recall@4: 0.1671

Sample hyperbolic trajectory 1:
  Visit 1: ['C022', 'C14', 'C141', 'C142']
  Visit 2: ['C022', 'C14', 'C142', 'C442']
  Visit 3: ['C02', 'C022', 'C14', 'C142']
  Visit 4: ['C011', 'C210', 'C214', 'C331']
  Visit 5: ['C022', 'C14', 'C141', 'C142']
  Visit 6: ['C011', 'C210', 'C214', 'C331']

Sample hyperbolic trajectory 2:
  Visit 1: ['C14', 'C142', 'C44', 'C442']
  Visit 2: ['C011', 'C210', 'C330', 'C331']
  Visit 3: ['C011', 'C210', 'C330', 'C331']
  Visit 4: ['C011', 'C210', 'C330', 'C331']
  Visit 5: ['C011', 'C210', 'C330', 'C331']
  Visit 6: ['C14', 'C142', 'C44', 'C442']

Sample hyperbolic trajectory 3:
  Visit 1: ['C14', 'C141', 'C142', 'C442']
  Visit 2: ['C022', 'C14', 'C142', 'C442']
  Visit 3: ['C022', 'C14', 'C142', 'C442']
  Visit 4: ['C022', 'C14', 'C142', 'C442']
  Visit 5: ['C011', 'C210', 'C214', 'C331']
  Visit 6: ['C011', 'C210', 'C214', 'C331']
Tree/embedding correlation: 0.6091
Synthetic stats (N=1000): {'mean_depth': 1.8627916666666666, 'std_depth': 0.34406715419700007, 'mean_tree_dist': 1.5095157538591668, 'std_tree_dist': 0.4999094422277816, 'mean_root_purity': 0.5572083333333333, 'std_root_purity': 0.10521861335914966}

Training hyperbolic DDPM | lambda_recon=3000
Epoch   1 | Train 353.463761 | Val 333.773761 | lambda_recon=3000 | lambda_pair=0.01
Epoch   2 | Train 331.452107 | Val 327.539309 | lambda_recon=3000 | lambda_pair=0.01
Epoch   3 | Train 327.156698 | Val 324.263096 | lambda_recon=3000 | lambda_pair=0.01
Epoch   4 | Train 323.924677 | Val 320.421141 | lambda_recon=3000 | lambda_pair=0.01
Epoch   5 | Train 319.374043 | Val 315.378400 | lambda_recon=3000 | lambda_pair=0.01
Epoch   6 | Train 311.745276 | Val 303.395623 | lambda_recon=3000 | lambda_pair=0.01
Epoch   7 | Train 297.583158 | Val 290.566453 | lambda_recon=3000 | lambda_pair=0.01
Epoch   8 | Train 286.452516 | Val 279.386557 | lambda_recon=3000 | lambda_pair=0.01
Epoch   9 | Train 273.103279 | Val 265.076486 | lambda_recon=3000 | lambda_pair=0.01
Epoch  10 | Train 260.831070 | Val 255.052161 | lambda_recon=3000 | lambda_pair=0.01
Epoch  11 | Train 251.969800 | Val 247.280113 | lambda_recon=3000 | lambda_pair=0.01
Epoch  12 | Train 245.260329 | Val 241.618614 | lambda_recon=3000 | lambda_pair=0.01
Epoch  13 | Train 240.483301 | Val 237.665804 | lambda_recon=3000 | lambda_pair=0.01
Epoch  14 | Train 236.970683 | Val 234.523968 | lambda_recon=3000 | lambda_pair=0.01
Epoch  15 | Train 234.126189 | Val 232.201316 | lambda_recon=3000 | lambda_pair=0.01
Epoch  16 | Train 231.559053 | Val 229.753639 | lambda_recon=3000 | lambda_pair=0.01
Epoch  17 | Train 229.209068 | Val 227.584893 | lambda_recon=3000 | lambda_pair=0.01
Epoch  18 | Train 227.167289 | Val 226.132578 | lambda_recon=3000 | lambda_pair=0.01
Epoch  19 | Train 225.611811 | Val 224.573963 | lambda_recon=3000 | lambda_pair=0.01
Epoch  20 | Train 224.136045 | Val 223.614381 | lambda_recon=3000 | lambda_pair=0.01
Epoch  21 | Train 223.156195 | Val 222.728966 | lambda_recon=3000 | lambda_pair=0.01
Epoch  22 | Train 222.239279 | Val 222.002605 | lambda_recon=3000 | lambda_pair=0.01
Epoch  23 | Train 221.534713 | Val 221.517677 | lambda_recon=3000 | lambda_pair=0.01
Epoch  24 | Train 221.052949 | Val 221.049468 | lambda_recon=3000 | lambda_pair=0.01
Epoch  25 | Train 220.456235 | Val 220.538700 | lambda_recon=3000 | lambda_pair=0.01
Epoch  26 | Train 219.815943 | Val 219.884353 | lambda_recon=3000 | lambda_pair=0.01
Epoch  27 | Train 219.254841 | Val 219.294933 | lambda_recon=3000 | lambda_pair=0.01
Epoch  28 | Train 218.537223 | Val 218.541828 | lambda_recon=3000 | lambda_pair=0.01
Epoch  29 | Train 217.853500 | Val 218.057226 | lambda_recon=3000 | lambda_pair=0.01
Epoch  30 | Train 217.361399 | Val 217.696632 | lambda_recon=3000 | lambda_pair=0.01
Epoch  31 | Train 216.813308 | Val 217.194610 | lambda_recon=3000 | lambda_pair=0.01
Epoch  32 | Train 216.285007 | Val 216.630875 | lambda_recon=3000 | lambda_pair=0.01
Epoch  33 | Train 215.760700 | Val 216.278324 | lambda_recon=3000 | lambda_pair=0.01
Epoch  34 | Train 215.325101 | Val 215.667543 | lambda_recon=3000 | lambda_pair=0.01
Epoch  35 | Train 214.834054 | Val 215.590125 | lambda_recon=3000 | lambda_pair=0.01
Epoch  36 | Train 214.494120 | Val 215.051908 | lambda_recon=3000 | lambda_pair=0.01
Epoch  37 | Train 214.117146 | Val 214.758595 | lambda_recon=3000 | lambda_pair=0.01
Epoch  38 | Train 213.848674 | Val 214.459396 | lambda_recon=3000 | lambda_pair=0.01
Epoch  39 | Train 213.636130 | Val 214.260647 | lambda_recon=3000 | lambda_pair=0.01
Epoch  40 | Train 213.411529 | Val 213.996355 | lambda_recon=3000 | lambda_pair=0.01
Best validation loss: 213.996355
Test Recall@4: 0.2422

Sample hyperbolic trajectory 1:
  Visit 1: ['C22', 'C221', 'C314', 'C414']
  Visit 2: ['C200', 'C42', 'C420', 'C421']
  Visit 3: ['C200', 'C42', 'C420', 'C421']
  Visit 4: ['C22', 'C221', 'C314', 'C414']
  Visit 5: ['C200', 'C42', 'C420', 'C421']
  Visit 6: ['C200', 'C42', 'C420', 'C421']

Sample hyperbolic trajectory 2:
  Visit 1: ['C03', 'C221', 'C314', 'C414']
  Visit 2: ['C200', 'C42', 'C420', 'C421']
  Visit 3: ['C03', 'C221', 'C314', 'C414']
  Visit 4: ['C200', 'C42', 'C420', 'C421']
  Visit 5: ['C200', 'C42', 'C420', 'C421']
  Visit 6: ['C200', 'C42', 'C420', 'C421']

Sample hyperbolic trajectory 3:
  Visit 1: ['C200', 'C42', 'C420', 'C421']
  Visit 2: ['C200', 'C42', 'C420', 'C421']
  Visit 3: ['C221', 'C314', 'C414', 'C442']
  Visit 4: ['C200', 'C42', 'C420', 'C421']
  Visit 5: ['C200', 'C42', 'C420', 'C421']
  Visit 6: ['C22', 'C221', 'C314', 'C414']
Tree/embedding correlation: 0.7138
Synthetic stats (N=1000): {'mean_depth': 1.721875, 'std_depth': 0.4480753110527292, 'mean_tree_dist': 1.3778958554729013, 'std_tree_dist': 0.602880293281723, 'mean_root_purity': 0.64475, 'std_root_purity': 0.18732085886698968}

Training hyperbolic DDPM | lambda_recon=4000
Epoch   1 | Train 449.531702 | Val 438.458363 | lambda_recon=4000 | lambda_pair=0.01
Epoch   2 | Train 435.454753 | Val 430.306397 | lambda_recon=4000 | lambda_pair=0.01
Epoch   3 | Train 426.424889 | Val 420.621983 | lambda_recon=4000 | lambda_pair=0.01
Epoch   4 | Train 415.396890 | Val 407.270236 | lambda_recon=4000 | lambda_pair=0.01
Epoch   5 | Train 398.542169 | Val 385.106397 | lambda_recon=4000 | lambda_pair=0.01
Epoch   6 | Train 372.093850 | Val 360.431045 | lambda_recon=4000 | lambda_pair=0.01
Epoch   7 | Train 354.240305 | Val 346.321943 | lambda_recon=4000 | lambda_pair=0.01
Epoch   8 | Train 341.496715 | Val 335.984668 | lambda_recon=4000 | lambda_pair=0.01
Epoch   9 | Train 332.128292 | Val 327.333305 | lambda_recon=4000 | lambda_pair=0.01
Epoch  10 | Train 323.904953 | Val 320.118666 | lambda_recon=4000 | lambda_pair=0.01
Epoch  11 | Train 317.478930 | Val 314.698766 | lambda_recon=4000 | lambda_pair=0.01
Epoch  12 | Train 312.613525 | Val 310.409601 | lambda_recon=4000 | lambda_pair=0.01
Epoch  13 | Train 308.360737 | Val 306.138334 | lambda_recon=4000 | lambda_pair=0.01
Epoch  14 | Train 304.292645 | Val 302.324768 | lambda_recon=4000 | lambda_pair=0.01
Epoch  15 | Train 300.905680 | Val 299.401109 | lambda_recon=4000 | lambda_pair=0.01
Epoch  16 | Train 297.989580 | Val 297.026429 | lambda_recon=4000 | lambda_pair=0.01
Epoch  17 | Train 295.697058 | Val 294.933996 | lambda_recon=4000 | lambda_pair=0.01
Epoch  18 | Train 293.902619 | Val 293.524584 | lambda_recon=4000 | lambda_pair=0.01
Epoch  19 | Train 292.429636 | Val 292.178958 | lambda_recon=4000 | lambda_pair=0.01
Epoch  20 | Train 291.207520 | Val 291.168991 | lambda_recon=4000 | lambda_pair=0.01
Epoch  21 | Train 290.124602 | Val 290.205284 | lambda_recon=4000 | lambda_pair=0.01
Epoch  22 | Train 289.298369 | Val 289.452062 | lambda_recon=4000 | lambda_pair=0.01
Epoch  23 | Train 288.600774 | Val 288.803007 | lambda_recon=4000 | lambda_pair=0.01
Epoch  24 | Train 287.949891 | Val 288.156958 | lambda_recon=4000 | lambda_pair=0.01
Epoch  25 | Train 287.411878 | Val 287.612297 | lambda_recon=4000 | lambda_pair=0.01
Epoch  26 | Train 287.009672 | Val 287.414711 | lambda_recon=4000 | lambda_pair=0.01
Epoch  27 | Train 286.578510 | Val 287.243143 | lambda_recon=4000 | lambda_pair=0.01
Epoch  28 | Train 286.203662 | Val 286.784536 | lambda_recon=4000 | lambda_pair=0.01
Epoch  29 | Train 285.790181 | Val 286.205058 | lambda_recon=4000 | lambda_pair=0.01
Epoch  30 | Train 285.326831 | Val 285.963639 | lambda_recon=4000 | lambda_pair=0.01
Epoch  31 | Train 284.657878 | Val 285.283693 | lambda_recon=4000 | lambda_pair=0.01
Epoch  32 | Train 284.073656 | Val 284.746787 | lambda_recon=4000 | lambda_pair=0.01
Epoch  33 | Train 283.530528 | Val 284.377925 | lambda_recon=4000 | lambda_pair=0.01
Epoch  34 | Train 283.028946 | Val 283.951636 | lambda_recon=4000 | lambda_pair=0.01
Epoch  35 | Train 282.626329 | Val 283.491765 | lambda_recon=4000 | lambda_pair=0.01
Epoch  36 | Train 282.259948 | Val 283.181442 | lambda_recon=4000 | lambda_pair=0.01
Epoch  37 | Train 281.796541 | Val 282.918358 | lambda_recon=4000 | lambda_pair=0.01
Epoch  38 | Train 281.463732 | Val 282.487168 | lambda_recon=4000 | lambda_pair=0.01
Epoch  39 | Train 281.061776 | Val 282.353341 | lambda_recon=4000 | lambda_pair=0.01
Epoch  40 | Train 280.707832 | Val 281.875351 | lambda_recon=4000 | lambda_pair=0.01
Best validation loss: 281.875351
Test Recall@4: 0.2486

Sample hyperbolic trajectory 1:
  Visit 1: ['C01', 'C013', 'C11', 'C110']
  Visit 2: ['C030', 'C10', 'C101', 'C32']
  Visit 3: ['C10', 'C101', 'C32', 'C323']
  Visit 4: ['C01', 'C013', 'C11', 'C110']
  Visit 5: ['C10', 'C101', 'C32', 'C323']
  Visit 6: ['C030', 'C10', 'C101', 'C32']

Sample hyperbolic trajectory 2:
  Visit 1: ['C030', 'C10', 'C101', 'C32']
  Visit 2: ['C030', 'C10', 'C101', 'C32']
  Visit 3: ['C01', 'C013', 'C110', 'C30']
  Visit 4: ['C01', 'C013', 'C110', 'C30']
  Visit 5: ['C030', 'C10', 'C101', 'C32']
  Visit 6: ['C030', 'C10', 'C101', 'C32']

Sample hyperbolic trajectory 3:
  Visit 1: ['C101', 'C32', 'C321', 'C323']
  Visit 2: ['C01', 'C013', 'C11', 'C110']
  Visit 3: ['C01', 'C013', 'C11', 'C110']
  Visit 4: ['C01', 'C013', 'C11', 'C110']
  Visit 5: ['C101', 'C32', 'C321', 'C323']
  Visit 6: ['C01', 'C013', 'C11', 'C110']
Tree/embedding correlation: 0.6964
Synthetic stats (N=1000): {'mean_depth': 1.5161666666666667, 'std_depth': 0.49973857054352816, 'mean_tree_dist': 1.0494788546783276, 'std_tree_dist': 0.21686562110682384, 'mean_root_purity': 0.5170833333333333, 'std_root_purity': 0.06357273830468178}
[Summary] depth=2 | lambda_recon=2000: best_val=155.304683, test_recall=0.1671, corr=0.6091
[Summary] depth=2 | lambda_recon=3000: best_val=213.996355, test_recall=0.2422, corr=0.7138
[Summary] depth=2 | lambda_recon=4000: best_val=281.875351, test_recall=0.2486, corr=0.6964

