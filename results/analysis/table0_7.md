Using device: mps

depth2_final | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

--- Running hyperbolic ---

Training HYPERBOLIC | Depth 2 | lambda_recon=2000.0
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

Training HYPERBOLIC | Depth 7 | lambda_recon=2000.0
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
