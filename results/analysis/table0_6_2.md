Using device: mps

depth2_final | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

--- Running hyperbolic ---

Training HYPERBOLIC | Depth 2 | lambda_recon=2000.0
Epoch   1 | Train 35.85829 | Val 26.79634 | lambda_recon=2000.0
Epoch   2 | Train 26.38775 | Val 21.90384 | lambda_recon=2000.0
Epoch   3 | Train 22.99043 | Val 19.14310 | lambda_recon=2000.0
Epoch   4 | Train 21.05081 | Val 17.37051 | lambda_recon=2000.0
Epoch   5 | Train 19.49521 | Val 16.02302 | lambda_recon=2000.0
Epoch   6 | Train 18.08940 | Val 14.58368 | lambda_recon=2000.0
Epoch   7 | Train 16.90734 | Val 13.39435 | lambda_recon=2000.0
Epoch   8 | Train 15.87960 | Val 12.54220 | lambda_recon=2000.0
Epoch   9 | Train 15.01677 | Val 11.77425 | lambda_recon=2000.0
Epoch  10 | Train 14.36659 | Val 11.30276 | lambda_recon=2000.0
Epoch  11 | Train 13.91404 | Val 10.92649 | lambda_recon=2000.0
Epoch  12 | Train 13.68058 | Val 10.81766 | lambda_recon=2000.0
Epoch  13 | Train 13.48241 | Val 10.63548 | lambda_recon=2000.0
Epoch  14 | Train 13.26812 | Val 10.51346 | lambda_recon=2000.0
Epoch  15 | Train 13.08271 | Val 10.42924 | lambda_recon=2000.0
Epoch  16 | Train 12.96136 | Val 10.25529 | lambda_recon=2000.0
Epoch  17 | Train 12.82800 | Val 10.33973 | lambda_recon=2000.0
Epoch  18 | Train 12.69981 | Val 10.27242 | lambda_recon=2000.0
Epoch  19 | Train 12.62950 | Val 10.10245 | lambda_recon=2000.0
Epoch  20 | Train 12.44973 | Val 9.73226 | lambda_recon=2000.0
Epoch  21 | Train 12.27751 | Val 9.60100 | lambda_recon=2000.0
Epoch  22 | Train 12.15828 | Val 9.36093 | lambda_recon=2000.0
Epoch  23 | Train 11.97269 | Val 9.30726 | lambda_recon=2000.0
Epoch  24 | Train 11.88403 | Val 9.35184 | lambda_recon=2000.0
Epoch  25 | Train 11.85632 | Val 9.23392 | lambda_recon=2000.0
Epoch  26 | Train 11.71296 | Val 9.16763 | lambda_recon=2000.0
Epoch  27 | Train 11.67467 | Val 9.14081 | lambda_recon=2000.0
Epoch  28 | Train 11.67486 | Val 9.10835 | lambda_recon=2000.0
Epoch  29 | Train 11.61035 | Val 9.05114 | lambda_recon=2000.0
Epoch  30 | Train 11.55312 | Val 8.95485 | lambda_recon=2000.0
Epoch  31 | Train 11.51632 | Val 8.98020 | lambda_recon=2000.0
Epoch  32 | Train 11.48160 | Val 8.98919 | lambda_recon=2000.0
Epoch  33 | Train 11.43001 | Val 8.77829 | lambda_recon=2000.0
Epoch  34 | Train 11.41095 | Val 8.95177 | lambda_recon=2000.0
Epoch  35 | Train 11.39271 | Val 8.84092 | lambda_recon=2000.0
Epoch  36 | Train 11.31099 | Val 8.74515 | lambda_recon=2000.0
Epoch  37 | Train 11.22339 | Val 8.84670 | lambda_recon=2000.0
Epoch  38 | Train 11.21044 | Val 8.91696 | lambda_recon=2000.0
Epoch  39 | Train 11.22259 | Val 8.79498 | lambda_recon=2000.0
Epoch  40 | Train 11.18484 | Val 8.64098 | lambda_recon=2000.0
Epoch  41 | Train 11.17070 | Val 8.69639 | lambda_recon=2000.0
Epoch  42 | Train 11.12288 | Val 8.66080 | lambda_recon=2000.0
Epoch  43 | Train 11.14403 | Val 8.78843 | lambda_recon=2000.0
Epoch  44 | Train 11.09012 | Val 8.60714 | lambda_recon=2000.0
Epoch  45 | Train 11.10076 | Val 8.61357 | lambda_recon=2000.0
Epoch  46 | Train 11.15448 | Val 8.62086 | lambda_recon=2000.0
Epoch  47 | Train 11.14332 | Val 8.74004 | lambda_recon=2000.0
Epoch  48 | Train 11.27316 | Val 8.70283 | lambda_recon=2000.0
Epoch  49 | Train 11.43691 | Val 8.95014 | lambda_recon=2000.0
Early stopping triggered.
Best validation loss (lambda_recon=2000.0): 8.607137
Test Recall@4 (lambda_recon=2000.0): 0.5919

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C232', 'C34', 'C341', 'C343']
  Visit 2: ['C022', 'C10', 'C100', 'C104']
  Visit 3: ['C001', 'C210', 'C420', 'C422']
  Visit 4: ['C001', 'C210', 'C213', 'C220']
  Visit 5: ['C104', 'C341', 'C40', 'C402']
  Visit 6: ['C0', 'C1', 'C2', 'C4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C001', 'C210', 'C213', 'C220']
  Visit 2: ['C341', 'C343', 'C423', 'C443']
  Visit 3: ['C001', 'C1', 'C123', 'C210']
  Visit 4: ['C022', 'C101', 'C241']
  Visit 5: ['C001', 'C21', 'C210', 'C220']
  Visit 6: ['C001', 'C210', 'C213', 'C220']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C341', 'C343', 'C401', 'C443']
  Visit 2: ['C1', 'C123', 'C222', 'C323']
  Visit 3: ['C001', 'C1', 'C210', 'C220']
  Visit 4: ['C032', 'C32', 'C321', 'C343']
  Visit 5: ['C022', 'C10', 'C100', 'C104']
  Visit 6: ['C123', 'C210', 'C212', 'C413']
Tree-Embedding Correlation (lambda_recon=2000.0): 0.6206
Synthetic (hyperbolic, lambda_recon=2000.0) stats (N=1000): {'mean_depth': 1.8036470341647453, 'std_depth': 0.5111329309578114, 'mean_tree_dist': 2.190301060554225, 'std_tree_dist': 1.1437696637931298, 'mean_root_purity': 0.595111111111111, 'std_root_purity': 0.17338590726903436}

Training HYPERBOLIC | Depth 2 | lambda_recon=3000.0
Epoch   1 | Train 41.65308 | Val 32.51307 | lambda_recon=3000.0
Epoch   2 | Train 32.39197 | Val 27.60993 | lambda_recon=3000.0
Epoch   3 | Train 28.73075 | Val 24.61372 | lambda_recon=3000.0
Epoch   4 | Train 26.60719 | Val 23.06860 | lambda_recon=3000.0
Epoch   5 | Train 25.34226 | Val 21.90958 | lambda_recon=3000.0
Epoch   6 | Train 23.86708 | Val 20.19127 | lambda_recon=3000.0
Epoch   7 | Train 22.44525 | Val 19.18529 | lambda_recon=3000.0
Epoch   8 | Train 21.11553 | Val 17.47124 | lambda_recon=3000.0
Epoch   9 | Train 19.58201 | Val 16.20119 | lambda_recon=3000.0
Epoch  10 | Train 18.64246 | Val 15.63607 | lambda_recon=3000.0
Epoch  11 | Train 18.13055 | Val 15.18494 | lambda_recon=3000.0
Epoch  12 | Train 17.63592 | Val 14.78301 | lambda_recon=3000.0
Epoch  13 | Train 17.14330 | Val 14.44084 | lambda_recon=3000.0
Epoch  14 | Train 16.84595 | Val 14.24163 | lambda_recon=3000.0
Epoch  15 | Train 16.75513 | Val 14.00376 | lambda_recon=3000.0
Epoch  16 | Train 16.54301 | Val 13.84039 | lambda_recon=3000.0
Epoch  17 | Train 16.43130 | Val 13.83203 | lambda_recon=3000.0
Epoch  18 | Train 16.24919 | Val 13.57041 | lambda_recon=3000.0
Epoch  19 | Train 16.08191 | Val 13.42899 | lambda_recon=3000.0
Epoch  20 | Train 15.99738 | Val 13.22259 | lambda_recon=3000.0
Epoch  21 | Train 15.84772 | Val 13.22933 | lambda_recon=3000.0
Epoch  22 | Train 15.71787 | Val 13.00640 | lambda_recon=3000.0
Epoch  23 | Train 15.57367 | Val 12.95068 | lambda_recon=3000.0
Epoch  24 | Train 15.47624 | Val 12.89582 | lambda_recon=3000.0
Epoch  25 | Train 15.33523 | Val 12.77144 | lambda_recon=3000.0
Epoch  26 | Train 15.32263 | Val 12.72760 | lambda_recon=3000.0
Epoch  27 | Train 15.24635 | Val 12.57813 | lambda_recon=3000.0
Epoch  28 | Train 15.14237 | Val 12.57133 | lambda_recon=3000.0
Epoch  29 | Train 15.14599 | Val 12.56705 | lambda_recon=3000.0
Epoch  30 | Train 15.05697 | Val 12.44142 | lambda_recon=3000.0
Epoch  31 | Train 15.02684 | Val 12.54551 | lambda_recon=3000.0
Epoch  32 | Train 14.99604 | Val 12.47362 | lambda_recon=3000.0
Epoch  33 | Train 14.93127 | Val 12.45847 | lambda_recon=3000.0
Epoch  34 | Train 14.93705 | Val 12.35311 | lambda_recon=3000.0
Epoch  35 | Train 14.81668 | Val 12.37198 | lambda_recon=3000.0
Epoch  36 | Train 14.75500 | Val 12.18585 | lambda_recon=3000.0
Epoch  37 | Train 14.71165 | Val 12.26103 | lambda_recon=3000.0
Epoch  38 | Train 14.64987 | Val 12.12201 | lambda_recon=3000.0
Epoch  39 | Train 14.68254 | Val 12.01294 | lambda_recon=3000.0
Epoch  40 | Train 14.59357 | Val 12.08876 | lambda_recon=3000.0
Epoch  41 | Train 14.60808 | Val 12.08400 | lambda_recon=3000.0
Epoch  42 | Train 14.60412 | Val 12.04631 | lambda_recon=3000.0
Epoch  43 | Train 14.53936 | Val 12.05332 | lambda_recon=3000.0
Epoch  44 | Train 14.54693 | Val 12.05269 | lambda_recon=3000.0
Early stopping triggered.
Best validation loss (lambda_recon=3000.0): 12.012943
Test Recall@4 (lambda_recon=3000.0): 0.5743

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C02', 'C023', 'C302', 'C414']
  Visit 2: ['C31', 'C310', 'C311', 'C401']
  Visit 3: ['C02', 'C023', 'C302', 'C414']
  Visit 4: ['C02', 'C021', 'C023', 'C414']
  Visit 5: ['C023', 'C043', 'C102', 'C302']
  Visit 6: ['C31', 'C310', 'C401', 'C404']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C31', 'C310', 'C401', 'C404']
  Visit 2: ['C02', 'C020', 'C021', 'C023']
  Visit 3: ['C310', 'C343', 'C400', 'C401']
  Visit 4: ['C021', 'C023', 'C102', 'C414']
  Visit 5: ['C020', 'C101', 'C111', 'C231']
  Visit 6: ['C02', 'C023', 'C302', 'C414']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C02', 'C023', 'C224', 'C414']
  Visit 2: ['C31', 'C310', 'C311', 'C401']
  Visit 3: ['C02', 'C021', 'C023', 'C414']
  Visit 4: ['C020', 'C023', 'C304', 'C414']
  Visit 5: ['C004', 'C31', 'C310', 'C400']
  Visit 6: ['C31', 'C310', 'C401', 'C404']
Tree-Embedding Correlation (lambda_recon=3000.0): 0.6204
Synthetic (hyperbolic, lambda_recon=3000.0) stats (N=1000): {'mean_depth': 1.8162083333333334, 'std_depth': 0.4106039737555993, 'mean_tree_dist': 1.7415293960091067, 'std_tree_dist': 0.9390956255103249, 'mean_root_purity': 0.6585833333333333, 'std_root_purity': 0.17986474470804134}
[Summary] depth2_final | hyperbolic | lambda_recon=2000.0: best_val=8.607137, test_recall=0.5919, corr=0.6206
[Summary] depth2_final | hyperbolic | lambda_recon=3000.0: best_val=12.012943, test_recall=0.5743, corr=0.6204
