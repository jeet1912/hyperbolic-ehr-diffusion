
depth2_final | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

--- Running euclidean ---
Using device: mps

Training EUCLIDEAN | Depth 2 | lambda_recon=1.0
Epoch   1 | Train 24.21398 | Val 15.86340 | lambda_recon=1.0
Epoch   2 | Train 15.32503 | Val 11.43425 | lambda_recon=1.0
Epoch   3 | Train 12.43584 | Val 9.12880 | lambda_recon=1.0
Epoch   4 | Train 10.70047 | Val 7.54874 | lambda_recon=1.0
Epoch   5 | Train 9.47242 | Val 6.29994 | lambda_recon=1.0
Epoch   6 | Train 8.65779 | Val 5.64102 | lambda_recon=1.0
Epoch   7 | Train 8.12382 | Val 5.20094 | lambda_recon=1.0
Epoch   8 | Train 7.51460 | Val 4.33987 | lambda_recon=1.0
Epoch   9 | Train 6.97539 | Val 3.96775 | lambda_recon=1.0
Epoch  10 | Train 6.47648 | Val 3.42997 | lambda_recon=1.0
Epoch  11 | Train 6.19043 | Val 3.11697 | lambda_recon=1.0
Epoch  12 | Train 5.89911 | Val 2.99351 | lambda_recon=1.0
Epoch  13 | Train 5.71622 | Val 2.73643 | lambda_recon=1.0
Epoch  14 | Train 5.54469 | Val 2.57878 | lambda_recon=1.0
Epoch  15 | Train 5.31895 | Val 2.23897 | lambda_recon=1.0
Epoch  16 | Train 5.21885 | Val 2.32647 | lambda_recon=1.0
Epoch  17 | Train 5.12019 | Val 2.16812 | lambda_recon=1.0
Epoch  18 | Train 5.03210 | Val 2.18046 | lambda_recon=1.0
Epoch  19 | Train 4.99169 | Val 2.08798 | lambda_recon=1.0
Epoch  20 | Train 4.92923 | Val 2.12235 | lambda_recon=1.0
Epoch  21 | Train 4.87181 | Val 2.02235 | lambda_recon=1.0
Epoch  22 | Train 4.85625 | Val 2.02870 | lambda_recon=1.0
Epoch  23 | Train 4.82541 | Val 2.00882 | lambda_recon=1.0
Epoch  24 | Train 4.77686 | Val 2.03164 | lambda_recon=1.0
Epoch  25 | Train 4.78052 | Val 1.92805 | lambda_recon=1.0
Epoch  26 | Train 4.71190 | Val 2.00836 | lambda_recon=1.0
Epoch  27 | Train 4.73880 | Val 1.90242 | lambda_recon=1.0
Epoch  28 | Train 4.71873 | Val 1.94350 | lambda_recon=1.0
Epoch  29 | Train 4.67727 | Val 1.90095 | lambda_recon=1.0
Epoch  30 | Train 4.67755 | Val 1.94463 | lambda_recon=1.0
Epoch  31 | Train 4.66054 | Val 1.95018 | lambda_recon=1.0
Epoch  32 | Train 4.67159 | Val 1.86198 | lambda_recon=1.0
Epoch  33 | Train 4.61104 | Val 1.89670 | lambda_recon=1.0
Epoch  34 | Train 4.59372 | Val 1.89721 | lambda_recon=1.0
Epoch  35 | Train 4.59123 | Val 1.83208 | lambda_recon=1.0
Epoch  36 | Train 4.60224 | Val 1.88010 | lambda_recon=1.0
Epoch  37 | Train 4.60496 | Val 1.90348 | lambda_recon=1.0
Epoch  38 | Train 4.58410 | Val 1.84501 | lambda_recon=1.0
Epoch  39 | Train 4.60481 | Val 1.91776 | lambda_recon=1.0
Epoch  40 | Train 4.55416 | Val 1.89135 | lambda_recon=1.0
Early stopping triggered.
Best validation loss (lambda_recon=1.0): 1.832076
Saved loss curves to results/plots
Test Recall@4 (lambda_recon=1.0): 0.0672

Sample trajectory (euclidean) 1:
  Visit 1: ['C12', 'C120', 'C13', 'C133']
  Visit 2: ['C12', 'C120', 'C13', 'C133']
  Visit 3: ['C030', 'C104', 'C142', 'C334']
  Visit 4: ['C030', 'C104', 'C142', 'C334']
  Visit 5: ['C01', 'C12', 'C13', 'C44']
  Visit 6: ['C12', 'C120', 'C13', 'C133']

Sample trajectory (euclidean) 2:
  Visit 1: ['C030', 'C104', 'C142', 'C334']
  Visit 2: ['C12', 'C120', 'C13', 'C133']
  Visit 3: ['C030', 'C104', 'C142', 'C334']
  Visit 4: ['C12', 'C120', 'C13', 'C133']
  Visit 5: ['C12', 'C120', 'C13', 'C133']
  Visit 6: ['C01', 'C13', 'C133', 'C44']

Sample trajectory (euclidean) 3:
  Visit 1: ['C030', 'C104', 'C142', 'C334']
  Visit 2: ['C030', 'C104', 'C142', 'C334']
  Visit 3: ['C12', 'C120', 'C13', 'C133']
  Visit 4: ['C030', 'C104', 'C142', 'C334']
  Visit 5: ['C12', 'C120', 'C13', 'C133']
  Visit 6: ['C030', 'C104', 'C142', 'C334']
Tree-Embedding Correlation (lambda_recon=1.0): -0.0216
Synthetic (euclidean, lambda_recon=1.0) stats (N=1000): {'mean_depth': 1.6917083333333334, 'std_depth': 0.4617877379603702, 'mean_tree_dist': 2.5390587609678277, 'std_tree_dist': 1.170009018299364, 'mean_root_purity': 0.71275, 'std_root_purity': 0.24712501053785177}

Training EUCLIDEAN | Depth 2 | lambda_recon=10.0
Epoch   1 | Train 23.37926 | Val 15.23693 | lambda_recon=10.0
Epoch   2 | Train 14.76522 | Val 10.79747 | lambda_recon=10.0
Epoch   3 | Train 11.86631 | Val 8.54615 | lambda_recon=10.0
Epoch   4 | Train 10.10943 | Val 6.74886 | lambda_recon=10.0
Epoch   5 | Train 8.96441 | Val 5.81039 | lambda_recon=10.0
Epoch   6 | Train 8.10940 | Val 5.00740 | lambda_recon=10.0
Epoch   7 | Train 7.45981 | Val 4.39121 | lambda_recon=10.0
Epoch   8 | Train 6.78575 | Val 3.57834 | lambda_recon=10.0
Epoch   9 | Train 6.38700 | Val 3.35905 | lambda_recon=10.0
Epoch  10 | Train 6.16875 | Val 3.19145 | lambda_recon=10.0
Epoch  11 | Train 5.96072 | Val 2.94904 | lambda_recon=10.0
Epoch  12 | Train 5.68259 | Val 2.73566 | lambda_recon=10.0
Epoch  13 | Train 5.52615 | Val 2.46321 | lambda_recon=10.0
Epoch  14 | Train 5.38321 | Val 2.51069 | lambda_recon=10.0
Epoch  15 | Train 5.23840 | Val 2.33203 | lambda_recon=10.0
Epoch  16 | Train 5.14730 | Val 2.16762 | lambda_recon=10.0
Epoch  17 | Train 5.10236 | Val 2.24189 | lambda_recon=10.0
Epoch  18 | Train 5.02917 | Val 2.10936 | lambda_recon=10.0
Epoch  19 | Train 4.95555 | Val 2.19464 | lambda_recon=10.0
Epoch  20 | Train 4.91864 | Val 2.08798 | lambda_recon=10.0
Epoch  21 | Train 4.90394 | Val 2.00379 | lambda_recon=10.0
Epoch  22 | Train 4.83114 | Val 2.08199 | lambda_recon=10.0
Epoch  23 | Train 4.85217 | Val 2.08910 | lambda_recon=10.0
Epoch  24 | Train 4.78496 | Val 1.98173 | lambda_recon=10.0
Epoch  25 | Train 4.77172 | Val 2.01107 | lambda_recon=10.0
Epoch  26 | Train 4.75487 | Val 1.94184 | lambda_recon=10.0
Epoch  27 | Train 4.74128 | Val 1.95270 | lambda_recon=10.0
Epoch  28 | Train 4.70718 | Val 1.96747 | lambda_recon=10.0
Epoch  29 | Train 4.72667 | Val 1.95787 | lambda_recon=10.0
Epoch  30 | Train 4.71259 | Val 1.92588 | lambda_recon=10.0
Epoch  31 | Train 4.67866 | Val 1.97140 | lambda_recon=10.0
Epoch  32 | Train 4.68652 | Val 1.93047 | lambda_recon=10.0
Epoch  33 | Train 4.67405 | Val 1.95555 | lambda_recon=10.0
Epoch  34 | Train 4.64544 | Val 1.97327 | lambda_recon=10.0
Epoch  35 | Train 4.62644 | Val 1.91038 | lambda_recon=10.0
Epoch  36 | Train 4.59482 | Val 1.94956 | lambda_recon=10.0
Epoch  37 | Train 4.62654 | Val 1.97666 | lambda_recon=10.0
Epoch  38 | Train 4.60948 | Val 1.87448 | lambda_recon=10.0
Epoch  39 | Train 4.60812 | Val 1.94771 | lambda_recon=10.0
Epoch  40 | Train 4.59389 | Val 1.96488 | lambda_recon=10.0
Epoch  41 | Train 4.59796 | Val 1.91900 | lambda_recon=10.0
Epoch  42 | Train 4.59206 | Val 1.89257 | lambda_recon=10.0
Epoch  43 | Train 4.59234 | Val 1.88513 | lambda_recon=10.0
Early stopping triggered.
Best validation loss (lambda_recon=10.0): 1.874482
Saved loss curves to results/plots
Test Recall@4 (lambda_recon=10.0): 0.2493

Sample trajectory (euclidean) 1:
  Visit 1: ['C21', 'C30', 'C33', 'C422']
  Visit 2: ['C003', 'C21', 'C33', 'C422']
  Visit 3: ['C21', 'C30', 'C33', 'C422']
  Visit 4: ['C03', 'C22', 'C42', 'C43']
  Visit 5: ['C003', 'C21', 'C33', 'C422']
  Visit 6: ['C003', 'C21', 'C33', 'C422']

Sample trajectory (euclidean) 2:
  Visit 1: ['C04', 'C042', 'C32', 'C321']
  Visit 2: ['C21', 'C30', 'C33', 'C422']
  Visit 3: ['C003', 'C21', 'C33', 'C422']
  Visit 4: ['C21', 'C30', 'C33', 'C422']
  Visit 5: ['C04', 'C141', 'C32', 'C321']
  Visit 6: ['C21', 'C30', 'C33', 'C422']

Sample trajectory (euclidean) 3:
  Visit 1: ['C003', 'C21', 'C33', 'C422']
  Visit 2: ['C21', 'C30', 'C33', 'C422']
  Visit 3: ['C21', 'C30', 'C33', 'C422']
  Visit 4: ['C003', 'C21', 'C33', 'C422']
  Visit 5: ['C003', 'C21', 'C33', 'C422']
  Visit 6: ['C003', 'C21', 'C33', 'C422']
Tree-Embedding Correlation (lambda_recon=10.0): 0.0488
Synthetic (euclidean, lambda_recon=10.0) stats (N=1000): {'mean_depth': 1.3805, 'std_depth': 0.48722316926298437, 'mean_tree_dist': 1.8119549032120825, 'std_tree_dist': 0.6289762914153608, 'mean_root_purity': 0.41329166666666667, 'std_root_purity': 0.15475921597939024}

Training EUCLIDEAN | Depth 2 | lambda_recon=100.0
Epoch   1 | Train 23.96959 | Val 15.73676 | lambda_recon=100.0
Epoch   2 | Train 15.19281 | Val 10.99239 | lambda_recon=100.0
Epoch   3 | Train 12.09674 | Val 8.63492 | lambda_recon=100.0
Epoch   4 | Train 10.76086 | Val 7.69065 | lambda_recon=100.0
Epoch   5 | Train 9.79802 | Val 6.79982 | lambda_recon=100.0
Epoch   6 | Train 9.23035 | Val 6.54380 | lambda_recon=100.0
Epoch   7 | Train 8.80664 | Val 5.98136 | lambda_recon=100.0
Epoch   8 | Train 8.43282 | Val 5.58858 | lambda_recon=100.0
Epoch   9 | Train 8.07148 | Val 5.19421 | lambda_recon=100.0
Epoch  10 | Train 7.63777 | Val 4.65115 | lambda_recon=100.0
Epoch  11 | Train 7.14769 | Val 4.30135 | lambda_recon=100.0
Epoch  12 | Train 6.92110 | Val 4.08309 | lambda_recon=100.0
Epoch  13 | Train 6.76921 | Val 3.88276 | lambda_recon=100.0
Epoch  14 | Train 6.59349 | Val 3.73530 | lambda_recon=100.0
Epoch  15 | Train 6.40859 | Val 3.50718 | lambda_recon=100.0
Epoch  16 | Train 6.28203 | Val 3.52864 | lambda_recon=100.0
Epoch  17 | Train 6.13045 | Val 3.49901 | lambda_recon=100.0
Epoch  18 | Train 6.10403 | Val 3.42271 | lambda_recon=100.0
Epoch  19 | Train 6.04489 | Val 3.31148 | lambda_recon=100.0
Epoch  20 | Train 5.97947 | Val 3.22205 | lambda_recon=100.0
Epoch  21 | Train 5.88135 | Val 3.12182 | lambda_recon=100.0
Epoch  22 | Train 5.88545 | Val 3.15404 | lambda_recon=100.0
Epoch  23 | Train 5.86104 | Val 3.11234 | lambda_recon=100.0
Epoch  24 | Train 5.83419 | Val 3.15277 | lambda_recon=100.0
Epoch  25 | Train 5.79439 | Val 3.08096 | lambda_recon=100.0
Epoch  26 | Train 5.74536 | Val 3.11572 | lambda_recon=100.0
Epoch  27 | Train 5.76694 | Val 3.13847 | lambda_recon=100.0
Epoch  28 | Train 5.72989 | Val 3.01556 | lambda_recon=100.0
Epoch  29 | Train 5.70462 | Val 3.01460 | lambda_recon=100.0
Epoch  30 | Train 5.66098 | Val 3.02005 | lambda_recon=100.0
Epoch  31 | Train 5.67999 | Val 3.01637 | lambda_recon=100.0
Epoch  32 | Train 5.63585 | Val 2.98768 | lambda_recon=100.0
Epoch  33 | Train 5.60855 | Val 2.95478 | lambda_recon=100.0
Epoch  34 | Train 5.63260 | Val 2.96877 | lambda_recon=100.0
Epoch  35 | Train 5.60139 | Val 2.93472 | lambda_recon=100.0
Epoch  36 | Train 5.54744 | Val 3.03152 | lambda_recon=100.0
Epoch  37 | Train 5.56958 | Val 2.98953 | lambda_recon=100.0
Epoch  38 | Train 5.57732 | Val 2.97288 | lambda_recon=100.0
Epoch  39 | Train 5.51951 | Val 2.95653 | lambda_recon=100.0
Epoch  40 | Train 5.50358 | Val 2.88992 | lambda_recon=100.0
Epoch  41 | Train 5.50931 | Val 2.88059 | lambda_recon=100.0
Epoch  42 | Train 5.49382 | Val 2.84539 | lambda_recon=100.0
Epoch  43 | Train 5.49673 | Val 2.83800 | lambda_recon=100.0
Epoch  44 | Train 5.49089 | Val 2.85715 | lambda_recon=100.0
Epoch  45 | Train 5.44395 | Val 2.83032 | lambda_recon=100.0
Epoch  46 | Train 5.41122 | Val 2.82640 | lambda_recon=100.0
Epoch  47 | Train 5.39890 | Val 2.79745 | lambda_recon=100.0
Epoch  48 | Train 5.41388 | Val 2.78579 | lambda_recon=100.0
Epoch  49 | Train 5.36783 | Val 2.84097 | lambda_recon=100.0
