
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
Epoch   1 | Train 24.88237 | Val 16.43887 | lambda_recon=100.0
Epoch   2 | Train 15.89218 | Val 11.96854 | lambda_recon=100.0
Epoch   3 | Train 12.90101 | Val 9.54124 | lambda_recon=100.0
Epoch   4 | Train 11.09863 | Val 7.96730 | lambda_recon=100.0
Epoch   5 | Train 9.90529 | Val 6.72330 | lambda_recon=100.0
Epoch   6 | Train 9.08699 | Val 6.04310 | lambda_recon=100.0
Epoch   7 | Train 8.53449 | Val 5.63224 | lambda_recon=100.0
Epoch   8 | Train 8.13363 | Val 5.03535 | lambda_recon=100.0
Epoch   9 | Train 7.50094 | Val 4.50516 | lambda_recon=100.0
Epoch  10 | Train 7.11783 | Val 4.17177 | lambda_recon=100.0
Epoch  11 | Train 6.80491 | Val 3.68452 | lambda_recon=100.0
Epoch  12 | Train 6.38175 | Val 3.42891 | lambda_recon=100.0
Epoch  13 | Train 6.18427 | Val 3.23248 | lambda_recon=100.0
Epoch  14 | Train 6.09384 | Val 3.23144 | lambda_recon=100.0
Epoch  15 | Train 5.93534 | Val 2.95078 | lambda_recon=100.0
Epoch  16 | Train 5.83638 | Val 2.94911 | lambda_recon=100.0
Epoch  17 | Train 5.70653 | Val 2.71734 | lambda_recon=100.0
Epoch  18 | Train 5.57840 | Val 2.65671 | lambda_recon=100.0
Epoch  19 | Train 5.47585 | Val 2.52172 | lambda_recon=100.0
Epoch  20 | Train 5.37255 | Val 2.48673 | lambda_recon=100.0
Epoch  21 | Train 5.28794 | Val 2.39753 | lambda_recon=100.0
Epoch  22 | Train 5.25267 | Val 2.37762 | lambda_recon=100.0
Epoch  23 | Train 5.21032 | Val 2.32684 | lambda_recon=100.0
Epoch  24 | Train 5.15044 | Val 2.37961 | lambda_recon=100.0
Epoch  25 | Train 5.16087 | Val 2.25337 | lambda_recon=100.0
Epoch  26 | Train 5.06497 | Val 2.32796 | lambda_recon=100.0
Epoch  27 | Train 5.08968 | Val 2.18876 | lambda_recon=100.0
Epoch  28 | Train 5.05852 | Val 2.24969 | lambda_recon=100.0
Epoch  29 | Train 5.00358 | Val 2.18094 | lambda_recon=100.0
Epoch  30 | Train 5.00270 | Val 2.21701 | lambda_recon=100.0
Epoch  31 | Train 4.97800 | Val 2.23234 | lambda_recon=100.0
Epoch  32 | Train 4.98383 | Val 2.10872 | lambda_recon=100.0
Epoch  33 | Train 4.89858 | Val 2.13225 | lambda_recon=100.0
Epoch  34 | Train 4.87762 | Val 2.16577 | lambda_recon=100.0
Epoch  35 | Train 4.86355 | Val 2.06735 | lambda_recon=100.0
Epoch  36 | Train 4.87986 | Val 2.09959 | lambda_recon=100.0
Epoch  37 | Train 4.88434 | Val 2.15084 | lambda_recon=100.0
Epoch  38 | Train 4.84343 | Val 2.05428 | lambda_recon=100.0
Epoch  39 | Train 4.86425 | Val 2.11436 | lambda_recon=100.0
Epoch  40 | Train 4.80834 | Val 2.09226 | lambda_recon=100.0
Epoch  41 | Train 4.84491 | Val 2.05834 | lambda_recon=100.0
Epoch  42 | Train 4.82595 | Val 2.07257 | lambda_recon=100.0
Epoch  43 | Train 4.80713 | Val 2.05197 | lambda_recon=100.0
Epoch  44 | Train 4.81460 | Val 2.04270 | lambda_recon=100.0
Epoch  45 | Train 4.80272 | Val 2.06263 | lambda_recon=100.0
Epoch  46 | Train 4.78569 | Val 2.05516 | lambda_recon=100.0
Epoch  47 | Train 4.76585 | Val 2.05562 | lambda_recon=100.0
Epoch  48 | Train 4.77074 | Val 2.03055 | lambda_recon=100.0
Epoch  49 | Train 4.79940 | Val 2.07971 | lambda_recon=100.0
Epoch  50 | Train 4.78450 | Val 2.11989 | lambda_recon=100.0
Best validation loss (lambda_recon=100.0): 2.030545
Test Recall@4 (lambda_recon=100.0): 0.8702

Sample trajectory (euclidean) 1:
  Visit 1: ['C043', 'C144', 'C300', 'C440']
  Visit 2: ['C22', 'C320', 'C403', 'C404']
  Visit 3: ['C101', 'C241', 'C3', 'C433']
  Visit 4: ['C023', 'C134', 'C204', 'C301']
  Visit 5: ['C042', 'C131', 'C200', 'C31']
  Visit 6: ['C10', 'C101', 'C303', 'C433']

Sample trajectory (euclidean) 2:
  Visit 1: ['C042', 'C131', 'C200', 'C210']
  Visit 2: ['C101', 'C133', 'C34', 'C443']
  Visit 3: ['C013', 'C304', 'C334', 'C433']
  Visit 4: ['C014', 'C021', 'C112', 'C143']
  Visit 5: ['C014', 'C342', 'C432', 'C433']
  Visit 6: ['C01', 'C022', 'C131', 'C20']

Sample trajectory (euclidean) 3:
  Visit 1: ['C02', 'C100', 'C243', 'C433']
  Visit 2: ['C10', 'C100', 'C303', 'C433']
  Visit 3: ['C022', 'C111', 'C13', 'C31']
  Visit 4: ['C042', 'C131', 'C200', 'C210']
  Visit 5: ['C10', 'C100', 'C303', 'C432']
  Visit 6: ['C111', 'C230', 'C401', 'C403']
Tree-Embedding Correlation (lambda_recon=100.0): -0.0155
Synthetic (euclidean, lambda_recon=100.0) stats (N=1000): {'mean_depth': 1.743, 'std_depth': 0.4482755848805509, 'mean_tree_dist': 2.6802887889933253, 'std_tree_dist': 1.21990471889981, 'mean_root_purity': 0.4925833333333333, 'std_root_purity': 0.14939877193456294}

Training EUCLIDEAN | Depth 2 | lambda_recon=1000.0
Epoch   1 | Train 30.12401 | Val 19.36360 | lambda_recon=1000.0
Epoch   2 | Train 17.96088 | Val 13.17634 | lambda_recon=1000.0
Epoch   3 | Train 13.88033 | Val 10.16595 | lambda_recon=1000.0
Epoch   4 | Train 11.84622 | Val 8.53419 | lambda_recon=1000.0
Epoch   5 | Train 10.59898 | Val 7.40571 | lambda_recon=1000.0
Epoch   6 | Train 9.69031 | Val 6.66769 | lambda_recon=1000.0
Epoch   7 | Train 9.18916 | Val 6.09711 | lambda_recon=1000.0
Epoch   8 | Train 8.45090 | Val 5.38571 | lambda_recon=1000.0
Epoch   9 | Train 8.00808 | Val 5.19033 | lambda_recon=1000.0
Epoch  10 | Train 7.78968 | Val 4.90193 | lambda_recon=1000.0
Epoch  11 | Train 7.54519 | Val 4.64427 | lambda_recon=1000.0
Epoch  12 | Train 7.27127 | Val 4.54405 | lambda_recon=1000.0
Epoch  13 | Train 7.10932 | Val 4.32205 | lambda_recon=1000.0
Epoch  14 | Train 6.86838 | Val 4.06941 | lambda_recon=1000.0
Epoch  15 | Train 6.78130 | Val 4.06434 | lambda_recon=1000.0
Epoch  16 | Train 6.62425 | Val 3.88200 | lambda_recon=1000.0
Epoch  17 | Train 6.52936 | Val 3.76972 | lambda_recon=1000.0
Epoch  18 | Train 6.37827 | Val 3.69779 | lambda_recon=1000.0
Epoch  19 | Train 6.36296 | Val 3.61454 | lambda_recon=1000.0
Epoch  20 | Train 6.25571 | Val 3.57686 | lambda_recon=1000.0
Epoch  21 | Train 6.16751 | Val 3.62938 | lambda_recon=1000.0
Epoch  22 | Train 6.14495 | Val 3.52609 | lambda_recon=1000.0
Epoch  23 | Train 6.07144 | Val 3.43951 | lambda_recon=1000.0
Epoch  24 | Train 6.02503 | Val 3.50107 | lambda_recon=1000.0
Epoch  25 | Train 5.96174 | Val 3.37929 | lambda_recon=1000.0
Epoch  26 | Train 5.92876 | Val 3.41296 | lambda_recon=1000.0
Epoch  27 | Train 5.91171 | Val 3.37986 | lambda_recon=1000.0
Epoch  28 | Train 5.85720 | Val 3.26740 | lambda_recon=1000.0
Epoch  29 | Train 5.80074 | Val 3.31171 | lambda_recon=1000.0
Epoch  30 | Train 5.80269 | Val 3.33193 | lambda_recon=1000.0
Epoch  31 | Train 5.77723 | Val 3.26439 | lambda_recon=1000.0
Epoch  32 | Train 5.74782 | Val 3.16831 | lambda_recon=1000.0
Epoch  33 | Train 5.71054 | Val 3.16980 | lambda_recon=1000.0
Epoch  34 | Train 5.68743 | Val 3.19826 | lambda_recon=1000.0
Epoch  35 | Train 5.67155 | Val 3.15904 | lambda_recon=1000.0
Epoch  36 | Train 5.63987 | Val 3.19949 | lambda_recon=1000.0
Epoch  37 | Train 5.62383 | Val 3.14938 | lambda_recon=1000.0
Epoch  38 | Train 5.58714 | Val 3.09630 | lambda_recon=1000.0
Epoch  39 | Train 5.56068 | Val 2.96309 | lambda_recon=1000.0
Epoch  40 | Train 5.50271 | Val 2.96045 | lambda_recon=1000.0
Epoch  41 | Train 5.34296 | Val 2.79018 | lambda_recon=1000.0
Epoch  42 | Train 5.29009 | Val 2.62825 | lambda_recon=1000.0
Epoch  43 | Train 5.22583 | Val 2.64855 | lambda_recon=1000.0
Epoch  44 | Train 5.20030 | Val 2.58720 | lambda_recon=1000.0
Epoch  45 | Train 5.17975 | Val 2.46742 | lambda_recon=1000.0
Epoch  46 | Train 5.13027 | Val 2.48881 | lambda_recon=1000.0
Epoch  47 | Train 5.11116 | Val 2.47784 | lambda_recon=1000.0
Epoch  48 | Train 5.08590 | Val 2.43948 | lambda_recon=1000.0
Epoch  49 | Train 5.12594 | Val 2.44411 | lambda_recon=1000.0
Epoch  50 | Train 5.09506 | Val 2.45557 | lambda_recon=1000.0
Best validation loss (lambda_recon=1000.0): 2.439485
Test Recall@4 (lambda_recon=1000.0): 0.9227

Sample trajectory (euclidean) 1:
  Visit 1: ['C120', 'C21', 'C213', 'C42']
  Visit 2: ['C01', 'C030', 'C40', 'C42']
  Visit 3: ['C0', 'C011', 'C142', 'C204']
  Visit 4: ['C144', 'C214', 'C334', 'C343']
  Visit 5: ['C100', 'C133', 'C23', 'C303']
  Visit 6: ['C102', 'C104', 'C110', 'C330']

Sample trajectory (euclidean) 2:
  Visit 1: ['C103', 'C33', 'C333', 'C34']
  Visit 2: ['C14', 'C40', 'C414', 'C43']
  Visit 3: ['C00', 'C134', 'C340', 'C421']
  Visit 4: ['C004', 'C01', 'C421', 'C424']
  Visit 5: ['C04', 'C233', 'C33', 'C414']
  Visit 6: ['C04', 'C1', 'C34', 'C423']

Sample trajectory (euclidean) 3:
  Visit 1: ['C01', 'C013', 'C13', 'C23']
  Visit 2: ['C1', 'C11', 'C312', 'C423']
  Visit 3: ['C013', 'C124', 'C130', 'C232']
  Visit 4: ['C1', 'C123', 'C224', 'C34']
  Visit 5: ['C004', 'C124', 'C203', 'C33']
  Visit 6: ['C113', 'C134', 'C201', 'C30']
Tree-Embedding Correlation (lambda_recon=1000.0): 0.0527
Synthetic (euclidean, lambda_recon=1000.0) stats (N=1000): {'mean_depth': 1.642375, 'std_depth': 0.5588494365285996, 'mean_tree_dist': 2.889705882352941, 'std_tree_dist': 1.0497615416368438, 'mean_root_purity': 0.5011666666666666, 'std_root_purity': 0.15240671099251357}
[Summary] depth2_final | euclidean | lambda_recon=100.0: best_val=2.030545, test_recall=0.8702, corr=-0.0155
[Summary] depth2_final | euclidean | lambda_recon=1000.0: best_val=2.439485, test_recall=0.9227, corr=0.0527

