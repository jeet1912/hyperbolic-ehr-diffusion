
depth7_final | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

--- Running euclidean ---
Using device: mps

Training EUCLIDEAN | Depth 7 | lambda_recon=1.0
Epoch   1 | Train 22.97804 | Val 15.01984 | lambda_recon=1.0
Epoch   2 | Train 14.46133 | Val 10.42288 | lambda_recon=1.0
Epoch   3 | Train 11.40272 | Val 8.03328 | lambda_recon=1.0
Epoch   4 | Train 9.83752 | Val 6.97662 | lambda_recon=1.0
Epoch   5 | Train 9.14786 | Val 6.34838 | lambda_recon=1.0
Epoch   6 | Train 8.65437 | Val 5.82616 | lambda_recon=1.0
Epoch   7 | Train 8.16841 | Val 5.44554 | lambda_recon=1.0
Epoch   8 | Train 7.88464 | Val 5.18122 | lambda_recon=1.0
Epoch   9 | Train 7.52223 | Val 4.66139 | lambda_recon=1.0
Epoch  10 | Train 6.99130 | Val 4.31142 | lambda_recon=1.0
Epoch  11 | Train 6.81321 | Val 4.15681 | lambda_recon=1.0
Epoch  12 | Train 6.62707 | Val 3.92649 | lambda_recon=1.0
Epoch  13 | Train 6.20774 | Val 3.27829 | lambda_recon=1.0
Epoch  14 | Train 5.85498 | Val 3.16941 | lambda_recon=1.0
Epoch  15 | Train 5.72575 | Val 2.93555 | lambda_recon=1.0
Epoch  16 | Train 5.69084 | Val 3.07749 | lambda_recon=1.0
Epoch  17 | Train 5.60586 | Val 2.95437 | lambda_recon=1.0
Epoch  18 | Train 5.52701 | Val 3.00296 | lambda_recon=1.0
Epoch  19 | Train 5.48488 | Val 2.94382 | lambda_recon=1.0
Epoch  20 | Train 5.44575 | Val 2.97255 | lambda_recon=1.0
Early stopping triggered.
Best validation loss (lambda_recon=1.0): 2.935548
Saved loss curves to results/plots
Test Recall@4 (lambda_recon=1.0): 0.0098

Sample trajectory (euclidean) 1:
  Visit 1: ['C230', 'C212d0', 'C310d0', 'C323d1']
  Visit 2: ['C131d3', 'C131d4', 'C244d3', 'C340d3']
  Visit 3: ['C131d3', 'C244d3', 'C331d4', 'C340d3']
  Visit 4: ['C003', 'C001d2', 'C323d1', 'C441d1']
  Visit 5: ['C230', 'C212d0', 'C310d0', 'C323d1']
  Visit 6: ['C002d3', 'C131d3', 'C440d4', 'C443d3']

Sample trajectory (euclidean) 2:
  Visit 1: ['C103d4', 'C331d3', 'C331d4', 'C340d3']
  Visit 2: ['C230', 'C212d0', 'C310d0', 'C323d1']
  Visit 3: ['C230', 'C002d4', 'C022d1', 'C410d3']
  Visit 4: ['C230', 'C212d0', 'C310d0', 'C323d1']
  Visit 5: ['C131d3', 'C131d4', 'C244d3', 'C340d3']
  Visit 6: ['C230', 'C111d1', 'C310d0', 'C323d1']

Sample trajectory (euclidean) 3:
  Visit 1: ['C131d3', 'C131d4', 'C244d3', 'C340d3']
  Visit 2: ['C103d4', 'C214d3', 'C331d4', 'C420d3']
  Visit 3: ['C001d2', 'C002d4', 'C323d1', 'C441d1']
  Visit 4: ['C111d1', 'C212d0', 'C310d0', 'C323d1']
  Visit 5: ['C230', 'C212d0', 'C301d1', 'C310d0']
  Visit 6: ['C230', 'C212d0', 'C310d0', 'C323d1']
Tree-Embedding Correlation (lambda_recon=1.0): 0.0303
Synthetic (euclidean, lambda_recon=1.0) stats (N=1000): {'mean_depth': 5.4627083333333335, 'std_depth': 1.5876689405951594, 'mean_tree_dist': 7.234317343173432, 'std_tree_dist': 4.887808030028154, 'mean_root_purity': 0.523125, 'std_root_purity': 0.11900903764700674}

Training EUCLIDEAN | Depth 7 | lambda_recon=10.0
Epoch   1 | Train 23.30467 | Val 15.64291 | lambda_recon=10.0
Epoch   2 | Train 14.70863 | Val 10.54333 | lambda_recon=10.0
Epoch   3 | Train 11.70147 | Val 8.30895 | lambda_recon=10.0
Epoch   4 | Train 10.04336 | Val 6.98375 | lambda_recon=10.0
Epoch   5 | Train 8.89160 | Val 5.70671 | lambda_recon=10.0
Epoch   6 | Train 8.06681 | Val 4.96411 | lambda_recon=10.0
Epoch   7 | Train 7.55462 | Val 4.64950 | lambda_recon=10.0
Epoch   8 | Train 7.16904 | Val 4.30240 | lambda_recon=10.0
Epoch   9 | Train 6.85512 | Val 4.01923 | lambda_recon=10.0
Epoch  10 | Train 6.52713 | Val 3.67795 | lambda_recon=10.0
Epoch  11 | Train 6.19276 | Val 3.35631 | lambda_recon=10.0
Epoch  12 | Train 6.03507 | Val 3.17970 | lambda_recon=10.0
Epoch  13 | Train 5.89524 | Val 3.18830 | lambda_recon=10.0
Epoch  14 | Train 5.83224 | Val 3.11959 | lambda_recon=10.0
Epoch  15 | Train 5.75098 | Val 3.04455 | lambda_recon=10.0
Epoch  16 | Train 5.66280 | Val 2.98456 | lambda_recon=10.0
Epoch  17 | Train 5.61277 | Val 2.99591 | lambda_recon=10.0
Epoch  18 | Train 5.61871 | Val 3.03305 | lambda_recon=10.0
Epoch  19 | Train 5.49558 | Val 2.91291 | lambda_recon=10.0
Epoch  20 | Train 5.47975 | Val 2.96032 | lambda_recon=10.0
Epoch  21 | Train 5.46550 | Val 2.90426 | lambda_recon=10.0
Epoch  22 | Train 5.40823 | Val 2.94190 | lambda_recon=10.0
Epoch  23 | Train 5.37815 | Val 2.89399 | lambda_recon=10.0
Epoch  24 | Train 5.35456 | Val 2.92756 | lambda_recon=10.0
Epoch  25 | Train 5.32530 | Val 2.86597 | lambda_recon=10.0
Epoch  26 | Train 5.29788 | Val 2.91831 | lambda_recon=10.0
Epoch  27 | Train 5.28231 | Val 2.90523 | lambda_recon=10.0
Epoch  28 | Train 5.25490 | Val 2.90944 | lambda_recon=10.0
Epoch  29 | Train 5.22722 | Val 2.88170 | lambda_recon=10.0
Epoch  30 | Train 5.19278 | Val 2.89426 | lambda_recon=10.0
Early stopping triggered.
Best validation loss (lambda_recon=10.0): 2.865968
Saved loss curves to results/plots
Test Recall@4 (lambda_recon=10.0): 0.0111

Sample trajectory (euclidean) 1:
  Visit 1: ['C100d4', 'C134d3', 'C213d3', 'C301d4']
  Visit 2: ['C100d4', 'C134d3', 'C213d3', 'C301d4']
  Visit 3: ['C100d4', 'C134d3', 'C213d3', 'C414d4']
  Visit 4: ['C100d4', 'C134d3', 'C213d3', 'C414d4']
  Visit 5: ['C100d4', 'C134d3', 'C213d3', 'C414d4']
  Visit 6: ['C100d4', 'C134d3', 'C213d3', 'C301d4']

Sample trajectory (euclidean) 2:
  Visit 1: ['C134d3', 'C134d4', 'C213d3', 'C304d3']
  Visit 2: ['C100d4', 'C134d3', 'C213d3', 'C301d4']
  Visit 3: ['C100d4', 'C134d3', 'C213d3', 'C301d4']
  Visit 4: ['C134d3', 'C213d3', 'C304d3', 'C414d4']
  Visit 5: ['C100d4', 'C134d3', 'C213d3', 'C414d4']
  Visit 6: ['C224', 'C43', 'C431', 'C441d1']

Sample trajectory (euclidean) 3:
  Visit 1: ['C43', 'C214d1', 'C242d2', 'C441d1']
  Visit 2: ['C100d4', 'C134d3', 'C213d3', 'C414d4']
  Visit 3: ['C010d3', 'C121d4', 'C342d4', 'C414d4']
  Visit 4: ['C224', 'C43', 'C431', 'C441d1']
  Visit 5: ['C100d4', 'C134d3', 'C213d3', 'C414d4']
  Visit 6: ['C100d4', 'C134d3', 'C213d3', 'C301d4']
Tree-Embedding Correlation (lambda_recon=10.0): -0.0285
Synthetic (euclidean, lambda_recon=10.0) stats (N=1000): {'mean_depth': 5.340708333333334, 'std_depth': 1.972319150542635, 'mean_tree_dist': 7.972628160519601, 'std_tree_dist': 4.747333674560458, 'mean_root_purity': 0.542625, 'std_root_purity': 0.14040783231358572}

Training EUCLIDEAN | Depth 7 | lambda_recon=100.0
Epoch   1 | Train 23.83525 | Val 16.10357 | lambda_recon=100.0
Epoch   2 | Train 15.44911 | Val 11.73222 | lambda_recon=100.0
Epoch   3 | Train 12.36081 | Val 8.78517 | lambda_recon=100.0
Epoch   4 | Train 10.51571 | Val 7.24652 | lambda_recon=100.0
Epoch   5 | Train 9.29091 | Val 5.96248 | lambda_recon=100.0
Epoch   6 | Train 8.29017 | Val 5.43443 | lambda_recon=100.0
Epoch   7 | Train 7.92256 | Val 5.07064 | lambda_recon=100.0
Epoch   8 | Train 7.50609 | Val 4.61038 | lambda_recon=100.0
Epoch   9 | Train 7.16779 | Val 4.29049 | lambda_recon=100.0
Epoch  10 | Train 6.78076 | Val 3.77556 | lambda_recon=100.0
Epoch  11 | Train 6.45625 | Val 3.53637 | lambda_recon=100.0
Epoch  12 | Train 6.23760 | Val 3.41298 | lambda_recon=100.0
Epoch  13 | Train 6.06092 | Val 3.31744 | lambda_recon=100.0
Epoch  14 | Train 5.96236 | Val 3.25716 | lambda_recon=100.0
Epoch  15 | Train 5.93361 | Val 3.16713 | lambda_recon=100.0
Epoch  16 | Train 5.86184 | Val 3.23117 | lambda_recon=100.0
Epoch  17 | Train 5.79706 | Val 3.11266 | lambda_recon=100.0
Epoch  18 | Train 5.73412 | Val 3.17393 | lambda_recon=100.0
Epoch  19 | Train 5.71081 | Val 3.11923 | lambda_recon=100.0
Epoch  20 | Train 5.65336 | Val 3.00845 | lambda_recon=100.0
Epoch  21 | Train 5.60565 | Val 3.03174 | lambda_recon=100.0
Epoch  22 | Train 5.56762 | Val 3.03613 | lambda_recon=100.0
Epoch  23 | Train 5.52207 | Val 3.09364 | lambda_recon=100.0
Epoch  24 | Train 5.49226 | Val 3.01683 | lambda_recon=100.0
Epoch  25 | Train 5.46102 | Val 3.01617 | lambda_recon=100.0
Early stopping triggered.
Best validation loss (lambda_recon=100.0): 3.008448
Saved loss curves to results/plots
Test Recall@4 (lambda_recon=100.0): 0.0296

Sample trajectory (euclidean) 1:
  Visit 1: ['C10', 'C231d3', 'C231d4', 'C331d0']
  Visit 2: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 3: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 4: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 5: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 6: ['C231d3', 'C231d4', 'C420d3', 'C420d4']

Sample trajectory (euclidean) 2:
  Visit 1: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 2: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 3: ['C10', 'C231d3', 'C231d4', 'C420d4']
  Visit 4: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 5: ['C10', 'C231d3', 'C231d4', 'C420d4']
  Visit 6: ['C231d3', 'C231d4', 'C420d3', 'C420d4']

Sample trajectory (euclidean) 3:
  Visit 1: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 2: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 3: ['C10', 'C231d3', 'C231d4', 'C420d4']
  Visit 4: ['C10', 'C231d3', 'C231d4', 'C420d4']
  Visit 5: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
  Visit 6: ['C231d3', 'C231d4', 'C420d3', 'C420d4']
Tree-Embedding Correlation (lambda_recon=100.0): 0.0109
Synthetic (euclidean, lambda_recon=100.0) stats (N=1000): {'mean_depth': 5.736166666666667, 'std_depth': 1.826396809446281, 'mean_tree_dist': 1.4805168170631666, 'std_tree_dist': 2.337593974615817, 'mean_root_purity': 0.5031666666666667, 'std_root_purity': 0.03579998448168503}

Training EUCLIDEAN | Depth 7 | lambda_recon=1000.0
Epoch   1 | Train 25.99765 | Val 16.57386 | lambda_recon=1000.0
Epoch   2 | Train 16.14643 | Val 12.27963 | lambda_recon=1000.0
Epoch   3 | Train 13.26014 | Val 9.72179 | lambda_recon=1000.0
Epoch   4 | Train 11.33221 | Val 8.02328 | lambda_recon=1000.0
Epoch   5 | Train 10.31403 | Val 7.37549 | lambda_recon=1000.0
Epoch   6 | Train 9.74630 | Val 6.91752 | lambda_recon=1000.0
Epoch   7 | Train 9.34897 | Val 6.60095 | lambda_recon=1000.0
Epoch   8 | Train 8.94440 | Val 6.21827 | lambda_recon=1000.0
Epoch   9 | Train 8.57257 | Val 5.72011 | lambda_recon=1000.0
Epoch  10 | Train 8.19470 | Val 5.46049 | lambda_recon=1000.0
Epoch  11 | Train 8.01476 | Val 5.36001 | lambda_recon=1000.0
Epoch  12 | Train 7.73533 | Val 4.99574 | lambda_recon=1000.0
Epoch  13 | Train 7.47737 | Val 4.65020 | lambda_recon=1000.0
Epoch  14 | Train 7.16854 | Val 4.34013 | lambda_recon=1000.0
Epoch  15 | Train 7.04810 | Val 4.43730 | lambda_recon=1000.0
Epoch  16 | Train 6.95905 | Val 4.30415 | lambda_recon=1000.0
Epoch  17 | Train 6.88703 | Val 4.25280 | lambda_recon=1000.0
Epoch  18 | Train 6.83655 | Val 4.26350 | lambda_recon=1000.0
Epoch  19 | Train 6.76795 | Val 4.25633 | lambda_recon=1000.0
Epoch  20 | Train 6.73148 | Val 4.07752 | lambda_recon=1000.0
Epoch  21 | Train 6.69191 | Val 4.10275 | lambda_recon=1000.0
Epoch  22 | Train 6.66854 | Val 4.12542 | lambda_recon=1000.0
Epoch  23 | Train 6.56348 | Val 4.02318 | lambda_recon=1000.0
Epoch  24 | Train 6.56888 | Val 4.03929 | lambda_recon=1000.0
Epoch  25 | Train 6.53207 | Val 4.04666 | lambda_recon=1000.0
Epoch  26 | Train 6.53026 | Val 3.98774 | lambda_recon=1000.0
Epoch  27 | Train 6.53439 | Val 3.88804 | lambda_recon=1000.0
Epoch  28 | Train 6.49320 | Val 4.06251 | lambda_recon=1000.0
Epoch  29 | Train 6.42336 | Val 3.92233 | lambda_recon=1000.0
Epoch  30 | Train 6.43339 | Val 3.93695 | lambda_recon=1000.0
Epoch  31 | Train 6.40606 | Val 4.01873 | lambda_recon=1000.0
Epoch  32 | Train 6.36714 | Val 3.91901 | lambda_recon=1000.0
Early stopping triggered.
Best validation loss (lambda_recon=1000.0): 3.888043
Saved loss curves to results/plots
Test Recall@4 (lambda_recon=1000.0): 0.4498

Sample trajectory (euclidean) 1:
  Visit 1: ['C114d3', 'C324d1', 'C331d4', 'C333d3']
  Visit 2: ['C104d4', 'C114d3', 'C333d3', 'C412d4']
  Visit 3: ['C014d1', 'C201d0', 'C231d2', 'C423d3']
  Visit 4: ['C230', 'C011d3', 'C201d0', 'C324d1']
  Visit 5: ['C114d3', 'C120d4', 'C324d1', 'C331d4']
  Visit 6: ['C114d3', 'C120d4', 'C324d1', 'C331d4']

Sample trajectory (euclidean) 2:
  Visit 1: ['C014d1', 'C033d3', 'C201d0', 'C321d1']
  Visit 2: ['C014d1', 'C201d0', 'C231d2', 'C423d3']
  Visit 3: ['C114d3', 'C320d2', 'C324d1', 'C333d3']
  Visit 4: ['C011d3', 'C114d3', 'C201d0', 'C324d1']
  Visit 5: ['C230', 'C014d1', 'C231d2', 'C423d3']
  Visit 6: ['C201d0', 'C222d3', 'C321d1', 'C431d4']

Sample trajectory (euclidean) 3:
  Visit 1: ['C201d0', 'C231d2', 'C324d1', 'C410d1']
  Visit 2: ['C014d1', 'C201d0', 'C231d2', 'C423d3']
  Visit 3: ['C230', 'C001d0', 'C040d1', 'C414d4']
  Visit 4: ['C014d1', 'C201d0', 'C231d2', 'C242d2']
  Visit 5: ['C114d3', 'C201d0', 'C222d3', 'C334d3']
  Visit 6: ['C302', 'C114d3', 'C324d1', 'C333d3']
Tree-Embedding Correlation (lambda_recon=1000.0): -0.0076
Synthetic (euclidean, lambda_recon=1000.0) stats (N=1000): {'mean_depth': 4.603916666666667, 'std_depth': 1.5346393690556603, 'mean_tree_dist': 8.847583219334245, 'std_tree_dist': 2.422563103126228, 'mean_root_purity': 0.530625, 'std_root_purity': 0.14379392444861266}
[Summary] depth7_final | euclidean | lambda_recon=1.0: best_val=2.935548, test_recall=0.0098, corr=0.0303
[Summary] depth7_final | euclidean | lambda_recon=10.0: best_val=2.865968, test_recall=0.0111, corr=-0.0285
[Summary] depth7_final | euclidean | lambda_recon=100.0: best_val=3.008448, test_recall=0.0296, corr=0.0109
[Summary] depth7_final | euclidean | lambda_recon=1000.0: best_val=3.888043, test_recall=0.4498, corr=-0.0076

--- Running hyperbolic ---
Using device: mps
