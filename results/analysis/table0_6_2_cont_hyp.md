Using device: mps

depth2_final | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

--- Running hyperbolic ---

Training HYPERBOLIC | Depth 2 | lambda_recon=1.0
Epoch   1 | Train 23.76987 | Val 15.37830 | lambda_recon=1.0
Epoch   2 | Train 14.69129 | Val 10.44064 | lambda_recon=1.0
Epoch   3 | Train 11.33262 | Val 7.66309 | lambda_recon=1.0
Epoch   4 | Train 9.58796 | Val 6.22485 | lambda_recon=1.0
Epoch   5 | Train 8.57221 | Val 5.49476 | lambda_recon=1.0
Epoch   6 | Train 8.00454 | Val 4.99529 | lambda_recon=1.0
Epoch   7 | Train 7.48506 | Val 4.31411 | lambda_recon=1.0
Epoch   8 | Train 6.92696 | Val 3.73814 | lambda_recon=1.0
Epoch   9 | Train 6.44815 | Val 3.36649 | lambda_recon=1.0
Epoch  10 | Train 6.17637 | Val 3.25031 | lambda_recon=1.0
Epoch  11 | Train 6.02641 | Val 3.06843 | lambda_recon=1.0
Epoch  12 | Train 5.94084 | Val 3.04304 | lambda_recon=1.0
Epoch  13 | Train 5.87169 | Val 3.00814 | lambda_recon=1.0
Epoch  14 | Train 5.76183 | Val 3.00349 | lambda_recon=1.0
Epoch  15 | Train 5.69516 | Val 2.97967 | lambda_recon=1.0
Epoch  16 | Train 5.69378 | Val 2.96279 | lambda_recon=1.0
Epoch  17 | Train 5.62657 | Val 3.03224 | lambda_recon=1.0
Epoch  18 | Train 5.56639 | Val 3.03708 | lambda_recon=1.0
Epoch  19 | Train 5.57313 | Val 2.90801 | lambda_recon=1.0
Epoch  20 | Train 5.49389 | Val 2.80336 | lambda_recon=1.0
Epoch  21 | Train 5.47361 | Val 2.84513 | lambda_recon=1.0
Epoch  22 | Train 5.45488 | Val 2.79313 | lambda_recon=1.0
Epoch  23 | Train 5.40732 | Val 2.79977 | lambda_recon=1.0
Epoch  24 | Train 5.36577 | Val 2.77643 | lambda_recon=1.0
Epoch  25 | Train 5.34267 | Val 2.73808 | lambda_recon=1.0
Epoch  26 | Train 5.18061 | Val 2.45371 | lambda_recon=1.0
Epoch  27 | Train 4.97036 | Val 2.24901 | lambda_recon=1.0
Epoch  28 | Train 4.85032 | Val 2.15094 | lambda_recon=1.0
Epoch  29 | Train 4.79043 | Val 2.14291 | lambda_recon=1.0
Epoch  30 | Train 4.72187 | Val 2.06802 | lambda_recon=1.0
Epoch  31 | Train 4.72904 | Val 2.10547 | lambda_recon=1.0
Epoch  32 | Train 4.71172 | Val 2.11105 | lambda_recon=1.0
Epoch  33 | Train 4.66233 | Val 2.03061 | lambda_recon=1.0
Epoch  34 | Train 4.64537 | Val 2.08256 | lambda_recon=1.0
Epoch  35 | Train 4.67352 | Val 2.01999 | lambda_recon=1.0
Epoch  36 | Train 4.61609 | Val 2.00764 | lambda_recon=1.0
Epoch  37 | Train 4.57035 | Val 2.08184 | lambda_recon=1.0
Epoch  38 | Train 4.58672 | Val 2.06168 | lambda_recon=1.0
Epoch  39 | Train 4.59037 | Val 2.04445 | lambda_recon=1.0
Epoch  40 | Train 4.55747 | Val 1.99099 | lambda_recon=1.0
Epoch  41 | Train 4.55797 | Val 2.04994 | lambda_recon=1.0
Epoch  42 | Train 4.56355 | Val 1.97926 | lambda_recon=1.0
Epoch  43 | Train 4.57297 | Val 2.06104 | lambda_recon=1.0
Epoch  44 | Train 4.53305 | Val 1.98568 | lambda_recon=1.0
Epoch  45 | Train 4.51081 | Val 1.99763 | lambda_recon=1.0
Epoch  46 | Train 4.55536 | Val 1.99566 | lambda_recon=1.0
Epoch  47 | Train 4.52662 | Val 1.99044 | lambda_recon=1.0
Early stopping triggered.
Best validation loss (lambda_recon=1.0): 1.979262
Test Recall@4 (lambda_recon=1.0): 0.0549

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C023', 'C11', 'C214', 'C223']
  Visit 2: ['C034', 'C11', 'C214', 'C223']
  Visit 3: ['C203', 'C222', 'C330', 'C333']
  Visit 4: ['C013', 'C203', 'C234', 'C321']
  Visit 5: ['C013', 'C041', 'C132', 'C321']
  Visit 6: ['C013', 'C203', 'C234', 'C321']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C013', 'C132', 'C203', 'C321']
  Visit 2: ['C023', 'C11', 'C214', 'C223']
  Visit 3: ['C023', 'C11', 'C214', 'C223']
  Visit 4: ['C013', 'C203', 'C234', 'C321']
  Visit 5: ['C023', 'C11', 'C223', 'C443']
  Visit 6: ['C034', 'C11', 'C214', 'C223']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C013', 'C203', 'C234', 'C321']
  Visit 2: ['C023', 'C034', 'C11', 'C223']
  Visit 3: ['C013', 'C041', 'C203', 'C321']
  Visit 4: ['C013', 'C203', 'C234', 'C321']
  Visit 5: ['C01', 'C013', 'C041', 'C302']
  Visit 6: ['C034', 'C11', 'C214', 'C223']
Tree-Embedding Correlation (lambda_recon=1.0): 0.0913
Synthetic (hyperbolic, lambda_recon=1.0) stats (N=1000): {'mean_depth': 1.8545120474380925, 'std_depth': 0.35507053011727324, 'mean_tree_dist': 3.8367957453880672, 'std_tree_dist': 0.5689508525547257, 'mean_root_purity': 0.49336111111111114, 'std_root_purity': 0.06711584213456685}

Training HYPERBOLIC | Depth 2 | lambda_recon=10.0
Epoch   1 | Train 23.02128 | Val 15.11228 | lambda_recon=10.0
Epoch   2 | Train 14.81809 | Val 11.28898 | lambda_recon=10.0
Epoch   3 | Train 12.24520 | Val 8.76167 | lambda_recon=10.0
Epoch   4 | Train 10.43734 | Val 7.20249 | lambda_recon=10.0
Epoch   5 | Train 9.49258 | Val 6.57403 | lambda_recon=10.0
Epoch   6 | Train 8.68570 | Val 5.56898 | lambda_recon=10.0
Epoch   7 | Train 7.75510 | Val 4.60670 | lambda_recon=10.0
Epoch   8 | Train 7.16784 | Val 4.25185 | lambda_recon=10.0
Epoch   9 | Train 6.86291 | Val 4.00304 | lambda_recon=10.0
Epoch  10 | Train 6.55410 | Val 3.60881 | lambda_recon=10.0
Epoch  11 | Train 6.25114 | Val 3.36871 | lambda_recon=10.0
Epoch  12 | Train 6.03998 | Val 3.28362 | lambda_recon=10.0
Epoch  13 | Train 5.93197 | Val 3.17653 | lambda_recon=10.0
Epoch  14 | Train 5.82569 | Val 3.19517 | lambda_recon=10.0
Epoch  15 | Train 5.70276 | Val 3.12661 | lambda_recon=10.0
Epoch  16 | Train 5.64427 | Val 3.12405 | lambda_recon=10.0
Epoch  17 | Train 5.65152 | Val 2.98880 | lambda_recon=10.0
Epoch  18 | Train 5.56482 | Val 3.01101 | lambda_recon=10.0
Epoch  19 | Train 5.54322 | Val 3.07320 | lambda_recon=10.0
Epoch  20 | Train 5.51556 | Val 2.95843 | lambda_recon=10.0
Epoch  21 | Train 5.43662 | Val 2.90540 | lambda_recon=10.0
Epoch  22 | Train 5.44125 | Val 2.87036 | lambda_recon=10.0
Epoch  23 | Train 5.41149 | Val 2.93125 | lambda_recon=10.0
Epoch  24 | Train 5.38925 | Val 2.95895 | lambda_recon=10.0
Epoch  25 | Train 5.34268 | Val 2.92981 | lambda_recon=10.0
Epoch  26 | Train 5.32526 | Val 2.95192 | lambda_recon=10.0
Epoch  27 | Train 5.27696 | Val 2.97527 | lambda_recon=10.0
Early stopping triggered.
Best validation loss (lambda_recon=10.0): 2.870362
Test Recall@4 (lambda_recon=10.0): 0.0516

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C320', 'C434', 'C443']
  Visit 2: ['C110', 'C123', 'C2']
  Visit 3: ['C041', 'C1', 'C233', 'C3']
  Visit 4: ['C13', 'C24', 'C31', 'C41']
  Visit 5: ['C13', 'C24', 'C31', 'C41']
  Visit 6: ['C10', 'C12', 'C21', 'C24']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C1', 'C320', 'C434']
  Visit 2: ['C13', 'C21', 'C24', 'C41']
  Visit 3: ['C02', 'C041', 'C24', 'C40']
  Visit 4: ['C022', 'C123', 'C42']
  Visit 5: ['C02', 'C041', 'C32', 'C40']
  Visit 6: ['C13', 'C24', 'C31', 'C41']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C02', 'C24', 'C32', 'C40']
  Visit 2: ['C13', 'C24', 'C41', 'C444']
  Visit 3: ['C024', 'C434', 'C443']
  Visit 4: ['C12', 'C21', 'C24', 'C44']
  Visit 5: ['C104', 'C424', 'C443']
  Visit 6: ['C13', 'C24', 'C41', 'C42']
Tree-Embedding Correlation (lambda_recon=10.0): -0.3309
Synthetic (hyperbolic, lambda_recon=10.0) stats (N=1000): {'mean_depth': 1.3005519779208832, 'std_depth': 0.6608444799581215, 'mean_tree_dist': 2.8232923207467118, 'std_tree_dist': 0.9318985897526333, 'mean_root_purity': 0.4626944444444444, 'std_root_purity': 0.13684425848477133}

Training HYPERBOLIC | Depth 2 | lambda_recon=100.0
Epoch   1 | Train 24.00755 | Val 16.13650 | lambda_recon=100.0
Epoch   2 | Train 15.27118 | Val 11.21677 | lambda_recon=100.0
Epoch   3 | Train 12.39578 | Val 8.93331 | lambda_recon=100.0
Epoch   4 | Train 10.92041 | Val 7.83495 | lambda_recon=100.0
Epoch   5 | Train 9.72231 | Val 6.59577 | lambda_recon=100.0
Epoch   6 | Train 8.92894 | Val 5.91352 | lambda_recon=100.0
Epoch   7 | Train 8.20095 | Val 5.05539 | lambda_recon=100.0
Epoch   8 | Train 7.70391 | Val 4.72115 | lambda_recon=100.0
Epoch   9 | Train 7.18282 | Val 3.98746 | lambda_recon=100.0
Epoch  10 | Train 6.77836 | Val 3.87276 | lambda_recon=100.0
Epoch  11 | Train 6.61005 | Val 3.76315 | lambda_recon=100.0
Epoch  12 | Train 6.48362 | Val 3.70175 | lambda_recon=100.0
Epoch  13 | Train 6.39280 | Val 3.59475 | lambda_recon=100.0
Epoch  14 | Train 6.33780 | Val 3.74744 | lambda_recon=100.0
Epoch  15 | Train 6.25600 | Val 3.48826 | lambda_recon=100.0
Epoch  16 | Train 6.20683 | Val 3.41528 | lambda_recon=100.0
Epoch  17 | Train 5.98521 | Val 3.11423 | lambda_recon=100.0
Epoch  18 | Train 5.79074 | Val 2.95279 | lambda_recon=100.0
Epoch  19 | Train 5.63380 | Val 2.96263 | lambda_recon=100.0
Epoch  20 | Train 5.58428 | Val 2.79820 | lambda_recon=100.0
Epoch  21 | Train 5.51515 | Val 2.78344 | lambda_recon=100.0
Epoch  22 | Train 5.46904 | Val 2.78333 | lambda_recon=100.0
Epoch  23 | Train 5.44175 | Val 2.79126 | lambda_recon=100.0
Epoch  24 | Train 5.35864 | Val 2.67395 | lambda_recon=100.0
Epoch  25 | Train 5.35170 | Val 2.64656 | lambda_recon=100.0
Epoch  26 | Train 5.29821 | Val 2.61318 | lambda_recon=100.0
Epoch  27 | Train 5.26282 | Val 2.63992 | lambda_recon=100.0
Epoch  28 | Train 5.27178 | Val 2.57751 | lambda_recon=100.0
Epoch  29 | Train 5.22209 | Val 2.55449 | lambda_recon=100.0
Epoch  30 | Train 5.19387 | Val 2.51482 | lambda_recon=100.0
Epoch  31 | Train 5.13820 | Val 2.52983 | lambda_recon=100.0
Epoch  32 | Train 5.17180 | Val 2.57547 | lambda_recon=100.0
Epoch  33 | Train 5.14810 | Val 2.68780 | lambda_recon=100.0
Epoch  34 | Train 5.15315 | Val 2.49924 | lambda_recon=100.0
Epoch  35 | Train 5.10688 | Val 2.51962 | lambda_recon=100.0
Epoch  36 | Train 5.09578 | Val 2.47054 | lambda_recon=100.0
Epoch  37 | Train 5.12223 | Val 2.49802 | lambda_recon=100.0
Epoch  38 | Train 5.06942 | Val 2.55101 | lambda_recon=100.0
Epoch  39 | Train 5.06198 | Val 2.50905 | lambda_recon=100.0
Epoch  40 | Train 5.04815 | Val 2.47183 | lambda_recon=100.0
Epoch  41 | Train 5.05403 | Val 2.45617 | lambda_recon=100.0
Epoch  42 | Train 5.00464 | Val 2.61374 | lambda_recon=100.0
Epoch  43 | Train 5.02253 | Val 2.43645 | lambda_recon=100.0
Epoch  44 | Train 5.01661 | Val 2.51565 | lambda_recon=100.0
Epoch  45 | Train 5.04026 | Val 2.48808 | lambda_recon=100.0
Epoch  46 | Train 5.06121 | Val 2.41965 | lambda_recon=100.0
Epoch  47 | Train 5.03680 | Val 2.56528 | lambda_recon=100.0
Epoch  48 | Train 5.02950 | Val 2.41563 | lambda_recon=100.0
Epoch  49 | Train 5.05334 | Val 2.43703 | lambda_recon=100.0
Epoch  50 | Train 5.05591 | Val 2.47466 | lambda_recon=100.0
Best validation loss (lambda_recon=100.0): 2.415631
Test Recall@4 (lambda_recon=100.0): 0.1587

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C21', 'C32', 'C33', 'C334']
  Visit 2: ['C01', 'C02', 'C10', 'C21']
  Visit 3: ['C21', 'C32', 'C33', 'C334']
  Visit 4: ['C21', 'C214', 'C32', 'C33']
  Visit 5: ['C00', 'C11', 'C42']
  Visit 6: ['C012', 'C023', 'C101', 'C131']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C00', 'C11', 'C41']
  Visit 2: ['C00', 'C12', 'C22', 'C42']
  Visit 3: ['C12', 'C22', 'C23', 'C34']
  Visit 4: ['C00', 'C11', 'C41', 'C42']
  Visit 5: ['C21', 'C32', 'C33', 'C334']
  Visit 6: ['C00', 'C11', 'C41']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C022', 'C101', 'C301', 'C303']
  Visit 2: ['C102', 'C30', 'C301', 'C303']
  Visit 3: ['C21', 'C32', 'C33', 'C334']
  Visit 4: ['C21', 'C214', 'C32', 'C33']
  Visit 5: ['C21', 'C32', 'C33', 'C40']
  Visit 6: ['C00', 'C11', 'C41']
Tree-Embedding Correlation (lambda_recon=100.0): -0.0071
Synthetic (hyperbolic, lambda_recon=100.0) stats (N=1000): {'mean_depth': 1.155748316934231, 'std_depth': 0.36781580386090157, 'mean_tree_dist': 1.9693782864212805, 'std_tree_dist': 0.7683491280815544, 'mean_root_purity': 0.5440138888888888, 'std_root_purity': 0.178973538784024}

Training HYPERBOLIC | Depth 2 | lambda_recon=1000.0
Epoch   1 | Train 29.10394 | Val 21.22764 | lambda_recon=1000.0
Epoch   2 | Train 20.89914 | Val 17.02229 | lambda_recon=1000.0
Epoch   3 | Train 18.02523 | Val 14.64741 | lambda_recon=1000.0
Epoch   4 | Train 16.16351 | Val 12.91462 | lambda_recon=1000.0
Epoch   5 | Train 14.95984 | Val 11.66513 | lambda_recon=1000.0
Epoch   6 | Train 13.62794 | Val 9.94032 | lambda_recon=1000.0
Epoch   7 | Train 12.39519 | Val 9.17821 | lambda_recon=1000.0
Epoch   8 | Train 11.86306 | Val 8.65698 | lambda_recon=1000.0
Epoch   9 | Train 11.40378 | Val 8.34722 | lambda_recon=1000.0
Epoch  10 | Train 10.97117 | Val 7.88855 | lambda_recon=1000.0
Epoch  11 | Train 10.51326 | Val 7.53600 | lambda_recon=1000.0
Epoch  12 | Train 10.11705 | Val 7.35750 | lambda_recon=1000.0
Epoch  13 | Train 9.92653 | Val 7.09446 | lambda_recon=1000.0
Epoch  14 | Train 9.73617 | Val 6.64872 | lambda_recon=1000.0
Epoch  15 | Train 9.42630 | Val 6.72915 | lambda_recon=1000.0
Epoch  16 | Train 9.27762 | Val 6.50899 | lambda_recon=1000.0
Epoch  17 | Train 9.11950 | Val 6.22769 | lambda_recon=1000.0
Epoch  18 | Train 8.97858 | Val 6.36405 | lambda_recon=1000.0
Epoch  19 | Train 8.89920 | Val 6.13150 | lambda_recon=1000.0
Epoch  20 | Train 8.84458 | Val 6.21867 | lambda_recon=1000.0
Epoch  21 | Train 8.73538 | Val 5.92797 | lambda_recon=1000.0
Epoch  22 | Train 8.60377 | Val 5.76387 | lambda_recon=1000.0
Epoch  23 | Train 8.64139 | Val 5.83866 | lambda_recon=1000.0
Epoch  24 | Train 8.50530 | Val 6.04738 | lambda_recon=1000.0
Epoch  25 | Train 8.50857 | Val 5.79851 | lambda_recon=1000.0
Epoch  26 | Train 8.42137 | Val 5.78107 | lambda_recon=1000.0
Epoch  27 | Train 8.27667 | Val 5.69726 | lambda_recon=1000.0
Epoch  28 | Train 8.28851 | Val 5.65791 | lambda_recon=1000.0
Epoch  29 | Train 8.26851 | Val 5.72212 | lambda_recon=1000.0
Epoch  30 | Train 8.26008 | Val 5.64422 | lambda_recon=1000.0
Epoch  31 | Train 8.18700 | Val 5.46578 | lambda_recon=1000.0
Epoch  32 | Train 8.19186 | Val 5.70144 | lambda_recon=1000.0
Epoch  33 | Train 8.12505 | Val 5.51571 | lambda_recon=1000.0
Epoch  34 | Train 8.08864 | Val 5.57093 | lambda_recon=1000.0
Epoch  35 | Train 8.07022 | Val 5.48782 | lambda_recon=1000.0
Epoch  36 | Train 7.99163 | Val 5.44046 | lambda_recon=1000.0
Epoch  37 | Train 7.95363 | Val 5.48636 | lambda_recon=1000.0
Epoch  38 | Train 7.97266 | Val 5.43639 | lambda_recon=1000.0
Epoch  39 | Train 7.95230 | Val 5.38075 | lambda_recon=1000.0
Epoch  40 | Train 7.92662 | Val 5.34960 | lambda_recon=1000.0
Epoch  41 | Train 7.88331 | Val 5.32722 | lambda_recon=1000.0
Epoch  42 | Train 7.92903 | Val 5.39571 | lambda_recon=1000.0
Epoch  43 | Train 7.83920 | Val 5.30313 | lambda_recon=1000.0
Epoch  44 | Train 7.84976 | Val 5.40382 | lambda_recon=1000.0
Epoch  45 | Train 7.84300 | Val 5.30007 | lambda_recon=1000.0
Epoch  46 | Train 7.80429 | Val 5.37209 | lambda_recon=1000.0
Epoch  47 | Train 7.83928 | Val 5.32279 | lambda_recon=1000.0
Epoch  48 | Train 7.94995 | Val 5.28248 | lambda_recon=1000.0
Epoch  49 | Train 7.99263 | Val 5.40879 | lambda_recon=1000.0
Epoch  50 | Train 8.18878 | Val 5.65008 | lambda_recon=1000.0
Best validation loss (lambda_recon=1000.0): 5.282482
Test Recall@4 (lambda_recon=1000.0): 0.5962

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C31', 'C311', 'C313', 'C314']
  Visit 2: ['C32', 'C320', 'C322']
  Visit 3: ['C32', 'C320', 'C322']
  Visit 4: ['C23', 'C232', 'C233', 'C304']
  Visit 5: ['C000', 'C001', 'C320', 'C322']
  Visit 6: ['C241', 'C243', 'C30', 'C304']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C000', 'C43', 'C430', 'C432']
  Visit 2: ['C23', 'C230', 'C232', 'C233']
  Visit 3: ['C32', 'C320', 'C322']
  Visit 4: ['C003', 'C021', 'C221', 'C432']
  Visit 5: ['C000', 'C32', 'C320', 'C322']
  Visit 6: ['C001', 'C32', 'C440']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C32', 'C320', 'C322']
  Visit 2: ['C13', 'C130', 'C243', 'C304']
  Visit 3: ['C024', 'C243', 'C304', 'C403']
  Visit 4: ['C000', 'C32', 'C320', 'C322']
  Visit 5: ['C32', 'C322', 'C440']
  Visit 6: ['C000', 'C32', 'C320', 'C324']
Tree-Embedding Correlation (lambda_recon=1000.0): 0.5987
Synthetic (hyperbolic, lambda_recon=1000.0) stats (N=1000): {'mean_depth': 1.812440645773979, 'std_depth': 0.3994327436511426, 'mean_tree_dist': 1.9616150127949958, 'std_tree_dist': 1.1097466745855, 'mean_root_purity': 0.6600416666666666, 'std_root_purity': 0.19845199094071425}
[Summary] depth2_final | hyperbolic | lambda_recon=1.0: best_val=1.979262, test_recall=0.0549, corr=0.0913
[Summary] depth2_final | hyperbolic | lambda_recon=10.0: best_val=2.870362, test_recall=0.0516, corr=-0.3309
[Summary] depth2_final | hyperbolic | lambda_recon=100.0: best_val=2.415631, test_recall=0.1587, corr=-0.0071
[Summary] depth2_final | hyperbolic | lambda_recon=1000.0: best_val=5.282482, test_recall=0.5962, corr=0.5987
