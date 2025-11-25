Using device: mps

depth7_final | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

--- Running hyperbolic ---

Training HYPERBOLIC | Depth 7 | lambda_recon=1.0
Epoch   1 | Train 23.24744 | Val 15.18705 | lambda_recon=1.0
Epoch   2 | Train 14.63316 | Val 10.57595 | lambda_recon=1.0
Epoch   3 | Train 11.71047 | Val 8.15232 | lambda_recon=1.0
Epoch   4 | Train 9.93428 | Val 6.69832 | lambda_recon=1.0
Epoch   5 | Train 8.84477 | Val 5.54862 | lambda_recon=1.0
Epoch   6 | Train 7.79767 | Val 4.70059 | lambda_recon=1.0
Epoch   7 | Train 7.34013 | Val 4.39518 | lambda_recon=1.0
Epoch   8 | Train 7.08335 | Val 4.25219 | lambda_recon=1.0
Epoch   9 | Train 6.85324 | Val 4.12940 | lambda_recon=1.0
Epoch  10 | Train 6.50874 | Val 3.46138 | lambda_recon=1.0
Epoch  11 | Train 6.00369 | Val 3.12648 | lambda_recon=1.0
Epoch  12 | Train 5.86994 | Val 3.10388 | lambda_recon=1.0
Epoch  13 | Train 5.79694 | Val 3.03493 | lambda_recon=1.0
Epoch  14 | Train 5.71455 | Val 2.99702 | lambda_recon=1.0
Epoch  15 | Train 5.62668 | Val 3.00418 | lambda_recon=1.0
Epoch  16 | Train 5.61552 | Val 2.93889 | lambda_recon=1.0
Epoch  17 | Train 5.53974 | Val 3.01609 | lambda_recon=1.0
Epoch  18 | Train 5.33214 | Val 2.62412 | lambda_recon=1.0
Epoch  19 | Train 5.08275 | Val 2.31348 | lambda_recon=1.0
Epoch  20 | Train 4.92926 | Val 2.11829 | lambda_recon=1.0
Epoch  21 | Train 4.87622 | Val 2.13282 | lambda_recon=1.0
Epoch  22 | Train 4.83583 | Val 2.06034 | lambda_recon=1.0
Epoch  23 | Train 4.75259 | Val 2.06570 | lambda_recon=1.0
Epoch  24 | Train 4.69610 | Val 2.01696 | lambda_recon=1.0
Epoch  25 | Train 4.68030 | Val 2.07924 | lambda_recon=1.0
Epoch  26 | Train 4.63074 | Val 2.01193 | lambda_recon=1.0
Epoch  27 | Train 4.63278 | Val 1.99604 | lambda_recon=1.0
Epoch  28 | Train 4.60156 | Val 1.99943 | lambda_recon=1.0
Epoch  29 | Train 4.57397 | Val 2.00336 | lambda_recon=1.0
Epoch  30 | Train 4.56127 | Val 1.94438 | lambda_recon=1.0
Epoch  31 | Train 4.55517 | Val 1.96906 | lambda_recon=1.0
Epoch  32 | Train 4.55169 | Val 2.00459 | lambda_recon=1.0
Epoch  33 | Train 4.51194 | Val 1.93723 | lambda_recon=1.0
Epoch  34 | Train 4.50396 | Val 2.00101 | lambda_recon=1.0
Epoch  35 | Train 4.53232 | Val 1.95422 | lambda_recon=1.0
Epoch  36 | Train 4.49150 | Val 1.92973 | lambda_recon=1.0
Epoch  37 | Train 4.44362 | Val 1.99586 | lambda_recon=1.0
Epoch  38 | Train 4.44545 | Val 1.97923 | lambda_recon=1.0
Epoch  39 | Train 4.46650 | Val 1.96711 | lambda_recon=1.0
Epoch  40 | Train 4.43730 | Val 1.90177 | lambda_recon=1.0
Epoch  41 | Train 4.44084 | Val 1.94792 | lambda_recon=1.0
Epoch  42 | Train 4.41838 | Val 1.87700 | lambda_recon=1.0
Epoch  43 | Train 4.44368 | Val 1.99005 | lambda_recon=1.0
Epoch  44 | Train 4.41678 | Val 1.90715 | lambda_recon=1.0
Epoch  45 | Train 4.40437 | Val 1.88532 | lambda_recon=1.0
Epoch  46 | Train 4.44517 | Val 1.90721 | lambda_recon=1.0
Epoch  47 | Train 4.39146 | Val 1.90867 | lambda_recon=1.0
Early stopping triggered.
Best validation loss (lambda_recon=1.0): 1.877002
Test Recall@4 (lambda_recon=1.0): 0.0051

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C123d0', 'C220d2', 'C322d0', 'C424d1']
  Visit 2: ['C123d0', 'C220d2', 'C322d0', 'C424d1']
  Visit 3: ['C024d0', 'C331d0', 'C401d0', 'C431d2']
  Visit 4: ['C233d0', 'C242d0', 'C312d4', 'C342d0']
  Visit 5: ['C242d0', 'C312d4', 'C321d2', 'C342d0']
  Visit 6: ['C10', 'C233d0', 'C242d0', 'C342d0']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C233d0', 'C242d0', 'C342d0', 'C424d0']
  Visit 2: ['C303', 'C323d2', 'C331d0', 'C404d1']
  Visit 3: ['C123d0', 'C220d2', 'C322d0', 'C424d1']
  Visit 4: ['C242d0', 'C342d0', 'C433d0']
  Visit 5: ['C322d0', 'C402d4', 'C424d1']
  Visit 6: ['C123d0', 'C220d2', 'C322d0', 'C424d1']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C242d0', 'C342d0', 'C433d0']
  Visit 2: ['C011d1', 'C100d2', 'C411d1']
  Visit 3: ['C233d0', 'C242d0', 'C321d2', 'C342d0']
  Visit 4: ['C233d0', 'C242d0', 'C321d2', 'C342d0']
  Visit 5: ['C111d1', 'C231d0', 'C321d2', 'C433d4']
  Visit 6: ['C303', 'C123d0', 'C220d2', 'C320d1']
Tree-Embedding Correlation (lambda_recon=1.0): -0.2411
Synthetic (hyperbolic, lambda_recon=1.0) stats (N=1000): {'mean_depth': 3.6497801082543977, 'std_depth': 1.2106016879682144, 'mean_tree_dist': 7.480277986476334, 'std_tree_dist': 1.8087009624060906, 'mean_root_purity': 0.3989305555555555, 'std_root_purity': 0.13618143803895605}

Training HYPERBOLIC | Depth 7 | lambda_recon=10.0
Epoch   1 | Train 23.46687 | Val 15.59233 | lambda_recon=10.0
Epoch   2 | Train 15.00040 | Val 11.10397 | lambda_recon=10.0
Epoch   3 | Train 12.11704 | Val 8.70710 | lambda_recon=10.0
Epoch   4 | Train 10.54937 | Val 7.30297 | lambda_recon=10.0
Epoch   5 | Train 9.21678 | Val 5.89949 | lambda_recon=10.0
Epoch   6 | Train 8.27941 | Val 5.37641 | lambda_recon=10.0
Epoch   7 | Train 7.74075 | Val 4.70232 | lambda_recon=10.0
Epoch   8 | Train 7.22914 | Val 4.26368 | lambda_recon=10.0
Epoch   9 | Train 6.87985 | Val 3.95662 | lambda_recon=10.0
Epoch  10 | Train 6.45778 | Val 3.45892 | lambda_recon=10.0
Epoch  11 | Train 6.11590 | Val 3.23768 | lambda_recon=10.0
Epoch  12 | Train 5.92965 | Val 3.13685 | lambda_recon=10.0
Epoch  13 | Train 5.82411 | Val 3.03529 | lambda_recon=10.0
Epoch  14 | Train 5.75738 | Val 3.09144 | lambda_recon=10.0
Epoch  15 | Train 5.64621 | Val 2.99090 | lambda_recon=10.0
Epoch  16 | Train 5.56192 | Val 2.98039 | lambda_recon=10.0
Epoch  17 | Train 5.56295 | Val 2.88699 | lambda_recon=10.0
Epoch  18 | Train 5.43427 | Val 2.65192 | lambda_recon=10.0
Epoch  19 | Train 5.24311 | Val 2.46105 | lambda_recon=10.0
Epoch  20 | Train 5.04102 | Val 2.17967 | lambda_recon=10.0
Epoch  21 | Train 4.91483 | Val 2.13315 | lambda_recon=10.0
Epoch  22 | Train 4.88689 | Val 2.09074 | lambda_recon=10.0
Epoch  23 | Train 4.84497 | Val 2.12174 | lambda_recon=10.0
Epoch  24 | Train 4.83388 | Val 2.08550 | lambda_recon=10.0
Epoch  25 | Train 4.77153 | Val 2.03299 | lambda_recon=10.0
Epoch  26 | Train 4.74591 | Val 2.07427 | lambda_recon=10.0
Epoch  27 | Train 4.70120 | Val 2.06929 | lambda_recon=10.0
Epoch  28 | Train 4.71478 | Val 2.00918 | lambda_recon=10.0
Epoch  29 | Train 4.66772 | Val 1.99325 | lambda_recon=10.0
Epoch  30 | Train 4.62932 | Val 2.01084 | lambda_recon=10.0
Epoch  31 | Train 4.63616 | Val 1.93953 | lambda_recon=10.0
Epoch  32 | Train 4.59563 | Val 1.92239 | lambda_recon=10.0
Epoch  33 | Train 4.59548 | Val 2.07948 | lambda_recon=10.0
Epoch  34 | Train 4.55959 | Val 1.98166 | lambda_recon=10.0
Epoch  35 | Train 4.56161 | Val 2.00607 | lambda_recon=10.0
Epoch  36 | Train 4.57981 | Val 1.94344 | lambda_recon=10.0
Epoch  37 | Train 4.54212 | Val 1.97492 | lambda_recon=10.0
Early stopping triggered.
Best validation loss (lambda_recon=10.0): 1.922391
Test Recall@4 (lambda_recon=10.0): 0.0125

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C034d2', 'C310d3', 'C313d0', 'C420d3']
  Visit 2: ['C304d1', 'C311d0', 'C313d0', 'C320d0']
  Visit 3: ['C424', 'C201d1', 'C210d1']
  Visit 4: ['C304d1', 'C313d0', 'C401d1']
  Visit 5: ['C424', 'C201d1', 'C210d1', 'C414d0']
  Visit 6: ['C424', 'C201d1', 'C210d1', 'C414d0']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C03', 'C424', 'C201d1']
  Visit 2: ['C424', 'C201d1', 'C210d1', 'C414d0']
  Visit 3: ['C04', 'C121d1', 'C142d1', 'C331d1']
  Visit 4: ['C03', 'C424', 'C201d1', 'C414d0']
  Visit 5: ['C013', 'C010d2', 'C401d1']
  Visit 6: ['C432', 'C304d1', 'C311d0', 'C313d0']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C210d1', 'C222d1', 'C231d1']
  Visit 2: ['C03', 'C424', 'C210d1', 'C414d0']
  Visit 3: ['C304d1', 'C311d0', 'C313d0', 'C401d1']
  Visit 4: ['C03', 'C424', 'C210d1', 'C414d0']
  Visit 5: ['C03', 'C21', 'C424']
  Visit 6: ['C03', 'C424', 'C210d1', 'C414d0']
Tree-Embedding Correlation (lambda_recon=10.0): -0.2620
Synthetic (hyperbolic, lambda_recon=10.0) stats (N=1000): {'mean_depth': 3.1753202966958867, 'std_depth': 1.1717098878421541, 'mean_tree_dist': 6.2169167803547065, 'std_tree_dist': 1.4446165916110696, 'mean_root_purity': 0.6068888888888889, 'std_root_purity': 0.18248096467614314}

Training HYPERBOLIC | Depth 7 | lambda_recon=100.0
Epoch   1 | Train 24.01901 | Val 16.03961 | lambda_recon=100.0
Epoch   2 | Train 15.41652 | Val 11.28944 | lambda_recon=100.0
Epoch   3 | Train 12.26617 | Val 8.35458 | lambda_recon=100.0
Epoch   4 | Train 10.25159 | Val 7.06997 | lambda_recon=100.0
Epoch   5 | Train 9.39301 | Val 6.42442 | lambda_recon=100.0
Epoch   6 | Train 8.74477 | Val 5.57676 | lambda_recon=100.0
Epoch   7 | Train 8.06405 | Val 5.20610 | lambda_recon=100.0
Epoch   8 | Train 7.69210 | Val 4.77224 | lambda_recon=100.0
Epoch   9 | Train 7.31306 | Val 4.58228 | lambda_recon=100.0
Epoch  10 | Train 7.11409 | Val 4.37910 | lambda_recon=100.0
Epoch  11 | Train 6.96165 | Val 4.24831 | lambda_recon=100.0
Epoch  12 | Train 6.85749 | Val 4.17076 | lambda_recon=100.0
Epoch  13 | Train 6.59002 | Val 3.60118 | lambda_recon=100.0
Epoch  14 | Train 6.10034 | Val 3.23911 | lambda_recon=100.0
Epoch  15 | Train 5.96670 | Val 3.13636 | lambda_recon=100.0
Epoch  16 | Train 5.83384 | Val 2.87321 | lambda_recon=100.0
Epoch  17 | Train 5.46432 | Val 2.51660 | lambda_recon=100.0
Epoch  18 | Train 5.31649 | Val 2.30405 | lambda_recon=100.0
Epoch  19 | Train 5.17948 | Val 2.25715 | lambda_recon=100.0
Epoch  20 | Train 5.12484 | Val 2.18796 | lambda_recon=100.0
Epoch  21 | Train 5.03438 | Val 2.21367 | lambda_recon=100.0
Epoch  22 | Train 5.03406 | Val 2.25281 | lambda_recon=100.0
Epoch  23 | Train 5.00332 | Val 2.29045 | lambda_recon=100.0
Epoch  24 | Train 4.99658 | Val 2.15478 | lambda_recon=100.0
Epoch  25 | Train 4.92130 | Val 2.15045 | lambda_recon=100.0
Epoch  26 | Train 4.90302 | Val 2.07046 | lambda_recon=100.0
Epoch  27 | Train 4.90467 | Val 2.11021 | lambda_recon=100.0
Epoch  28 | Train 4.88050 | Val 2.15596 | lambda_recon=100.0
Epoch  29 | Train 4.84384 | Val 2.11352 | lambda_recon=100.0
Epoch  30 | Train 4.85241 | Val 2.06679 | lambda_recon=100.0
Epoch  31 | Train 4.83529 | Val 2.04292 | lambda_recon=100.0
Epoch  32 | Train 4.78096 | Val 2.16068 | lambda_recon=100.0
Epoch  33 | Train 4.79411 | Val 2.03452 | lambda_recon=100.0
Epoch  34 | Train 4.78284 | Val 2.05943 | lambda_recon=100.0
Epoch  35 | Train 4.77450 | Val 2.03153 | lambda_recon=100.0
Epoch  36 | Train 4.80066 | Val 1.96424 | lambda_recon=100.0
Epoch  37 | Train 4.76582 | Val 2.14392 | lambda_recon=100.0
Epoch  38 | Train 4.75458 | Val 2.01880 | lambda_recon=100.0
Epoch  39 | Train 4.75750 | Val 2.00369 | lambda_recon=100.0
Epoch  40 | Train 4.72241 | Val 2.03815 | lambda_recon=100.0
Epoch  41 | Train 4.71394 | Val 1.98530 | lambda_recon=100.0
Early stopping triggered.
Best validation loss (lambda_recon=100.0): 1.964240
Test Recall@4 (lambda_recon=100.0): 0.0137

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C243', 'C140d0', 'C201d2', 'C300d0']
  Visit 2: ['C111d4', 'C123d3', 'C300d3', 'C324d3']
  Visit 3: ['C233', 'C234d2', 'C422d0']
  Visit 4: ['C111d4', 'C230d3', 'C300d3', 'C410d4']
  Visit 5: ['C111d4', 'C230d3', 'C300d3', 'C420d3']
  Visit 6: ['C233', 'C234d2', 'C422d0']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C233', 'C234d2', 'C422d0']
  Visit 2: ['C233', 'C234d2', 'C422d0']
  Visit 3: ['C002d3', 'C034d4', 'C134d4', 'C324d3']
  Visit 4: ['C233', 'C234d2', 'C422d0']
  Visit 5: ['C002d3', 'C034d4', 'C340d4', 'C400d3']
  Visit 6: ['C233', 'C234d2', 'C422d0']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C140d0', 'C201d2', 'C300d0', 'C333d2']
  Visit 2: ['C002d3', 'C032d4', 'C111d4', 'C324d3']
  Visit 3: ['C002d3', 'C034d4', 'C134d4', 'C324d3']
  Visit 4: ['C233', 'C234d2', 'C422d0']
  Visit 5: ['C301', 'C024d0', 'C303d2']
  Visit 6: ['C233', 'C234d2', 'C422d0']
Tree-Embedding Correlation (lambda_recon=100.0): 0.0583
Synthetic (hyperbolic, lambda_recon=100.0) stats (N=1000): {'mean_depth': 5.184031973231713, 'std_depth': 1.7606742083378322, 'mean_tree_dist': 9.068279293189226, 'std_tree_dist': 3.8531284529350187, 'mean_root_purity': 0.54925, 'std_root_purity': 0.12337984295721949}

Training HYPERBOLIC | Depth 7 | lambda_recon=1000.0
Epoch   1 | Train 25.90133 | Val 16.75089 | lambda_recon=1000.0
Epoch   2 | Train 16.02102 | Val 12.02379 | lambda_recon=1000.0
Epoch   3 | Train 13.19391 | Val 9.92307 | lambda_recon=1000.0
Epoch   4 | Train 11.64561 | Val 8.29274 | lambda_recon=1000.0
Epoch   5 | Train 10.60913 | Val 7.54433 | lambda_recon=1000.0
Epoch   6 | Train 10.02517 | Val 7.12915 | lambda_recon=1000.0
Epoch   7 | Train 9.62343 | Val 6.77802 | lambda_recon=1000.0
Epoch   8 | Train 9.16242 | Val 6.07293 | lambda_recon=1000.0
Epoch   9 | Train 8.62585 | Val 5.65822 | lambda_recon=1000.0
Epoch  10 | Train 8.00498 | Val 4.95557 | lambda_recon=1000.0
Epoch  11 | Train 7.64100 | Val 4.89733 | lambda_recon=1000.0
Epoch  12 | Train 7.54731 | Val 4.83161 | lambda_recon=1000.0
Epoch  13 | Train 7.48969 | Val 4.63907 | lambda_recon=1000.0
Epoch  14 | Train 7.39420 | Val 4.79591 | lambda_recon=1000.0
Epoch  15 | Train 7.34427 | Val 4.65299 | lambda_recon=1000.0
Epoch  16 | Train 7.30314 | Val 4.57828 | lambda_recon=1000.0
Epoch  17 | Train 7.24707 | Val 4.66319 | lambda_recon=1000.0
Epoch  18 | Train 7.21227 | Val 4.62751 | lambda_recon=1000.0
Epoch  19 | Train 7.20255 | Val 4.60012 | lambda_recon=1000.0
Epoch  20 | Train 7.16086 | Val 4.56970 | lambda_recon=1000.0
Epoch  21 | Train 7.11479 | Val 4.46086 | lambda_recon=1000.0
Epoch  22 | Train 7.10021 | Val 4.54894 | lambda_recon=1000.0
Epoch  23 | Train 7.05102 | Val 4.59424 | lambda_recon=1000.0
Epoch  24 | Train 7.05710 | Val 4.49812 | lambda_recon=1000.0
Epoch  25 | Train 6.99031 | Val 4.50851 | lambda_recon=1000.0
Epoch  26 | Train 6.92616 | Val 4.45147 | lambda_recon=1000.0
Epoch  27 | Train 6.93978 | Val 4.52493 | lambda_recon=1000.0
Epoch  28 | Train 6.94473 | Val 4.55275 | lambda_recon=1000.0
Epoch  29 | Train 6.92953 | Val 4.45238 | lambda_recon=1000.0
Epoch  30 | Train 6.87338 | Val 4.38493 | lambda_recon=1000.0
Epoch  31 | Train 6.85657 | Val 4.49086 | lambda_recon=1000.0
Epoch  32 | Train 6.82496 | Val 4.41747 | lambda_recon=1000.0
Epoch  33 | Train 6.78157 | Val 4.44200 | lambda_recon=1000.0
Epoch  34 | Train 6.81143 | Val 4.39028 | lambda_recon=1000.0
Epoch  35 | Train 6.77660 | Val 4.40147 | lambda_recon=1000.0
Early stopping triggered.
Best validation loss (lambda_recon=1000.0): 4.384935
Test Recall@4 (lambda_recon=1000.0): 0.0617

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C244d3', 'C244d4', 'C303d3', 'C303d4']
  Visit 2: ['C103d4', 'C244d4', 'C312d3', 'C312d4']
  Visit 3: ['C312d3', 'C312d4', 'C424d3', 'C424d4']
  Visit 4: ['C032d2', 'C112d0', 'C134d2']
  Visit 5: ['C331d4', 'C332d4', 'C414d3', 'C414d4']
  Visit 6: ['C103d4', 'C312d3', 'C312d4', 'C330d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C101d3', 'C244d3', 'C244d4', 'C303d4']
  Visit 2: ['C312d3', 'C312d4', 'C424d3', 'C424d4']
  Visit 3: ['C012d3', 'C034d3', 'C213d4', 'C334d3']
  Visit 4: ['C012d3', 'C034d3', 'C401d4']
  Visit 5: ['C114d4', 'C244d3', 'C244d4', 'C312d4']
  Visit 6: ['C312d3', 'C312d4', 'C424d3', 'C424d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C214d3', 'C232d4', 'C413d3', 'C432d4']
  Visit 2: ['C012d3', 'C032d2', 'C034d3']
  Visit 3: ['C332d4', 'C402d3', 'C402d4', 'C414d4']
  Visit 4: ['C143d3', 'C332d3', 'C332d4', 'C440d4']
  Visit 5: ['C114d4', 'C244d3', 'C303d3', 'C303d4']
  Visit 6: ['C312d4', 'C424d3', 'C424d4']
Tree-Embedding Correlation (lambda_recon=1000.0): 0.3997
Synthetic (hyperbolic, lambda_recon=1000.0) stats (N=1000): {'mean_depth': 6.146926536731634, 'std_depth': 1.2355959185724372, 'mean_tree_dist': 5.9719415179531286, 'std_tree_dist': 5.526209289717405, 'mean_root_purity': 0.5665555555555555, 'std_root_purity': 0.13075984017453193}
[Summary] depth7_final | hyperbolic | lambda_recon=1.0: best_val=1.877002, test_recall=0.0051, corr=-0.2411
[Summary] depth7_final | hyperbolic | lambda_recon=10.0: best_val=1.922391, test_recall=0.0125, corr=-0.2620
[Summary] depth7_final | hyperbolic | lambda_recon=100.0: best_val=1.964240, test_recall=0.0137, corr=0.0583
[Summary] depth7_final | hyperbolic | lambda_recon=1000.0: best_val=4.384935, test_recall=0.0617, corr=0.3997
