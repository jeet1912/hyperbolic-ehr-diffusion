Using device: mps

depth7_final | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

--- Running hyperbolic ---

Training HYPERBOLIC | Depth 7 | lambda_recon=3000.0
Epoch   1 | Train 28.34289 | Val 20.10995 | lambda_recon=3000.0
Epoch   2 | Train 19.61869 | Val 15.52271 | lambda_recon=3000.0
Epoch   3 | Train 16.69272 | Val 13.14069 | lambda_recon=3000.0
Epoch   4 | Train 14.89527 | Val 11.69834 | lambda_recon=3000.0
Epoch   5 | Train 13.79010 | Val 10.52227 | lambda_recon=3000.0
Epoch   6 | Train 12.70836 | Val 9.64449 | lambda_recon=3000.0
Epoch   7 | Train 12.23616 | Val 9.33458 | lambda_recon=3000.0
Epoch   8 | Train 11.97053 | Val 9.20000 | lambda_recon=3000.0
Epoch   9 | Train 11.73229 | Val 9.04984 | lambda_recon=3000.0
Epoch  10 | Train 11.34455 | Val 8.33609 | lambda_recon=3000.0
Epoch  11 | Train 10.80985 | Val 7.94755 | lambda_recon=3000.0
Epoch  12 | Train 10.60418 | Val 7.85665 | lambda_recon=3000.0
Epoch  13 | Train 10.44825 | Val 7.73145 | lambda_recon=3000.0
Epoch  14 | Train 10.31654 | Val 7.62747 | lambda_recon=3000.0
Epoch  15 | Train 10.16131 | Val 7.60600 | lambda_recon=3000.0
Epoch  16 | Train 10.08219 | Val 7.38358 | lambda_recon=3000.0
Epoch  17 | Train 9.80798 | Val 7.11708 | lambda_recon=3000.0
Epoch  18 | Train 9.39348 | Val 6.79539 | lambda_recon=3000.0
Epoch  19 | Train 9.19616 | Val 6.57970 | lambda_recon=3000.0
Epoch  20 | Train 9.03337 | Val 6.27848 | lambda_recon=3000.0
Epoch  21 | Train 8.92678 | Val 6.27596 | lambda_recon=3000.0
Epoch  22 | Train 8.83982 | Val 6.03903 | lambda_recon=3000.0
Epoch  23 | Train 8.61737 | Val 5.99106 | lambda_recon=3000.0
Epoch  24 | Train 8.47735 | Val 5.99900 | lambda_recon=3000.0
Epoch  25 | Train 8.41899 | Val 5.93206 | lambda_recon=3000.0
Epoch  26 | Train 8.27117 | Val 5.80454 | lambda_recon=3000.0
Epoch  27 | Train 8.22952 | Val 5.73602 | lambda_recon=3000.0
Epoch  28 | Train 8.20758 | Val 5.71228 | lambda_recon=3000.0
Epoch  29 | Train 8.08824 | Val 5.61036 | lambda_recon=3000.0
Epoch  30 | Train 8.03487 | Val 5.45380 | lambda_recon=3000.0
Epoch  31 | Train 7.94187 | Val 5.45938 | lambda_recon=3000.0
Epoch  32 | Train 7.86489 | Val 5.49649 | lambda_recon=3000.0
Epoch  33 | Train 7.78110 | Val 5.20556 | lambda_recon=3000.0
Epoch  34 | Train 7.75782 | Val 5.43780 | lambda_recon=3000.0
Epoch  35 | Train 7.70713 | Val 5.30605 | lambda_recon=3000.0
Epoch  36 | Train 7.60067 | Val 5.18974 | lambda_recon=3000.0
Epoch  37 | Train 7.51397 | Val 5.26015 | lambda_recon=3000.0
Epoch  38 | Train 7.46897 | Val 5.30567 | lambda_recon=3000.0
Epoch  39 | Train 7.47898 | Val 5.16084 | lambda_recon=3000.0
Epoch  40 | Train 7.43154 | Val 4.98454 | lambda_recon=3000.0
Epoch  41 | Train 7.42958 | Val 5.04181 | lambda_recon=3000.0
Epoch  42 | Train 7.33340 | Val 4.99138 | lambda_recon=3000.0
Epoch  43 | Train 7.36273 | Val 5.16315 | lambda_recon=3000.0
Epoch  44 | Train 7.32377 | Val 4.96368 | lambda_recon=3000.0
Epoch  45 | Train 7.34932 | Val 4.93415 | lambda_recon=3000.0
Epoch  46 | Train 7.44366 | Val 5.03107 | lambda_recon=3000.0
Epoch  47 | Train 7.42931 | Val 5.14354 | lambda_recon=3000.0
Epoch  48 | Train 7.61688 | Val 5.22234 | lambda_recon=3000.0
Epoch  49 | Train 7.81530 | Val 5.52690 | lambda_recon=3000.0
Epoch  50 | Train 8.21757 | Val 5.74884 | lambda_recon=3000.0
Early stopping triggered.
Best validation loss (lambda_recon=3000.0): 4.934151
Test Recall@4 (lambda_recon=3000.0): 0.4835

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C121d4', 'C333d3', 'C333d4', 'C430d4']
  Visit 2: ['C231d3', 'C314d4', 'C443d3', 'C443d4']
  Visit 3: ['C032d3', 'C032d4', 'C443d3', 'C443d4']
  Visit 4: ['C022d3', 'C022d4', 'C442d3', 'C442d4']
  Visit 5: ['C032d3', 'C032d4', 'C040d3', 'C040d4']
  Visit 6: ['C022d3', 'C022d4', 'C112d3', 'C112d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C022d4', 'C114d4', 'C412d3', 'C412d4']
  Visit 2: ['C430d3', 'C430d4', 'C444d3', 'C444d4']
  Visit 3: ['C123d3', 'C123d4', 'C134d4', 'C313d2']
  Visit 4: ['C022d3', 'C022d4', 'C440d3', 'C440d4']
  Visit 5: ['C022d3', 'C022d4', 'C124d3', 'C124d4']
  Visit 6: ['C022d4', 'C112d3', 'C112d4', 'C114d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C031d3', 'C031d4', 'C424d0', 'C434d1']
  Visit 2: ['C102d3', 'C440d3', 'C440d4', 'C442d4']
  Visit 3: ['C021d1', 'C111d3', 'C111d4', 'C334d4']
  Visit 4: ['C041d0', 'C123d4', 'C210d3', 'C210d4']
  Visit 5: ['C022d3', 'C022d4', 'C112d3', 'C112d4']
  Visit 6: ['C102d3', 'C440d3', 'C442d3', 'C442d4']
Tree-Embedding Correlation (lambda_recon=3000.0): -0.1185
Synthetic (hyperbolic, lambda_recon=3000.0) stats (N=1000): {'mean_depth': 6.1220510250949935, 'std_depth': 1.206068501308584, 'mean_tree_dist': 5.255817367687, 'std_tree_dist': 5.4067979189035045, 'mean_root_purity': 0.5872083333333333, 'std_root_purity': 0.1732519390223382}

Training HYPERBOLIC | Depth 7 | lambda_recon=4000.0
Epoch   1 | Train 30.70836 | Val 21.63928 | lambda_recon=4000.0
Epoch   2 | Train 21.19351 | Val 17.21737 | lambda_recon=4000.0
Epoch   3 | Train 18.20244 | Val 14.81023 | lambda_recon=4000.0
Epoch   4 | Train 16.63190 | Val 13.49650 | lambda_recon=4000.0
Epoch   5 | Train 15.41306 | Val 12.02015 | lambda_recon=4000.0
Epoch   6 | Train 14.42714 | Val 11.40967 | lambda_recon=4000.0
Epoch   7 | Train 13.88133 | Val 10.93433 | lambda_recon=4000.0
Epoch   8 | Train 13.36250 | Val 10.29514 | lambda_recon=4000.0
Epoch   9 | Train 12.87181 | Val 10.02412 | lambda_recon=4000.0
Epoch  10 | Train 12.55290 | Val 9.66976 | lambda_recon=4000.0
Epoch  11 | Train 12.30180 | Val 9.56166 | lambda_recon=4000.0
Epoch  12 | Train 12.02513 | Val 9.31185 | lambda_recon=4000.0
Epoch  13 | Train 11.75869 | Val 9.14361 | lambda_recon=4000.0
Epoch  14 | Train 11.57436 | Val 8.84208 | lambda_recon=4000.0
Epoch  15 | Train 11.26173 | Val 8.59501 | lambda_recon=4000.0
Epoch  16 | Train 11.02008 | Val 8.27562 | lambda_recon=4000.0
Epoch  17 | Train 10.63458 | Val 7.77000 | lambda_recon=4000.0
Epoch  18 | Train 10.38767 | Val 7.61970 | lambda_recon=4000.0
Epoch  19 | Train 10.28366 | Val 7.43767 | lambda_recon=4000.0
Epoch  20 | Train 10.11286 | Val 7.46218 | lambda_recon=4000.0
Epoch  21 | Train 9.98907 | Val 7.27149 | lambda_recon=4000.0
Epoch  22 | Train 9.85518 | Val 7.29457 | lambda_recon=4000.0
Epoch  23 | Train 9.74485 | Val 7.26947 | lambda_recon=4000.0
Epoch  24 | Train 9.58145 | Val 7.06807 | lambda_recon=4000.0
Epoch  25 | Train 9.52376 | Val 6.97979 | lambda_recon=4000.0
Epoch  26 | Train 9.34546 | Val 6.72673 | lambda_recon=4000.0
Epoch  27 | Train 9.16750 | Val 6.65830 | lambda_recon=4000.0
Epoch  28 | Train 9.12447 | Val 6.56887 | lambda_recon=4000.0
Epoch  29 | Train 8.95705 | Val 6.38364 | lambda_recon=4000.0
Epoch  30 | Train 8.89526 | Val 6.49010 | lambda_recon=4000.0
Epoch  31 | Train 8.78998 | Val 6.35939 | lambda_recon=4000.0
Epoch  32 | Train 8.71535 | Val 6.29333 | lambda_recon=4000.0
Epoch  33 | Train 8.65972 | Val 6.22158 | lambda_recon=4000.0
Epoch  34 | Train 8.53663 | Val 6.22683 | lambda_recon=4000.0
Epoch  35 | Train 8.48381 | Val 5.95412 | lambda_recon=4000.0
Epoch  36 | Train 8.41842 | Val 6.03075 | lambda_recon=4000.0
Epoch  37 | Train 8.34444 | Val 5.86353 | lambda_recon=4000.0
Epoch  38 | Train 8.37283 | Val 5.75691 | lambda_recon=4000.0
Epoch  39 | Train 8.26652 | Val 5.82990 | lambda_recon=4000.0
Epoch  40 | Train 8.30649 | Val 5.88844 | lambda_recon=4000.0
Epoch  41 | Train 8.25187 | Val 5.80353 | lambda_recon=4000.0
Epoch  42 | Train 8.19384 | Val 5.75923 | lambda_recon=4000.0
Epoch  43 | Train 8.20482 | Val 5.81836 | lambda_recon=4000.0
Early stopping triggered.
Best validation loss (lambda_recon=4000.0): 5.756914
Test Recall@4 (lambda_recon=4000.0): 0.5059

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C110d3', 'C110d4', 'C134d3', 'C134d4']
  Visit 2: ['C324', 'C044d3', 'C332d1']
  Visit 3: ['C110d3', 'C110d4', 'C323d3', 'C323d4']
  Visit 4: ['C110d4', 'C323d4', 'C443d3', 'C443d4']
  Visit 5: ['C034d4', 'C100d4', 'C321d3', 'C411d4']
  Visit 6: ['C324', 'C044d3', 'C332d1', 'C413d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C324', 'C044d3', 'C044d4', 'C332d1']
  Visit 2: ['C110d3', 'C110d4', 'C323d3', 'C323d4']
  Visit 3: ['C324', 'C332d1', 'C343d2']
  Visit 4: ['C110d3', 'C110d4', 'C323d3', 'C323d4']
  Visit 5: ['C040', 'C110d3', 'C110d4', 'C413d0']
  Visit 6: ['C100d4', 'C110d3', 'C110d4', 'C323d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C232', 'C211d0', 'C323d4', 'C442d3']
  Visit 2: ['C324', 'C044d3', 'C332d2', 'C413d4']
  Visit 3: ['C110d3', 'C110d4', 'C323d3', 'C323d4']
  Visit 4: ['C321d3', 'C323d4', 'C342d3', 'C342d4']
  Visit 5: ['C324', 'C332d1', 'C332d2']
  Visit 6: ['C324', 'C044d3', 'C413d3', 'C413d4']
Tree-Embedding Correlation (lambda_recon=4000.0): -0.0369
Synthetic (hyperbolic, lambda_recon=4000.0) stats (N=1000): {'mean_depth': 5.457463884430177, 'std_depth': 1.706123847964029, 'mean_tree_dist': 4.975821508588499, 'std_tree_dist': 4.763984737177185, 'mean_root_purity': 0.5583333333333333, 'std_root_purity': 0.1502313031443329}

Training HYPERBOLIC | Depth 7 | lambda_recon=5000.0
Epoch   1 | Train 32.74227 | Val 22.91899 | lambda_recon=5000.0
Epoch   2 | Train 22.71863 | Val 18.58131 | lambda_recon=5000.0
Epoch   3 | Train 19.75527 | Val 16.23523 | lambda_recon=5000.0
Epoch   4 | Train 17.92406 | Val 14.81997 | lambda_recon=5000.0
Epoch   5 | Train 16.84052 | Val 13.50470 | lambda_recon=5000.0
Epoch   6 | Train 15.81122 | Val 12.71987 | lambda_recon=5000.0
Epoch   7 | Train 15.10071 | Val 12.16557 | lambda_recon=5000.0
Epoch   8 | Train 14.40954 | Val 11.54585 | lambda_recon=5000.0
Epoch   9 | Train 13.92782 | Val 10.92395 | lambda_recon=5000.0
Epoch  10 | Train 13.42779 | Val 10.52700 | lambda_recon=5000.0
Epoch  11 | Train 13.01069 | Val 10.03331 | lambda_recon=5000.0
Epoch  12 | Train 12.49180 | Val 9.90518 | lambda_recon=5000.0
Epoch  13 | Train 12.11706 | Val 9.48067 | lambda_recon=5000.0
Epoch  14 | Train 11.72161 | Val 9.13373 | lambda_recon=5000.0
Epoch  15 | Train 11.44850 | Val 8.74691 | lambda_recon=5000.0
Epoch  16 | Train 11.14500 | Val 8.62918 | lambda_recon=5000.0
Epoch  17 | Train 10.96144 | Val 8.30518 | lambda_recon=5000.0
Epoch  18 | Train 10.75680 | Val 8.32892 | lambda_recon=5000.0
Epoch  19 | Train 10.58580 | Val 8.08516 | lambda_recon=5000.0
Epoch  20 | Train 10.37198 | Val 7.88777 | lambda_recon=5000.0
Epoch  21 | Train 10.24273 | Val 7.73905 | lambda_recon=5000.0
Epoch  22 | Train 10.12743 | Val 7.63405 | lambda_recon=5000.0
Epoch  23 | Train 9.88336 | Val 7.74658 | lambda_recon=5000.0
Epoch  24 | Train 9.89691 | Val 7.38456 | lambda_recon=5000.0
Epoch  25 | Train 9.70519 | Val 7.49333 | lambda_recon=5000.0
Epoch  26 | Train 9.63819 | Val 7.26666 | lambda_recon=5000.0
Epoch  27 | Train 9.63093 | Val 7.17173 | lambda_recon=5000.0
Epoch  28 | Train 9.51146 | Val 7.14051 | lambda_recon=5000.0
Epoch  29 | Train 9.39319 | Val 6.98709 | lambda_recon=5000.0
Epoch  30 | Train 9.34439 | Val 6.97144 | lambda_recon=5000.0
Epoch  31 | Train 9.26077 | Val 7.00440 | lambda_recon=5000.0
Epoch  32 | Train 9.19956 | Val 6.99755 | lambda_recon=5000.0
Epoch  33 | Train 9.19783 | Val 6.83876 | lambda_recon=5000.0
Epoch  34 | Train 9.07043 | Val 6.93039 | lambda_recon=5000.0
Epoch  35 | Train 9.10793 | Val 6.79759 | lambda_recon=5000.0
Epoch  36 | Train 9.07051 | Val 6.76470 | lambda_recon=5000.0
Epoch  37 | Train 8.98886 | Val 6.81310 | lambda_recon=5000.0
Epoch  38 | Train 8.91549 | Val 6.85145 | lambda_recon=5000.0
Epoch  39 | Train 8.93894 | Val 6.62002 | lambda_recon=5000.0
Epoch  40 | Train 8.87701 | Val 6.58602 | lambda_recon=5000.0
Epoch  41 | Train 8.88489 | Val 6.64225 | lambda_recon=5000.0
Epoch  42 | Train 8.89191 | Val 6.49752 | lambda_recon=5000.0
Epoch  43 | Train 8.86848 | Val 6.63916 | lambda_recon=5000.0
Epoch  44 | Train 8.84315 | Val 6.61762 | lambda_recon=5000.0
Epoch  45 | Train 8.90502 | Val 6.55609 | lambda_recon=5000.0
Epoch  46 | Train 8.94522 | Val 6.64401 | lambda_recon=5000.0
Epoch  47 | Train 9.06225 | Val 6.91153 | lambda_recon=5000.0
Early stopping triggered.
Best validation loss (lambda_recon=5000.0): 6.497522
Test Recall@4 (lambda_recon=5000.0): 0.5283

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C111', 'C202d0', 'C211d2', 'C311d2']
  Visit 2: ['C111d3', 'C111d4', 'C414d3', 'C414d4']
  Visit 3: ['C031d3', 'C031d4', 'C111d3', 'C111d4']
  Visit 4: ['C111d3', 'C111d4', 'C241d3', 'C241d4']
  Visit 5: ['C213d3', 'C213d4', 'C220d3', 'C220d4']
  Visit 6: ['C111d3', 'C111d4', 'C444d3', 'C444d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C111d3', 'C111d4', 'C241d3', 'C241d4']
  Visit 2: ['C111d3', 'C111d4', 'C241d3', 'C241d4']
  Visit 3: ['C0', 'C010d3', 'C010d4', 'C103d4']
  Visit 4: ['C111d3', 'C111d4', 'C444d3', 'C444d4']
  Visit 5: ['C111d3', 'C111d4', 'C444d3', 'C444d4']
  Visit 6: ['C213d3', 'C213d4', 'C233d1', 'C341d3']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C111d3', 'C111d4', 'C444d3', 'C444d4']
  Visit 2: ['C111d3', 'C111d4', 'C241d3', 'C241d4']
  Visit 3: ['C111', 'C241', 'C121d1', 'C311d2']
  Visit 4: ['C110d0', 'C134d4', 'C213d3', 'C213d4']
  Visit 5: ['C0', 'C120', 'C341d3', 'C341d4']
  Visit 6: ['C111d3', 'C111d4', 'C414d3', 'C414d4']
Tree-Embedding Correlation (lambda_recon=5000.0): -0.1266
Synthetic (hyperbolic, lambda_recon=5000.0) stats (N=1000): {'mean_depth': 5.7633256928526775, 'std_depth': 1.611726751620238, 'mean_tree_dist': 3.4129241292412926, 'std_tree_dist': 4.224863768607494, 'mean_root_purity': 0.5346388888888889, 'std_root_purity': 0.12277335022182687}
[Summary] depth7_final | hyperbolic | lambda_recon=3000.0: best_val=4.934151, test_recall=0.4835, corr=-0.1185
[Summary] depth7_final | hyperbolic | lambda_recon=4000.0: best_val=5.756914, test_recall=0.5059, corr=-0.0369
[Summary] depth7_final | hyperbolic | lambda_recon=5000.0: best_val=6.497522, test_recall=0.5283, corr=-0.1266
