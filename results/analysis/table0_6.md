### depth2_final | max_depth = 2 | Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

### Experiment depth2_final | depth 2 | euclidean | lambda_recon=1.0
![Loss curves](../plots/depth2_final_euclidean_lrecon1_loss.png)

Best validation loss: 1.832076  
Test Recall@4: 0.0672  
Tree-Embedding Correlation: -0.0216  
Synthetic stats: mean_depth=1.6917, std_depth=0.4618, mean_tree_dist=2.5391, std_tree_dist=1.1700, mean_root_purity=0.7128, std_root_purity=0.2471

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

### Experiment depth2_final | depth 2 | euclidean | lambda_recon=10.0
![Loss curves](../plots/depth2_final_euclidean_lrecon10_loss.png)

Best validation loss: 1.874482  
Test Recall@4: 0.2493  
Tree-Embedding Correlation: 0.0488  
Synthetic stats: mean_depth=1.3805, std_depth=0.4872, mean_tree_dist=1.8120, std_tree_dist=0.6290, mean_root_purity=0.4133, std_root_purity=0.1548

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

### Experiment depth2_final | depth 2 | euclidean | lambda_recon=100.0
![Loss curves](../plots/depth2_final_euclidean_lrecon100_loss.png)

Best validation loss: 2.030545  
Test Recall@4: 0.8702  
Tree-Embedding Correlation: -0.0155  
Synthetic stats: mean_depth=1.7430, std_depth=0.4483, mean_tree_dist=2.6803, std_tree_dist=1.2199, mean_root_purity=0.4926, std_root_purity=0.1494

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

### Experiment depth2_final | depth 2 | euclidean | lambda_recon=1000.0
![Loss curves](../plots/depth2_final_euclidean_lrecon1000_loss.png)

Best validation loss: 2.439485  
Test Recall@4: 0.9227  
Tree-Embedding Correlation: 0.0527  
Synthetic stats: mean_depth=1.6424, std_depth=0.5588, mean_tree_dist=2.8897, std_tree_dist=1.0498, mean_root_purity=0.5012, std_root_purity=0.1524

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

### Experiment depth2_final | depth 2 | hyperbolic | lambda_recon=1.0
![Loss curves](../plots/depth2_final_hyperbolic_lrecon1_loss.png)

Best validation loss: 1.979262  
Test Recall@4: 0.0549  
Tree-Embedding Correlation: 0.0913  
Synthetic stats: mean_depth=1.8545, std_depth=0.3551, mean_tree_dist=3.8368, std_tree_dist=0.5690, mean_root_purity=0.4934, std_root_purity=0.0671

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

### Experiment depth2_final | depth 2 | hyperbolic | lambda_recon=10.0
![Loss curves](../plots/depth2_final_hyperbolic_lrecon10_loss.png)

Best validation loss: 2.870362  
Test Recall@4: 0.0516  
Tree-Embedding Correlation: -0.3309  
Synthetic stats: mean_depth=1.3006, std_depth=0.6608, mean_tree_dist=2.8233, std_tree_dist=0.9319, mean_root_purity=0.4627, std_root_purity=0.1368

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

### Experiment depth2_final | depth 2 | hyperbolic | lambda_recon=100.0
![Loss curves](../plots/depth2_final_hyperbolic_lrecon100_loss.png)

Best validation loss: 2.415631  
Test Recall@4: 0.1587  
Tree-Embedding Correlation: -0.0071  
Synthetic stats: mean_depth=1.1557, std_depth=0.3678, mean_tree_dist=1.9694, std_tree_dist=0.7683, mean_root_purity=0.5440, std_root_purity=0.1790

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

### Experiment depth2_final | depth 2 | hyperbolic | lambda_recon=1000.0
![Loss curves](../plots/depth2_final_hyperbolic_lrecon1000_loss.png)

Best validation loss: 5.282482  
Test Recall@4: 0.5962  
Tree-Embedding Correlation: 0.5987  
Synthetic stats: mean_depth=1.8124, std_depth=0.3994, mean_tree_dist=1.9616, std_tree_dist=1.1097, mean_root_purity=0.6600, std_root_purity=0.1985

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

### depth7_final | max_depth = 7 | Real stats: {'mean_depth': 5.374459093875327, 'std_depth': 1.7322915840970905, 'mean_tree_dist': 5.76189079147913, 'std_tree_dist': 4.753635709372622, 'mean_root_purity': 0.6272780762911319, 'std_root_purity': 0.2050864797754083}

### Experiment depth7_final | depth 7 | euclidean | lambda_recon=1.0
![Loss curves](../plots/depth7_final_euclidean_lrecon1_loss.png)

Best validation loss: 2.935548  
Test Recall@4: 0.0098  
Tree-Embedding Correlation: 0.0303  
Synthetic stats: mean_depth=5.4627, std_depth=1.5877, mean_tree_dist=7.2343, std_tree_dist=4.8878, mean_root_purity=0.5231, std_root_purity=0.1190

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

### Experiment depth7_final | depth 7 | euclidean | lambda_recon=10.0
![Loss curves](../plots/depth7_final_euclidean_lrecon10_loss.png)

Best validation loss: 2.865968  
Test Recall@4: 0.0111  
Tree-Embedding Correlation: -0.0285  
Synthetic stats: mean_depth=5.3407, std_depth=1.9723, mean_tree_dist=7.9726, std_tree_dist=4.7473, mean_root_purity=0.5426, std_root_purity=0.1404

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

### Experiment depth7_final | depth 7 | euclidean | lambda_recon=100.0
![Loss curves](../plots/depth7_final_euclidean_lrecon100_loss.png)

Best validation loss: 3.008448  
Test Recall@4: 0.0296  
Tree-Embedding Correlation: 0.0109  
Synthetic stats: mean_depth=5.7362, std_depth=1.8264, mean_tree_dist=1.4805, std_tree_dist=2.3376, mean_root_purity=0.5032, std_root_purity=0.0358

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

### Experiment depth7_final | depth 7 | euclidean | lambda_recon=1000.0
![Loss curves](../plots/depth7_final_euclidean_lrecon1000_loss.png)

Best validation loss: 3.888043  
Test Recall@4: 0.4498  
Tree-Embedding Correlation: -0.0076  
Synthetic stats: mean_depth=4.6039, std_depth=1.5346, mean_tree_dist=8.8476, std_tree_dist=2.4226, mean_root_purity=0.5306, std_root_purity=0.1438

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

### Experiment depth7_final | depth 7 | hyperbolic | lambda_recon=1.0
![Loss curves](../plots/depth7_final_hyperbolic_lrecon1_loss.png)

Best validation loss: 1.877002  
Test Recall@4: 0.0051  
Tree-Embedding Correlation: -0.2411  
Synthetic stats: mean_depth=3.6498, std_depth=1.2106, mean_tree_dist=7.4803, std_tree_dist=1.8087, mean_root_purity=0.3989, std_root_purity=0.1362

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

### Experiment depth7_final | depth 7 | hyperbolic | lambda_recon=10.0
![Loss curves](../plots/depth7_final_hyperbolic_lrecon10_loss.png)

Best validation loss: 1.922391  
Test Recall@4: 0.0125  
Tree-Embedding Correlation: -0.2620  
Synthetic stats: mean_depth=3.1753, std_depth=1.1717, mean_tree_dist=6.2169, std_tree_dist=1.4446, mean_root_purity=0.6069, std_root_purity=0.1825

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

### Experiment depth7_final | depth 7 | hyperbolic | lambda_recon=100.0
![Loss curves](../plots/depth7_final_hyperbolic_lrecon100_loss.png)

Best validation loss: 1.964240  
Test Recall@4: 0.0137  
Tree-Embedding Correlation: 0.0583  
Synthetic stats: mean_depth=5.1840, std_depth=1.7607, mean_tree_dist=9.0683, std_tree_dist=3.8531, mean_root_purity=0.5493, std_root_purity=0.1234

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
  Visit 2: ['C002d3', 'C032d2', 'C111d4', 'C324d3']
  Visit 3: ['C002d3', 'C034d4', 'C134d4', 'C324d3']
  Visit 4: ['C233', 'C234d2', 'C422d0']
  Visit 5: ['C301', 'C024d0', 'C303d2']
  Visit 6: ['C233', 'C234d2', 'C422d0']

### Experiment depth7_final | depth 7 | hyperbolic | lambda_recon=1000.0
![Loss curves](../plots/depth7_final_hyperbolic_lrecon1000_loss.png)

Best validation loss: 4.384935  
Test Recall@4: 0.0617  
Tree-Embedding Correlation: 0.3997  
Synthetic stats: mean_depth=6.1469, std_depth=1.2356, mean_tree_dist=5.9719, std_tree_dist=5.5262, mean_root_purity=0.5666, std_root_purity=0.1308

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

## Comparison Table
| Depth / Experiment | Source | Embedding | lambda_recon | Mean depth | Depth std | Mean tree dist | Tree dist std | Mean root purity | Root purity std | Corr(tree, emb) | Best val loss | Test Recall@4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth2_final | Real | — | — | 1.6439 | 0.5138 | 2.1206 | 1.2453 | 0.6242 | 0.2040 | — | — | — |
| depth2_final | Synthetic | Euclidean | 1 | 1.6917 | 0.4618 | 2.5391 | 1.1700 | 0.7128 | 0.2471 | -0.0216 | 1.8321 | 0.0672 |
| depth2_final | Synthetic | Euclidean | 10 | 1.3805 | 0.4872 | 1.8120 | 0.6290 | 0.4133 | 0.1548 | 0.0488 | 1.8745 | 0.2493 |
| depth2_final | Synthetic | Euclidean | 100 | 1.7430 | 0.4483 | 2.6803 | 1.2199 | 0.4926 | 0.1494 | -0.0155 | 2.0305 | 0.8702 |
| depth2_final | Synthetic | Euclidean | 1000 | 1.6424 | 0.5588 | 2.8897 | 1.0498 | 0.5012 | 0.1524 | 0.0527 | 2.4395 | 0.9227 |
| depth2_final | Synthetic | Hyperbolic | 1 | 1.8545 | 0.3551 | 3.8368 | 0.5690 | 0.4934 | 0.0671 | 0.0913 | 1.9793 | 0.0549 |
| depth2_final | Synthetic | Hyperbolic | 10 | 1.3006 | 0.6608 | 2.8233 | 0.9319 | 0.4627 | 0.1368 | -0.3309 | 2.8704 | 0.0516 |
| depth2_final | Synthetic | Hyperbolic | 100 | 1.1557 | 0.3678 | 1.9694 | 0.7683 | 0.5440 | 0.1790 | -0.0071 | 2.4156 | 0.1587 |
| depth2_final | Synthetic | Hyperbolic | 1000 | 1.8124 | 0.3994 | 1.9616 | 1.1097 | 0.6600 | 0.1985 | 0.5987 | 5.2825 | 0.5962 |
| depth7_final | Real | — | — | 5.3745 | 1.7323 | 5.7619 | 4.7536 | 0.6273 | 0.2051 | — | — | — |
| depth7_final | Synthetic | Euclidean | 1 | 5.4627 | 1.5877 | 7.2343 | 4.8878 | 0.5231 | 0.1190 | 0.0303 | 2.9355 | 0.0098 |
| depth7_final | Synthetic | Euclidean | 10 | 5.3407 | 1.9723 | 7.9726 | 4.7473 | 0.5426 | 0.1404 | -0.0285 | 2.8660 | 0.0111 |
| depth7_final | Synthetic | Euclidean | 100 | 5.7362 | 1.8264 | 1.4805 | 2.3376 | 0.5032 | 0.0358 | 0.0109 | 3.0084 | 0.0296 |
| depth7_final | Synthetic | Euclidean | 1000 | 4.6039 | 1.5346 | 8.8476 | 2.4226 | 0.5306 | 0.1438 | -0.0076 | 3.8880 | 0.4498 |
| depth7_final | Synthetic | Hyperbolic | 1 | 3.6498 | 1.2106 | 7.4803 | 1.8087 | 0.3989 | 0.1362 | -0.2411 | 1.8770 | 0.0051 |
| depth7_final | Synthetic | Hyperbolic | 10 | 3.1753 | 1.1717 | 6.2169 | 1.4446 | 0.6069 | 0.1825 | -0.2620 | 1.9224 | 0.0125 |
| depth7_final | Synthetic | Hyperbolic | 100 | 5.1840 | 1.7607 | 9.0683 | 3.8531 | 0.5493 | 0.1234 | 0.0583 | 1.9642 | 0.0137 |
| depth7_final | Synthetic | Hyperbolic | 1000 | 6.1469 | 1.2356 | 5.9719 | 5.5262 | 0.5666 | 0.1308 | 0.3997 | 4.3849 | 0.0617 |

## Results
These experiments apply the rectified-flow training loop in `src/train_rectified_flow.py` (depth2_final, all embeddings) and the depth-7 variant in `src/train_rectified_flow2.py`. Each script learns a visit encoder/decoder and a trajectory velocity model that predicts tangent-space velocities along rectified flows, combines that loss with a reconstruction term weighted by `lambda_recon`, and monitors performance through test Recall@4 and a tree-vs-embedding distance correlation.

Across depth2_final, increasing `lambda_recon` consistently improves reconstruction recall, especially for Euclidean embeddings (Recall@4 rises from 0.07 to 0.92) but does not guarantee stronger alignment with the ICD tree—correlations remain near zero until very large weights in the hyperbolic model drive a positive correlation (0.60) at the cost of higher validation loss. Hyperbolic models produce more concentrated visit depths and maintain higher root purity, but mid-range `lambda_recon` values (10–100) can over-compress the hierarchy, producing negative correlations.

For the deeper depth7_final hierarchy trained with `train_rectified_flow2.py`, Euclidean models struggle to push Recall@4 beyond 0.45 even at `lambda_recon=1000`, and their tree correlations hover around zero. Hyperbolic models keep the flow stable at low `lambda_recon`, but only the most aggressive reconstruction weight (1000) raises correlation to 0.40 while nudging Recall@4 to 0.06. The derived synthetic statistics show that neither geometry recovers the real visit-depth distribution without additional structure (e.g., the geometric regularizers used in earlier diffusion experiments).

Overall, these rectified-flow baselines confirm that (1) strong reconstruction weights dominate the objective and boost Recall@4 regardless of geometry, (2) hyperbolic embeddings preserve root-level purity better than Euclidean ones, but (3) without the auxiliary regularization from `train_toy_withDecHypNoise.py`, both geometries provide weak alignment with the ICD tree. These tables serve as a reference point for future modifications to the rectified-flow scripts.
