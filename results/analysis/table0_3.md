
Real stats (depth2_base_w/DecHypNoise_hdd, max_depth=2): {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Experiment depth2_base_w/DecHypNoise_hdd | depth 2 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.427652, Val Loss: 1.120670, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.823479, Val Loss: 0.750248, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.531852, Val Loss: 0.590492, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.417815, Val Loss: 0.528251, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.354095, Val Loss: 0.495525, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.331733, Val Loss: 0.488008, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.320655, Val Loss: 0.482749, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.313527, Val Loss: 0.481798, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.308920, Val Loss: 0.482923, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.304356, Val Loss: 0.483869, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.294482, Val Loss: 0.458599, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.253800, Val Loss: 0.440864, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.242862, Val Loss: 0.439169, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.238956, Val Loss: 0.433903, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.234942, Val Loss: 0.435085, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.231663, Val Loss: 0.438831, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.228746, Val Loss: 0.432703, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.225572, Val Loss: 0.431846, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.223154, Val Loss: 0.434699, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.221155, Val Loss: 0.436990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Best validation loss: 0.431846
Saved loss curves to results/plots
Test recall@4: 0.0471

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C03', 'C12', 'C30', 'C41']
  Visit 2: ['C03', 'C12', 'C30', 'C41']
  Visit 3: ['C03', 'C12', 'C30', 'C41']
  Visit 4: ['C03', 'C12', 'C30', 'C41']
  Visit 5: ['C00', 'C01', 'C33', 'C44']
  Visit 6: ['C00', 'C01', 'C33', 'C44']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C00', 'C01', 'C33', 'C44']
  Visit 2: ['C00', 'C01', 'C33', 'C44']
  Visit 3: ['C03', 'C12', 'C30', 'C41']
  Visit 4: ['C03', 'C12', 'C30', 'C41']
  Visit 5: ['C00', 'C01', 'C33', 'C44']
  Visit 6: ['C03', 'C12', 'C30', 'C41']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C00', 'C01', 'C33', 'C44']
  Visit 2: ['C00', 'C01', 'C33', 'C44']
  Visit 3: ['C03', 'C12', 'C30', 'C41']
  Visit 4: ['C00', 'C01', 'C33', 'C44']
  Visit 5: ['C03', 'C12', 'C30', 'C41']
  Visit 6: ['C00', 'C01', 'C33', 'C44']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.0463

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.0, 'std_depth': 0.0, 'mean_tree_dist': 2.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.371625, 'std_root_purity': 0.12495442919320626}

=== Experiment depth2_base_w/DecHypNoise_hdd | depth 2 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.474472, Val Loss: 1.170265, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.944327, Val Loss: 0.919415, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.762500, Val Loss: 0.828219, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.669603, Val Loss: 0.779058, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.625948, Val Loss: 0.754990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.591662, Val Loss: 0.727655, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.563086, Val Loss: 0.715106, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.553368, Val Loss: 0.705919, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.529209, Val Loss: 0.687594, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.502110, Val Loss: 0.682125, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.498790, Val Loss: 0.676293, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.489174, Val Loss: 0.667226, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.477639, Val Loss: 0.674488, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.475990, Val Loss: 0.675020, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.468953, Val Loss: 0.657662, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.461091, Val Loss: 0.647391, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.461387, Val Loss: 0.658540, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.461810, Val Loss: 0.656768, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.453207, Val Loss: 0.647204, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.452916, Val Loss: 0.654236, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Best validation loss: 0.647204
Saved loss curves to results/plots
Test recall@4: 0.2446

Sample trajectory (euclidean) 1:
  Visit 1: ['C03', 'C034', 'C340', 'C442']
  Visit 2: ['C023', 'C211', 'C31', 'C313']
  Visit 3: ['C01', 'C02', 'C30', 'C402']
  Visit 4: ['C112', 'C233', 'C32', 'C402']
  Visit 5: ['C224', 'C33', 'C330', 'C340']
  Visit 6: ['C01', 'C02', 'C30', 'C402']

Sample trajectory (euclidean) 2:
  Visit 1: ['C10', 'C114', 'C131', 'C312']
  Visit 2: ['C34', 'C344', 'C433', 'C44']
  Visit 3: ['C01', 'C02', 'C310', 'C402']
  Visit 4: ['C224', 'C33', 'C330', 'C340']
  Visit 5: ['C33', 'C330', 'C334', 'C340']
  Visit 6: ['C13', 'C342', 'C420', 'C433']

Sample trajectory (euclidean) 3:
  Visit 1: ['C224', 'C33', 'C330', 'C340']
  Visit 2: ['C01', 'C31', 'C310', 'C402']
  Visit 3: ['C034', 'C33', 'C330', 'C340']
  Visit 4: ['C121', 'C144', 'C224', 'C23']
  Visit 5: ['C224', 'C23', 'C33', 'C330']
  Visit 6: ['C204', 'C413', 'C44', 'C442']
Correlation(tree_dist, euclidean_embedding_dist) = 0.0483

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.6091666666666666, 'std_depth': 0.4890459817054243, 'mean_tree_dist': 2.572414380644062, 'std_tree_dist': 1.143962936121926, 'mean_root_purity': 0.60025, 'std_root_purity': 0.16147735909408475}

=== Experiment depth2_base_w/DecHypNoise_hdd | depth 2 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.534775, Val Loss: 1.220632, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.916012, Val Loss: 0.840983, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.635595, Val Loss: 0.694215, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.535511, Val Loss: 0.656921, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.499887, Val Loss: 0.610388, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.432646, Val Loss: 0.574339, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.387720, Val Loss: 0.552761, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.378374, Val Loss: 0.566712, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.388448, Val Loss: 0.571355, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.394951, Val Loss: 0.586295, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.398149, Val Loss: 0.585008, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.397432, Val Loss: 0.590505, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.394578, Val Loss: 0.587571, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.396515, Val Loss: 0.585254, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.396161, Val Loss: 0.579339, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.392207, Val Loss: 0.594033, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.393378, Val Loss: 0.583179, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.389552, Val Loss: 0.585042, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.391171, Val Loss: 0.584910, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.386581, Val Loss: 0.589116, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Best validation loss: 0.552761
Saved loss curves to results/plots
Test recall@4: 0.0829

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C012', 'C22', 'C222', 'C323']
  Visit 2: ['C03', 'C04', 'C21', 'C42']
  Visit 3: ['C012', 'C22', 'C222', 'C323']
  Visit 4: ['C03', 'C04', 'C21', 'C42']
  Visit 5: ['C012', 'C22', 'C222', 'C323']
  Visit 6: ['C012', 'C22', 'C222', 'C323']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C012', 'C22', 'C222', 'C323']
  Visit 2: ['C012', 'C22', 'C222', 'C323']
  Visit 3: ['C03', 'C04', 'C21', 'C42']
  Visit 4: ['C03', 'C04', 'C21', 'C42']
  Visit 5: ['C03', 'C04', 'C21', 'C42']
  Visit 6: ['C03', 'C04', 'C21', 'C42']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C03', 'C04', 'C21', 'C42']
  Visit 2: ['C03', 'C04', 'C21', 'C42']
  Visit 3: ['C03', 'C04', 'C21', 'C42']
  Visit 4: ['C012', 'C22', 'C222', 'C323']
  Visit 5: ['C03', 'C04', 'C21', 'C42']
  Visit 6: ['C012', 'C22', 'C222', 'C323']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9753

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.370625, 'std_depth': 0.48297216211185506, 'mean_tree_dist': 1.5058333333333334, 'std_tree_dist': 0.4999659710642537, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

=== Experiment depth2_base_w/DecHypNoise_hdd | depth 2 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 1.777736, Val Loss: 1.490787, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 1.239938, Val Loss: 1.177433, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 1.007472, Val Loss: 1.065307, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.884419, Val Loss: 0.977141, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.816099, Val Loss: 0.936890, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.764983, Val Loss: 0.896079, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.715587, Val Loss: 0.859988, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.682835, Val Loss: 0.833555, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.657185, Val Loss: 0.803361, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.630723, Val Loss: 0.790453, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.609553, Val Loss: 0.765700, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.584977, Val Loss: 0.743215, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.564988, Val Loss: 0.741825, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.549976, Val Loss: 0.716120, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.526187, Val Loss: 0.701024, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.508930, Val Loss: 0.680025, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.476242, Val Loss: 0.640676, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.451147, Val Loss: 0.630855, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.434989, Val Loss: 0.627143, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.421167, Val Loss: 0.617589, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Best validation loss: 0.617589
Saved loss curves to results/plots
Test recall@4: 0.1557

Sample trajectory (euclidean) 1:
  Visit 1: ['C111', 'C24', 'C32', 'C41']
  Visit 2: ['C111', 'C24', 'C32', 'C41']
  Visit 3: ['C032', 'C13', 'C131', 'C40']
  Visit 4: ['C13', 'C40', 'C400', 'C401']
  Visit 5: ['C11', 'C113', 'C12', 'C431']
  Visit 6: ['C032', 'C13', 'C131', 'C40']

Sample trajectory (euclidean) 2:
  Visit 1: ['C111', 'C24', 'C32', 'C41']
  Visit 2: ['C032', 'C13', 'C131', 'C40']
  Visit 3: ['C032', 'C13', 'C131', 'C40']
  Visit 4: ['C023', 'C032', 'C13', 'C400']
  Visit 5: ['C031', 'C032', 'C341', 'C400']
  Visit 6: ['C032', 'C13', 'C131', 'C40']

Sample trajectory (euclidean) 3:
  Visit 1: ['C111', 'C24', 'C32', 'C41']
  Visit 2: ['C032', 'C13', 'C131', 'C40']
  Visit 3: ['C032', 'C13', 'C131', 'C40']
  Visit 4: ['C032', 'C13', 'C131', 'C40']
  Visit 5: ['C032', 'C13', 'C131', 'C40']
  Visit 6: ['C111', 'C24', 'C32', 'C41']
Correlation(tree_dist, euclidean_embedding_dist) = 0.1516

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.399625, 'std_depth': 0.49607612928024136, 'mean_tree_dist': 1.6287387575820957, 'std_tree_dist': 0.9982217293461216, 'mean_root_purity': 0.42270833333333335, 'std_root_purity': 0.1475316291417614}

Real stats (depth7_extended_w/DecHypNoise_hdd, max_depth=7): {'mean_depth': 5.3797976334479465, 'std_depth': 1.7294582012361523, 'mean_tree_dist': 5.7591450057100175, 'std_tree_dist': 4.756650684766557, 'mean_root_purity': 0.6289569269083962, 'std_root_purity': 0.20468844637974468}

=== Experiment depth7_extended_w/DecHypNoise_hdd | depth 7 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.395746, Val Loss: 1.096705, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.806977, Val Loss: 0.741233, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.529464, Val Loss: 0.582528, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.420430, Val Loss: 0.530158, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.378490, Val Loss: 0.507089, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.327186, Val Loss: 0.456345, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.274091, Val Loss: 0.408032, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.239408, Val Loss: 0.404198, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.230463, Val Loss: 0.400251, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.224661, Val Loss: 0.396832, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.219433, Val Loss: 0.393142, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.183093, Val Loss: 0.355993, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.163049, Val Loss: 0.356095, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.157729, Val Loss: 0.349400, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.153410, Val Loss: 0.351691, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.150482, Val Loss: 0.349687, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.147421, Val Loss: 0.351269, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.144713, Val Loss: 0.352689, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.142519, Val Loss: 0.348961, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.141717, Val Loss: 0.347258, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Best validation loss: 0.347258
Saved loss curves to results/plots
Test recall@4: 0.0089

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C013d3', 'C202d4', 'C211d3', 'C322d4']
  Visit 2: ['C013d3', 'C202d4', 'C211d3', 'C322d4']
  Visit 3: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 4: ['C013d3', 'C202d4', 'C211d3', 'C322d4']
  Visit 5: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 6: ['C013d3', 'C202d4', 'C211d3', 'C322d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C013d3', 'C202d4', 'C211d3', 'C322d4']
  Visit 2: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 3: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 4: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 5: ['C013d3', 'C202d4', 'C211d3', 'C322d4']
  Visit 6: ['C013d3', 'C202d4', 'C211d3', 'C322d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 2: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 3: ['C013d3', 'C202d4', 'C211d3', 'C322d4']
  Visit 4: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 5: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
  Visit 6: ['C101d4', 'C112d4', 'C330d4', 'C400d3']
Correlation(tree_dist, hyperbolic_embedding_dist) = -0.0078

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 6.623625, 'std_depth': 0.48447586046675223, 'mean_tree_dist': 13.4945, 'std_tree_dist': 0.4999697490848822, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

=== Experiment depth7_extended_w/DecHypNoise_hdd | depth 7 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.507807, Val Loss: 1.204589, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.964578, Val Loss: 0.907627, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.738377, Val Loss: 0.777566, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.623729, Val Loss: 0.717126, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.575660, Val Loss: 0.698815, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.531689, Val Loss: 0.662607, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.510887, Val Loss: 0.657249, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.501390, Val Loss: 0.655470, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.489553, Val Loss: 0.647493, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.489216, Val Loss: 0.659185, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.483494, Val Loss: 0.652740, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.474272, Val Loss: 0.652343, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.475330, Val Loss: 0.629542, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.474073, Val Loss: 0.659801, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.473372, Val Loss: 0.645905, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.466228, Val Loss: 0.649576, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.464464, Val Loss: 0.653201, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.463900, Val Loss: 0.651090, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.465445, Val Loss: 0.644821, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.460262, Val Loss: 0.648887, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_recon=1.0000
Best validation loss: 0.629542
Saved loss curves to results/plots
Test recall@4: 0.1262

Sample trajectory (euclidean) 1:
  Visit 1: ['C140d4', 'C334d3', 'C342d4', 'C420d3']
  Visit 2: ['C032d4', 'C111d3', 'C441d4', 'C442d4']
  Visit 3: ['C011d3', 'C101d3', 'C101d4', 'C140d4']
  Visit 4: ['C032d4', 'C104d3', 'C104d4', 'C442d4']
  Visit 5: ['C034d3', 'C103d3', 'C103d4', 'C113d4']
  Visit 6: ['C014d3', 'C140d3', 'C140d4', 'C334d3']

Sample trajectory (euclidean) 2:
  Visit 1: ['C223d4', 'C333d4', 'C402d3', 'C402d4']
  Visit 2: ['C104d4', 'C242d4', 'C440d3', 'C441d4']
  Visit 3: ['C032d3', 'C032d4', 'C111d3', 'C304d3']
  Visit 4: ['C014d3', 'C044d3', 'C140d4', 'C143d3']
  Visit 5: ['C104d3', 'C104d4', 'C434d3', 'C442d4']
  Visit 6: ['C103d3', 'C103d4', 'C302d3', 'C342d4']

Sample trajectory (euclidean) 3:
  Visit 1: ['C032d4', 'C040d3', 'C104d3', 'C104d4']
  Visit 2: ['C323d4', 'C333d3', 'C333d4', 'C402d3']
  Visit 3: ['C032d3', 'C032d4', 'C111d3', 'C304d3']
  Visit 4: ['C140d3', 'C140d4', 'C144d3', 'C144d4']
  Visit 5: ['C140d3', 'C140d4', 'C334d3', 'C342d4']
  Visit 6: ['C223d3', 'C333d4', 'C402d3', 'C402d4']
Correlation(tree_dist, euclidean_embedding_dist) = -0.0135

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.539458333333333, 'std_depth': 0.4984406082278565, 'mean_tree_dist': 8.028088403627718, 'std_tree_dist': 5.58009424420139, 'mean_root_purity': 0.5815833333333333, 'std_root_purity': 0.14459135424437458}

=== Experiment depth7_extended_w/DecHypNoise_hdd | depth 7 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.550303, Val Loss: 1.214092, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.907264, Val Loss: 0.814137, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.599283, Val Loss: 0.627562, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.447690, Val Loss: 0.531454, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.368684, Val Loss: 0.497136, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.342329, Val Loss: 0.487781, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.329918, Val Loss: 0.490092, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.313587, Val Loss: 0.449959, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.282088, Val Loss: 0.450676, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.279146, Val Loss: 0.453844, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.283265, Val Loss: 0.469048, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.286637, Val Loss: 0.477314, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.291809, Val Loss: 0.485053, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.294429, Val Loss: 0.474794, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.300986, Val Loss: 0.487735, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.304223, Val Loss: 0.490444, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.307027, Val Loss: 0.497856, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.309111, Val Loss: 0.503892, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.315823, Val Loss: 0.512582, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.317247, Val Loss: 0.512863, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_recon=1.0000
Best validation loss: 0.449959
Saved loss curves to results/plots
Test recall@4: 0.0116

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 2: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 3: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 4: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 5: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 6: ['C130d4', 'C200d3', 'C411d3', 'C412d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 2: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 3: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 4: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 5: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 6: ['C121d3', 'C140d4', 'C244d4', 'C341d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 2: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 3: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 4: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 5: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 6: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9145

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 6.622791666666667, 'std_depth': 0.48468774133169723, 'mean_tree_dist': 11.982333333333333, 'std_tree_dist': 0.9998439322658754, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

=== Experiment depth7_extended_w/DecHypNoise_hdd | depth 7 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 1.814268, Val Loss: 1.521894, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 1.253803, Val Loss: 1.169030, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.980116, Val Loss: 1.006479, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.846015, Val Loss: 0.927682, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.770356, Val Loss: 0.890397, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.716894, Val Loss: 0.848500, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.676349, Val Loss: 0.802186, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.647740, Val Loss: 0.778316, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.603478, Val Loss: 0.754125, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.572307, Val Loss: 0.744290, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.558705, Val Loss: 0.726647, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.537004, Val Loss: 0.716899, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.519262, Val Loss: 0.702720, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.510848, Val Loss: 0.677811, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.492030, Val Loss: 0.668020, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.476879, Val Loss: 0.661233, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.464404, Val Loss: 0.638014, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.450759, Val Loss: 0.633966, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.438387, Val Loss: 0.626959, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.425050, Val Loss: 0.618803, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_recon=1.0000
Best validation loss: 0.618803
Saved loss curves to results/plots
Test recall@4: 0.1206

Sample trajectory (euclidean) 1:
  Visit 1: ['C202d4', 'C324d4', 'C411d3', 'C411d4']
  Visit 2: ['C031d3', 'C031d4', 'C202d4', 'C411d3']
  Visit 3: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 4: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 5: ['C031d3', 'C031d4', 'C240d3', 'C240d4']
  Visit 6: ['C113d4', 'C202d3', 'C202d4', 'C411d3']

Sample trajectory (euclidean) 2:
  Visit 1: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 2: ['C003d3', 'C010d3', 'C110d4', 'C301d4']
  Visit 3: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 4: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 5: ['C220d3', 'C220d4', 'C402d4', 'C414d3']
  Visit 6: ['C031d3', 'C031d4', 'C202d4', 'C411d3']

Sample trajectory (euclidean) 3:
  Visit 1: ['C010d4', 'C031d4', 'C240d3', 'C411d3']
  Visit 2: ['C202d4', 'C240d3', 'C411d3', 'C411d4']
  Visit 3: ['C010d4', 'C202d4', 'C333d4', 'C420d3']
  Visit 4: ['C413d3', 'C421d3', 'C421d4', 'C422d3']
  Visit 5: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 6: ['C031d3', 'C031d4', 'C202d4', 'C411d3']
Correlation(tree_dist, euclidean_embedding_dist) = 0.1633

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.496791666666667, 'std_depth': 0.49998970649126584, 'mean_tree_dist': 7.143839899937461, 'std_tree_dist': 5.7003843732801, 'mean_root_purity': 0.521625, 'std_root_purity': 0.11218040251458065}
