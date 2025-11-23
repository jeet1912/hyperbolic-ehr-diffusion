
Real stats (depth2_base_w/DecHypNoise_hgd, max_depth=2): {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Experiment depth2_base_w/DecHypNoise_hgd | depth 2 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.427652, Val Loss: 1.120670, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.823479, Val Loss: 0.750248, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.531852, Val Loss: 0.590492, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.417815, Val Loss: 0.528251, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.354095, Val Loss: 0.495525, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.331733, Val Loss: 0.488008, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.320655, Val Loss: 0.482749, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.313527, Val Loss: 0.481798, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.308920, Val Loss: 0.482923, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.304356, Val Loss: 0.483869, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.294482, Val Loss: 0.458599, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.253800, Val Loss: 0.440864, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.242862, Val Loss: 0.439169, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.238956, Val Loss: 0.433903, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.234942, Val Loss: 0.435085, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.231663, Val Loss: 0.438831, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.228746, Val Loss: 0.432703, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.225572, Val Loss: 0.431846, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.223154, Val Loss: 0.434699, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.221155, Val Loss: 0.436990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
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

=== Experiment depth2_base_w/DecHypNoise_hgd | depth 2 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.474472, Val Loss: 1.170265, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.944327, Val Loss: 0.919415, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.762500, Val Loss: 0.828219, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.669603, Val Loss: 0.779058, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.625948, Val Loss: 0.754990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.591662, Val Loss: 0.727655, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.563086, Val Loss: 0.715106, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.553368, Val Loss: 0.705919, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.529209, Val Loss: 0.687594, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.502110, Val Loss: 0.682125, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.498790, Val Loss: 0.676293, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.489174, Val Loss: 0.667226, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.477639, Val Loss: 0.674488, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.475990, Val Loss: 0.675020, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.468953, Val Loss: 0.657662, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.461091, Val Loss: 0.647391, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.461387, Val Loss: 0.658540, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.461810, Val Loss: 0.656768, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.453207, Val Loss: 0.647204, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.452916, Val Loss: 0.654236, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
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

=== Experiment depth2_base_w/DecHypNoise_hgd | depth 2 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.524104, Val Loss: 1.210753, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.906363, Val Loss: 0.831405, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.626398, Val Loss: 0.685067, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.526270, Val Loss: 0.647308, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.489928, Val Loss: 0.599639, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.421117, Val Loss: 0.561989, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.373951, Val Loss: 0.538264, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.362607, Val Loss: 0.550211, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.370663, Val Loss: 0.553750, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.375371, Val Loss: 0.566700, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.377613, Val Loss: 0.565658, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.376525, Val Loss: 0.570187, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.373577, Val Loss: 0.567166, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.375181, Val Loss: 0.565522, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.374560, Val Loss: 0.559788, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.370693, Val Loss: 0.573499, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.371937, Val Loss: 0.563395, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.368105, Val Loss: 0.564853, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.369659, Val Loss: 0.564531, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.365145, Val Loss: 0.568433, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.538264
Saved loss curves to results/plots
Test recall@4: 0.0763

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C012', 'C221', 'C222', 'C323']
  Visit 2: ['C03', 'C04', 'C21', 'C42']
  Visit 3: ['C012', 'C221', 'C222', 'C323']
  Visit 4: ['C03', 'C04', 'C21', 'C42']
  Visit 5: ['C012', 'C221', 'C222', 'C323']
  Visit 6: ['C03', 'C04', 'C21', 'C42']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C012', 'C221', 'C222', 'C323']
  Visit 2: ['C012', 'C221', 'C222', 'C323']
  Visit 3: ['C03', 'C04', 'C21', 'C42']
  Visit 4: ['C03', 'C04', 'C21', 'C42']
  Visit 5: ['C03', 'C04', 'C21', 'C42']
  Visit 6: ['C03', 'C04', 'C21', 'C42']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C03', 'C04', 'C21', 'C42']
  Visit 2: ['C03', 'C04', 'C21', 'C42']
  Visit 3: ['C03', 'C04', 'C21', 'C42']
  Visit 4: ['C012', 'C221', 'C222', 'C323']
  Visit 5: ['C03', 'C04', 'C21', 'C42']
  Visit 6: ['C012', 'C221', 'C222', 'C323']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9837

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.49225, 'std_depth': 0.4999399338920627, 'mean_tree_dist': 1.9976666666666667, 'std_tree_dist': 0.04824820088758635, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

=== Experiment depth2_base_w/DecHypNoise_hgd | depth 2 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 1.803505, Val Loss: 1.516151, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 1.264521, Val Loss: 1.201386, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 1.030839, Val Loss: 1.087553, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.906524, Val Loss: 0.998601, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.836860, Val Loss: 0.956923, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.784522, Val Loss: 0.915053, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.734216, Val Loss: 0.877849, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.699932, Val Loss: 0.850581, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.673401, Val Loss: 0.818852, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.645565, Val Loss: 0.805293, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.623341, Val Loss: 0.779090, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.597668, Val Loss: 0.755756, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.576521, Val Loss: 0.752769, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.560558, Val Loss: 0.726225, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.535874, Val Loss: 0.710313, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.517478, Val Loss: 0.687969, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.483575, Val Loss: 0.648122, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.457706, Val Loss: 0.637087, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.440658, Val Loss: 0.632532, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.425973, Val Loss: 0.621947, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.621947
Saved loss curves to results/plots
Test recall@4: 0.1527

Sample trajectory (euclidean) 1:
  Visit 1: ['C111', 'C24', 'C32', 'C41']
  Visit 2: ['C111', 'C24', 'C32', 'C41']
  Visit 3: ['C032', 'C13', 'C131', 'C40']
  Visit 4: ['C13', 'C40', 'C400', 'C401']
  Visit 5: ['C11', 'C113', 'C12', 'C431']
  Visit 6: ['C032', 'C13', 'C231', 'C40']

Sample trajectory (euclidean) 2:
  Visit 1: ['C111', 'C24', 'C32', 'C41']
  Visit 2: ['C032', 'C13', 'C131', 'C40']
  Visit 3: ['C032', 'C13', 'C131', 'C40']
  Visit 4: ['C023', 'C032', 'C13', 'C400']
  Visit 5: ['C031', 'C032', 'C341', 'C400']
  Visit 6: ['C032', 'C13', 'C131', 'C40']

Sample trajectory (euclidean) 3:
  Visit 1: ['C11', 'C111', 'C32', 'C41']
  Visit 2: ['C032', 'C13', 'C131', 'C40']
  Visit 3: ['C032', 'C13', 'C131', 'C40']
  Visit 4: ['C032', 'C13', 'C131', 'C40']
  Visit 5: ['C032', 'C13', 'C131', 'C40']
  Visit 6: ['C111', 'C24', 'C32', 'C41']
Correlation(tree_dist, euclidean_embedding_dist) = 0.1591

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.3995, 'std_depth': 0.4957147197061364, 'mean_tree_dist': 1.6360225140712945, 'std_tree_dist': 1.0051457464263533, 'mean_root_purity': 0.4232916666666667, 'std_root_purity': 0.14776529226182386}

Real stats (depth7_extended_w/DecHypNoise_hgd, max_depth=7): {'mean_depth': 5.3797976334479465, 'std_depth': 1.7294582012361523, 'mean_tree_dist': 5.7591450057100175, 'std_tree_dist': 4.756650684766557, 'mean_root_purity': 0.6289569269083962, 'std_root_purity': 0.20468844637974468}

=== Experiment depth7_extended_w/DecHypNoise_hgd | depth 7 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.395746, Val Loss: 1.096705, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.806977, Val Loss: 0.741233, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.529464, Val Loss: 0.582528, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.420430, Val Loss: 0.530158, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.378490, Val Loss: 0.507089, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.327186, Val Loss: 0.456345, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.274091, Val Loss: 0.408032, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.239408, Val Loss: 0.404198, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.230463, Val Loss: 0.400251, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.224661, Val Loss: 0.396832, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.219433, Val Loss: 0.393142, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.183093, Val Loss: 0.355993, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.163049, Val Loss: 0.356095, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.157729, Val Loss: 0.349400, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.153410, Val Loss: 0.351691, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.150482, Val Loss: 0.349687, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.147421, Val Loss: 0.351269, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.144713, Val Loss: 0.352689, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.142519, Val Loss: 0.348961, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.141717, Val Loss: 0.347258, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
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

=== Experiment depth7_extended_w/DecHypNoise_hgd | depth 7 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.507807, Val Loss: 1.204589, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.964578, Val Loss: 0.907627, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.738377, Val Loss: 0.777566, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.623729, Val Loss: 0.717126, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.575660, Val Loss: 0.698815, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.531689, Val Loss: 0.662607, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.510887, Val Loss: 0.657249, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.501390, Val Loss: 0.655470, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.489553, Val Loss: 0.647493, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.489216, Val Loss: 0.659185, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.483494, Val Loss: 0.652740, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.474272, Val Loss: 0.652343, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.475330, Val Loss: 0.629542, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.474073, Val Loss: 0.659801, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.473372, Val Loss: 0.645905, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.466228, Val Loss: 0.649576, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.464464, Val Loss: 0.653201, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.463900, Val Loss: 0.651090, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.465445, Val Loss: 0.644821, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.460262, Val Loss: 0.648887, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
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

=== Experiment depth7_extended_w/DecHypNoise_hgd | depth 7 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.528660, Val Loss: 1.193132, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.886601, Val Loss: 0.794024, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.579349, Val Loss: 0.607779, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.427965, Val Loss: 0.511915, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.349026, Val Loss: 0.477591, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.322584, Val Loss: 0.468279, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.310238, Val Loss: 0.469602, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.292506, Val Loss: 0.428842, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.260567, Val Loss: 0.428830, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.256758, Val Loss: 0.431252, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.259425, Val Loss: 0.444510, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.261246, Val Loss: 0.450905, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.265135, Val Loss: 0.458431, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.266991, Val Loss: 0.447701, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.272219, Val Loss: 0.459197, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.274479, Val Loss: 0.461120, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.276645, Val Loss: 0.467793, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.278141, Val Loss: 0.474435, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.283885, Val Loss: 0.481368, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.285453, Val Loss: 0.481438, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.428830
Saved loss curves to results/plots
Test recall@4: 0.0117

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 2: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 3: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 4: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 5: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
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
  Visit 2: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 3: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 4: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
  Visit 5: ['C130d4', 'C200d3', 'C411d3', 'C412d4']
  Visit 6: ['C121d3', 'C140d4', 'C244d4', 'C341d4']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9354

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 6.623, 'std_depth': 0.48463491413640425, 'mean_tree_dist': 11.984, 'std_tree_dist': 0.9998719918069512, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

=== Experiment depth7_extended_w/DecHypNoise_hgd | depth 7 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 1.862411, Val Loss: 1.568997, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 1.300141, Val Loss: 1.214230, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 1.024577, Val Loss: 1.050151, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.888732, Val Loss: 0.969598, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.811229, Val Loss: 0.930864, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.756051, Val Loss: 0.886779, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.713908, Val Loss: 0.838697, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.683474, Val Loss: 0.813451, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.637465, Val Loss: 0.787014, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.604512, Val Loss: 0.775743, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.589242, Val Loss: 0.756256, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.565922, Val Loss: 0.744968, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.546463, Val Loss: 0.729232, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.536464, Val Loss: 0.702834, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.515918, Val Loss: 0.691025, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.499263, Val Loss: 0.682761, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.485195, Val Loss: 0.658191, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.470057, Val Loss: 0.652572, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.456141, Val Loss: 0.643811, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.441321, Val Loss: 0.634645, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.634645
Saved loss curves to results/plots
Test recall@4: 0.1185

Sample trajectory (euclidean) 1:
  Visit 1: ['C202d4', 'C324d4', 'C411d3', 'C411d4']
  Visit 2: ['C031d3', 'C031d4', 'C202d4', 'C411d3']
  Visit 3: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 4: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 5: ['C031d3', 'C031d4', 'C240d3', 'C240d4']
  Visit 6: ['C113d4', 'C202d3', 'C202d4', 'C411d3']

Sample trajectory (euclidean) 2:
  Visit 1: ['C112d4', 'C300d3', 'C344d3', 'C402d4']
  Visit 2: ['C010d3', 'C110d4', 'C301d4', 'C443d3']
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
Correlation(tree_dist, euclidean_embedding_dist) = 0.1532

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.4969166666666665, 'std_depth': 0.49999049296517195, 'mean_tree_dist': 7.1533039647577095, 'std_tree_dist': 5.697595087058764, 'mean_root_purity': 0.5207916666666667, 'std_root_purity': 0.11140671702021483}
