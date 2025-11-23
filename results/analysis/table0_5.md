
Real stats (depth2_base_w/DecHypNoise_hgd2, max_depth=2): {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Experiment depth2_base_w/DecHypNoise_hgd2 | depth 2 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.427652, Val Loss: 1.120670, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.823479, Val Loss: 0.750248, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.531852, Val Loss: 0.590492, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.417815, Val Loss: 0.528251, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.354095, Val Loss: 0.495525, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.331733, Val Loss: 0.488008, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.320655, Val Loss: 0.482749, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.313527, Val Loss: 0.481798, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.308920, Val Loss: 0.482923, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.304356, Val Loss: 0.483869, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.294482, Val Loss: 0.458599, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.253800, Val Loss: 0.440864, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.242862, Val Loss: 0.439169, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.238956, Val Loss: 0.433903, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.234942, Val Loss: 0.435085, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.231663, Val Loss: 0.438831, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.228746, Val Loss: 0.432703, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.225572, Val Loss: 0.431846, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.223154, Val Loss: 0.434699, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.221155, Val Loss: 0.436990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
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

=== Experiment depth2_base_w/DecHypNoise_hgd2 | depth 2 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.474472, Val Loss: 1.170265, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.944327, Val Loss: 0.919415, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.762500, Val Loss: 0.828219, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.669603, Val Loss: 0.779058, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.625948, Val Loss: 0.754990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.591662, Val Loss: 0.727655, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.563086, Val Loss: 0.715106, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.553368, Val Loss: 0.705919, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.529209, Val Loss: 0.687594, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.502110, Val Loss: 0.682125, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.498790, Val Loss: 0.676293, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.489174, Val Loss: 0.667226, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.477639, Val Loss: 0.674488, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.475990, Val Loss: 0.675020, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.468953, Val Loss: 0.657662, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.461091, Val Loss: 0.647391, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.461387, Val Loss: 0.658540, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.461810, Val Loss: 0.656768, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.453207, Val Loss: 0.647204, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.452916, Val Loss: 0.654236, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
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

=== Experiment depth2_base_w/DecHypNoise_hgd2 | depth 2 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.575221, Val Loss: 1.252169, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.945937, Val Loss: 0.872377, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.658326, Val Loss: 0.717922, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.559878, Val Loss: 0.685659, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.522311, Val Loss: 0.649693, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.460455, Val Loss: 0.599848, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.418213, Val Loss: 0.589322, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.404568, Val Loss: 0.580542, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.401305, Val Loss: 0.582147, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.402789, Val Loss: 0.585795, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.396042, Val Loss: 0.585509, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.391187, Val Loss: 0.587480, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.390175, Val Loss: 0.584964, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.388699, Val Loss: 0.586076, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.388509, Val Loss: 0.580481, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.384614, Val Loss: 0.586375, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.383287, Val Loss: 0.577852, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.381827, Val Loss: 0.582003, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.378117, Val Loss: 0.569686, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.375327, Val Loss: 0.578254, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.569686
Saved loss curves to results/plots
Test recall@4: 0.1454

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C14', 'C20', 'C22', 'C41']
  Visit 2: ['C14', 'C20', 'C22', 'C41']
  Visit 3: ['C03', 'C04', 'C24', 'C42']
  Visit 4: ['C03', 'C04', 'C24', 'C42']
  Visit 5: ['C14', 'C20', 'C22', 'C41']
  Visit 6: ['C03', 'C04', 'C24', 'C42']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C14', 'C20', 'C22', 'C41']
  Visit 2: ['C14', 'C20', 'C22', 'C41']
  Visit 3: ['C03', 'C04', 'C24', 'C42']
  Visit 4: ['C03', 'C04', 'C24', 'C42']
  Visit 5: ['C03', 'C04', 'C24', 'C42']
  Visit 6: ['C14', 'C20', 'C22', 'C41']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C03', 'C04', 'C24', 'C42']
  Visit 2: ['C14', 'C20', 'C22', 'C41']
  Visit 3: ['C14', 'C20', 'C22', 'C41']
  Visit 4: ['C03', 'C04', 'C24', 'C42']
  Visit 5: ['C03', 'C04', 'C24', 'C42']
  Visit 6: ['C03', 'C04', 'C24', 'C42']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9777

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.0, 'std_depth': 0.0, 'mean_tree_dist': 2.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

=== Experiment depth2_base_w/DecHypNoise_hgd2 | depth 2 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 2.185449, Val Loss: 1.886549, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 1.608333, Val Loss: 1.514210, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 1.328799, Val Loss: 1.341782, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 1.171956, Val Loss: 1.241195, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 1.063003, Val Loss: 1.149060, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.983065, Val Loss: 1.094844, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.921419, Val Loss: 1.038879, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.852101, Val Loss: 0.971401, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.802509, Val Loss: 0.941745, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.759643, Val Loss: 0.909973, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.719126, Val Loss: 0.870575, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.686462, Val Loss: 0.846288, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.649279, Val Loss: 0.822270, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.622454, Val Loss: 0.786741, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.597129, Val Loss: 0.764253, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.571779, Val Loss: 0.746602, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.551272, Val Loss: 0.713788, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.525487, Val Loss: 0.704982, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.507149, Val Loss: 0.699033, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.487249, Val Loss: 0.665742, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.665742
Saved loss curves to results/plots
Test recall@4: 0.1406

Sample trajectory (euclidean) 1:
  Visit 1: ['C20', 'C204', 'C322', 'C43']
  Visit 2: ['C303', 'C323', 'C42', 'C420']
  Visit 3: ['C30', 'C322', 'C404', 'C42']
  Visit 4: ['C100', 'C130', 'C312', 'C442']
  Visit 5: ['C303', 'C323', 'C42', 'C420']
  Visit 6: ['C10', 'C121', 'C34', 'C342']

Sample trajectory (euclidean) 2:
  Visit 1: ['C303', 'C32', 'C42', 'C420']
  Visit 2: ['C00', 'C22', 'C404', 'C43']
  Visit 3: ['C22', 'C223', 'C322', 'C404']
  Visit 4: ['C10', 'C112', 'C34', 'C342']
  Visit 5: ['C012', 'C10', 'C34', 'C342']
  Visit 6: ['C30', 'C303', 'C323', 'C42']

Sample trajectory (euclidean) 3:
  Visit 1: ['C00', 'C10', 'C12', 'C14']
  Visit 2: ['C02', 'C124', 'C233', 'C442']
  Visit 3: ['C012', 'C10', 'C34', 'C342']
  Visit 4: ['C10', 'C12', 'C34', 'C342']
  Visit 5: ['C00', 'C12', 'C14', 'C43']
  Visit 6: ['C30', 'C303', 'C323', 'C42']
Correlation(tree_dist, euclidean_embedding_dist) = 0.0790

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.4269166666666666, 'std_depth': 0.49496682689067384, 'mean_tree_dist': 2.2105077987568897, 'std_tree_dist': 1.0299341989813853, 'mean_root_purity': 0.498375, 'std_root_purity': 0.1622994435449487}

Real stats (depth7_extended_w/DecHypNoise_hgd2, max_depth=7): {'mean_depth': 5.3797976334479465, 'std_depth': 1.7294582012361523, 'mean_tree_dist': 5.7591450057100175, 'std_tree_dist': 4.756650684766557, 'mean_root_purity': 0.6289569269083962, 'std_root_purity': 0.20468844637974468}

=== Experiment depth7_extended_w/DecHypNoise_hgd2 | depth 7 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.513236, Val Loss: 1.144987, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.830042, Val Loss: 0.747377, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.551956, Val Loss: 0.589704, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.425210, Val Loss: 0.518762, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.336324, Val Loss: 0.430100, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.252367, Val Loss: 0.380936, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.211797, Val Loss: 0.362760, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.195853, Val Loss: 0.355825, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.186672, Val Loss: 0.353056, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.180082, Val Loss: 0.355258, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.174781, Val Loss: 0.351241, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.169847, Val Loss: 0.351740, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.166514, Val Loss: 0.350794, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.162336, Val Loss: 0.345884, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.158349, Val Loss: 0.348994, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.156125, Val Loss: 0.353734, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.152778, Val Loss: 0.352676, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.150500, Val Loss: 0.349005, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.148479, Val Loss: 0.348610, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.146271, Val Loss: 0.348765, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Best validation loss: 0.345884
Saved loss curves to results/plots
Test recall@4: 0.0093

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C023d4', 'C210d4', 'C420d3', 'C431d4']
  Visit 2: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 3: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 4: ['C023d4', 'C210d4', 'C420d3', 'C431d4']
  Visit 5: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 6: ['C104d4', 'C320d4', 'C333d4', 'C411d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C023d4', 'C210d4', 'C420d3', 'C431d4']
  Visit 2: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 3: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 4: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 5: ['C023d4', 'C210d4', 'C420d3', 'C431d4']
  Visit 6: ['C023d4', 'C210d4', 'C420d3', 'C431d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C023d4', 'C210d4', 'C420d3', 'C431d4']
  Visit 2: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 3: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 4: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 5: ['C104d4', 'C320d4', 'C333d4', 'C411d4']
  Visit 6: ['C023d4', 'C210d4', 'C420d3', 'C431d4']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.0168

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 6.874583333333334, 'std_depth': 0.33119077038602523, 'mean_tree_dist': 13.498333333333333, 'std_tree_dist': 0.49999722221450604, 'mean_root_purity': 0.5, 'std_root_purity': 0.0}

=== Experiment depth7_extended_w/DecHypNoise_hgd2 | depth 7 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.502012, Val Loss: 1.207229, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.959915, Val Loss: 0.894763, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.730661, Val Loss: 0.778960, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.625938, Val Loss: 0.722572, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.573387, Val Loss: 0.690291, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.531935, Val Loss: 0.668852, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.510733, Val Loss: 0.659136, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.501495, Val Loss: 0.656288, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.495232, Val Loss: 0.646762, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.487302, Val Loss: 0.654863, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.482126, Val Loss: 0.645241, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.462578, Val Loss: 0.611237, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.446310, Val Loss: 0.631075, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.446016, Val Loss: 0.624256, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.439414, Val Loss: 0.614293, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.434608, Val Loss: 0.625264, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.435822, Val Loss: 0.625180, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.429677, Val Loss: 0.634708, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.434207, Val Loss: 0.620580, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.427482, Val Loss: 0.621802, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_hdd=0.0000, lambda_hgd=0.0000, lambda_recon=1.0000
Best validation loss: 0.611237
Saved loss curves to results/plots
Test recall@4: 0.1379

Sample trajectory (euclidean) 1:
  Visit 1: ['C041d4', 'C223d3', 'C223d4', 'C400d3']
  Visit 2: ['C011d3', 'C011d4', 'C103d3', 'C103d4']
  Visit 3: ['C023d3', 'C023d4', 'C240d4', 'C410d3']
  Visit 4: ['C011d3', 'C041d4', 'C223d3', 'C400d3']
  Visit 5: ['C021d4', 'C034d4', 'C421d3', 'C421d4']
  Visit 6: ['C023d4', 'C144d4', 'C332d4', 'C403d4']

Sample trajectory (euclidean) 2:
  Visit 1: ['C204d4', 'C214d4', 'C331d3', 'C331d4']
  Visit 2: ['C021d4', 'C230d4', 'C341d3', 'C442d3']
  Visit 3: ['C041d4', 'C223d3', 'C223d4', 'C400d3']
  Visit 4: ['C021d4', 'C213d3', 'C213d4', 'C333d3']
  Visit 5: ['C023d3', 'C023d4', 'C204d4', 'C402d3']
  Visit 6: ['C110d3', 'C110d4', 'C213d3', 'C224d4']

Sample trajectory (euclidean) 3:
  Visit 1: ['C021d4', 'C110d4', 'C213d3', 'C213d4']
  Visit 2: ['C032d4', 'C223d3', 'C223d4', 'C400d3']
  Visit 3: ['C023d4', 'C041d4', 'C223d3', 'C400d3']
  Visit 4: ['C041d4', 'C223d3', 'C223d4', 'C400d3']
  Visit 5: ['C030d3', 'C333d3', 'C333d4', 'C434d3']
  Visit 6: ['C011d3', 'C041d4', 'C223d3', 'C223d4']
Correlation(tree_dist, euclidean_embedding_dist) = -0.0796

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.501625, 'std_depth': 0.4999973593680271, 'mean_tree_dist': 6.112317073170732, 'std_tree_dist': 5.89579796597222, 'mean_root_purity': 0.5165, 'std_root_purity': 0.11346181442817374}

=== Experiment depth7_extended_w/DecHypNoise_hgd2 | depth 7 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.511804, Val Loss: 1.196742, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.885270, Val Loss: 0.809540, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.578717, Val Loss: 0.621681, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.418408, Val Loss: 0.509755, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.333740, Val Loss: 0.484085, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.309406, Val Loss: 0.480804, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.301145, Val Loss: 0.477531, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.300493, Val Loss: 0.486222, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.304707, Val Loss: 0.488880, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.310714, Val Loss: 0.496934, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.318124, Val Loss: 0.509209, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.319058, Val Loss: 0.508724, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.324341, Val Loss: 0.512808, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.323250, Val Loss: 0.523423, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.324783, Val Loss: 0.524912, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.327784, Val Loss: 0.521932, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.328116, Val Loss: 0.517117, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.329357, Val Loss: 0.523222, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.327813, Val Loss: 0.527023, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.328004, Val Loss: 0.529022, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.477531
Saved loss curves to results/plots
Test recall@4: 0.0127

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 2: ['C331', 'C002d0', 'C241d3', 'C314d4']
  Visit 3: ['C331', 'C002d0', 'C241d3', 'C314d4']
  Visit 4: ['C331', 'C002d0', 'C241d3', 'C314d4']
  Visit 5: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 6: ['C331', 'C002d0', 'C241d3', 'C314d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C331', 'C002d0', 'C241d3', 'C314d4']
  Visit 2: ['C331', 'C002d0', 'C241d3', 'C314d4']
  Visit 3: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 4: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 5: ['C331', 'C002d0', 'C241d3', 'C314d4']
  Visit 6: ['C033d3', 'C314d3', 'C423d3', 'C432d3']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 2: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 3: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 4: ['C331', 'C002d0', 'C241d3', 'C314d4']
  Visit 5: ['C033d3', 'C314d3', 'C423d3', 'C432d3']
  Visit 6: ['C331', 'C002d0', 'C241d3', 'C314d4']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.8070

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 5.2605, 'std_depth': 1.6352287556587712, 'mean_tree_dist': 10.51314459049545, 'std_tree_dist': 1.499942405474526, 'mean_root_purity': 0.49725, 'std_root_purity': 0.02607561121047789}

=== Experiment depth7_extended_w/DecHypNoise_hgd2 | depth 7 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 2.140487, Val Loss: 1.834130, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 2/20, Train Loss: 1.534960, Val Loss: 1.434922, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 3/20, Train Loss: 1.245577, Val Loss: 1.277476, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 4/20, Train Loss: 1.109039, Val Loss: 1.183944, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 5/20, Train Loss: 1.007984, Val Loss: 1.099746, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.920170, Val Loss: 1.033922, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.861304, Val Loss: 1.010982, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.816436, Val Loss: 0.961740, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.757563, Val Loss: 0.899935, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.706756, Val Loss: 0.874165, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.669845, Val Loss: 0.826427, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.636430, Val Loss: 0.802257, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.609986, Val Loss: 0.787343, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.578376, Val Loss: 0.763016, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.552489, Val Loss: 0.737422, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.527970, Val Loss: 0.712610, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.509110, Val Loss: 0.694076, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.491558, Val Loss: 0.671415, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.465778, Val Loss: 0.669262, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.450399, Val Loss: 0.644523, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_hdd=0.0200, lambda_hgd=0.0200, lambda_recon=1.0000
Best validation loss: 0.644523
Saved loss curves to results/plots
Test recall@4: 0.1054

Sample trajectory (euclidean) 1:
  Visit 1: ['C311d4', 'C434d4', 'C441d3', 'C441d4']
  Visit 2: ['C140d3', 'C434d4', 'C441d3', 'C441d4']
  Visit 3: ['C140d3', 'C434d4', 'C441d3', 'C441d4']
  Visit 4: ['C012d4', 'C142d4', 'C302d3', 'C333d4']
  Visit 5: ['C140d3', 'C434d4', 'C441d3', 'C441d4']
  Visit 6: ['C023d4', 'C030d3', 'C132d3', 'C132d4']

Sample trajectory (euclidean) 2:
  Visit 1: ['C434d3', 'C434d4', 'C441d3', 'C441d4']
  Visit 2: ['C203d3', 'C213d3', 'C240d3', 'C341d4']
  Visit 3: ['C140d3', 'C311d3', 'C441d3', 'C441d4']
  Visit 4: ['C024d4', 'C210d3', 'C221d3', 'C403d3']
  Visit 5: ['C033d3', 'C203d3', 'C244d3', 'C341d4']
  Visit 6: ['C320d4', 'C434d4', 'C441d3', 'C441d4']

Sample trajectory (euclidean) 3:
  Visit 1: ['C142d4', 'C434d4', 'C441d3', 'C441d4']
  Visit 2: ['C033d3', 'C203d3', 'C244d3', 'C341d4']
  Visit 3: ['C320d4', 'C434d4', 'C441d3', 'C441d4']
  Visit 4: ['C142d4', 'C434d4', 'C441d3', 'C441d4']
  Visit 5: ['C033d3', 'C203d3', 'C244d3', 'C341d4']
  Visit 6: ['C142d4', 'C434d4', 'C441d3', 'C441d4']
Correlation(tree_dist, euclidean_embedding_dist) = 0.1373

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.481041666666667, 'std_depth': 0.49964045232269, 'mean_tree_dist': 9.448052524177092, 'std_tree_dist': 5.5281386708163165, 'mean_root_purity': 0.59625, 'std_root_purity': 0.14572555541153379}
