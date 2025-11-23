
Real stats (depth2_base_w-DecHypNoise, max_depth=2): {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Experiment depth2_base_w-DecHypNoise | depth 2 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.427652, Val Loss: 1.120670, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.823479, Val Loss: 0.750248, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.531852, Val Loss: 0.590492, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.417815, Val Loss: 0.528251, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.354095, Val Loss: 0.495525, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.331733, Val Loss: 0.488008, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.320655, Val Loss: 0.482749, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.313527, Val Loss: 0.481798, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.308920, Val Loss: 0.482923, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.304356, Val Loss: 0.483869, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.294482, Val Loss: 0.458599, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.253800, Val Loss: 0.440864, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.242862, Val Loss: 0.439169, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.238956, Val Loss: 0.433903, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.234942, Val Loss: 0.435085, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.231663, Val Loss: 0.438831, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.228746, Val Loss: 0.432703, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.225572, Val Loss: 0.431846, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.223154, Val Loss: 0.434699, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.221155, Val Loss: 0.436990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
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

=== Experiment depth2_base_w-DecHypNoise | depth 2 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.474472, Val Loss: 1.170265, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.944327, Val Loss: 0.919415, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.762500, Val Loss: 0.828219, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.669603, Val Loss: 0.779058, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.625948, Val Loss: 0.754990, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.591662, Val Loss: 0.727655, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.563086, Val Loss: 0.715106, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.553368, Val Loss: 0.705919, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.529209, Val Loss: 0.687594, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.502110, Val Loss: 0.682125, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.498790, Val Loss: 0.676293, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.489174, Val Loss: 0.667226, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.477639, Val Loss: 0.674488, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.475990, Val Loss: 0.675020, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.468953, Val Loss: 0.657662, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.461091, Val Loss: 0.647391, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.461387, Val Loss: 0.658540, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.461810, Val Loss: 0.656768, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.453207, Val Loss: 0.647204, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.452916, Val Loss: 0.654236, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
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

=== Experiment depth2_base_w-DecHypNoise | depth 2 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.471839, Val Loss: 1.160602, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.860726, Val Loss: 0.789452, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.584265, Val Loss: 0.640176, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.482702, Val Loss: 0.597767, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.430341, Val Loss: 0.538554, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.353189, Val Loss: 0.487853, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.290472, Val Loss: 0.450598, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.267336, Val Loss: 0.444274, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.260020, Val Loss: 0.443002, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.255192, Val Loss: 0.439399, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.251489, Val Loss: 0.443261, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.247829, Val Loss: 0.440770, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.244309, Val Loss: 0.442138, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.240401, Val Loss: 0.439326, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.239345, Val Loss: 0.443343, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.236985, Val Loss: 0.444020, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.234315, Val Loss: 0.442510, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.232469, Val Loss: 0.441729, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.230040, Val Loss: 0.446271, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.229237, Val Loss: 0.444918, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Best validation loss: 0.439326
Saved loss curves to results/plots
Test recall@4: 0.0514

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C01', 'C14', 'C20', 'C41']
  Visit 2: ['C01', 'C14', 'C20', 'C41']
  Visit 3: ['C01', 'C14', 'C20', 'C41']
  Visit 4: ['C01', 'C14', 'C20', 'C41']
  Visit 5: ['C01', 'C14', 'C20', 'C41']
  Visit 6: ['C01', 'C14', 'C20', 'C41']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C03', 'C04', 'C30', 'C42']
  Visit 2: ['C03', 'C04', 'C30', 'C42']
  Visit 3: ['C01', 'C14', 'C20', 'C41']
  Visit 4: ['C03', 'C04', 'C30', 'C42']
  Visit 5: ['C03', 'C04', 'C30', 'C42']
  Visit 6: ['C01', 'C14', 'C20', 'C41']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C03', 'C04', 'C30', 'C42']
  Visit 2: ['C03', 'C04', 'C30', 'C42']
  Visit 3: ['C01', 'C14', 'C20', 'C41']
  Visit 4: ['C03', 'C04', 'C30', 'C42']
  Visit 5: ['C01', 'C14', 'C20', 'C41']
  Visit 6: ['C03', 'C04', 'C30', 'C42']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9879

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.0, 'std_depth': 0.0, 'mean_tree_dist': 2.0, 'std_tree_dist': 0.0, 'mean_root_purity': 0.376125, 'std_root_purity': 0.12499493739748024}

=== Experiment depth2_base_w-DecHypNoise | depth 2 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 1.518372, Val Loss: 1.198585, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.964447, Val Loss: 0.930130, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.768609, Val Loss: 0.828256, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.686010, Val Loss: 0.789062, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.647194, Val Loss: 0.780441, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.630631, Val Loss: 0.779954, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.613124, Val Loss: 0.763886, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.603475, Val Loss: 0.750512, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.579728, Val Loss: 0.736511, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.557208, Val Loss: 0.715678, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.549445, Val Loss: 0.718298, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.529633, Val Loss: 0.697278, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.505477, Val Loss: 0.692288, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.505152, Val Loss: 0.673880, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.494663, Val Loss: 0.684874, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.485185, Val Loss: 0.677147, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.484470, Val Loss: 0.652734, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.475954, Val Loss: 0.671516, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.473294, Val Loss: 0.665658, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.471953, Val Loss: 0.666103, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Best validation loss: 0.652734
Saved loss curves to results/plots
Test recall@4: 0.2237

Sample trajectory (euclidean) 1:
  Visit 1: ['C100', 'C30', 'C301', 'C31']
  Visit 2: ['C211', 'C22', 'C224', 'C402']
  Visit 3: ['C004', 'C030', 'C123', 'C413']
  Visit 4: ['C021', 'C023', 'C123', 'C31']
  Visit 5: ['C211', 'C22', 'C221', 'C402']
  Visit 6: ['C211', 'C22', 'C224', 'C402']

Sample trajectory (euclidean) 2:
  Visit 1: ['C211', 'C22', 'C224', 'C402']
  Visit 2: ['C00', 'C020', 'C120', 'C31']
  Visit 3: ['C04', 'C104', 'C110', 'C22']
  Visit 4: ['C030', 'C123', 'C2', 'C31']
  Visit 5: ['C221', 'C231', 'C32', 'C402']
  Visit 6: ['C110', 'C114', 'C402', 'C423']

Sample trajectory (euclidean) 3:
  Visit 1: ['C123', 'C20', 'C201', 'C31']
  Visit 2: ['C013', 'C211', 'C22', 'C224']
  Visit 3: ['C01', 'C020', 'C12', 'C301']
  Visit 4: ['C030', 'C100', 'C123', 'C31']
  Visit 5: ['C123', 'C20', 'C202', 'C31']
  Visit 6: ['C140', 'C221', 'C402', 'C422']
Correlation(tree_dist, euclidean_embedding_dist) = 0.5077

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.6279166666666667, 'std_depth': 0.5250910965939359, 'mean_tree_dist': 2.4264346764346763, 'std_tree_dist': 1.2214926895503617, 'mean_root_purity': 0.5520833333333334, 'std_root_purity': 0.16309453206312247}

Real stats (depth7_extended_wDecHypNoise, max_depth=7): {'mean_depth': 5.3797976334479465, 'std_depth': 1.7294582012361523, 'mean_tree_dist': 5.7591450057100175, 'std_tree_dist': 4.756650684766557, 'mean_root_purity': 0.6289569269083962, 'std_root_purity': 0.20468844637974468}

=== Experiment depth7_extended_wDecHypNoise | depth 7 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.438177, Val Loss: 1.118426, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.822551, Val Loss: 0.735579, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.516850, Val Loss: 0.561179, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.377913, Val Loss: 0.480567, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.297079, Val Loss: 0.424235, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.258993, Val Loss: 0.411938, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.243285, Val Loss: 0.403790, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.233733, Val Loss: 0.399157, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.224496, Val Loss: 0.384016, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.184779, Val Loss: 0.355065, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.168296, Val Loss: 0.351672, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.163147, Val Loss: 0.348764, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.159075, Val Loss: 0.348925, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.155889, Val Loss: 0.349568, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.152108, Val Loss: 0.348900, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.150055, Val Loss: 0.349976, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.146969, Val Loss: 0.349954, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.146054, Val Loss: 0.351433, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.144442, Val Loss: 0.349417, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.143528, Val Loss: 0.346793, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Best validation loss: 0.346793
Saved loss curves to results/plots
Test recall@4: 0.0098

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 2: ['C000', 'C113d3', 'C223d3', 'C400d0']
  Visit 3: ['C000', 'C113d3', 'C223d3', 'C400d0']
  Visit 4: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 5: ['C000', 'C113d3', 'C223d3', 'C400d0']
  Visit 6: ['C000', 'C113d3', 'C223d3', 'C400d0']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 2: ['C000', 'C113d3', 'C223d3', 'C400d0']
  Visit 3: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 4: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 5: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 6: ['C000', 'C113d3', 'C223d3', 'C400d0']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 2: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
  Visit 3: ['C000', 'C113d3', 'C223d3', 'C400d0']
  Visit 4: ['C000', 'C113d3', 'C223d3', 'C400d0']
  Visit 5: ['C000', 'C113d3', 'C223d3', 'C400d0']
  Visit 6: ['C121d3', 'C411d3', 'C420d3', 'C441d4']
Correlation(tree_dist, hyperbolic_embedding_dist) = -0.0415

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 5.261375, 'std_depth': 1.6341077410547322, 'mean_tree_dist': 12.655878467635404, 'std_tree_dist': 0.4750809439744333, 'mean_root_purity': 0.504375, 'std_root_purity': 0.24687974678980854}

=== Experiment depth7_extended_wDecHypNoise | depth 7 | euclidean | regularization=off ===
Epoch 1/20, Train Loss: 1.500497, Val Loss: 1.212608, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.941795, Val Loss: 0.890056, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.702886, Val Loss: 0.762403, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.601558, Val Loss: 0.708888, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.543972, Val Loss: 0.683037, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.518432, Val Loss: 0.681648, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.508848, Val Loss: 0.664360, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.502130, Val Loss: 0.661869, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.472186, Val Loss: 0.636099, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.463133, Val Loss: 0.635940, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.448055, Val Loss: 0.629208, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.446382, Val Loss: 0.625313, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.441872, Val Loss: 0.634846, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.443139, Val Loss: 0.638036, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.437929, Val Loss: 0.618249, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.435912, Val Loss: 0.619855, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.435208, Val Loss: 0.623755, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.428370, Val Loss: 0.612466, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.428033, Val Loss: 0.619466, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.428026, Val Loss: 0.626818, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Best validation loss: 0.612466
Saved loss curves to results/plots
Test recall@4: 0.1416

Sample trajectory (euclidean) 1:
  Visit 1: ['C112d3', 'C334d3', 'C334d4', 'C432d3']
  Visit 2: ['C021d4', 'C334d3', 'C334d4', 'C432d3']
  Visit 3: ['C002d3', 'C002d4', 'C114d4', 'C332d3']
  Visit 4: ['C002d3', 'C002d4', 'C230d3', 'C332d3']
  Visit 5: ['C112d3', 'C334d3', 'C334d4', 'C432d3']
  Visit 6: ['C021d4', 'C334d3', 'C334d4', 'C432d3']

Sample trajectory (euclidean) 2:
  Visit 1: ['C002d3', 'C002d4', 'C102d4', 'C114d4']
  Visit 2: ['C021d4', 'C334d3', 'C334d4', 'C432d3']
  Visit 3: ['C002d3', 'C002d4', 'C114d4', 'C344d3']
  Visit 4: ['C021d4', 'C334d3', 'C334d4', 'C432d3']
  Visit 5: ['C114d4', 'C121d4', 'C332d3', 'C332d4']
  Visit 6: ['C041d3', 'C041d4', 'C114d3', 'C114d4']

Sample trajectory (euclidean) 3:
  Visit 1: ['C014d4', 'C021d4', 'C334d3', 'C334d4']
  Visit 2: ['C112d3', 'C334d3', 'C334d4', 'C432d3']
  Visit 3: ['C002d3', 'C002d4', 'C230d3', 'C332d3']
  Visit 4: ['C002d3', 'C002d4', 'C114d4', 'C130d3']
  Visit 5: ['C002d3', 'C002d4', 'C324d4', 'C423d4']
  Visit 6: ['C004d3', 'C021d4', 'C334d3', 'C334d4']
Correlation(tree_dist, euclidean_embedding_dist) = -0.0843

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.471375, 'std_depth': 0.49917993687146517, 'mean_tree_dist': 5.385145050231403, 'std_tree_dist': 5.733773465551494, 'mean_root_purity': 0.5239583333333333, 'std_root_purity': 0.10810372301893932}

=== Experiment depth7_extended_wDecHypNoise | depth 7 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.465265, Val Loss: 1.137522, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.819561, Val Loss: 0.729055, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.516913, Val Loss: 0.551919, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.365131, Val Loss: 0.452143, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.274911, Val Loss: 0.396785, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.226002, Val Loss: 0.373075, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.206452, Val Loss: 0.365599, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.196114, Val Loss: 0.361286, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.187868, Val Loss: 0.363749, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.182054, Val Loss: 0.362159, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.177333, Val Loss: 0.360356, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.173464, Val Loss: 0.356085, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.170025, Val Loss: 0.358045, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.165963, Val Loss: 0.354010, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.163293, Val Loss: 0.361376, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.161497, Val Loss: 0.358162, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.158575, Val Loss: 0.356063, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.155093, Val Loss: 0.357948, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.154141, Val Loss: 0.358982, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.152709, Val Loss: 0.357725, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Best validation loss: 0.354010
Saved loss curves to results/plots
Test recall@4: 0.0106

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 2: ['C203d3', 'C304d4', 'C324d3', 'C330d4']
  Visit 3: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 4: ['C203d3', 'C304d4', 'C324d3', 'C330d4']
  Visit 5: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 6: ['C203d3', 'C304d4', 'C324d3', 'C330d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 2: ['C203d3', 'C304d4', 'C324d3', 'C330d4']
  Visit 3: ['C203d3', 'C304d4', 'C324d3', 'C330d4']
  Visit 4: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 5: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 6: ['C203d3', 'C304d4', 'C324d3', 'C330d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 2: ['C203d3', 'C304d4', 'C324d3', 'C330d4']
  Visit 3: ['C203d3', 'C304d4', 'C324d3', 'C330d4']
  Visit 4: ['C203d3', 'C304d4', 'C324d3', 'C330d4']
  Visit 5: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
  Visit 6: ['C110d4', 'C114d4', 'C204d4', 'C402d4']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9725

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 6.7491666666666665, 'std_depth': 0.4334927591347083, 'mean_tree_dist': 13.001663893510816, 'std_tree_dist': 0.7065163017956323, 'mean_root_purity': 0.6254166666666666, 'std_root_purity': 0.12499930555362654}

=== Experiment depth7_extended_wDecHypNoise | depth 7 | euclidean | regularization=on ===
Epoch 1/20, Train Loss: 1.529125, Val Loss: 1.239995, lambda_tree_eff=0.0017, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.984008, Val Loss: 0.907263, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.738314, Val Loss: 0.778129, lambda_tree_eff=0.0050, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.637098, Val Loss: 0.740370, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.593386, Val Loss: 0.709507, lambda_tree_eff=0.0083, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.577174, Val Loss: 0.705274, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.556398, Val Loss: 0.683477, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.534283, Val Loss: 0.693107, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.524901, Val Loss: 0.680640, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.518494, Val Loss: 0.680558, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.511560, Val Loss: 0.664361, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.506093, Val Loss: 0.675055, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.492730, Val Loss: 0.652081, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.471127, Val Loss: 0.646709, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.461905, Val Loss: 0.650309, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.465475, Val Loss: 0.630772, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.459733, Val Loss: 0.642562, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.453572, Val Loss: 0.651698, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.453355, Val Loss: 0.656259, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.454126, Val Loss: 0.644338, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000, lambda_recon=1.0000
Best validation loss: 0.630772
Saved loss curves to results/plots
Test recall@4: 0.1903

Sample trajectory (euclidean) 1:
  Visit 1: ['C030d4', 'C231d3', 'C323d3', 'C440d4']
  Visit 2: ['C230d3', 'C230d4', 'C312d3', 'C312d4']
  Visit 3: ['C224d4', 'C234d3', 'C301d3', 'C413d3']
  Visit 4: ['C114d3', 'C114d4', 'C211d3', 'C222d4']
  Visit 5: ['C230d3', 'C312d3', 'C312d4', 'C430d3']
  Visit 6: ['C230d3', 'C232d4', 'C312d3', 'C312d4']

Sample trajectory (euclidean) 2:
  Visit 1: ['C220d4', 'C231d4', 'C310d3', 'C310d4']
  Visit 2: ['C120d3', 'C120d4', 'C234d3', 'C334d3']
  Visit 3: ['C120d3', 'C234d3', 'C234d4', 'C334d3']
  Visit 4: ['C011d3', 'C312d3', 'C312d4', 'C441d4']
  Visit 5: ['C230d3', 'C232d4', 'C312d3', 'C312d4']
  Visit 6: ['C120d3', 'C143d4', 'C234d3', 'C334d3']

Sample trajectory (euclidean) 3:
  Visit 1: ['C120d3', 'C120d4', 'C234d3', 'C334d3']
  Visit 2: ['C230d3', 'C233d3', 'C312d4', 'C421d3']
  Visit 3: ['C212d3', 'C212d4', 'C233d4', 'C412d4']
  Visit 4: ['C230d3', 'C232d4', 'C312d3', 'C312d4']
  Visit 5: ['C120d3', 'C120d4', 'C143d4', 'C334d3']
  Visit 6: ['C032d4', 'C223d3', 'C331d3', 'C331d4']
Correlation(tree_dist, euclidean_embedding_dist) = 0.4818

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.4267916666666665, 'std_depth': 0.49461150404186477, 'mean_tree_dist': 5.95960720793683, 'std_tree_dist': 5.609165149866249, 'mean_root_purity': 0.5315416666666667, 'std_root_purity': 0.11307684671889684}
