
Real stats (depth2_base_w/Dec, max_depth=2): {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Experiment depth2_base_w/Dec | depth 2 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.420406, Val Loss: 1.115433, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.818002, Val Loss: 0.747522, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.528572, Val Loss: 0.589247, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.412958, Val Loss: 0.523214, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.351586, Val Loss: 0.495023, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.330843, Val Loss: 0.487745, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.320096, Val Loss: 0.482589, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.313115, Val Loss: 0.481770, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.308652, Val Loss: 0.482867, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.304350, Val Loss: 0.484257, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.299351, Val Loss: 0.482692, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.274030, Val Loss: 0.443411, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.244769, Val Loss: 0.439425, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.239676, Val Loss: 0.434053, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.235388, Val Loss: 0.435170, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.232002, Val Loss: 0.438892, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.229020, Val Loss: 0.432752, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.225859, Val Loss: 0.431862, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.223412, Val Loss: 0.434686, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.221387, Val Loss: 0.437002, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Best validation loss: 0.431862
Saved loss curves to results/plots
Test recall@4: 0.0493

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C020', 'C104', 'C3', 'C410']
  Visit 2: ['C01', 'C043', 'C402', 'C413']
  Visit 3: ['C00', 'C10', 'C23', 'C32']
  Visit 4: ['C00', 'C10', 'C31', 'C40']
  Visit 5: ['C004', 'C020', 'C111', 'C402']
  Visit 6: ['C0', 'C043', 'C402', 'C421']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C00', 'C01', 'C10', 'C24']
  Visit 2: ['C02', 'C11', 'C30', 'C44']
  Visit 3: ['C00', 'C01', 'C402', 'C413']
  Visit 4: ['C020', 'C211', 'C3', 'C410']
  Visit 5: ['C021', 'C143', 'C302', 'C410']
  Visit 6: ['C0', 'C241', 'C3', 'C4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C03', 'C23', 'C30', 'C42']
  Visit 2: ['C043', 'C102', 'C402', 'C413']
  Visit 3: ['C0', 'C022', 'C4', 'C443']
  Visit 4: ['C020', 'C021', 'C3', 'C410']
  Visit 5: ['C043', 'C241', 'C402', 'C413']
  Visit 6: ['C00', 'C01', 'C102', 'C413']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.0463

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.3694583333333334, 'std_depth': 0.6919481242096276, 'mean_tree_dist': 2.658041697691735, 'std_tree_dist': 0.9128868276723565, 'mean_root_purity': 0.4290833333333333, 'std_root_purity': 0.13644288080446784}

=== Experiment depth2_base_w/Dec | depth 2 | euclidean | regularization=off ===
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
  Visit 1: ['C03', 'C143', 'C340', 'C442']
  Visit 2: ['C013', 'C134', 'C30', 'C301']
  Visit 3: ['C014', 'C02', 'C30', 'C402']
  Visit 4: ['C112', 'C134', 'C32', 'C402']
  Visit 5: ['C321', 'C33', 'C34', 'C340']
  Visit 6: ['C01', 'C014', 'C040', 'C402']

Sample trajectory (euclidean) 2:
  Visit 1: ['C10', 'C131', 'C301', 'C312']
  Visit 2: ['C34', 'C344', 'C433', 'C442']
  Visit 3: ['C211', 'C310', 'C313', 'C402']
  Visit 4: ['C231', 'C33', 'C330', 'C340']
  Visit 5: ['C10', 'C144', 'C33', 'C330']
  Visit 6: ['C00', 'C010', 'C31', 'C40']

Sample trajectory (euclidean) 3:
  Visit 1: ['C224', 'C23', 'C330', 'C411']
  Visit 2: ['C01', 'C31', 'C310', 'C402']
  Visit 3: ['C11', 'C23', 'C400', 'C442']
  Visit 4: ['C011', 'C144', 'C224', 'C23']
  Visit 5: ['C23', 'C301', 'C433', 'C442']
  Visit 6: ['C204', 'C413', 'C44', 'C442']
Correlation(tree_dist, euclidean_embedding_dist) = 0.0483

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.631375, 'std_depth': 0.48424230440452015, 'mean_tree_dist': 2.6212617779598526, 'std_tree_dist': 1.165768145120876, 'mean_root_purity': 0.5488333333333333, 'std_root_purity': 0.16479372628295724}

=== Experiment depth2_base_w/Dec | depth 2 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.462018, Val Loss: 1.153012, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.852555, Val Loss: 0.783507, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.578059, Val Loss: 0.637432, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.479055, Val Loss: 0.594632, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.416334, Val Loss: 0.517424, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.340931, Val Loss: 0.485708, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.289025, Val Loss: 0.449283, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.266206, Val Loss: 0.442966, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.259009, Val Loss: 0.441952, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.254132, Val Loss: 0.438347, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.250429, Val Loss: 0.441955, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.246596, Val Loss: 0.439199, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.243063, Val Loss: 0.440750, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.239198, Val Loss: 0.437931, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.237778, Val Loss: 0.441545, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.235192, Val Loss: 0.442039, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.232582, Val Loss: 0.440032, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.230367, Val Loss: 0.439030, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.227990, Val Loss: 0.443246, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.227005, Val Loss: 0.442023, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Best validation loss: 0.437931
Saved loss curves to results/plots
Test recall@4: 0.0488

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C04', 'C21', 'C30', 'C42']
  Visit 2: ['C020', 'C123', 'C201', 'C313']
  Visit 3: ['C04', 'C21', 'C30', 'C42']
  Visit 4: ['C00', 'C14', 'C244', 'C313']
  Visit 5: ['C03', 'C04', 'C30', 'C42']
  Visit 6: ['C34', 'C404', 'C41', 'C43']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C020', 'C123', 'C222', 'C313']
  Visit 2: ['C03', 'C04', 'C42', 'C43']
  Visit 3: ['C020', 'C123', 'C201', 'C41']
  Visit 4: ['C03', 'C04', 'C30', 'C42']
  Visit 5: ['C132', 'C14', 'C22', 'C42']
  Visit 6: ['C14', 'C22', 'C33', 'C34']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C04', 'C21', 'C30', 'C42']
  Visit 2: ['C03', 'C04', 'C30', 'C42']
  Visit 3: ['C00', 'C14', 'C41', 'C42']
  Visit 4: ['C041', 'C1', 'C314', 'C404']
  Visit 5: ['C020', 'C123', 'C201', 'C313']
  Visit 6: ['C03', 'C14', 'C22', 'C30']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9879

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.3542083333333332, 'std_depth': 0.5106317557012642, 'mean_tree_dist': 2.3730242360379346, 'std_tree_dist': 0.8250231790610231, 'mean_root_purity': 0.38979166666666665, 'std_root_purity': 0.13945563188300747}

=== Experiment depth2_base_w/Dec | depth 2 | euclidean | regularization=on ===
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
  Visit 1: ['C10', 'C100', 'C41', 'C411']
  Visit 2: ['C211', 'C224', 'C402', 'C42']
  Visit 3: ['C004', 'C030', 'C123', 'C413']
  Visit 4: ['C021', 'C023', 'C14', 'C31']
  Visit 5: ['C04', 'C22', 'C224', 'C402']
  Visit 6: ['C04', 'C143', 'C211', 'C402']

Sample trajectory (euclidean) 2:
  Visit 1: ['C110', 'C211', 'C224', 'C402']
  Visit 2: ['C03', 'C032', 'C11', 'C244']
  Visit 3: ['C10', 'C11', 'C114', 'C443']
  Visit 4: ['C01', 'C11', 'C111', 'C120']
  Visit 5: ['C024', 'C23', 'C231', 'C32']
  Visit 6: ['C114', 'C13', 'C422', 'C423']

Sample trajectory (euclidean) 3:
  Visit 1: ['C20', 'C200', 'C31', 'C312']
  Visit 2: ['C013', 'C211', 'C22', 'C223']
  Visit 3: ['C01', 'C020', 'C12', 'C301']
  Visit 4: ['C030', 'C10', 'C100', 'C344']
  Visit 5: ['C01', 'C20', 'C32', 'C410']
  Visit 6: ['C140', 'C221', 'C314', 'C422']
Correlation(tree_dist, euclidean_embedding_dist) = 0.5077

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.6311666666666667, 'std_depth': 0.5165223185454386, 'mean_tree_dist': 2.4414581066376497, 'std_tree_dist': 1.181283327773499, 'mean_root_purity': 0.53125, 'std_root_purity': 0.15904172670507993}

Real stats (depth7_extended_w/Dec, max_depth=7): {'mean_depth': 5.3797976334479465, 'std_depth': 1.7294582012361523, 'mean_tree_dist': 5.7591450057100175, 'std_tree_dist': 4.756650684766557, 'mean_root_purity': 0.6289569269083962, 'std_root_purity': 0.20468844637974468}

=== Experiment depth7_extended_w/Dec | depth 7 | hyperbolic | regularization=off ===
Epoch 1/20, Train Loss: 1.430760, Val Loss: 1.111367, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.813900, Val Loss: 0.730096, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.511168, Val Loss: 0.556867, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.373589, Val Loss: 0.476626, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.293212, Val Loss: 0.423022, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.257887, Val Loss: 0.411590, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.242609, Val Loss: 0.403563, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.232968, Val Loss: 0.398285, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.217473, Val Loss: 0.366989, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.178011, Val Loss: 0.354202, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.167469, Val Loss: 0.351539, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.162781, Val Loss: 0.348699, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.158882, Val Loss: 0.348899, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.155746, Val Loss: 0.349208, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.152073, Val Loss: 0.348886, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.150049, Val Loss: 0.349974, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.147007, Val Loss: 0.349975, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.146111, Val Loss: 0.351437, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.144503, Val Loss: 0.349446, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.143605, Val Loss: 0.346782, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000, lambda_recon=1.0000
Best validation loss: 0.346782
Saved loss curves to results/plots
Test recall@4: 0.0094

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C121d3', 'C131d4', 'C334d3', 'C414d4']
  Visit 2: ['C010d4', 'C043d3', 'C111d4', 'C324d4']
  Visit 3: ['C000', 'C004', 'C231d0', 'C302d2']
  Visit 4: ['C1', 'C401', 'C213d1', 'C224d0']
  Visit 5: ['C121d3', 'C131d4', 'C232d4', 'C414d4']
  Visit 6: ['C004', 'C412', 'C422', 'C230d0']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C121d3', 'C131d4', 'C411d3', 'C414d4']
  Visit 2: ['C000', 'C204', 'C241', 'C314d0']
  Visit 3: ['C000', 'C41', 'C412', 'C134d0']
  Visit 4: ['C114d3', 'C121d3', 'C232d4', 'C420d3']
  Visit 5: ['C020d4', 'C023d3', 'C131d4', 'C414d4']
  Visit 6: ['C004', 'C231d0', 'C314d0', 'C344d0']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C104', 'C022d2', 'C231d0', 'C344d0']
  Visit 2: ['C023d3', 'C213d4', 'C240d3', 'C341d3']
  Visit 3: ['C123', 'C41', 'C303d0', 'C310d3']
  Visit 4: ['C121d3', 'C232d4', 'C334d3', 'C414d4']
  Visit 5: ['C000', 'C004', 'C022d2', 'C231d0']
  Visit 6: ['C004', 'C231d0', 'C314d0', 'C344d0']
Correlation(tree_dist, hyperbolic_embedding_dist) = -0.0415

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 4.6375416666666665, 'std_depth': 2.019323060647773, 'mean_tree_dist': 8.488310969705353, 'std_tree_dist': 4.06970429546887, 'mean_root_purity': 0.49391666666666667, 'std_root_purity': 0.13793292955474973}

=== Experiment depth7_extended_w/Dec | depth 7 | euclidean | regularization=off ===
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
  Visit 1: ['C014d3', 'C014d4', 'C334d3', 'C334d4']
  Visit 2: ['C021d4', 'C334d3', 'C334d4', 'C432d3']
  Visit 3: ['C002d3', 'C002d4', 'C042d3', 'C130d4']
  Visit 4: ['C002d3', 'C002d4', 'C114d4', 'C344d3']
  Visit 5: ['C112d3', 'C310d3', 'C310d4', 'C342d4']
  Visit 6: ['C021d4', 'C334d3', 'C334d4', 'C432d3']

Sample trajectory (euclidean) 2:
  Visit 1: ['C002d3', 'C002d4', 'C114d4', 'C304d3']
  Visit 2: ['C112d3', 'C334d3', 'C334d4', 'C432d3']
  Visit 3: ['C002d4', 'C114d4', 'C130d3', 'C232d4']
  Visit 4: ['C021d4', 'C334d3', 'C334d4', 'C432d3']
  Visit 5: ['C121d4', 'C314d4', 'C332d3', 'C332d4']
  Visit 6: ['C041d3', 'C041d4', 'C114d3', 'C114d4']

Sample trajectory (euclidean) 3:
  Visit 1: ['C014d4', 'C204d4', 'C331d3', 'C401d3']
  Visit 2: ['C112d3', 'C112d4', 'C334d3', 'C334d4']
  Visit 3: ['C002d3', 'C002d4', 'C304d3', 'C423d4']
  Visit 4: ['C002d3', 'C002d4', 'C102d4', 'C304d3']
  Visit 5: ['C002d3', 'C002d4', 'C324d4', 'C423d4']
  Visit 6: ['C004d3', 'C021d4', 'C213d3', 'C334d3']
Correlation(tree_dist, euclidean_embedding_dist) = -0.0843

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.48525, 'std_depth': 0.49978239014595144, 'mean_tree_dist': 6.4741301963911875, 'std_tree_dist': 5.905505160937394, 'mean_root_purity': 0.54575, 'std_root_purity': 0.138182623726719}

=== Experiment depth7_extended_w/Dec | depth 7 | hyperbolic | regularization=on ===
Epoch 1/20, Train Loss: 1.459589, Val Loss: 1.133107, lambda_tree_eff=0.0017, lambda_radius_eff=0.0005, lambda_recon=1.0000
Epoch 2/20, Train Loss: 0.814512, Val Loss: 0.725051, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010, lambda_recon=1.0000
Epoch 3/20, Train Loss: 0.511978, Val Loss: 0.548245, lambda_tree_eff=0.0050, lambda_radius_eff=0.0015, lambda_recon=1.0000
Epoch 4/20, Train Loss: 0.360520, Val Loss: 0.450352, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020, lambda_recon=1.0000
Epoch 5/20, Train Loss: 0.272600, Val Loss: 0.395281, lambda_tree_eff=0.0083, lambda_radius_eff=0.0025, lambda_recon=1.0000
Epoch 6/20, Train Loss: 0.223779, Val Loss: 0.371771, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 7/20, Train Loss: 0.204515, Val Loss: 0.364138, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 8/20, Train Loss: 0.193891, Val Loss: 0.359699, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 9/20, Train Loss: 0.185610, Val Loss: 0.361625, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 10/20, Train Loss: 0.179999, Val Loss: 0.360226, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 11/20, Train Loss: 0.175075, Val Loss: 0.358029, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 12/20, Train Loss: 0.171198, Val Loss: 0.354042, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 13/20, Train Loss: 0.167650, Val Loss: 0.355874, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 14/20, Train Loss: 0.163510, Val Loss: 0.351564, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 15/20, Train Loss: 0.160708, Val Loss: 0.358605, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 16/20, Train Loss: 0.158743, Val Loss: 0.355445, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 17/20, Train Loss: 0.155536, Val Loss: 0.353068, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 18/20, Train Loss: 0.152092, Val Loss: 0.354853, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 19/20, Train Loss: 0.150857, Val Loss: 0.356098, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Epoch 20/20, Train Loss: 0.149635, Val Loss: 0.354824, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030, lambda_recon=1.0000
Best validation loss: 0.351564
Saved loss curves to results/plots
Test recall@4: 0.0106

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C331', 'C002d0', 'C032d1', 'C122d1']
  Visit 2: ['C331', 'C022d1', 'C122d1', 'C403d1']
  Visit 3: ['C012d4', 'C114d3', 'C120d3', 'C204d4']
  Visit 4: ['C422', 'C022d1', 'C122d1', 'C122d2']
  Visit 5: ['C331', 'C022d1', 'C122d1', 'C403d1']
  Visit 6: ['C012d4', 'C013d3', 'C204d4', 'C304d3']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C122d1', 'C203d1', 'C320d1', 'C400d0']
  Visit 2: ['C321', 'C331', 'C203d1', 'C403d1']
  Visit 3: ['C422', 'C022d1', 'C122d1', 'C143d1']
  Visit 4: ['C331', 'C122d1', 'C304d4', 'C324d3']
  Visit 5: ['C122d1', 'C122d2', 'C131d2', 'C320d1']
  Visit 6: ['C120d3', 'C204d4', 'C402d4', 'C423d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C022d4', 'C103d3', 'C322d4', 'C444d4']
  Visit 2: ['C321', 'C331', 'C121d2', 'C122d1']
  Visit 3: ['C114d4', 'C204d4', 'C314d4', 'C402d4']
  Visit 4: ['C331', 'C422', 'C022d1', 'C122d1']
  Visit 5: ['C331', 'C022d1', 'C122d1', 'C403d1']
  Visit 6: ['C331', 'C032d1', 'C122d1', 'C403d1']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9725

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 5.208208333333333, 'std_depth': 1.8587919795565855, 'mean_tree_dist': 9.419073463849584, 'std_tree_dist': 4.187826142776691, 'mean_root_purity': 0.42995833333333333, 'std_root_purity': 0.1530944640754706}

=== Experiment depth7_extended_w/Dec | depth 7 | euclidean | regularization=on ===
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
  Visit 1: ['C030d3', 'C030d4', 'C220d4', 'C440d4']
  Visit 2: ['C012d3', 'C012d4', 'C230d3', 'C230d4']
  Visit 3: ['C224d4', 'C413d3', 'C440d3', 'C440d4']
  Visit 4: ['C102d3', 'C102d4', 'C203d4', 'C304d3']
  Visit 5: ['C312d4', 'C331d4', 'C430d3', 'C430d4']
  Visit 6: ['C230d3', 'C230d4', 'C312d3', 'C312d4']

Sample trajectory (euclidean) 2:
  Visit 1: ['C112d3', 'C220d4', 'C310d3', 'C310d4']
  Visit 2: ['C120d3', 'C234d3', 'C334d3', 'C334d4']
  Visit 3: ['C104d3', 'C104d4', 'C322d3', 'C322d4']
  Visit 4: ['C011d3', 'C011d4', 'C312d3', 'C312d4']
  Visit 5: ['C230d3', 'C232d4', 'C312d3', 'C312d4']
  Visit 6: ['C120d3', 'C144d4', 'C234d3', 'C334d3']

Sample trajectory (euclidean) 3:
  Visit 1: ['C120d3', 'C120d4', 'C234d3', 'C334d3']
  Visit 2: ['C233d3', 'C233d4', 'C421d3', 'C421d4']
  Visit 3: ['C014d4', 'C233d4', 'C304d4', 'C324d4']
  Visit 4: ['C122d3', 'C230d3', 'C230d4', 'C231d4']
  Visit 5: ['C004d4', 'C120d3', 'C120d4', 'C233d4']
  Visit 6: ['C032d4', 'C113d3', 'C223d3', 'C331d3']
Correlation(tree_dist, euclidean_embedding_dist) = 0.4818

Synthetic (euclidean) stats (N=1000): {'mean_depth': 6.464666666666667, 'std_depth': 0.4987499930381509, 'mean_tree_dist': 6.646397497311566, 'std_tree_dist': 5.8206486833263495, 'mean_root_purity': 0.5469583333333333, 'std_root_purity': 0.13491169308312587}
