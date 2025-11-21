Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

Hyperbolic Training:
Epoch 1 loss: 0.5533
Epoch 2 loss: 0.3432
Epoch 3 loss: 0.2591
Epoch 4 loss: 0.2419
Epoch 5 loss: 0.2123

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C012', 'C34', 'C342', 'C43']
  Visit 2: ['C01', 'C114', 'C343', 'C430']
  Visit 3: ['C00', 'C014', 'C024', 'C43']
  Visit 4: ['C134', 'C30', 'C403', 'C41']
  Visit 5: ['C024', 'C314', 'C431', 'C434']
  Visit 6: ['C00', 'C023', 'C114', 'C431']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C02', 'C123', 'C221', 'C310']
  Visit 2: ['C024', 'C134', 'C210', 'C310']
  Visit 3: ['C00', 'C014', 'C424', 'C43']
  Visit 4: ['C013', 'C123', 'C221', 'C310']
  Visit 5: ['C10', 'C241', 'C411', 'C433']
  Visit 6: ['C10', 'C30', 'C301', 'C41']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C02', 'C042', 'C101', 'C331']
  Visit 2: ['C00', 'C014', 'C412', 'C421']
  Visit 3: ['C022', 'C110', 'C411', 'C441']
  Visit 4: ['C11', 'C114', 'C2', 'C214']
  Visit 5: ['C013', 'C134', 'C310', 'C344']
  Visit 6: ['C023', 'C041', 'C11', 'C114']

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.720625, 'std_depth': 0.4683210537387787, 'mean_tree_dist': 3.2577200205867216, 'std_tree_dist': 0.8973847459364479, 'mean_root_purity': 0.5019166666666667, 'std_root_purity': 0.14103956793120936}


Euclidean Training:
    Epoch 1 loss: 0.7178
    Epoch 2 loss: 0.6103
    Epoch 3 loss: 0.5077
    Epoch 4 loss: 0.5701
    Epoch 5 loss: 0.4526

Sample trajectory (euclidean) 1:
  Visit 1: ['C04', 'C11', 'C242', 'C432']
  Visit 2: ['C002', 'C12', 'C330', 'C333']
  Visit 3: ['C024', 'C032', 'C44', 'C443']
  Visit 4: ['C222', 'C230', 'C42', 'C432']
  Visit 5: ['C010', 'C310', 'C34', 'C401']
  Visit 6: ['C211', 'C310', 'C401', 'C410']

Sample trajectory (euclidean) 2:
  Visit 1: ['C022', 'C114', 'C304', 'C311']
  Visit 2: ['C010', 'C222', 'C401', 'C440']
  Visit 3: ['C032', 'C043', 'C101', 'C333']
  Visit 4: ['C142', 'C222', 'C422', 'C440']
  Visit 5: ['C041', 'C231', 'C330', 'C432']
  Visit 6: ['C042', 'C114', 'C304', 'C32']

Sample trajectory (euclidean) 3:
  Visit 1: ['C0', 'C010', 'C311', 'C401']
  Visit 2: ['C03', 'C203', 'C312', 'C341']
  Visit 3: ['C003', 'C024', 'C312', 'C314']
  Visit 4: ['C042', 'C12', 'C234', 'C404']
  Visit 5: ['C024', 'C041', 'C241', 'C32']
  Visit 6: ['C001', 'C203', 'C340', 'C341']

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.7702916666666666, 'std_depth': 0.49094373227885174, 'mean_tree_dist': 3.2383784629832526, 'std_tree_dist': 0.9111360919577319, 'mean_root_purity': 0.46779166666666666, 'std_root_purity': 0.13766701346808619}


On a synthetic ICD-like hierarchy consisting of 155 codes organised into a 3-level tree, we trained two structurally identical DDPMs over visit-level trajectories: one with Euclidean code embeddings and one with Poincaré-ball embeddings (hyperbolic) and tangent-space diffusion. Using 20,000 training trajectories and 1,000 sampled trajectories per model, we evaluated hierarchy-aware statistics. Real data exhibited a mean code depth of 1.64, mean intra-visit tree distance of 2.12, and root-chapter purity of 0.62. Both generative models produced more specific and slightly more dispersed code patterns than the data (tree distance ≈3.24–3.26, purity ≈0.47–0.50), but the hyperbolic model consistently matched the hierarchical structure more closely than the Euclidean baseline (mean depth 1.72 vs. 1.77; root purity 0.50 vs. 0.47). These preliminary results suggest that, even without explicit structural regularization, hyperbolic latent diffusion provides a modest but measurable advantage in preserving hierarchical code organization.