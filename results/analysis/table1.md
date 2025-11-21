
Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

Hyperbolic Training
Epoch 1/20, Train Loss: 0.883233, Val Loss: 0.609470
Epoch 2/20, Train Loss: 0.513966, Val Loss: 0.372927
Epoch 3/20, Train Loss: 0.365628, Val Loss: 0.251284
Epoch 4/20, Train Loss: 0.282176, Val Loss: 0.193553
Epoch 5/20, Train Loss: 0.247410, Val Loss: 0.173019
Epoch 6/20, Train Loss: 0.233024, Val Loss: 0.160133
Epoch 7/20, Train Loss: 0.199268, Val Loss: 0.111353
Epoch 8/20, Train Loss: 0.176201, Val Loss: 0.105316
Epoch 9/20, Train Loss: 0.168579, Val Loss: 0.105536
Epoch 10/20, Train Loss: 0.162226, Val Loss: 0.103101
Epoch 11/20, Train Loss: 0.157004, Val Loss: 0.101257
Epoch 12/20, Train Loss: 0.151370, Val Loss: 0.099701
Epoch 13/20, Train Loss: 0.146367, Val Loss: 0.100235
Epoch 14/20, Train Loss: 0.142561, Val Loss: 0.099512
Epoch 15/20, Train Loss: 0.137327, Val Loss: 0.099653
Epoch 16/20, Train Loss: 0.132849, Val Loss: 0.099266
Epoch 17/20, Train Loss: 0.128985, Val Loss: 0.098041
Epoch 18/20, Train Loss: 0.125216, Val Loss: 0.097271
Epoch 19/20, Train Loss: 0.120691, Val Loss: 0.095608
Epoch 20/20, Train Loss: 0.117318, Val Loss: 0.097124

Best validation loss: 0.095608
Test loss: 0.099007

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C013', 'C123', 'C144', 'C30']
  Visit 2: ['C10', 'C112', 'C123', 'C403']
  Visit 3: ['C114', 'C412', 'C43', 'C44']
  Visit 4: ['C144', 'C241', 'C30', 'C321']
  Visit 5: ['C00', 'C114', 'C43', 'C44']
  Visit 6: ['C121', 'C332', 'C403', 'C42']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C00', 'C114', 'C43', 'C44']
  Visit 2: ['C122', 'C322', 'C412', 'C44']
  Visit 3: ['C00', 'C114', 'C421', 'C44']
  Visit 4: ['C014', 'C043', 'C42', 'C431']
  Visit 5: ['C03', 'C10', 'C30', 'C301']
  Visit 6: ['C223', 'C342', 'C412', 'C444']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C1', 'C132', 'C22', 'C314']
  Visit 2: ['C014', 'C043', 'C234', 'C431']
  Visit 3: ['C102', 'C214', 'C43', 'C44']
  Visit 4: ['C00', 'C114', 'C421', 'C44']
  Visit 5: ['C123', 'C144', 'C331', 'C41']
  Visit 6: ['C10', 'C123', 'C133', 'C30']

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.6587083333333332, 'std_depth': 0.4897907698843343, 'mean_tree_dist': 3.3919328514805316, 'std_tree_dist': 0.8137545825094897, 'mean_root_purity': 0.5281666666666667, 'std_root_purity': 0.14789626619883128}


Euclidean Training
Epoch 1/20, Train Loss: 0.903884, Val Loss: 0.704113
Epoch 2/20, Train Loss: 0.644800, Val Loss: 0.558521
Epoch 3/20, Train Loss: 0.554957, Val Loss: 0.488182
Epoch 4/20, Train Loss: 0.506796, Val Loss: 0.447437
Epoch 5/20, Train Loss: 0.483163, Val Loss: 0.431772
Epoch 6/20, Train Loss: 0.456827, Val Loss: 0.405081
Epoch 7/20, Train Loss: 0.446182, Val Loss: 0.397027
Epoch 8/20, Train Loss: 0.434274, Val Loss: 0.402800
Epoch 9/20, Train Loss: 0.427188, Val Loss: 0.377822
Epoch 10/20, Train Loss: 0.425925, Val Loss: 0.385557
Epoch 11/20, Train Loss: 0.424724, Val Loss: 0.390947
Epoch 12/20, Train Loss: 0.420160, Val Loss: 0.382441
Epoch 13/20, Train Loss: 0.416689, Val Loss: 0.390343
Epoch 14/20, Train Loss: 0.413130, Val Loss: 0.387129
Epoch 15/20, Train Loss: 0.407964, Val Loss: 0.370545
Epoch 16/20, Train Loss: 0.408260, Val Loss: 0.379202
Epoch 17/20, Train Loss: 0.407189, Val Loss: 0.388355
Epoch 18/20, Train Loss: 0.407513, Val Loss: 0.379266
Epoch 19/20, Train Loss: 0.407863, Val Loss: 0.375980
Epoch 20/20, Train Loss: 0.406894, Val Loss: 0.375313

Best validation loss: 0.370545
Test loss: 0.379503

Sample trajectory (euclidean) 1:
  Visit 1: ['C202', 'C311', 'C321', 'C412']
  Visit 2: ['C04', 'C200', 'C31', 'C344']
  Visit 3: ['C033', 'C341', 'C344', 'C42']
  Visit 4: ['C024', 'C124', 'C24', 'C333']
  Visit 5: ['C200', 'C302', 'C31', 'C41']
  Visit 6: ['C200', 'C211', 'C300', 'C340']

Sample trajectory (euclidean) 2:
  Visit 1: ['C030', 'C040', 'C200', 'C412']
  Visit 2: ['C243', 'C321', 'C342', 'C412']
  Visit 3: ['C034', 'C311', 'C321', 'C412']
  Visit 4: ['C034', 'C100', 'C124', 'C311']
  Visit 5: ['C12', 'C140', 'C431', 'C442']
  Visit 6: ['C030', 'C034', 'C311', 'C321']

Sample trajectory (euclidean) 3:
  Visit 1: ['C124', 'C311', 'C321', 'C412']
  Visit 2: ['C200', 'C211', 'C302', 'C41']
  Visit 3: ['C00', 'C142', 'C200', 'C211']
  Visit 4: ['C014', 'C140', 'C211', 'C442']
  Visit 5: ['C033', 'C34', 'C404', 'C44']
  Visit 6: ['C014', 'C142', 'C211', 'C401']

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.8326666666666667, 'std_depth': 0.3963368376632292, 'mean_tree_dist': 3.5144363545066533, 'std_tree_dist': 0.8266706484071835, 'mean_root_purity': 0.5090833333333333, 'std_root_purity': 0.14684456540467847}


We evaluated the ability of Euclidean vs. hyperbolic latent diffusion models to preserve hierarchical structure in synthetic ICD-style trajectories. Real data exhibited a mean code depth of 1.64, an intra-visit tree distance of 2.12, and a chapter-level purity of 0.62. Both generative models produced more dispersed code sets than the real distribution, but the hyperbolic model consistently remained closer to the ground truth hierarchy. Hyperbolic diffusion generated trajectories with a mean depth of 1.66 (vs. 1.83 for Euclidean), a tree distance of 3.39 (vs. 3.51), and a purity of 0.53 (vs. 0.51). These results demonstrate that, even without any explicit hierarchical regularization or domain-specific constraints, hyperbolic latent diffusion better preserves the underlying medical taxonomy than its Euclidean counterpart, supporting the proposed use of hyperbolic geometry for EHR generation.