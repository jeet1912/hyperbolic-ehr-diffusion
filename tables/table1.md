Hyperbolic :
  Epoch 1 loss: 0.5533
  Epoch 2 loss: 0.3432
  Epoch 3 loss: 0.2591
  Epoch 4 loss: 0.2419
  Epoch 5 loss: 0.2123

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C012', 'C34', 'C342', 'C43']
  Visit 2: ['C01', 'C304', 'C343', 'C430']
  Visit 3: ['C014', 'C024', 'C20', 'C43']
  Visit 4: ['C103', 'C134', 'C403', 'C41']
  Visit 5: ['C024', 'C221', 'C431', 'C434']
  Visit 6: ['C00', 'C023', 'C114', 'C431']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C02', 'C221', 'C310', 'C331']
  Visit 2: ['C024', 'C134', 'C210', 'C234']
  Visit 3: ['C00', 'C014', 'C424', 'C43']
  Visit 4: ['C013', 'C123', 'C221', 'C310']
  Visit 5: ['C10', 'C241', 'C321', 'C411']
  Visit 6: ['C000', 'C10', 'C301', 'C41']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C02', 'C042', 'C101', 'C331']
  Visit 2: ['C00', 'C014', 'C122', 'C44']
  Visit 3: ['C022', 'C110', 'C21', 'C411']
  Visit 4: ['C102', 'C114', 'C2', 'C314']
  Visit 5: ['C013', 'C123', 'C134', 'C310']
  Visit 6: ['C023', 'C11', 'C114', 'C24']

Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}
Synthetic (hyperbolic) stats: {'mean_depth': 1.7083333333333333, 'std_depth': 0.4841229182759271, 'mean_tree_dist': 3.125, 'std_tree_dist': 0.9921567416492215, 'mean_root_purity': 0.4444444444444444, 'std_root_purity': 0.10393492741038726}


Euclidean : 
  Epoch 1 loss: 0.7404
  Epoch 2 loss: 0.5522
  Epoch 3 loss: 0.4873
  Epoch 4 loss: 0.4475
  Epoch 5 loss: 0.5653

Sample trajectory (euclidean) 1:
  Visit 1: ['C00', 'C000', 'C1', 'C234']
  Visit 2: ['C303', 'C304', 'C423', 'C440']
  Visit 3: ['C000', 'C101', 'C2', 'C234']
  Visit 4: ['C24', 'C244', 'C304', 'C423']
  Visit 5: ['C024', 'C1', 'C421', 'C424']
  Visit 6: ['C112', 'C244', 'C310', 'C340']

Sample trajectory (euclidean) 2:
  Visit 1: ['C044', 'C300', 'C333', 'C422']
  Visit 2: ['C034', 'C24', 'C244', 'C304']
  Visit 3: ['C013', 'C041', 'C21', 'C423']
  Visit 4: ['C122', 'C13', 'C310', 'C430']
  Visit 5: ['C112', 'C13', 'C333', 'C424']
  Visit 6: ['C22', 'C303', 'C423', 'C440']

Sample trajectory (euclidean) 3:
  Visit 1: ['C000', 'C041', 'C102', 'C2']
  Visit 2: ['C000', 'C023', 'C11', 'C230']
  Visit 3: ['C13', 'C213', 'C310', 'C333']
  Visit 4: ['C00', 'C1', 'C134', 'C300']
  Visit 5: ['C24', 'C244', 'C304', 'C310']
  Visit 6: ['C13', 'C211', 'C231', 'C244']

Real stats: {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}
Synthetic (euclidean) stats: {'mean_depth': 1.6944444444444444, 'std_depth': 0.5925202502139316, 'mean_tree_dist': 3.0, 'std_tree_dist': 1.2060453783110545, 'mean_root_purity': 0.5138888888888888, 'std_root_purity': 0.0572653559113564}
