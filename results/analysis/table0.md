
Real stats (depth2_base, max_depth=2): {'mean_depth': 1.6439176670642877, 'std_depth': 0.5137982104125317, 'mean_tree_dist': 2.1205701725560644, 'std_tree_dist': 1.2453175481436083, 'mean_root_purity': 0.6241599458637942, 'std_root_purity': 0.2040143257571073}

=== Experiment depth2_base | depth 2 | hyperbolic | regularization=off ===
Epoch 1/10, Train Loss: 0.912778, Val Loss: 0.732496, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 2/10, Train Loss: 0.531126, Val Loss: 0.545545, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 3/10, Train Loss: 0.372695, Val Loss: 0.458432, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 4/10, Train Loss: 0.304756, Val Loss: 0.432610, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 5/10, Train Loss: 0.281576, Val Loss: 0.426236, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 6/10, Train Loss: 0.249554, Val Loss: 0.383552, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 7/10, Train Loss: 0.215768, Val Loss: 0.377297, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 8/10, Train Loss: 0.207289, Val Loss: 0.378918, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 9/10, Train Loss: 0.199021, Val Loss: 0.366868, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 10/10, Train Loss: 0.157815, Val Loss: 0.332425, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Best validation loss: 0.332425
Saved loss curves to results/plots
Test loss: 0.326676

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C002', 'C102', 'C210', 'C320']
  Visit 2: ['C144', 'C321', 'C400', 'C42']
  Visit 3: ['C013', 'C123', 'C133', 'C30']
  Visit 4: ['C00', 'C102', 'C114', 'C44']
  Visit 5: ['C013', 'C022', 'C133', 'C30']
  Visit 6: ['C013', 'C123', 'C144', 'C30']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C00', 'C102', 'C114', 'C44']
  Visit 2: ['C103', 'C123', 'C144', 'C223']
  Visit 3: ['C103', 'C123', 'C134', 'C144']
  Visit 4: ['C00', 'C010', 'C014', 'C424']
  Visit 5: ['C03', 'C112', 'C144', 'C344']
  Visit 6: ['C00', 'C114', 'C314', 'C44']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C02', 'C112', 'C411', 'C433']
  Visit 2: ['C124', 'C22', 'C23', 'C442']
  Visit 3: ['C103', 'C123', 'C134', 'C144']
  Visit 4: ['C023', 'C102', 'C214', 'C421']
  Visit 5: ['C10', 'C223', 'C30', 'C330']
  Visit 6: ['C024', 'C133', 'C221', 'C434']
Correlation(tree_dist, hyperbolic_embedding_dist) = -0.0031

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 1.6699583333333334, 'std_depth': 0.488140176176088, 'mean_tree_dist': 3.361720698254364, 'std_tree_dist': 0.8361865890852926, 'mean_root_purity': 0.5139166666666667, 'std_root_purity': 0.14560961411329343}

=== Experiment depth2_base | depth 2 | euclidean | regularization=off ===
Epoch 1/10, Train Loss: 0.947975, Val Loss: 0.828677, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 2/10, Train Loss: 0.679241, Val Loss: 0.717007, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 3/10, Train Loss: 0.571526, Val Loss: 0.666564, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 4/10, Train Loss: 0.514721, Val Loss: 0.642033, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 5/10, Train Loss: 0.485001, Val Loss: 0.635226, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 6/10, Train Loss: 0.473099, Val Loss: 0.634152, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 7/10, Train Loss: 0.471229, Val Loss: 0.629261, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 8/10, Train Loss: 0.464363, Val Loss: 0.646630, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 9/10, Train Loss: 0.455250, Val Loss: 0.623683, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 10/10, Train Loss: 0.451568, Val Loss: 0.627336, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Best validation loss: 0.623683
Saved loss curves to results/plots
Test loss: 0.618369

Sample trajectory (euclidean) 1:
  Visit 1: ['C040', 'C323', 'C403', 'C411']
  Visit 2: ['C033', 'C213', 'C24', 'C423']
  Visit 3: ['C012', 'C233', 'C323', 'C400']
  Visit 4: ['C303', 'C342', 'C343', 'C413']
  Visit 5: ['C104', 'C341', 'C440', 'C441']
  Visit 6: ['C003', 'C121', 'C130', 'C210']

Sample trajectory (euclidean) 2:
  Visit 1: ['C104', 'C341', 'C43', 'C440']
  Visit 2: ['C043', 'C14', 'C24', 'C312']
  Visit 3: ['C001', 'C010', 'C223', 'C341']
  Visit 4: ['C012', 'C034', 'C323', 'C434']
  Visit 5: ['C043', 'C121', 'C14', 'C210']
  Visit 6: ['C020', 'C040', 'C44', 'C444']

Sample trajectory (euclidean) 3:
  Visit 1: ['C040', 'C31', 'C41', 'C43']
  Visit 2: ['C012', 'C040', 'C240', 'C411']
  Visit 3: ['C023', 'C044', 'C14', 'C440']
  Visit 4: ['C044', 'C122', 'C413', 'C441']
  Visit 5: ['C104', 'C3', 'C341', 'C440']
  Visit 6: ['C02', 'C023', 'C122', 'C14']
Correlation(tree_dist, euclidean_embedding_dist) = 0.0515

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.8534166666666667, 'std_depth': 0.37562835319265003, 'mean_tree_dist': 3.377860135352884, 'std_tree_dist': 0.8634556089105296, 'mean_root_purity': 0.4648333333333333, 'std_root_purity': 0.13886734277319807}

=== Experiment depth2_base | depth 2 | hyperbolic | regularization=on ===
Epoch 1/10, Train Loss: 0.893015, Val Loss: 0.711625, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010
Epoch 2/10, Train Loss: 0.533323, Val Loss: 0.550441, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020
Epoch 3/10, Train Loss: 0.391487, Val Loss: 0.467570, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 4/10, Train Loss: 0.316703, Val Loss: 0.436577, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 5/10, Train Loss: 0.268763, Val Loss: 0.391133, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 6/10, Train Loss: 0.230685, Val Loss: 0.385636, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 7/10, Train Loss: 0.220027, Val Loss: 0.383998, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 8/10, Train Loss: 0.213740, Val Loss: 0.380746, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 9/10, Train Loss: 0.208762, Val Loss: 0.380426, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 10/10, Train Loss: 0.204308, Val Loss: 0.377999, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Best validation loss: 0.377999
Saved loss curves to results/plots
Test loss: 0.376347

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C203', 'C342', 'C420', 'C421']
  Visit 2: ['C030', 'C102', 'C213', 'C311']
  Visit 3: ['C010', 'C031', 'C421', 'C424']
  Visit 4: ['C010', 'C132', 'C223', 'C344']
  Visit 5: ['C132', 'C134', 'C223', 'C301']
  Visit 6: ['C310', 'C311', 'C313', 'C410']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C034', 'C310', 'C311', 'C410']
  Visit 2: ['C002', 'C203', 'C322', 'C404']
  Visit 3: ['C012', 'C134', 'C223', 'C303']
  Visit 4: ['C310', 'C311', 'C313', 'C410']
  Visit 5: ['C032', 'C203', 'C313', 'C414']
  Visit 6: ['C030', 'C034', 'C311', 'C313']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C123', 'C124', 'C340', 'C343']
  Visit 2: ['C130', 'C132', 'C134', 'C224']
  Visit 3: ['C030', 'C034', 'C102', 'C311']
  Visit 4: ['C010', 'C131', 'C132', 'C304']
  Visit 5: ['C030', 'C034', 'C102', 'C311']
  Visit 6: ['C132', 'C134', 'C223', 'C301']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9868

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 2.0, 'std_depth': 0.0, 'mean_tree_dist': 2.440640703517588, 'std_tree_dist': 0.8289253147532056, 'mean_root_purity': 0.46758333333333335, 'std_root_purity': 0.1536066178768205}

=== Experiment depth2_base | depth 2 | euclidean | regularization=on ===
Epoch 1/10, Train Loss: 0.980752, Val Loss: 0.853825, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000
Epoch 2/10, Train Loss: 0.730873, Val Loss: 0.765562, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000
Epoch 3/10, Train Loss: 0.651374, Val Loss: 0.733302, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 4/10, Train Loss: 0.589709, Val Loss: 0.698304, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 5/10, Train Loss: 0.552023, Val Loss: 0.663967, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 6/10, Train Loss: 0.514207, Val Loss: 0.655232, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 7/10, Train Loss: 0.495204, Val Loss: 0.654696, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 8/10, Train Loss: 0.485092, Val Loss: 0.653050, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 9/10, Train Loss: 0.484861, Val Loss: 0.635257, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 10/10, Train Loss: 0.473076, Val Loss: 0.643631, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Best validation loss: 0.635257
Saved loss curves to results/plots
Test loss: 0.655352

Sample trajectory (euclidean) 1:
  Visit 1: ['C001', 'C022', 'C311', 'C413']
  Visit 2: ['C043', 'C113', 'C133', 'C244']
  Visit 3: ['C202', 'C230', 'C413', 'C414']
  Visit 4: ['C001', 'C022', 'C044', 'C202']
  Visit 5: ['C122', 'C123', 'C133', 'C413']
  Visit 6: ['C002', 'C133', 'C202', 'C402']

Sample trajectory (euclidean) 2:
  Visit 1: ['C012', 'C33', 'C41', 'C431']
  Visit 2: ['C140', 'C230', 'C413', 'C424']
  Visit 3: ['C240', 'C301', 'C342', 'C404']
  Visit 4: ['C123', 'C202', 'C221', 'C3']
  Visit 5: ['C122', 'C123', 'C230', 'C413']
  Visit 6: ['C123', 'C221', 'C233', 'C41']

Sample trajectory (euclidean) 3:
  Visit 1: ['C023', 'C201', 'C404', 'C421']
  Visit 2: ['C010', 'C240', 'C342', 'C404']
  Visit 3: ['C120', 'C122', 'C342', 'C413']
  Visit 4: ['C120', 'C122', 'C342', 'C413']
  Visit 5: ['C240', 'C301', 'C404', 'C424']
  Visit 6: ['C024', 'C101', 'C124', 'C204']
Correlation(tree_dist, euclidean_embedding_dist) = 0.3726

Synthetic (euclidean) stats (N=1000): {'mean_depth': 1.9552083333333334, 'std_depth': 0.2234995300454915, 'mean_tree_dist': 3.445792502722041, 'std_tree_dist': 0.9052446250955426, 'mean_root_purity': 0.47275, 'std_root_purity': 0.13550499683283515}

Real stats (depth7_extended, max_depth=7): {'mean_depth': 5.3797976334479465, 'std_depth': 1.7294582012361523, 'mean_tree_dist': 5.7591450057100175, 'std_tree_dist': 4.756650684766557, 'mean_root_purity': 0.6289569269083962, 'std_root_purity': 0.20468844637974468}

=== Experiment depth7_extended | depth 7 | hyperbolic | regularization=off ===
Epoch 1/10, Train Loss: 0.912479, Val Loss: 0.725803, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 2/10, Train Loss: 0.540240, Val Loss: 0.545033, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 3/10, Train Loss: 0.384740, Val Loss: 0.466284, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 4/10, Train Loss: 0.306083, Val Loss: 0.420165, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 5/10, Train Loss: 0.250685, Val Loss: 0.382678, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 6/10, Train Loss: 0.223724, Val Loss: 0.365706, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 7/10, Train Loss: 0.184834, Val Loss: 0.331074, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 8/10, Train Loss: 0.162772, Val Loss: 0.330852, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 9/10, Train Loss: 0.154966, Val Loss: 0.329067, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 10/10, Train Loss: 0.149176, Val Loss: 0.326480, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Best validation loss: 0.326480
Saved loss curves to results/plots
Test loss: 0.330172

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C244', 'C310', 'C400', 'C103d1']
  Visit 2: ['C244', 'C031d3', 'C212d0', 'C442d2']
  Visit 3: ['C030', 'C422', 'C144d0', 'C431d2']
  Visit 4: ['C030', 'C142', 'C021d2', 'C123d3']
  Visit 5: ['C023', 'C030', 'C142', 'C422']
  Visit 6: ['C142', 'C422', 'C201d2', 'C431d2']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C030', 'C142', 'C422', 'C431d2']
  Visit 2: ['C030', 'C142', 'C422', 'C021d2']
  Visit 3: ['C112d3', 'C204d4', 'C300d3', 'C324d2']
  Visit 4: ['C030', 'C142', 'C030d3', 'C123d3']
  Visit 5: ['C030', 'C142', 'C234d3', 'C404d4']
  Visit 6: ['C031d3', 'C131d2', 'C240d2', 'C442d2']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C112d3', 'C142d0', 'C143d2', 'C431d4']
  Visit 2: ['C030', 'C234', 'C131d0', 'C431d2']
  Visit 3: ['C030', 'C030d3', 'C041d0', 'C343d0']
  Visit 4: ['C203', 'C031d3', 'C111d4', 'C432d2']
  Visit 5: ['C030', 'C142', 'C422', 'C021d2']
  Visit 6: ['C031d3', 'C131d2', 'C212d0', 'C442d2']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.0509

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 4.125416666666666, 'std_depth': 1.6717318344725296, 'mean_tree_dist': 7.964722483537159, 'std_tree_dist': 2.544956785908991, 'mean_root_purity': 0.4735416666666667, 'std_root_purity': 0.13431135443645692}

=== Experiment depth7_extended | depth 7 | euclidean | regularization=off ===
Epoch 1/10, Train Loss: 0.964533, Val Loss: 0.851455, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 2/10, Train Loss: 0.714400, Val Loss: 0.747548, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 3/10, Train Loss: 0.626180, Val Loss: 0.700517, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 4/10, Train Loss: 0.565685, Val Loss: 0.675115, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 5/10, Train Loss: 0.512445, Val Loss: 0.642529, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 6/10, Train Loss: 0.487090, Val Loss: 0.643835, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 7/10, Train Loss: 0.477612, Val Loss: 0.637954, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 8/10, Train Loss: 0.474798, Val Loss: 0.633878, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 9/10, Train Loss: 0.470369, Val Loss: 0.636830, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Epoch 10/10, Train Loss: 0.465050, Val Loss: 0.633295, lambda_tree_eff=0.0000, lambda_radius_eff=0.0000
Best validation loss: 0.633295
Saved loss curves to results/plots
Test loss: 0.635111

Sample trajectory (euclidean) 1:
  Visit 1: ['C044', 'C023d3', 'C134d0', 'C433d1']
  Visit 2: ['C401', 'C101d3', 'C134d0', 'C141d2']
  Visit 3: ['C113d3', 'C224d4', 'C413d2', 'C444d2']
  Visit 4: ['C011d0', 'C233d1', 'C301d3', 'C414d3']
  Visit 5: ['C134d0', 'C240d2', 'C331d4', 'C443d0']
  Visit 6: ['C014', 'C111', 'C130', 'C211d2']

Sample trajectory (euclidean) 2:
  Visit 1: ['C044', 'C424', 'C232d0', 'C340d2']
  Visit 2: ['C044', 'C014d2', 'C110d1', 'C134d0']
  Visit 3: ['C413', 'C133d0', 'C400d3', 'C401d0']
  Visit 4: ['C144d0', 'C202d3', 'C202d4', 'C301d1']
  Visit 5: ['C113d3', 'C124d2', 'C202d4', 'C210d3']
  Visit 6: ['C120d2', 'C134d0', 'C203d0', 'C240d2']

Sample trajectory (euclidean) 3:
  Visit 1: ['C124d2', 'C210d3', 'C222d0', 'C432d3']
  Visit 2: ['C044', 'C120d1', 'C331d3', 'C442d4']
  Visit 3: ['C044', 'C424', 'C232d0', 'C331d3']
  Visit 4: ['C044', 'C424', 'C331d3', 'C442d4']
  Visit 5: ['C030d0', 'C222d0', 'C310d2', 'C401d0']
  Visit 6: ['C424', 'C002d0', 'C424d0', 'C430d0']
Correlation(tree_dist, euclidean_embedding_dist) = -0.0043

Synthetic (euclidean) stats (N=1000): {'mean_depth': 4.300833333333333, 'std_depth': 1.7453030220820935, 'mean_tree_dist': 8.03245078071962, 'std_tree_dist': 2.8330137017922876, 'mean_root_purity': 0.49266666666666664, 'std_root_purity': 0.1380231462070362}

=== Experiment depth7_extended | depth 7 | hyperbolic | regularization=on ===
Epoch 1/10, Train Loss: 0.953402, Val Loss: 0.769077, lambda_tree_eff=0.0033, lambda_radius_eff=0.0010
Epoch 2/10, Train Loss: 0.578639, Val Loss: 0.575850, lambda_tree_eff=0.0067, lambda_radius_eff=0.0020
Epoch 3/10, Train Loss: 0.431770, Val Loss: 0.504912, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 4/10, Train Loss: 0.344645, Val Loss: 0.446072, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 5/10, Train Loss: 0.298213, Val Loss: 0.428704, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 6/10, Train Loss: 0.282778, Val Loss: 0.425511, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 7/10, Train Loss: 0.274603, Val Loss: 0.420530, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 8/10, Train Loss: 0.255767, Val Loss: 0.388646, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 9/10, Train Loss: 0.220120, Val Loss: 0.375289, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Epoch 10/10, Train Loss: 0.212186, Val Loss: 0.370899, lambda_tree_eff=0.0100, lambda_radius_eff=0.0030
Best validation loss: 0.370899
Saved loss curves to results/plots
Test loss: 0.381992

Sample trajectory (hyperbolic) 1:
  Visit 1: ['C010d3', 'C010d4', 'C011d4', 'C014d4']
  Visit 2: ['C103d4', 'C203d2', 'C203d3', 'C203d4']
  Visit 3: ['C001d4', 'C010d3', 'C010d4', 'C133d4']
  Visit 4: ['C010d3', 'C010d4', 'C011d4', 'C410d4']
  Visit 5: ['C010d3', 'C010d4', 'C233d3', 'C233d4']
  Visit 6: ['C141d4', 'C331d2', 'C331d3', 'C331d4']

Sample trajectory (hyperbolic) 2:
  Visit 1: ['C041d3', 'C041d4', 'C141d3', 'C141d4']
  Visit 2: ['C010d4', 'C011d3', 'C011d4', 'C424d4']
  Visit 3: ['C010d3', 'C010d4', 'C211d4', 'C403d4']
  Visit 4: ['C011d3', 'C011d4', 'C221d4', 'C444d4']
  Visit 5: ['C122d4', 'C240d3', 'C240d4', 'C323d4']
  Visit 6: ['C011d3', 'C011d4', 'C410d4', 'C424d4']

Sample trajectory (hyperbolic) 3:
  Visit 1: ['C122d3', 'C122d4', 'C200d4', 'C203d4']
  Visit 2: ['C011d3', 'C011d4', 'C424d4', 'C444d4']
  Visit 3: ['C100d4', 'C240d4', 'C321d4', 'C400d4']
  Visit 4: ['C033d3', 'C033d4', 'C240d4', 'C442d4']
  Visit 5: ['C011d3', 'C011d4', 'C221d4', 'C444d4']
  Visit 6: ['C122d4', 'C203d4', 'C411d3', 'C411d4']
Correlation(tree_dist, hyperbolic_embedding_dist) = 0.9704

Synthetic (hyperbolic) stats (N=1000): {'mean_depth': 6.625416666666666, 'std_depth': 0.5911322410331624, 'mean_tree_dist': 3.844302967886902, 'std_tree_dist': 4.816374209092423, 'mean_root_purity': 0.566375, 'std_root_purity': 0.14332576893799195}

=== Experiment depth7_extended | depth 7 | euclidean | regularization=on ===
Epoch 1/10, Train Loss: 0.935955, Val Loss: 0.820722, lambda_tree_eff=0.0033, lambda_radius_eff=0.0000
Epoch 2/10, Train Loss: 0.685128, Val Loss: 0.729737, lambda_tree_eff=0.0067, lambda_radius_eff=0.0000
Epoch 3/10, Train Loss: 0.600562, Val Loss: 0.699745, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 4/10, Train Loss: 0.560039, Val Loss: 0.690076, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 5/10, Train Loss: 0.540990, Val Loss: 0.680080, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 6/10, Train Loss: 0.516456, Val Loss: 0.653006, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 7/10, Train Loss: 0.496505, Val Loss: 0.648082, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 8/10, Train Loss: 0.488895, Val Loss: 0.654562, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 9/10, Train Loss: 0.487095, Val Loss: 0.648217, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Epoch 10/10, Train Loss: 0.481671, Val Loss: 0.648298, lambda_tree_eff=0.0100, lambda_radius_eff=0.0000
Best validation loss: 0.648082
Saved loss curves to results/plots
Test loss: 0.646227

Sample trajectory (euclidean) 1:
  Visit 1: ['C130d1', 'C232d4', 'C343d3', 'C411d0']
  Visit 2: ['C012d1', 'C233d3', 'C234d1', 'C303d2']
  Visit 3: ['C303d2', 'C303d3', 'C320d0', 'C411d0']
  Visit 4: ['C242d0', 'C344d4', 'C400d0', 'C412d3']
  Visit 5: ['C100d3', 'C231d2', 'C304d4', 'C342d3']
  Visit 6: ['C043d1', 'C114d3', 'C203d3', 'C412d3']

Sample trajectory (euclidean) 2:
  Visit 1: ['C004', 'C112d4', 'C242d4', 'C301d4']
  Visit 2: ['C110d2', 'C110d3', 'C312d4', 'C411d0']
  Visit 3: ['C112d4', 'C113d4', 'C412d3', 'C434d4']
  Visit 4: ['C43', 'C103d4', 'C232d1', 'C423d3']
  Visit 5: ['C312d4', 'C313d4', 'C320d4', 'C411d0']
  Visit 6: ['C004', 'C043d1', 'C103d4', 'C423d1']

Sample trajectory (euclidean) 3:
  Visit 1: ['C110d3', 'C232d4', 'C303d2', 'C320d4']
  Visit 2: ['C110d2', 'C320d4', 'C333d1', 'C342d4']
  Visit 3: ['C223d2', 'C230d0', 'C242d4', 'C301d4']
  Visit 4: ['C430', 'C043d1', 'C240d4', 'C324d0']
  Visit 5: ['C003d1', 'C141d1', 'C212d3', 'C324d0']
  Visit 6: ['C230d0', 'C310d2', 'C423d3', 'C441d4']
Correlation(tree_dist, euclidean_embedding_dist) = 0.2246

Synthetic (euclidean) stats (N=1000): {'mean_depth': 5.063625, 'std_depth': 1.731904594959453, 'mean_tree_dist': 9.632374309793665, 'std_tree_dist': 3.0116963459061554, 'mean_root_purity': 0.477875, 'std_root_purity': 0.1517264019268455}
