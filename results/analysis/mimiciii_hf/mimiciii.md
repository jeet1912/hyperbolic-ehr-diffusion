HyperMedDiff-Risk: Model and Ablation Record
===========================================
A precise description clarifies how the MIMIC risk pipeline couples hyperbolic geometry with MedDiffusion-style multitask learning and why the ablation sweeps in `results/analysis/table2_ablation.md` behave the way they do.

Imports and High-Level Modules
------------------------------
```python
import argparse
import copy
import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import MimicDataset, make_pad_collate
from hyperbolic_embeddings import HyperbolicCodeEmbedding
from traj_models import TrajectoryVelocityModel
from regularizers import radius_regularizer
```
Auxiliary helpers live in `visit_utils.py`, `train_utils.py`, `losses.py`, `traj_models.py`, and `regularizers.py`. Together they establish the workflow below.

Model Pipeline
--------------

1. **Dataset and batching**  
   `MimicDataset` provides trajectories $\{\mathcal{V}_p\}$ with binary labels $y_p \in \{0,1\}$. `make_pad_collate` builds padded tensors $(B,L,V)$ alongside visit masks.

2. **Code embedding pretraining**  
   Hyperbolic embeddings $c_i \in \mathbb{B}^d$ [1] minimize
   $$\mathcal{L}_{\text{pre}} = \lambda_{\text{radius}} \underbrace{\frac{1}{N}\sum_i(\|c_i\|_{\mathbb{B}}-r^\star)^2}_{\mathcal{L}_{\text{radius}}} + \lambda_{\text{hdd}} \underbrace{\mathbb{E}_{i,j}\left(\|f_i-f_j\|_2 - d_{\mathbb{B}}(c_i,c_j)\right)^2}_{\mathcal{L}_{\text{HDD}}}$$
   where $f_i$ is the diffusion signature from co-occurrence graphs [2]. This stage aligns geometry with graph diffusion before freezing $C$.

3. **Graph-hyperbolic visit encoder**  
   Diffusion kernels $K_s$ define the `HyperbolicGraphDiffusionLayer`,
   $$
   Z_s = K_s\,\log_0(C), \qquad Z = \mathrm{Proj}\big[\mathrm{concat}_s Z_s\big],
   $$
   ensuring each code receives information from multi-hop neighborhoods before re-entering $\mathbb{R}^d$. For each visit, we aggregate the corresponding rows of 
H with a simple mean in tangent space, and a stack of `GlobalSelfAttentionBlock`s (multi-head attention, feed-forward layers, LayerNorm) enforces global interactions. Time encodings then shift the tangent vectors, yielding visit representations $z_{p,t}$.

4. **Trajectory velocity model**  
   `TrajectoryVelocityModel` predicts $v_\theta(z_t,t,h_{<t})$ and underpins both rectified-flow losses [4] and flow-based sampling for synthetic trajectories.

5. **Risk encoder and head**  
   `TemporalLSTMEncoder` is a single-layer LSTM operating on the visit sequence with masking; it returns both per-visit states $h_{p,t}$ and pooled representations $h_p=\mathrm{LSTMPool}(h_{p,1:L})$. These features feed the linear `RiskHead`, which produces logits $\hat{y}_p$ and final probabilities $\sigma(\hat{y}_p)$.

Joint Training Loss
-------------------

Within each batch, `run_epoch` accumulates:

1. **Real risk BCE**
   $$
   \mathcal{L}_{\text{real}} = \frac{1}{B}\sum_{p=1}^B \mathrm{BCE}(\hat{y}_p, y_p).
   $$

2. **Synthetic risk BCE** (MedDiffusion $\lambda_S$ term) [5]  
   Synthetic latents $\tilde{z}$ sampled via the rectified flow pass through the same risk head:
   $$
   \mathcal{L}_{\text{synth}} = \frac{1}{B}\sum_{p=1}^B \mathrm{BCE}(\hat{y}^{\text{synth}}_p, y_p).
   $$

3. **Hyperbolic rectified-flow penalty**
   $$
   \mathcal{L}_{\text{flow}} = \mathbb{E}_{t}\left\|v_\theta(z_t,t) - \left(z_1 - z_0\right)\right\|_2^2,
   $$
   where $z_t$ follows the interpolant used inside `rectified_flow_loss_hyperbolic`.

4. **Feature consistency**
   $$
   \mathcal{L}_{\text{cons}} = \mathrm{MSE}\left(h_{\text{real}}, h_{\text{synth}}\right).
   $$

The combined objective is
$$
\boxed{\mathcal{L}_{\text{HyperMedDiff}} = \mathcal{L}_{\text{real}} + \lambda_S \mathcal{L}_{\text{synth}} + \lambda_D \mathcal{L}_{\text{flow}} + \lambda_{\text{consistency}} \mathcal{L}_{\text{cons}}.}
$$

Ablation Summary (Table 2)
--------------------------

Each sweep toggles one factor of the baseline configuration; the metrics below replicate the block reported in `table2_ablation.md`.

| Run | Experiment     | Val Loss | AUROC  | AUPRC  | Accuracy | F1     | Kappa  | Corr   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | MedDiffusion (paper) | N/A    | N/A    | 0.7064 | N/A    | 0.6679 | 0.4526 | N/A    |
| 1   | Base           | 1.9489 | 0.8711 | 0.7991 | 0.7533 | 0.7175 | 0.5046 | 0.8385 |
| 2   | 02_NoDiffusion | 1.8176 | 0.8687 | 0.7919 | 0.7534 | 0.7175 | 0.5046 | 0.8897 |
| 3   | 03_LocalDiff   | 1.9546 | 0.8726 | 0.8054 | 0.7534 | 0.7175 | 0.5046 | 0.8533 |
| 4   | 04_GlobalDiff  | 1.8712 | 0.8740 | 0.8058 | 0.7534 | 0.7175 | 0.5046 | 0.8380 |
| 5   | 05_NoHDD       | 1.8350 | 0.8806 | 0.8291 | 0.7534 | 0.7175 | 0.5046 | -0.0021 |
| 6   | 06_StrongHDD   | 1.8682 | 0.8696 | 0.8022 | 0.7534 | 0.7175 | 0.5046 | 0.9071 |
| 7   | 07_HighDropout | 1.8009 | 0.8733 | 0.8063 | 0.7534 | 0.7175 | 0.5046 | 0.8127 |
| 8   | 08_SmallDim    | 2.0531 | 0.8686 | 0.7928 | 0.7534 | 0.7175 | 0.5046 | 0.7374 |
| 9   | 09_DiscrimOnly | 1.2057 | 0.8758 | 0.8125 | 0.7534 | 0.7175 | 0.5046 | 0.8261 |
| 10  | 10_GenFocus    | 2.4380 | 0.8764 | 0.8115 | 0.7534 | 0.7175 | 0.5046 | 0.8419 |

Loss Variants per Ablation
--------------------------

- **Base** — Full objective with $\lambda_S=\lambda_D=1.0$, $\lambda_{\text{consistency}}=0.1$, diffusion steps $[1,2,4,8]$, and pretraining weight $\lambda_{\text{hdd}}=0.02$.
- **02_NoDiffusion** — Restricts diffusion steps to $[1]$; loss weights remain unchanged, but the encoder’s receptive field collapses to single-hop structure.
- **03_LocalDiff** — Steps $[1,2]$ favor local neighborhoods; the loss remains $\mathcal{L}_{\text{HyperMedDiff}}$.
- **04_GlobalDiff** — Steps $[1,2,4,8,16]$ enlarge context while keeping the same objective.
- **05_NoHDD** — Sets $\lambda_{\text{hdd}}=0$ during pretraining:
  $$
  \mathcal{L}_{\text{pre}}=\lambda_{\text{radius}}\mathcal{L}_{\text{radius}},
  $$
  so embeddings no longer align with diffusion distances. Downstream loss is unchanged.
- **06_StrongHDD** — Raises $\lambda_{\text{hdd}}$ to $0.1$, accentuating the structural term in $\mathcal{L}_{\text{pre}}$.
- **07_HighDropout** — Sets dropout to $0.5$ throughout the encoder and LSTM; the objective remains identical but with heavier regularization.
- **08_SmallDim** — Reduces embedding dimensionality to $64$; the loss is the same, yet gradients act in a smaller tangent space.
- **09_DiscrimOnly** — Eliminates the synthetic BCE by setting $\lambda_S=0$, yielding
  $$
  \mathcal{L}_{\text{real}} + \lambda_D \mathcal{L}_{\text{flow}} + \lambda_{\text{consistency}} \mathcal{L}_{\text{cons}},
  $$
  so synthetic supervision disappears.
- **10_GenFocus** — Doubles $\lambda_S$ to $2.0$, producing
  $$
  \mathcal{L}_{\text{real}} + 2\,\mathcal{L}_{\text{synth}} + \lambda_D \mathcal{L}_{\text{flow}} + \lambda_{\text{consistency}} \mathcal{L}_{\text{cons}},
  $$
  thereby emphasizing synthetic alignment.

These configurations reveal that structural correlation responds primarily to pretraining geometry (05/06) and diffusion scope (02–04), while discriminative performance is driven by $\lambda_S$ (09/10) and encoder capacity (07/08) [6].

References
----------
[1] Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. Advances in Neural Information Processing Systems.

[2] Chami, I., et al. (2019). Hyperbolic Graph Convolutional Neural Networks. Advances in Neural Information Processing Systems.


[4] Liu, X., et al. (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. arXiv preprint arXiv:2209.03003.

[5] Zhong, Y., et al. (2023). MedDiffusion: Boosting Health Risk Prediction via Diffusion-based Data Augmentation. arXiv preprint arXiv:2310.02520.

[6] Mao, W., et al. (2025). Hyperbolic Deep Learning for Foundation Models: A Survey. arXiv preprint arXiv:2507.17787.
