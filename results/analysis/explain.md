## `data_icd_toy.py`

#### `ToyICDHierarchy`

Constructs a directed hierarchy $G=(V,E)$ resembling ICD chapters.

- Roots: $C_i$ for $i\in\{0,1,2,3,4\}$.
- First level: nodes $C_{ij}$ with edges $C_i\rightarrow C_{ij}$.
- Leaves: $C_{ijk}$ with edges $C_{ij}\rightarrow C_{ijk}$.
- Optional `extra_depth` extends each leaf into a chain $C_{ijk} \rightarrow C_{ijk}d_0 \rightarrow \dots \rightarrow C_{ijk}d_{D-1}$.

Code maps $\text{idx}(c)=i$ and $\text{depth}(c)$ are stored.

#### `depth(code)`

Returns node depth $d(c)$ read from NetworkX attributes.

#### `tree_distance(c_1, c_2)`

Uses undirected shortest path length
\[
d_{\text{tree}}(c_1,c_2) = \text{length of shortest path in } G^{\text{undirected}}
\]
If no path exists, returns `None`.

#### `sample_toy_trajectories`

Generates $N$ patient trajectories. For each patient:

1. Sample $T \sim \mathcal{U}(\text{min}_T, \text{max}_T)$.
2. Sample cluster leaf $\ell$ and define $\mathcal{C}=\{\ell\}\cup\text{Ancestors}(\ell)$.
3. For each visit, sample visit length $K \sim \mathcal{U}(\text{min}_{\text{codes}}, \text{max}_{\text{codes}})$.
4. While visit codes < $K$, draw code $c$ via mixture distribution
   \[
   P(c) = 0.7\,\mathcal{U}(\mathcal{C}) + 0.3\,\mathcal{U}(V)
   \]
   Insert $\text{idx}(c)$ into visit list.

- Output: list of trajectories, each being a list of visits (sorted integer indices).

#### `_hierarchy_positions`

Groups nodes by depth and assigns layout coordinates $x=(k+1)/(n+1), y=-d$ for plotting.

#### `plot_hierarchy_graph`

Draws `hier.G` using the computed positions and saves PNG files.


## `diffusion.py`

#### `cosine_beta_schedule(T, s)`

Implements the cosine DDPM schedule from Nichol & Dhariwal (2021).

Define $x = 0,\dots,T$ and cumulative alphas
\[
\bar{\alpha}_t = \frac{\cos^2\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)}{\cos^2(s \cdot \frac{\pi}{2(1+s)})}
\]
Then per-step $\beta_t = 1 - \bar{\alpha}_{t+1}/\bar{\alpha}_t$. Clamp to $[10^{-5}, 0.999]$.

#### `TimeEmbedding`

Implements sinusoidal time embedding followed by 2-layer MLP.

1. For half-dim $d/2$, compute frequencies $\omega_k = 10000^{-2k/d}$.
2. Raw embedding: $\sin(t\omega_k), \cos(t\omega_k)$, zero-padding if dim odd.
3. Pass through `Linear -> SiLU -> Linear` to obtain $\mathbf{e}_t \in \mathbb{R}^d$.


## `euclidean_embeddings.py`

#### `EuclideanCodeEmbedding`

Learnable matrix $E\in\mathbb{R}^{(|V|+1)\times d}$. Forward map: `code_ids -> E[code_ids]`.

#### `EuclideanVisitEncoder`

For each visit tensor $s$, remove pad index `pad_idx`, embed codes, and mean-pool:
\[
v = \begin{cases}
\frac{1}{|s|} \sum_{i \in s} E_i & |s|>0\\
0 & |s|=0
\end{cases}
\]
Stack into `torch.stack` output of shape `[B*T, d]`.


## `hyperbolic_embeddings.py`

#### `HyperbolicCodeEmbedding`

- Manifold: Poincaré ball $\mathbb{B}^d_c$. Parameters are `geoopt.ManifoldParameter`.
- Initialization: small random vector projected via `projx` (ensures $||z|| < 1/\sqrt{c}$).

#### `VisitEncoder`

For visit codes `ids`:

1. Remove pads.
2. Map to tangent space with logarithmic map $\ell_i=\log_0(z_i)$.
3. Mean-pool: $v=\frac{1}{|s|}\sum \ell_i$, zero when empty.

Output is Euclidean vector per visit.


## `traj_model.py`

#### `TrajectoryEpsModel`

Predicts diffusion noise $\hat{\epsilon}$ for visit sequences.

**Inputs:**

- Noisy latent tensor $x_t \in \mathbb{R}^{B\times L \times d}$.
- Timesteps $t\in\mathbb{Z}^B$.
- Visit mask $M \in \{0,1\}^{B\times L}$ (True = real visit).

**Architecture:**

1. Compute time embeddings $\mathbf{e}_t$ using `TimeEmbedding` and broadcast along $L$: $\tilde{x}_t = x_t + \mathbf{e}_t$.
2. Apply TransformerEncoder (multi-head self-attention with `n_heads`, feed-forward dim `ff_dim`). Mask uses $\neg M$ because PyTorch expects True=pad.
3. Final linear projection yields $\hat{\epsilon} \in \mathbb{R}^{B\times L \times d}$.


## `metrics_toy.py`

#### `traj_stats`

Given trajectories (list of visits) and hierarchy object:

1. Remove pad entries (-1).
2. Map indices to codes, compute depths $d(c)$.
3. Aggregate mean and std of depths.
4. For each visit, compute all pairwise tree distances $d_{\text{tree}}(c_i,c_j)$ and accumulate mean/std.
5. Compute “root purity”: root labels are first two characters $C0..C4$; purity = `max_count / visit_len`.

Returns dictionary with aggregated statistics.


## `train_toy.py`

This file orchestrates datasets, diffusion training, evaluation, and (optionally) sampling.

#### `TrajDataset`

- Stores trajectories padded/truncated to length `max_len` using pad visit `[pad_idx]`.
- `__getitem__` returns list of visits (each a list of code indices).

#### `make_collate_fn`

Returns `collate_fn` that:

1. Stacks batch into `[B, L]` structure.
2. Flattens visits into list `flat_visits` of tensors.
3. Builds Boolean mask $M_{b\ell}=\text{True}$ if not pad visit.

#### `build_visit_tensor`

Runs visit encoder on `flat_visits` (no grad) to produce $x_0 \in \mathbb{R}^{B\times L \times d}$.

#### `decode_visit_vectors`

Given visit vectors $Z\in\mathbb{R}^{B\times L \times d}$, compute similarities vs code embeddings:

- Hyperbolic: log-map code embeddings, dot product as similarity.
- Euclidean: use negative squared distance.

Select top-$K$ codes per visit to approximate discrete visits.

#### `visits_from_indices`

Utility: convert tensors of code indices back to sorted unique lists.

#### `mean_tree_distance_from_visits`

Compute average tree distance between all code pairs from visit predictions (filtering pad codes).

#### `code_pair_loss`

Correlation-inspired regularizer:

1. Sample random code pairs $(i,j)$.
2. Compute tree distance $d_{\text{tree}}(i,j)$.
3. Compute embedding distance $d_{\text{emb}}(i,j)$: hyperbolic geodesic or Euclidean norm.
4. Normalize both via z-score, minimize MSE:
   \[
   \hat{d}_{\text{tree}} = \frac{d_{\text{tree}} - \mu_t}{\sigma_t},\quad
   \hat{d}_{\text{emb}} = \frac{d_{\text{emb}} - \mu_e}{\sigma_e},\quad
   \mathcal{L}_{\text{pair}} = \| \hat{d}_{\text{emb}} - \hat{d}_{\text{tree}} \|_2^2
   \]
   aligning manifold distances with the discrete hierarchy metric.

#### `sample_fake_visit_indices`

- Sample random noise $x_t \sim \mathcal{N}(0,I)$.
- Random timesteps $t$.
- Predict $x_0$ using $\hat{\epsilon}$ formula:
  \[
  \hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}
  \]
- Decode to code indices for quick inspection.

#### `sample_trajectories`

Implements reverse DDPM sampling. Iteratively for $t=T-1 \dots 0$:
\[
\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_t \right),
\qquad x_{t-1} =
\begin{cases}
\mu_t + \sqrt{\tilde{\beta}_t} z, & t>0\\
\mu_t, & t=0
\end{cases}
\]
Finally decode visits and print a few sample trajectories.

#### `split_trajectories`

Randomly permutes trajectory indices and splits into train/val/test per provided ratios.

#### `compute_batch_loss`

Main training objective:

1. Encode visits -> $x_0$.
2. Sample timesteps $t$ and noise $\epsilon$.
3. Form noisy $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.
4. Predict $\hat{\epsilon}$, compute MSE loss.
5. Optionally add `code_pair_loss` term and radius-depth penalty
   $\mathcal{L}_{\text{radius}} = \frac{1}{N}\sum (||z_i|| - d_i)^2$ for hyperbolic embeddings.


Total loss:

$\mathcal{L} = \mathbb{E}\big[\|\epsilon - \hat{\epsilon}(x_t, t)\|_2^2\big] + \lambda_{\text{tree}}\,\mathcal{L}_{\text{pair}} +\lambda_{\text{radius}}\,\mathcal{L}_{\text{radius}}$ with $\lambda_{\text{tree}}$ and $\lambda_{\text{radius}}$ linearly warmed up over the initial epochs.



#### `run_epoch`

Iterates over loader, computes loss, applies optimizer steps when provided. Returns mean loss.

#### `train_model`

Runs multiple epochs, uses Adam over both transformer and code embeddings, scheduler `ReduceLROnPlateau`. Tracks best validation loss.

#### `evaluate_loader`

Runs `compute_batch_loss` in no-grad mode over a loader.

#### `evaluate_test_accuracy`

Uses helper `compute_batch_accuracy` to estimate per-code recall (see below).

#### `compute_batch_accuracy`

1. Reconstruct $x_0$ and sample $t$.
2. Denoise to $\hat{x}_0$, decode visits.
3. For each real visit, count hits where predicted top-$K$ contains the true code.
4. Return `(correct, total)` counts.

#### `_format_float_for_name` & `_plot_single_curve`

Utility functions for naming plot files and drawing loss curves (currently commented out in `main`).

#### `save_loss_curves`

Saves training/validation loss PNGs with metadata tags.

#### `correlation_tree_vs_embedding`

Monte Carlo Pearson correlation between tree distances and embedding distances. For sampled pairs $(i,j)$:
\[
\text{corr} = \frac{\text{Cov}(d_{\text{tree}}, d_{\text{emb}})}{\sigma(d_{\text{tree}})\,\sigma(d_{\text{emb}})}
\]
where $d_{\text{emb}}$ uses hyperbolic geodesic length or Euclidean norm. High correlation indicates the embedding geometry preserves the ontology structure.

#### `main`

High-level orchestration:

1. Generate or receive trajectory splits.
2. Instantiate embeddings (Euclidean vs hyperbolic) and visit encoder.
3. Build cosine schedule (`betas`, `alphas`, `alphas_cumprod`).
4. Train `TrajectoryEpsModel` for specified epochs.
5. Print test accuracy (`evaluate_test_accuracy`).
6. (Commented) Optional sampling, correlation, and synthetic stats.


## Model Sketch

```
codes --[{Euclid/Hyperbolic embedding}]--> visit vectors --[{Transformer DDPM}]--> noisy latent sequences
```

Latent diffusion process:

1. $x_0$: visit representations.
2. Forward noising: $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$.
3. Reverse model approximates $p_\theta(x_{t-1}|x_t)$ by predicting $\hat{\epsilon}_t$ and using DDPM update.

## Narrative Summary

We frame the system as a generative model over visit representations: the DDPM learns to push random noise toward clean visit vectors in latent space. The foundation is a geometry-aware embedding space for codes, where hyperbolic embeddings combined with the tree regularizer force the code manifold to mirror the ICD hierarchy. Decoding remains heuristic: latent visit vectors yield discrete codes by taking the top-K nearest neighbors (log-map plus dot product on the hyperbolic setup).

With these ingredients, we can sample trajectories: the trained diffusion model produces sequences of latent visit vectors, which are then decoded into ICD codes. Yet the crucial limitation is structural. The pipeline does **not** establish a reversible path from ICD codes → latent visit vectors → ICD codes. Instead, real visits are encoded via simple mean pooling; the DDPM denoises those pooled vectors; and decoding relies on a nearest-neighbor lookup. No part of this is an end-to-end autoencoder, so the system never learns to reconstruct discrete visits in the latent space. The DDPM emits continuous vectors only, expecting a downstream lookup to convert them into codes. Consequently, this process genuinely learns to synthesize visit vectors, not actual medical visits, and the decoding procedure is approximate, untrained, and inherently lossy.

## Extended Variants and Regularizers

After the narrative summary I keep a running log of every architectural or loss modification I implemented. Everything below is phrased in first person so it reads the way I actually narrate progress in my notes.

### Baseline context

In `train_toy.py` I train a DDPM over visit vectors \(x_0\) that come from mean-pooled code embeddings. The only geometry-aware terms act directly on the *code* embeddings (tree/radius regularizers). Decoding is still a heuristic nearest-neighbor lookup from latent visits back to codes. Every variant below changes one of three ingredients: how I decode, how I inject noise (Euclidean vs. hyperbolic), and how I constrain the embedding geometry (HDD/HGD). I list them in the order I built them.

### 1. Adding a Learnable Visit Decoder (`train_toyWithDecoder.py`)

Here I insert a parametric visit decoder so I can reconstruct code sets with an actual supervised loss instead of relying on nearest neighbors. (VisitDecoder)

- **Input:** visit latents \(v\in\mathbb{R}^d\) (from encoder or DDPM).
- **Output:** logits over true codes (no pad token), shape \([B,L,C]\).
- **Usage:** multi-label prediction because each visit contains a *set* of ICD codes.

**Reconstruction loss.** For each batch I encode visits into \(x_0\in\mathbb{R}^{B\times L\times d}\), sample \(t,\epsilon\), and form
\[
x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\qquad
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}(x_t,t)}{\sqrt{\bar{\alpha}_t}}.
\]
Multi-hot targets \(y\in\{0,1\}^{B\times L\times C}\) come from ground-truth visits. I decode both \(x_0\) and \(\hat{x}_0\) and apply BCE-with-logits:
\[
\mathcal{L}_{\text{recon}}=\operatorname{BCE}(f_{\text{dec}}(x_0),y)+\operatorname{BCE}(f_{\text{dec}}(\hat{x}_0),y).
\]

**Total loss.**
\[
\mathcal{L}=\mathbb{E}\| \epsilon-\hat{\epsilon}(x_t,t)\|_2^2
+\lambda_{\text{tree}}\mathcal{L}_{\text{pair}}
+\lambda_{\text{radius}}\mathcal{L}_{\text{radius}}
+\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}.
\]
Now the model is explicitly penalized when decoded codes do not match the original visit.

**Sketch**

```
codes --[Euclid/Hyperbolic embedding]--> visit encoder --> x0
     --> DDPM forward (Euclidean noise) --> x_t
     --> eps_model --> ε̂_t --> x̂0 --> VisitDecoder --> logits/top-K
```

**Why hyperbolic lagged here.**
- Decoder is purely Euclidean, so hyperbolic structure is flattened by logmap and mean pooling before the MLP sees it.
- Encoder is still mean pooling; nothing enforces an invertible path between encoder and decoder.
- Forward diffusion is Euclidean even for hyperbolic embeddings.
- Result: Euclidean setups get reasonable recall@4, while hyperbolic setups remain geometry-aware but harder to decode.

### 2. Hyperbolic Forward Noise + Decoder (`train_toyWithDecHypNoise.py`)

I keep the VisitDecoder but make the forward noising process respect hyperbolic geometry whenever the embedding is hyperbolic.

- Treat \(x_0\) as tangent vectors at the origin.
- Map both \(x_0\) and \(\epsilon\) onto the manifold via `manifold.expmap0`.
- Use Möbius scalar multiplication and addition to approximate
  \[
  x_t \approx \sqrt{\bar{\alpha}_t}\otimes x_0 \;\oplus\; \sqrt{1-\bar{\alpha}_t}\otimes \epsilon.
  \]
- Reverse step stays in the tangent space but forward path travels along geodesics. Euclidean embeddings still use standard Gaussian noise.
- Reconstruction loss remains the same BCE on decoded logits from \(x_0\) and \(\hat{x}_0\).

**Sketch**

```
codes --[Hyperbolic embedding B_c^d]--> VisitEncoder (logmap0 + mean) --> x0 (tangent)
     --> hyperbolic forward noise (expmap0 + Möbius) --> x_t
     --> eps_model --> ε̂_t --> x̂0 --> VisitDecoder --> logits/top-K
```

**What I observed.**
- Hyperbolic models hit near-perfect tree-distance correlations (~0.97–0.99).
- Recall@4 stayed terrible (≈0.01–0.15) in deep hierarchies, whereas Euclidean stayed around 0.14–0.24.
- Reasons: the decoder is still a tiny Euclidean MLP, latent shells collapse under regularization, and diffusion supervision lives in tangent space while my metrics live on the manifold.

### 3. Hyperbolic Diffusion Distance (HDD) Regularizer (`train_toy_hdd.py`)

I keep the hyperbolic noise + decoder setup and add an HDD loss inspired by diffusion distances on the ICD graph.

- Build Laplacian \(L = D - A\) from the ICD tree and precompute \(\Phi_{t_k}=\exp(-t_k L)\).
- Stack/normalize to get diffusion embeddings \(\phi(i)\) for each code and define
  \[
  d_{\text{HDD}}(i,j)=\|\phi(i)-\phi(j)\|_2.
  \]
- HDD loss:
  \[
  \mathcal{L}_{\text{HDD}}=\mathbb{E}_{i,j}\big[(d_{\text{emb}}(i,j)-d_{\text{HDD}}(i,j))^2\big].
  \]
- Total loss becomes
  \[
  \mathcal{L}=\mathcal{L}_{\epsilon}
  +\lambda_{\text{tree}}\mathcal{L}_{\text{pair}}
  +\lambda_{\text{radius}}\mathcal{L}_{\text{radius}}
  +\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
  +\lambda_{\text{HDD}}\mathcal{L}_{\text{HDD}}.
  \]

**Sketch**

```
ICD tree --> diffusion embeddings φ(i) --> HDD loss
codes --> code_emb (hyperbolic/Euclid) --> VisitEncoder --> DDPM --> VisitDecoder --> codes
```

**Outcome.**
- Hyperbolic + HDD nailed the structure metrics but recall stayed ~0.08 (shallow) and ~0.01 (deep).
- Euclidean + HDD improved modestly but still beat hyperbolic in recall.
- The decoder/encoder mismatch remained the bottleneck.

### 4. Hyperbolic Graph Diffusion (HGD) Regularizer (`train_toy_hgd.py`)

Next I added the HGD idea: use a diffusion kernel \(K_t=\exp(-tL)\) to align manifold distances with graph diffusion similarities.

- Convert \(K_t\) to a similarity/distance \(s_t(i,j)\).
- Loss:
  \[
  \mathcal{L}_{\text{HGD}}=\mathbb{E}_{i,j}\big[(d_{\text{emb}}(i,j)-s_t(i,j))^2\big].
  \]
- Objective:
  \[
  \mathcal{L}=\mathcal{L}_{\epsilon}
  +\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
  +\lambda_{\text{tree}}\mathcal{L}_{\text{pair}}
  +\lambda_{\text{radius}}\mathcal{L}_{\text{radius}}
  +\lambda_{\text{HGD}}\mathcal{L}_{\text{HGD}}.
  \]

**Sketch**

```
ICD graph --> diffusion kernel K_t --> HGD loss
codes --> embeddings --> VisitEncoder --> DDPM (hyperbolic noise when needed)
     --> VisitDecoder --> logits --> codes
```

**Outcome.**
- Structural alignment kept improving, but recall hardly moved for hyperbolic models.
- Euclidean still delivered better recall because the decoder “saw” something more linear.

### 5. Combined HDD + HGD (`train_toy_hdd_hgd.py`)

Finally I stacked both graph-based regularizers on top of the prior losses:
\[
\mathcal{L}=\mathcal{L}_{\epsilon}+\lambda_{\text{recon}}\mathcal{L}_{\text{recon}}
+\lambda_{\text{tree}}\mathcal{L}_{\text{pair}}
+\lambda_{\text{radius}}\mathcal{L}_{\text{radius}}
+\lambda_{\text{HDD}}\mathcal{L}_{\text{HDD}}
+\lambda_{\text{HGD}}\mathcal{L}_{\text{HGD}}.
\]

**Sketch**

```
ICD tree + graph --> {pair, radius, HDD, HGD}
codes --> heavily constrained embeddings --> VisitEncoder --> DDPM
     --> VisitDecoder --> logits --> codes
```

**Outcome.**
- Geometry became almost perfect (tree statistics, diffusion alignment), but hyperbolic recall on deep hierarchies still sat near zero.
- Euclidean retained recall around 0.14 yet never matched the structural stats.

### Where I use hyperbolic noise

- `train_toy.py`, `train_toyWithDecoder.py`: always Euclidean forward noise, even if embeddings are hyperbolic.
- `train_toyWithDecHypNoise.py`, `train_toy_hdd.py`, `train_toy_hgd.py`, `train_toy_hdd_hgd.py`: `_hyperbolic_forward_noise` kicks in whenever the embedding type is hyperbolic; Euclidean embeddings keep Gaussian noise.

### High-level diagnosis: why hyperbolic ≠ Euclidean in recall

Empirically:

- Hyperbolic + geometry losses → extremely high tree–embedding correlations.
- Hyperbolic → very low recall@K, especially on deeper hierarchies.
- Euclidean → lower structural correlation but consistently better recall@4.

Main culprits in my setup:

1. **Decoder expressivity.** A small Euclidean MLP struggles to invert mean-pooled, curved, heavily regularized hyperbolic embeddings.
2. **Encoder non-invertibility.** Mean pooling discards combinatorial information, so the decoder never sees enough signal to reconstruct sets exactly.
3. **Objective mismatch.** Geometry losses optimize pairwise structure, not discrete reconstruction accuracy. The only discrete supervision is BCE, which has to fight all other penalties.
4. **Depth-induced collapse.** Radius/depth + diffusion losses push many codes onto almost identical shells, good for structure but disastrous for distinguishing leaves.

So far: geometry-aware regularizers excel at shaping the manifold yet fail to make the latent space easily decodable into exact visits. To close the gap I likely need stronger set-aware encoders, decoders that respect manifold geometry, or objectives that directly penalize wrong code predictions rather than just mismatched distances.
