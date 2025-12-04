# Project Narrative and Technical Analysis

I am attempting to understand the underlying structure of patient trajectories—specifically, how sequences of medical visits evolve over time when mapped into a continuous latent space. The core hypothesis driving this work is that medical concepts (ICD codes) naturally form a hierarchy, and therefore, the latent space representing them should respect that geometry.

This document is a record of the system I have built. It captures the architecture, the mathematical foundations, and the iterative failures that led to the current design. It merges the original experimental log with the final architectural decisions. To contextualize these decisions, we first define the clinical structures and the geometric priors that shape them.

---

## Clinical Structure: ICD Codes and Visits

Electronic health records organize diagnostic information using standardized medical ontologies. In this work, we model patient trajectories using ICD-coded visit sequences, where each visit is a bag of diagnosis codes drawn from a hierarchical taxonomy.

### ICD Code Hierarchy

The International Classification of Diseases (ICD) is a multi-level ontology with the following structure:
*   Chapters (broad disease systems; e.g., Nervous system disorders)
*   Blocks (narrower regional groupings)
*   Categories
*   Subcategories (fine-grained clinical conditions)

Let $\mathcal{V}$ denote the set of codes. The ICD hierarchy can be expressed as a directed tree (or DAG)
$G = (V, E)$,
where edges represent parent–child “is-a” relationships. The depth and branching factor of this tree induce a natural geometric prior: clinically similar codes lie close in the tree, while unrelated codes lie far apart.

This ontological structure motivates the use of hyperbolic embeddings, where geodesic distance increases exponentially with radius, mirroring the exponential expansion of hierarchical taxonomies.

### Definition of a Visit

A visit corresponds to one encounter with the healthcare system—such as an admission, outpatient consultation, or emergency room event. Each visit is represented as a set of ICD diagnosis codes:
$$ \text{Visit}_t = \{ c_1, c_2, \dots, c_k \}, \qquad c_i \in \mathcal{V}. $$

Visits contain no inherent ordering among codes, which is why our encoder operates on code sets using tangent-space pooling, attention, or Einstein midpoint averaging.

### Patient Trajectories

A patient trajectory is a temporally ordered sequence of visits:
$$ \mathcal{T} = [\text{Visit}_1, \text{Visit}_2, \dots, \text{Visit}_T]. $$

Each trajectory reflects longitudinal disease evolution and captures both local co-occurrence patterns (within visits) and global transitions across time.

### Relevance for Generative Modeling

In the hyperbolic diffusion framework described earlier, ICD codes serve as the discrete tokens that must be reconstructed from continuous latent visit representations. The mapping
$$ \text{codes} \rightarrow \text{visit vector} \rightarrow \text{denoised latent} \rightarrow \text{codes} $$
is only approximate unless an explicit decoder is trained. The ICD hierarchy is further used as a structural signal for evaluating embedding geometry, synthetic trajectory realism, and the preservation of clinical semantics.

Hyperbolic geometry is expected to excel at capturing:
*   hierarchical separation between disease families,
*   depth-sensitive representations,
*   and exponential branching seen in deeper ICD levels.

This section clarifies the clinical entities underlying the toy ICD datasets and motivates the architectural decisions in the remainder of the document.

---

## Part 1: The Codebase

This section maps the physical structure of the `src/` directory. It is the "what" and the "how" of the system.

### `data_icd_toy.py`
*The synthetic ground truth.*
This file constructs a controlled environment where I can debug the model's understanding of hierarchy.
-   **`ToyICDHierarchy`**: A directed graph $G=(V,E)$ mimicking ICD chapters ($C_0 \dots C_4$ branching to leaves).
-   **`sample_toy_trajectories`**: Generates patient lives. Visits are biased mixtures of a "condition" cluster and random noise: $P(c) = 0.7\,\mathcal{U}(\mathcal{C}) + 0.3\,\mathcal{U}(V)$.
-   **`tree_distance`**: The metric of truth—shortest path in the undirected graph.

### `euclidean_embeddings.py` & `hyperbolic_embeddings.py`
*The geometric substrates.*
-   **`EuclideanCodeEmbedding`**: Standard lookup table.
-   **`HyperbolicCodeEmbedding`**: Parameters on the Poincaré ball $\mathbb{B}^d_c$.
-   **Visit encoders**:
    -   *Euclidean*: Mean pooling or attention.
    -   *Hyperbolic Graph Visit Encoder*: Stacks Einstein-midpoint pooling with light-weight diffusion kernels so that each visit latent is built from both local co-occurrence structure and graph context before being log-mapped to the tangent space. Rectified Flow v2 adds an explicit **tangent projection layer** (see `train_graph_hyperbolic_rectified2.py`, `tangent_proj = nn.Linear(...)`) so that decoder inputs stay aligned with the deeper risk head—this projection proved essential for stabile training when the decoder became deeper.

### `diffusion.py` & `hyperbolic_noise.py`
*The stochastic engines.*
-   **`cosine_beta_schedule`**: DDPM noise schedule.
-   **`TimeEmbedding`**: Sinusoidal embeddings for temporal grounding.
-   **`hyperbolic_noise.py`**: Utilities for the Diffusion variants. Implements Möbius scalar multiplication and addition to perform "forward noising" directly on the manifold (`hyperbolic_forward_noise`).

### `traj_models.py`
*The learners.*
-   **`TrajectoryEpsModel`**: The legacy DDPM predictor. It takes a noisy trajectory $x_t$ and predicts the noise $\epsilon$.
-   **`TrajectoryVelocityModel`**: The new Rectified Flow predictor. It takes a position $x_t$ in the tangent space and predicts the **velocity field** $v$ that transports noise to data along straight lines.

### `decoders.py`
*The interface layer.*
-   **`VisitDecoder`**: A simple MLP used in early baselines.
-   **`StrongVisitDecoder`**: A deep residual MLP (6 blocks) for the Euclidean pipeline. It forces the network to "unpack" combinatorial set information from a flat vector.
-   **`HyperbolicDistanceDecoder`**: A geometry-aware decoder. It computes logits based on the negative squared distance in the Poincaré ball:
    $$ \text{logits}(v, c) \propto - \frac{d_{\mathbb{B}}(v, c)^2}{\tau} $$
    This physically couples semantic probability with geometric proximity.

### `train_utils.py` & `losses.py`
*The optimization landscape.*
-   **`code_pair_loss`**: Forces embedding distances to correlate with tree distances.
-   **`focal_loss`**: Handles the extreme sparsity of medical codes (most codes are 0).
-   **`get_cosine_temperature_schedule`**: Anneals the temperature $\tau$ for the hyperbolic decoder (1.0 $\to$ 0.07), sharpening the distribution over time.

### `train_toy.py` (and variants)
*The training loops.*
-   **`train_toy.py`**: The original DDPM training script.
-   **`train_rectified_flow.py`**: The modern Rectified Flow training script (formerly `train_toy_archFix*.py`). It implements the flow matching loss and Euler integration sampling, supporting variable hierarchy depths via command-line arguments.

---

## Part 2: Extended Variants and Regularizers

*This section preserves the original research log. It documents the sequence of experiments (1-5) that attempted—and failed—to make Hyperbolic DDPM work via regularization alone.*

### Baseline Context
In `train_toy.py`, I trained a DDPM over visit vectors $x_0$ from mean-pooled embeddings. The decoding was heuristic (nearest neighbors).

**Loss (`src/train_toy.py`, `compute_batch_loss`)**
$$\mathcal{L}_{\text{toy}} = \mathbb{E}_{t,\epsilon}\!\left[\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert_2^2\right] + \lambda_{\text{tree}}\mathcal{L}_{\text{pair}} + \lambda_{\text{radius}}\mathcal{L}_{\text{radius}}$$
Here $\mathcal{L}_{\text{pair}}=\texttt{code\_pair\_loss}(\text{code\_emb}, \text{hier})$ keeps geodesic distances aligned with tree distances, while $\mathcal{L}_{\text{radius}}=\frac{1}{N}\sum_i(\lVert \mathbf{z}_i\rVert_2-d_i)^2$ penalizes deviation between each code vector's norm $\mathbf{z}_i$ and its depth target $d_i$.

### 1. Adding a Learnable Visit Decoder (`train_toyWithDecoder.py`)
I inserted a parametric visit decoder (`VisitDecoder`) to reconstruct code sets with supervised loss.
-   **Result**: Euclidean worked okay (Recall@4 $\approx$ 0.14–0.24). Hyperbolic lagged because the decoder (Euclidean MLP) couldn't invert the curved space geometry.

**Loss (`src/train_toyWithDecoder.py`, `compute_batch_loss`)**
$$\mathcal{L}_{\text{dec}} = \mathcal{L}_{\epsilon} + \lambda_{\text{tree}}\mathcal{L}_{\text{pair}} + \lambda_{\text{radius}}\mathcal{L}_{\text{radius}} + \lambda_{\text{recon}}\mathcal{L}_{\text{BCE}}$$
The new term $\mathcal{L}_{\text{BCE}}=\frac{1}{|M|C}\sum_{(b,l)\in M}\sum_{c=1}^{C}\operatorname{BCE}(\text{logits}_{blc}, y_{blc})$ averages multi-label BCE across the set $M$ of real visits (mask-true positions) and $C=|\mathcal{V}|$ codes, exactly as implemented when flattening logits/targets in `compute_batch_loss`.

### 2. Hyperbolic Forward Noise + Decoder (`train_toyWithDecHypNoise.py`)
I made the forward noising process respect the manifold.
-   **Mechanism**: Map $x_0$ and $\epsilon$ to the manifold, use Möbius addition: $x_t \approx \sqrt{\bar{\alpha}_t} \otimes x_0 \oplus \sqrt{1-\bar{\alpha}_t} \otimes \epsilon$.
-   **Observation**: Tree-distance correlation hit near-perfect levels (0.99), but Recall@4 was terrible ($\approx$ 0.01). The geometry was perfect, but the model couldn't encode/decode semantic sets.

**Loss (`src/train_toyWithDecHypNoise.py`, `compute_batch_loss`)**
$$\mathcal{L}_{\text{hyp-dec}} = \mathcal{L}_{\epsilon}^{\mathbb{B}} + \lambda_{\text{tree}}\mathcal{L}_{\text{pair}} + \lambda_{\text{radius}}\mathcal{L}_{\text{radius}} + \lambda_{\text{recon}}\mathcal{L}_{\text{BCE}}$$
The term $\mathcal{L}_{\epsilon}^{\mathbb{B}}$ follows the same MSE on predicted noise but the latent $x_t$ is produced with Möbius scalar multiplication/addition (`hyperbolic_forward_noise`, `hyperbolic_remove_noise`) before evaluating $\epsilon_\theta$.

### 3. Hyperbolic Diffusion Distance (HDD) (`train_toy_hdd.py`)
Added a regularizer based on diffusion on the ICD graph.
-   **Idea**: Embeddings should match graph diffusion distance: $d_{\text{emb}}(i,j) \approx \|\phi(i) - \phi(j)\|$.
-   **Result**: Structure improved, recall didn't.

**Loss (`src/train_toy_hdd.py`, `compute_batch_loss`)**
$$\mathcal{L}_{\text{HDD}} = \mathcal{L}_{\epsilon}^{\mathbb{B}} + \lambda_{\text{tree}}\mathcal{L}_{\text{pair}} + \lambda_{\text{radius}}\mathcal{L}_{\text{radius}} + \lambda_{\text{hdd}}\mathcal{L}_{\text{diff}}^{\text{HDD}} + \lambda_{\text{recon}}\mathcal{L}_{\text{BCE}}$$
Here $\mathcal{L}_{\text{diff}}^{\text{HDD}}=\texttt{hdd\_metric.embedding\_loss}(code\_emb)$ aligns pairwise Poincaré distances with diffusion distances from the ICD graph metric.

### 4. Hyperbolic Graph Diffusion (HGD) (`train_toy_hgd.py`)
Aligned manifold distances with a graph diffusion kernel $K_t$.
-   **Result**: Same story. Structure $\uparrow$, Utility $\leftrightarrow$.

**Loss (`src/train_toy_hgd.py`, `compute_batch_loss`)**
$$\mathcal{L}_{\text{HGD}} = \mathcal{L}_{\epsilon}^{\mathbb{B}} + \lambda_{\text{tree}}\mathcal{L}_{\text{pair}} + \lambda_{\text{radius}}\mathcal{L}_{\text{radius}} + \lambda_{\text{hgd}}\mathcal{L}_{\text{diff}}^{\text{HGD}} + \lambda_{\text{recon}}\mathcal{L}_{\text{BCE}}$$
where $\mathcal{L}_{\text{diff}}^{\text{HGD}}=\texttt{hgd\_metric.diffusion\_loss}(code\_emb)$ penalizes deviations from the heat-kernel similarities computed on the ICD graph.

### 5. Combined HDD + HGD (`train_toy_hdd_hgd.py`)
Stacked all regularizers.
-   **Result**: Geometry became rigid. Deep hierarchy recall sat near zero.

**Loss (`src/train_toy_hdd_hgd.py`, `compute_batch_loss`)**
$$
\mathcal{L}_{\text{HDD+HGD}} = \mathcal{L}_{\epsilon}^{\mathbb{B}} + \lambda_{\text{tree}}\mathcal{L}_{\text{pair}} + \lambda_{\text{radius}}\mathcal{L}_{\text{radius}} + \lambda_{\text{hdd}}\mathcal{L}_{\text{diff}}^{\text{HDD}} + \lambda_{\text{hgd}}\mathcal{L}_{\text{diff}}^{\text{HGD}} + \lambda_{\text{recon}}\mathcal{L}_{\text{BCE}}$$
Both diffusion penalties are accumulated exactly as implemented in `src/train_toy_hdd_hgd.py`, keeping manifold geometry tethered to graph diffusion while still supervising reconstruction.

---

## Part 3: Diagnosis and Solution

The failure of the previous variants revealed four fundamental "culprits" that prevented the hyperbolic model from outperforming the Euclidean baseline.

### The Four Fatal Culprits

| Culprit | Original Problem | Final Fix |
| :--- | :--- | :--- |
| **1. Decoder Expressivity** | A Euclidean MLP could not invert curved, regularized hyperbolic latents. | **Hyperbolic Distance Decoder**: Logits $\propto -d^2/\tau$. |
| **2. Encoder Non-invertibility** | Mean pooling lost combinatorial information. | **Einstein Midpoint**: Preserves set geometry in a reversible way. |
| **3. Objective Mismatch** | Geometry losses (HDD/HGD/Pair) crushed the discrete signal. | **Focal Loss** ($\lambda=1000$) dominates optimization; radius loss removed. |
| **4. Depth-induced Collapse** | Radius-depth regularization forced leaves to the same shell. | **Radius regularization removed**. Let the model organize itself. |

### The Pivot: Rectified Flow

In `train_rectified_flow.py` (previously `train_toy_archFix{2,7}.py`), I abandoned the noisy DDPM for **Rectified Flow**. This models the problem as finding a deterministic velocity field in the tangent space.

#### 1. Interpolation
For data $X_1$ and noise $X_0$, we define a straight path:
$$ X_t = (1 - t) X_0 + t X_1 $$
Velocity: $v = X_1 - X_0$.

#### 2. Flow Matching Loss
The model $v_\theta$ learns to predict this constant velocity:
$$ \mathcal{L}_{\text{flow}} = \| v_\theta(X_t, t) - (X_1 - X_0) \|^2 $$

#### 3. Sampling (Euler Integration)
$$ x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t $$
*Hyperbolic Constraint*: After each step in tangent space, we project to the manifold (`expmap0`) and back (`logmap0`) to strictly enforce the curvature constraints.

#### 4. Total Objective
$$
\mathcal{L}_{\text{total}}
= \mathcal{L}_{\text{flow}}
+ \lambda_{\text{recon}}\mathcal{L}_{\text{focal}}
+ \lambda_{\text{pair}}\mathcal{L}_{\text{pair}},
$$
mirroring the call to `compute_batch_loss` in `src/train_rectified_flow.py` where `\mathcal{L}_{\text{focal}}=\texttt{focal\_loss}(f_{\text{dec}}(x), y)` and $\mathcal{L}_{\text{pair}}$ reuses the DDPM code-pair penalty inside the rectified-flow loop.

This architecture combining the stability of Rectified Flow with the geometry-aware Einstein Encoder and Distance Decoder finally bridges the gap, allowing the hyperbolic model to capture both the hierarchy and the discrete set structure of the visits.

---

## Part 4: Empirical Evidence from `results/analysis/table0_6.md`

The sweep summarized in `results/analysis/table0_6.md` quantifies how Euclidean and hyperbolic rectified-flow models behave across shallow (depth2_final) and deep (depth7_final) hierarchies when $\lambda_{\text{recon}}$ is varied.

### Depth 2 (Shallow Hierarchy)
- **Euclidean**: Recall@4 improves monotonically with higher $\lambda_{\text{recon}}$ (0.07 at $\lambda=1$ up to 0.92 at $\lambda=1000$), but tree/embedding correlations remain near zero (best 0.05). The decoder simply learns to memorize sets without recovering the ICD tree.
- **Hyperbolic**: Small $\lambda$ (1–100) collapse trajectories toward the root (mean depth $\approx 1.1$–1.8) and can even generate negative correlations. Once $\lambda_{\text{recon}} \ge 1000$, Recall stays in the 0.57–0.60 range while correlation rises to ~0.62, matching the real-tree metric without any HDD/HGD regularizers. These runs are the first to prove that the Einstein encoder + distance decoder can preserve shallow hierarchies for free.

### Depth 7 (Deep Hierarchy)
- **Euclidean**: All runs are unstable. Even at $\lambda_{\text{recon}} = 1000$, Recall peaks at 0.45 and correlation bounces around zero. Synthetic visits clump on a few leaves (root purity drops to 0.50 and tree-distance variance collapses), mirroring the visual samples in the log.
- **Hyperbolic**: Low $\lambda$ (1–100) keep mean depth near the real value but correlation is negative, indicating that visits wander across unrelated branches. Increasing $\lambda_{\text{recon}}$ into the 1800–2500 range finally pushes Recall past 0.38 and correlation above 0.13, yet the latent trajectories overshoot the hierarchy (mean depth as high as 6.4 with low variance). Extremely large weights (3000–5000) maximize Recall (0.48–0.53) at the expense of correlation (down to −0.13) and produce synthetic visits tightly concentrated in the deepest leaves. These numbers explain why the plain rectified-flow objective still needs auxiliary geometric guidance for very deep trees.

### Takeaways
1. **$\lambda_{\text{recon}}$ drives reconstruction, not structure.** High values are necessary to train the decoder but do not guarantee tree fidelity—especially in Euclidean space.
2. **Hyperbolic latents are necessary but not sufficient.** They carry the hierarchy in depth-2 experiments, yet in depth-7 they oscillate between under- and over-shooting the taxonomy depending on $\lambda_{\text{recon}}$.
3. **Geometric regularizers are still useful.** The diffusion-based `train_toy_withDecHypNoise.py` (with Möbius noise + decoder loss) achieves nearly perfect correlation because it blends stochastic gradients with explicit manifold operations. Rectified-flow models need additional priors (HDD/HGD or radius targets) to match that behavior at scale.

These outcomes motivated the introduction of the **HyperbolicGraphVisitEncoder** and the tangent projection layer in the later scripts: the encoder diffuses code embeddings before Einstein pooling so that visit latents evolve smoothly across hierarchy depth, while the projection keeps decoder inputs consistent even when the decoder becomes deeper (Rectified Flow v2). Tables 0.6–0.8 show that without these additions, deep hierarchies immediately collapse to a handful of leaves.

These observations close the loop between the architectural motivations above and the quantitative evidence in `table0_6.md`: they show precisely where the current hyperbolic rectified-flow pipeline excels (shallow hierarchies) and where further modeling work is required (deep ICD trees).

## Part 5: Rectified vs. Graph-DDPM Ablations (Tables 0.7 & 0.8)

`table0_7.md` juxtaposes four training scripts:

1. **Rectified Flow v1** (`train_graph_hyperbolic_rectified.py`). With Euclidean noise $X_t=(1-t)X_0+tX_1$, the loss
   $$\mathcal{L}_{\text{RF}}^{(1)} = \|v_\theta(X_t,t)-(X_1-X_0)\|_2^2 + \lambda_{\text{recon}}\mathcal{L}_{\text{focal}} + \lambda_{\text{pair}}\mathcal{L}_{\text{pair}}$$
   explains the depth-2 sweet spot (recall 0.57, corr 0.62 at $\lambda_{\text{recon}}=2000$) and depth-7 degradation recorded in the comparison table.

2. **Rectified Flow v2** (`train_graph_hyperbolic_rectified2.py`). Latents are mapped to the Poincaré ball and noise is sampled via $x_0=\exp_0(\epsilon)$. Interpolation takes place in log space,
   $$z_t=(1-t)\log_0(x_0)+t\log_0(x_1), \qquad v_\theta(z_t,t)\approx\log_0(x_1)-\log_0(x_0),$$
   giving the loss
   $$\mathcal{L}_{\text{RF}}^{(2)}=\|v_\theta(z_t,t)-(\log_0(x_1)-\log_0(x_0))\|_2^2 + \lambda_{\text{recon}}\mathcal{L}_{\text{focal}}.$$
   Coupled with the deeper decoder in that script, this produces the improved depth-2 row in `table0_8.md` (recall 0.133, corr 0.885).

3. **Graph DDPM** (`train_graph_hyperbolic_gd.py`). The diffusion loss
   $$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t, t)\|_2^2 + \lambda_{\text{recon}}\mathcal{L}_{\text{focal}} + \lambda_{\text{pair}}\mathcal{L}_{\text{pair}}$$
   operates on Einstein-pooled latents, yielding corr $\ge 0.89$ but recall stuck near 0.041 (see `table0_8.md`).

4. **HG-DDPM** (`train_graph_hyperbolic_gdrect.py`). The same loss is evaluated after graph-diffusion kernels and global attention. Even so, the rows in `table0_7.md`–`table0_8.md` keep recall below 0.034, underscoring the decoder bottleneck.

Across all DDPM rows, increasing $\lambda_{\text{recon}}$ to 2000 never pushes depth-7 recall beyond 0.02, whereas the rectified variants deliver the best geometry/recall compromise.

The encoder/decoder upgrades appear throughout these ablations: rectified v1 already benefits from Einstein pooling, rectified v2 adds the tangent projection plus a deeper risk decoder, and HG-DDPM injects graph attention prior to pooling. Each change is a direct response to the failure modes in Part 4: without richer encoders the depth-7 hierarchy either collapses (Euclidean runs) or becomes numerically unstable (hyperbolic runs). Tables 0.7 and 0.8 therefore double as a chronicle of why each architectural tweak was adopted.

## Part 6: Cross-Table Synthesis (Tables 0.1–0.8)

Reading the comparison tables reveals the empirical laws behind the project:

1. **Euclidean reconstruction scaling (Tables 0.1–0.6).** Doubling $\lambda_{\text{recon}}$ raises Recall@4 by ~5 points in `table0_1.md`–`table0_2.md` while correlation stays near zero, proving that Euclidean models simply memorise code sets.
2. **Depth scaling for hyperbolic latents (Tables 0.1–0.7).** Hyperbolic encoders keep $r_{\text{tree,emb}} \ge 0.9$, yet recall plunges from 0.57 (depth-2 rows in `table0_7.md`) to <$0.05$ (depth-7 rows in `table0_1.md`–`table0_5.md`) because the decoder cannot cover exponentially branching codes.
3. **Regulariser stacking (Tables 0.2–0.5).** Adding $\lambda_{\text{hdd}}\,\mathcal{L}_{\text{diff}}^{\text{HDD}} + \lambda_{\text{hgd}}\,\mathcal{L}_{\text{diff}}^{\text{HGD}}$ (see `table0_3.md`–`table0_5.md`) tightens geometry but suppresses recall unless decoder capacity increases.
4. **Rectified-flow resilience (Tables 0.6–0.8).** The Einstein encoder + distance decoder + $\mathcal{L}_{\text{RF}}$ combo scales cleanly at depth 2 (`table0_6.md`, `table0_7.md`), while DDPM pipelines need HDD/HGD or global attention to avoid collapse in the depth-7 rows of `table0_7.md`–`table0_8.md`.

### Detailed experiment log (Tables 0.1–0.8)
- **Table 0.1 (`table0_1.md` comparison):** Euclidean recall 0.24 vs. hyperbolic corr 0.99 without diffusion noise.
- **Table 0.2 (`table0_2.md` comparison):** Hyperbolic forward noise maintains Euclidean scaling but collapses hyperbolic radii.
- **Table 0.3 (`table0_3.md` comparison):** HDD raises corr to 0.98 yet recall only to 0.083.
- **Table 0.4 (`table0_4.md` comparison):** HGD mirrors HDD; Euclidean recall stays 0.244.
- **Table 0.5 (`table0_5.md` comparison):** HDD+HGD nudges hyperbolic recall to 0.145 but leaves depth-7 decoding unsolved.
- **Table 0.6 (`table0_6.md` comparison):** Rectified-flow sweeps balance recall and structure for depth 2.
- **Table 0.7 (`table0_7.md` comparison):** Large $\lambda_{\text{recon}}$ sweeps expose the limits of both rectified and DDPM models on deep hierarchies.
- **Table 0.8 (`table0_8.md` comparison):** Aggregates every pipeline, confirming that Rectified Flow v2 leads the geometry/recall compromise while Graph DDPM and HG-DDPM remain recall-poor.

These references justify the focus on rectified-flow objectives with hyperbolic latents: only those models scale from shallow ICD toy hierarchies to diffusion-heavy pipelines while keeping hierarchy fidelity and predictive accuracy in balance.

In short, the new encoders and tangent projections were not cosmetic upgrades—they are the mechanical pieces that finally reconciled reconstruction accuracy with hierarchical fidelity. Every table documents one step in that progression, from Euclidean mean pooling (Table 0.1) to Einstein pooling (Table 0.6) to graph-aware, tangent-projected encoders (Tables 0.7–0.8). The analysis above is therefore as much about the evolving codebase as it is about the resulting numbers.
