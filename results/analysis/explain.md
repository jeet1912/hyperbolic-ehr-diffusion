# Project Narrative and Technical Analysis

I am attempting to understand the underlying structure of patient trajectories—specifically, how sequences of medical visits evolve over time when mapped into a continuous latent space. The core hypothesis driving this work is that medical concepts (ICD codes) naturally form a hierarchy, and therefore, the latent space representing them should respect that geometry.

This document is a record of the system I have built. It captures the architecture, the mathematical foundations, and the iterative failures that led to the current design. It merges the original experimental log with the final architectural decisions.

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
-   **`VisitEncoder`**:
    -   *Euclidean*: Mean pooling or attention.
    -   *Hyperbolic*: **Einstein Midpoint** (barycenter) followed by a log-map to the tangent space.

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
-   **`train_toy_archFix{2,7}.py`**: The modern Rectified Flow training scripts. They implement the flow matching loss and Euler integration sampling.

---

## Part 2: Extended Variants and Regularizers

*This section preserves the original research log. It documents the sequence of experiments (1-5) that attempted—and failed—to make Hyperbolic DDPM work via regularization alone.*

### Baseline Context
In `train_toy.py`, I trained a DDPM over visit vectors $x_0$ from mean-pooled embeddings. The decoding was heuristic (nearest neighbors).

### 1. Adding a Learnable Visit Decoder (`train_toyWithDecoder.py`)
I inserted a parametric visit decoder (`VisitDecoder`) to reconstruct code sets with supervised loss.
-   **Loss**: $\mathcal{L} = \mathcal{L}_{\epsilon} + \mathcal{L}_{\text{pair}} + \mathcal{L}_{\text{radius}} + \lambda_{\text{recon}}\operatorname{BCE}(f_{\text{dec}}(x_0), y)$.
-   **Result**: Euclidean worked okay (Recall@4 $\approx$ 0.14–0.24). Hyperbolic lagged because the decoder (Euclidean MLP) couldn't invert the curved space geometry.

### 2. Hyperbolic Forward Noise + Decoder (`train_toyWithDecHypNoise.py`)
I made the forward noising process respect the manifold.
-   **Mechanism**: Map $x_0$ and $\epsilon$ to the manifold, use Möbius addition: $x_t \approx \sqrt{\bar{\alpha}_t} \otimes x_0 \oplus \sqrt{1-\bar{\alpha}_t} \otimes \epsilon$.
-   **Observation**: Tree-distance correlation hit near-perfect levels (0.99), but Recall@4 was terrible ($\approx$ 0.01). The geometry was perfect, but the model couldn't encode/decode semantic sets.

### 3. Hyperbolic Diffusion Distance (HDD) (`train_toy_hdd.py`)
Added a regularizer based on diffusion on the ICD graph.
-   **Idea**: Embeddings should match graph diffusion distance: $d_{\text{emb}}(i,j) \approx \|\phi(i) - \phi(j)\|$.
-   **Result**: Structure improved, recall didn't.

### 4. Hyperbolic Graph Diffusion (HGD) (`train_toy_hgd.py`)
Aligned manifold distances with a graph diffusion kernel $K_t$.
-   **Result**: Same story. Structure $\uparrow$, Utility $\leftrightarrow$.

### 5. Combined HDD + HGD (`train_toy_hdd_hgd.py`)
Stacked all regularizers.
-   **Result**: Geometry became rigid. Deep hierarchy recall sat near zero.

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

In `train_toy_archFix{2,7}.py`, I abandoned the noisy DDPM for **Rectified Flow**. This models the problem as finding a deterministic velocity field in the tangent space.

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
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{focal}} $$

This architecture—combining the stability of Rectified Flow with the geometry-aware Einstein Encoder and Distance Decoder—finally bridges the gap, allowing the hyperbolic model to capture both the hierarchy and the discrete set structure of the visits.
