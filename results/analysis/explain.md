# Project Narrative and Technical Analysis

I am attempting to understand the underlying structure of patient trajectories—specifically, how sequences of medical visits evolve over time when mapped into a continuous latent space. The core hypothesis driving this work is that medical concepts (ICD codes) naturally form a hierarchy, and therefore, the latent space representing them should respect that geometry.

This document is my attempt to map out the system I have built. It is a record of the architecture, the mathematical foundations, and the design choices I made while trying to force a neural network to "think" in hyperbolic space.

---

## 1. The Synthetic World: `data_icd_toy.py`

Before tackling real data, I needed a controlled environment. I cannot debug a model if I do not understand the ground truth it is trying to learn. So, I built a synthetic generator.

**`ToyICDHierarchy`**
The foundation is a directed graph $G=(V,E)$ that mimics the structure of ICD chapters.
-   **Roots:** I start with broad categories ($C_0 \dots C_4$).
-   **Expansion:** These branch into increasingly specific nodes ($C_{ij} \rightarrow C_{ijk}$).
-   **Verification:** I maintain explicit maps for $\text{idx}(c)$ and $\text{depth}(c)$ to ensure I can always trace a code back to its origin.

**`sample_toy_trajectories`**
This is where I generate the "lives" of my synthetic patients.
1.  **Trajectory Length:** Sample a duration $T$.
2.  **Cluster Assignment:** Assign the patient a "condition" (a cluster leaf $\ell$). Their visits will not be random; they will be biased toward this cluster and its ancestors.
3.  **Visit Generation:** For each timestep, I construct a visit by drawing codes from a mixture distribution:
    $$ P(c) = 0.7 \, \mathcal{U}(\mathcal{C}) + 0.3 \, \mathcal{U}(V) $$
    This balances structure (the condition) with noise (random ailments), creating a signal-to-noise ratio that challenges the model.

**`tree_distance`**
To measure success, I need a metric independent of the embedding. I use the shortest path in the undirected graph $G$. This allows me to ask: *Does the model understand that pneumonia and bronchitis are closer than pneumonia and a broken leg?*

---

## 2. The Geometry of Latent Space

The choice of geometry is the choice of constraints. Euclidean space is flat and forgiving; Hyperbolic space is exponential and opinionated. I implemented both to isolate the effect of curvature.

**`euclidean_embeddings.py`**
-   **`EuclideanCodeEmbedding`:** A standard lookup table.
-   **`EuclideanVisitEncoder`:** A simple mean-pooling operation. It is robust, but it discards the hierarchical relationships inherent in the data.

**`hyperbolic_embeddings.py`**
Here, I enforce the geometry of the Poincaré ball $\mathbb{B}^d_c$.
-   **`HyperbolicCodeEmbedding`:** Parameters are `geoopt.ManifoldParameter`. I initialize them with a projection `projx` to ensure they start strictly within the ball (norm $< 1/\sqrt{c}$).
-   **`VisitEncoder` (Hyperbolic):** This is where it gets interesting. I cannot just average vectors in curved space; the midpoint is not the linear average.
    1.  Map codes to the tangent space at the origin via $\log_0$.
    2.  Compute the mean in the tangent space (a simplified approximation of the Fréchet mean).
    3.  This results in a visit vector $v$ that lives in the tangent space, ready for the diffusion process.

---

## 3. The Interface Layer: Encoders and Decoders

The hardest part of this system is the bridge between the discrete set of codes and the continuous latent manifold. It feels like trying to translate poetry into calculus—something is always lost in the conversion.

### Euclidean Pipeline
-   **Encoder (`LearnableVisitEncoder`):** I use a learnable pooling module. It can employ an attention mechanism (`use_attention=True`) to weigh codes dynamically. This acknowledges that not all codes in a visit are equally important.
-   **Decoder (`StrongVisitDecoder`):** A deep residual MLP (6 layers, 512 dim). I made this deep because the Euclidean space lacks the intrinsic capacity to separate hierarchies efficiently; the network has to "memorize" the separation.

### Hyperbolic Pipeline
-   **Encoder (`HyperbolicVisitEncoder`):** I use the **Einstein Midpoint** (hyperbolic barycenter). This respects the manifold structure.
-   **Decoder (`HyperbolicDistanceDecoder`):** This is a geometry-aware decoder. Instead of an arbitrary projection, it calculates probabilities based on physical distance in the manifold:
    $$ \text{logits}(v, c) \propto - \frac{d_{\mathbb{B}}(v, c)^2}{\tau} $$
    where $\tau$ is a temperature parameter. This forces the latent visit vector to be geometrically close to its constituent codes. It fundamentally couples the latent position with semantic meaning.

---

## 4. The Generative Engine: Rectified Flow

In earlier iterations, I used a Denoising Diffusion Probabilistic Model (DDPM). It was noisy and slow. I have since shifted to **Rectified Flow** (implemented in `train_toy_archFix{2,7}.py`), which feels more like modeling a direct transport problem rather than a stochastic walk.

**The Core Concept**
Instead of predicting noise $\epsilon$, I predict the *velocity* field that transports the noise distribution $\pi_0$ to the data distribution $\pi_1$ along straight lines.

**Interpolation**
For a data sample $X_1$ and noise $X_0$, the path is linear:
$$ X_t = (1 - t) X_0 + t X_1 $$
The velocity is constant:
$$ v = \frac{dX_t}{dt} = X_1 - X_0 $$

**Training Objective**
The model $v_\theta$ learns to match this target velocity. The loss is cleaner, more direct:
$$ \mathcal{L}_{\text{flow}} = \| v_\theta(X_t, t) - (X_1 - X_0) \|^2 $$

**Sampling via Euler Integration**
To generate a patient trajectory, I solve the ODE numerically:
$$ x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t $$
*Constraint:* For hyperbolic embeddings, I must ensure the trajectory stays on the manifold. After each step in the tangent space, I project to the manifold and re-map to the tangent space (`expmap0` then `logmap0`) to correct numerical drift.

---

## 5. Optimization Landscape: `losses.py` & `train_toy.py`

Training this system requires balancing conflicting objectives. It feels like managing a process with unstable gradients—if I push too hard on structure, reconstruction fails; if I focus only on reconstruction, the geometry collapses.

**The Loss Function**
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{focal}} + \lambda_{\text{tree}} \cdot \mathcal{L}_{\text{pair}} $$

1.  **`rectified_flow_loss`:** The engine driving the generative process.
2.  **`focal_loss`:** A reconstruction term handling class imbalance. Standard BCE was insufficient because medical codes are sparse; the model learned to predict "no code" everywhere. Focal loss forces it to focus on the hard positives.
3.  **`code_pair_loss`:** A structural regularizer. I sample code pairs $(i, j)$ and force their embedding distance $d_{\text{emb}}$ to correlate with their tree distance $d_{\text{tree}}$.
    $$ \mathcal{L}_{\text{pair}} = \| \text{zscore}(d_{\text{emb}}) - \text{zscore}(d_{\text{tree}}) \|^2 $$
    This aligns the manifold with the ontology.

---

## 6. File Structure Analysis

To orient myself, here is how the codebase decomposes:

**Core Logic**
-   **`data_icd_toy.py`:** The synthetic generator.
-   **`diffusion.py`:** Legacy DDPM schedules and Time Embeddings.
-   **`traj_models.py`:** Contains `TrajectoryVelocityModel` (for Rectified Flow) and `TrajectoryEpsModel` (for DDPM).
-   **`hyperbolic_noise.py`:** Utilities for Möbius arithmetic and tangent space operations.

**Embeddings**
-   **`euclidean_embeddings.py`** / **`hyperbolic_embeddings.py`**: The distinct geometric substrates.

**Training & Execution**
-   **`train_toy_archFix{2,7}.py`:** The modern training scripts using Rectified Flow.
-   **`train_toy.py`:** The older DDPM training script.
-   **`train_utils.py`:** Helpers for temperature annealing (`get_cosine_temperature_schedule`) and frequency bias.

**Utilities**
-   **`decoders.py`:** The architectural definitions of `VisitDecoder`, `StrongVisitDecoder`, and `HyperbolicDistanceDecoder`.
-   **`losses.py`:** Definitions of `code_pair_loss` and `focal_loss`.
-   **`data_utils.py`:** Dataset handling (`TrajDataset`, `make_collate_fn`).
-   **`visit_utils.py`:** Metrics and decoding logic (`decode_visit_vectors`, `visits_from_indices`).

---

## 7. Reflection on Architecture

In comparing the Rectified Flow architecture (`train_toy_archFix*.py`) to the previous Hyperbolic DDPM (`train_toyWithDecHypNoise.py`), the shift is palpable. The DDPM felt like fighting the geometry—forcing curved noise injection steps (`hyperbolic_forward_noise`) and struggling with variance. Rectified Flow simplifies the problem to learning a vector field in the tangent space.

The removal of the radius-depth regularization was also a critical pivot. I initially thought I needed to force the hierarchy explicitly (deep nodes = large radius). But that was too rigid; it collapsed the leaves onto a single shell, making them indistinguishable. By removing it and relying on `focal_loss` and `code_pair_loss`, I allowed the model to find its own organization. It is a reminder that sometimes, the best way to build structure is to stop forcing it.
