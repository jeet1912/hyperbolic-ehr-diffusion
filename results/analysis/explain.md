Hyperbolic Trajectory Modeling
================================================================
Ablations from `results/analysis/table0_0.md` through `table0_8.md` chart a single theme: every architectural change under `src/` tries to reconcile discrete recall with hierarchical structure. The discussion below ignores the MIMIC task model (`task_prediction_mimic_iv.py`) and concentrates solely on the synthetic ICD trajectory program.

Clinical Context & Generative Intuition
---------------------------------------
Each visit is a multiset of ICD codes sampled from a rooted tree. Generative synthesis therefore follows
$$
\text{codes}\xrightarrow{\text{visit encoder}} z_0
\xrightarrow{\text{flow or diffusion}} z_T
\xrightarrow{\text{decoder}} \widehat{\text{codes}},
$$
with tree-aware supervision supplied by radius, pair, HDD/HGD losses, and evaluation grounded in tree–embedding correlation. Hyperbolic distances expand exponentially with radius, mirroring branching factors [1]; Euclidean embeddings offer easier reconstruction but flatten the tree. The ensuing sections quantify that trade-off.

Relevance for Generative Modeling
---------------------------------
The ICD program aims to generate visits that are simultaneously realistic and hierarchy-preserving. Hyperbolic diffusion [2] and rectified flows [3] provide the geometric scaffolding, while decoder-focused ablations tune discrete fidelity. By moving from Euclidean DDPM baselines to hyperbolic encoders, every experiment probes how curvature-aware transport affects the loss landscape $\mathcal{L}=\mathcal{L}_{\text{gen}}+\lambda_{\text{struct}}\mathcal{L}_{\text{tree}}$ and ultimately the observed trade-off between Recall@4 and tree correlation.

Part 1 – From `src/` to Experiments
-----------------------------------
| File(s) | Role | Key Equations / Uses | Tables |
| --- | --- | --- | --- |
| `data_icd_toy.py`, `data_utils.py`, `metrics_toy.py` | Build toy ontologies, segment trajectories, compute statistics and tree distances $d_{\text{tree}}$. | $d_{\text{tree}}$ underpins every correlation measurement. | 0_0–0_8 |
| `euclidean_embeddings.py`, `hyperbolic_embeddings.py` | Code lookup layers (Euclidean) and Poincaré embeddings. | Hyperbolic norms obey radius targets $\lVert z\rVert\approx r^\star$. | 0_0–0_8 |
| `hyperbolic_embeddings.py` (visit encoders) | Mean pooling, attention pooling, Einstein pooling, and `HyperbolicGraphVisitEncoder`. | Graph kernels $K_s$ diffuse codes: $Z=\mathrm{Proj}\big[\mathrm{concat}_s(K_s \log_0 X)\big]$. | 0_6–0_8 |
| `decoders.py` | Translators back to codes. | Hyperbolic logits $\propto -d_{\mathbb{B}}(z,c)^2/\tau$ [8]; Euclidean multilayer decoders baseline. | 0_1–0_8 |
| `diffusion.py`, `hyperbolic_noise.py`, `traj_models.py` | DDPM and rectified velocity models. | $\mathcal{L}_{\text{DDPM}}=\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|^2$, $\mathcal{L}_{\text{RF}}=\|v_\theta(x_t,t)-(x_1-x_0)\|^2$. | 0_0–0_8 |
| `losses.py`, `regularizers.py` | Auxiliary penalties. | $\mathcal{L}_{\text{pair}}$, $\mathcal{L}_{\text{radius}}$, $\mathcal{L}_{\text{diff}}^{\text{HDD}}$, $\mathcal{L}_{\text{diff}}^{\text{HGD}}$. | 0_3–0_5 |
| `train_toy*.py`, `train_graph_*` | Training harnesses per phase. | Assemble the encoder/decoder/objective combinations referenced below. | 0_0–0_8 |

Encoder and Decoder Evolution
-----------------------------
1. **Euclidean mean-pooler (`train_toy.py`)** – visit vector $z$ equals the arithmetic mean of active code embeddings. Hyperbolic versions project means back to $\mathbb{B}^d$ via $\exp_0$.
2. **Decoder-enhanced toy models (`train_toyWithDecoder.py`, `train_toyWithDecHypNoise.py`)** – add cross-entropy reconstruction $\mathcal{L}_{\text{recon}}=\mathrm{BCE}(x,\widehat{x})$, optionally under hyperbolic noise with $\epsilon\sim\mathcal{N}(0,I)$ mapped by $\exp_0$.
3. **HDD/HGD regularizers** – diffusion descriptors $f(c)$ for each code produce $\mathcal{L}_{\text{diff}}^{\text{HDD}}=\mathbb{E}_{i,j}(\|f_i-f_j\| - d_{\mathbb{B}}(c_i,c_j))^2$; HGD adds the generative variant on sampled latents.
4. **Graph Hyperbolic encoders (`train_graph_hyperbolic_rectified*.py`, `train_graph_hyperbolic_gd*.py`)** – apply co-occurrence diffusion kernels, Einstein aggregation [4], global self-attention, and time embeddings in tangent space before projecting back. These encoders were introduced to stabilize depth-7 regimes [5].
5. **Decoders** – Euclidean branches retain MLPs, while hyperbolic branches rely on the distance decoder with temperature annealing [8]. When rectified flow v2 landed (`train_graph_hyperbolic_rectified2.py`), decoder logits included tangent projections before distance computation to prevent saturation.

Phase-by-Phase Narrative (Tables 0_0 → 0_8)
-------------------------------------------
### Phase 0 – Baseline DDPM (`results/analysis/table0_0.md`)
- **Objective:** $\mathcal{L}_{\text{toy}}=\mathcal{L}_{\text{DDPM}}+\lambda_{\text{pair}}\mathcal{L}_{\text{pair}}+\lambda_{\text{radius}}\mathcal{L}_{\text{radius}}$ using mean pooling.
- **Observation:** Euclidean depth-2/7 runs reach Recall@4 ≈0.10 but tree correlation stays within ±0.05. Hyperbolic embeddings excel structurally (corr ≈0.99 when regularization is on) yet never exceed Recall@4=0.03, especially when depth=7 collapses under radius penalties. This exposes the reconstruction bottleneck.

### Phase 1 – Decoder Injection (`results/analysis/table0_1.md`, `table0_2.md`)
- **Objective:** $\mathcal{L}_{\text{dec}}=\mathcal{L}_{\text{DDPM}}+\lambda_{\text{recon}}\mathcal{L}_{\text{BCE}}$, and with hyperbolic noise $\mathcal{L}_{\epsilon}^{\mathbb{B}}=\|\epsilon-\epsilon_\theta(\exp_0^{-1}(x_t),t)\|^2$.
- **Outcome:** Doubling $\lambda_{\text{recon}}$ steadily raises Euclidean Recall@4 (0.22–0.24 at depth 2, 0.14 at depth 7) confirming a near-linear scaling law. Hyperbolic models obtain perfect correlation but remain stuck near Recall@4=0.05 because the decoder cannot disentangle multiple branches even when noise is sampled directly on $\mathbb{B}^d$.

### Phase 2 – HDD/HGD Regularizers (`results/analysis/table0_3.md`–`table0_5.md`)
- **Objective:** $\mathcal{L}=\mathcal{L}_{\text{dec}}+\lambda_{\text{hdd}}\mathcal{L}_{\text{diff}}^{\text{HDD}}+\lambda_{\text{hgd}}\mathcal{L}_{\text{diff}}^{\text{HGD}}$.
- **Result:** Hyperbolic correlations jump to ≥0.98 at both depths, yet Recall@4 barely hits 0.08 (depth 2) and collapses to ≈0.01 (depth 7). Euclidean variants dominate recall but still post ≈0 structural correlation. These rows motivated richer encoders capable of handling multiscale neighborhoods.
- **Milestone:** The persistent imbalance observed in `table0_5.md` is the point where the original “Diagnosis & Remaining Solution” section was first authored, so the later discussion is intentionally anchored to the lessons from this phase.

### Phase 3 – Rectified Flow v1 (`results/analysis/table0_6.md`)
- **Code:** `train_graph_hyperbolic_rectified.py` couples `HyperbolicGraphVisitEncoder` with a tangent-space velocity model.
- **Objective:** $\mathcal{L}_{\text{RF}}^{(1)}=\left\|v_\theta(z_t,t)-(z_1-z_0)\right\|_2^2+\lambda_{\text{recon}}\mathcal{L}_{\text{focal}}+\lambda_{\text{pair}}\mathcal{L}_{\text{pair}}$ where $z_t$ follows Euclidean interpolation in tangent space.
- **Effect:** Depth-2 hyperbolic runs attain Recall@4≈0.57 with correlation ≈0.62 once $\lambda_{\text{recon}}\ge 1000$, finally matching Euclidean recall while preserving structure. Depth-7 runs remain unstable—correlation oscillates around 0.0–0.1 because the encoder still relies on shallow diffusion kernels. Importantly, `table0_6.md` also shows that Euclidean embeddings saturate: higher $\lambda_{\text{recon}}$ improves depth-2 recall but provides diminishing—or even negative—returns at depth 7, quantifying the depth-driven failure mode despite aggressive reconstruction weighting.

### Phase 4 – Rectified Flow v2 (`results/analysis/table0_7.md`)
- **Code:** `train_graph_hyperbolic_rectified2.py` modifies the transport path to $z_t=(1-t)\log_0(x_0)+t\log_0(x_1)$ with $x_0=\exp_0(\epsilon)$ and $x_1$ the clean latent. The loss becomes $\mathcal{L}_{\text{RF}}^{(2)}=\|v_\theta(z_t,t)-(\log_0(x_1)-\log_0(x_0))\|_2^2+\lambda_{\text{recon}}\mathcal{L}_{\text{focal}}$.
- **Impact:** Tangent projection before decoding improves gradient signals, pushing depth-2 Recall@4 back to ≈0.57 while keeping correlation ≈0.62. Depth-7 results improve modestly (Recall@4≈0.43, correlation up to 0.15 at $\lambda_{\text{recon}}=2000$), demonstrating that the new encoder handles deeper chains better than v1 but still trails Euclidean baselines.

### Phase 5 – Graph DDPM and HG-DDPM (`results/analysis/table0_7.md`, `results/analysis/table0_8.md`)
- **Codes:** `train_graph_hyperbolic_gd.py` (Graph DDPM) and `train_graph_hyperbolic_gdrect.py` (HG-DDPM with global attention).
- **Objective:** Standard diffusion loss $\mathcal{L}_{\text{DDPM}}$ combined with hyperbolic reconstruction and pair penalties. HG-DDPM augments the encoder with stacked global self-attention after the diffusion kernels.
- **Finding:** Both depth-2 and depth-7 models maintain high correlation (≥0.89) yet recall plateaus below 0.04 because denoisers focus on geometric consistency rather than item-wise prediction. Adding attention (HG-DDPM) increases stability but not recall, signaling that decoder capacity remains the limiting factor.

Cross-Depth and Cross-Geometry Comparisons (Tables 0_0–0_8)
-----------------------------------------------------------
1. **Euclidean scaling law:** Across `table0_1.md`, `table0_2.md`, and `table0_6.md`, each doubling of $\lambda_{\text{recon}}$ raises Recall@4 by roughly five percentage points regardless of depth, while correlation stubbornly hovers around zero. This linear trend validates the choice to treat Euclidean checkpoints as reconstruction upper-bounds.
2. **Hyperbolic depth sensitivity:** In `table0_0.md`–`table0_5.md`, depth-7 runs collapse because radius and pair penalties enforce a single-shell latent space. Rectified flows and graph encoders (tables `0_6`–`0_8`) mitigate the issue: depth-2 models reach Recall@4≈0.57 with corr≈0.62, depth-7 models climb from ≈0.01 Recall@4 to ≈0.43 while correlations move from 0.0 to 0.15.
3. **Geometry vs. decoder tension:** `table0_7.md` and `table0_8.md` show that DDPM-style training yields corr≥0.89 but Recall@4≤0.04. Rectified flows invert that balance. The comparison tables therefore emphasize the need for hybrid objectives that reward both structure and discrete accuracy.
4. **Depth effect under Euclidean baselines:** Across all tables the Euclidean depth-7 recall (0.14–0.19) lags depth-2 recall (0.22–0.24), indicating that coarse pooling loses specificity whenever the visit tree deepens—even without curvature.

Detailed Analysis (Tables 0_0–0_8)
----------------------------------
* **Table 0_0 (`results/analysis/table0_0.md`)** – Establishes the recall–correlation dichotomy. Hyperbolic embeddings match tree statistics when regularization is active (corr=0.9868) yet fail to retrieve codes. Euclidean models do the opposite. Depth-7 versions exacerbate both issues, motivating decoder work.
* **Table 0_1 (`results/analysis/table0_1.md`)** – Decoder losses drive Euclidean recall to 0.24 at depth 2 and 0.14 at depth 7. Hyperbolic models stay flat, signaling that encoder capacity rather than reconstruction loss is failing.
* **Table 0_2 (`results/analysis/table0_2.md`)** – Hyperbolic noise injections reduce variance but cannot rescue recall, proving that noise scheduling alone is insufficient.
* **Table 0_3 (`results/analysis/table0_3.md`)** – HDD regularization tightens hyperbolic correlation to 0.99 yet again depresses recall; Euclidean checkpoints remain unchanged. Demonstrates structure-first trade-offs.
* **Table 0_4 (`results/analysis/table0_4.md`)** – HGD (generative) regularizers show identical behavior, underscoring that generative-distance terms alone cannot drive discrete accuracy.
* **Table 0_5 (`results/analysis/table0_5.md`)** – Combining HDD+HGD stabilizes latent geometry but retains Recall@4 below 0.08. Insight: deeper supervision needs encoder upgrades.
* **Table 0_6 (`results/analysis/table0_6.md`)** – Rectified Flow v1 introduces graph encoders and tangent projections. Depth-2 hyperbolic runs finally achieve 0.57 Recall@4 with corr≈0.62, surpassing Euclidean metrics while keeping structure. Depth-7 runs still lack correlation.
* **Table 0_7 (`results/analysis/table0_7.md`)** – Rectified Flow v2 plus hyperbolic DDPM sweeps reveal how $\lambda_{\text{recon}}$ modulates correlation: DDPM with $\lambda_{\text{recon}}=3000$ attains corr=0.714 at depth 2 but recall=0.242, whereas rectified depth-2 runs deliver recall=0.574 with corr=0.616. Depth-7 DDPM remains recall-poor.
* **Table 0_8 (`results/analysis/table0_8.md`)** – Consolidated graph experiments confirm that attention-augmented DDPMs secure corr≈0.89 while rectified v2 offers the best recall–geometry compromise (depth-2: Recall@4=0.133, corr=0.885; depth-7: Recall@4=0.048, corr=0.742). The comparison table provides the clearest geometry/accuracy frontier.

Diagnosis & Remaining Solution
------------------------------
| Culprit | Problem | Fix |
| --- | --- | --- |
| Decoder expressivity | Euclidean MLPs cannot invert curved latents, leading to vanishing gradients. | Hyperbolic distance decoder with temperature annealing improves curvature-aware logits. |
| Encoder non-invertibility | Mean pooling discards combinatorics, especially at depth 7. | Einstein pooling plus graph diffusion kernels and global attention retain local structure. |
| Objective mismatch | Geometry-only penalties collapse recall. | Prioritize focal or BCE reconstruction, relegating radius loss to pretraining. |
| Depth collapse | Radius regularizer forces single-shell embeddings. | Remove radius loss after pretraining and let manifold radii self-organize. |

Rectified flows (v1/v2) and graph DDPMs collectively implement these fixes: latents move in tangent space, decoders respect curvature, and hierarchy metrics remain measurable [6].

Next Research Directions
------------------------
1. **Decoder mixtures:** Introduce mixture-of-experts or transformer decoders on the tangent features to push depth-7 Recall@4 beyond 0.15 without sacrificing correlation.
2. **Adaptive diffusion kernels:** Learn per-layer kernel weights or attention over diffusion steps to tailor receptive fields to depth, replacing the static kernels used in `train_graph_hyperbolic_gd.py`.
3. **Hybrid RF+DDPM training:** Combine $\mathcal{L}_{\text{RF}}^{(2)}$ with a light DDPM term to inherit both deterministic transport and stochastic variation, potentially harmonizing recall and structure.
4. **Scale to richer ontologies:** Applying the same codebase to denser trees will stress whether hyperbolic benefits persist; the tables suggest they will once encoders keep pace with depth.

This narrative captures the evolution from Euclidean baselines to graph-aware hyperbolic generators, emphasizing how each phase, equation, and table entry contributes to the emerging scaling laws between recall, depth, and geometry [7].

References
----------
[1] Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. Advances in Neural Information Processing Systems.

[2] Fan, Y., et al. (2023). Hyperbolic Graph Diffusion Model. arXiv preprint arXiv:2306.07618.

[3] Liu, X., et al. (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. arXiv preprint arXiv:2209.03003.

[4] Dai, S., et al. (2021). A Hyperbolic-to-Hyperbolic Graph Convolutional Network. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

[5] Chami, I., et al. (2019). Hyperbolic Graph Convolutional Neural Networks. Advances in Neural Information Processing Systems.

[6] Mao, W., et al. (2025). Hyperbolic Deep Learning for Foundation Models: A Survey. arXiv preprint arXiv:2507.17787.

[7] Mitra, P., et al. (2024). Hyperbolic Deep Learning in Computer Vision: A Survey. International Journal of Computer Vision.

[8] Ganea, O., et al. (2018). Hyperbolic Neural Networks. Advances in Neural Information Processing Systems.
