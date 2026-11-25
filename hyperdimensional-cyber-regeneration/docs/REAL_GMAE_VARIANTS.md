# Two exploratory GraphMAE variants for captured telemetry

This follow-up tests two different changes to the GraphMAE-adapted flow classifier: **EDGE_GMAE** changes how neighbors contribute, while **CONS_GMAE** changes how the unchanged graph architecture is trained. The purpose is to discover whether either improves the previous captured-traffic result, not to assume that a more complicated model will win.

The preceding CTU-13 test results have already been examined and motivate this investigation. This is therefore an **exploratory reused-test follow-up**, not an untouched confirmatory evaluation. Freezing the new code and settings before their test predictions limits further tuning but does not erase that prior exposure. The previous studies and their scientific sources remain unchanged.

## Evidence motivating the variants

In the first external study, the unpretrained dense graph classifier had the largest descriptive clean macro attack AP, while the GraphMAE-adapted model and HDC augmentation did not establish an advantage over it. The shared-context model also had a marked deficit on capture 12. These observations motivate testing message quality and the reconstruction objective separately. They do not prove a causal diagnosis of the earlier errors.

Endpoint-sharing is an observed relationship, but neighbors need not have the same class or equally informative measurements. Uniform aggregation can mix poorly observed or incompatible reports. Separately, reconstructing individual feature values may not preserve representations or decisions when some measurements disappear. EDGE_GMAE and CONS_GMAE address these as hypotheses.

## Established lineage and novelty boundary

- **GraphMAE** uses masked feature reconstruction and latent remasking; its original scaled cosine reconstruction error differs from this project's observed-coordinate MSE adaptation. Our comparator is the already executed `GMAE` implementation, not an exact reproduction of the original paper. [Hou et al., KDD 2022](https://arxiv.org/abs/2205.10803).
- **GraphMAE2** already combines multiple remasked reconstruction views with latent representation prediction. The idea of adding representation consistency to a masked graph learner is established. Our two independently masked encoder views, symmetric detached alignment and flow-level training regularizer are a specific adaptation, not a claim to have invented latent prediction or multi-view learning. [Hou et al., WWW 2023](https://arxiv.org/abs/2304.04779).
- **Graph Attention Networks** learn unequal neighborhood contributions, and **Pro-GNN** studies graph refinement for robustness. EDGE_GMAE instead uses a fixed, parameter-free gate from visibly available coordinate overlap and discrepancy within an existing endpoint graph; it does not learn a new graph or a general attention mechanism. [Veličković et al., ICLR 2018](https://arxiv.org/abs/1710.10903), [Jin et al., KDD 2020](https://arxiv.org/abs/2005.10203).
- **DropEdge++** includes feature-dependent edge sampling. Similarity-dependent graph weighting is consequently not a new general principle; our gate is deterministic, availability-aware, and retains a positive off-diagonal floor. [Han et al., TNNLS 2023](https://arxiv.org/abs/2306.12091).
- **Equipping Graph Autoencoders** proposes reliability-based masking and decoupled feature reconstruction. Reliability-aware masked graph learning already has direct prior art. [Song et al., SDM 2025](https://epubs.siam.org/doi/abs/10.1137/1.9781611978520.39).
- **C2-GMAE** studies consistency and coverage of neighborhood context using structural references, density-partitioned masking and stronger heterophilic edges. This also cautions against assuming that feature similarity always identifies the useful edges. We neither reproduce these mechanisms nor claim that context consistency is novel. [Tang et al., ICML 2026](https://proceedings.mlr.press/v306/tang26l.html).

The contribution under investigation is the explicit telemetry adaptation and its measured tradeoffs against matched controls. This targeted literature check does not establish first-ever architectural novelty. No result in those papers guarantees an improvement on CTU-13.

## Variant 1: visible-evidence edge weighting

Let \(x_i\in\mathbb R^{14}\) be the sanitized, training-normalized measurement vector of a valid flow, \(m_i\in\{0,1\}^{14}\) its currently visible feature indicators, and \(A_{ij}\) the supplied endpoint-sharing adjacency. Define

\[
 s_{ij}=\sum_{k=1}^{14}m_{ik}m_{jk},\qquad
 r_{ij}=s_{ij}/14,\qquad
 d_{ij}=\frac{\sum_{k=1}^{14}m_{ik}m_{jk}(x_{ik}-x_{jk})^2}{\max(1,s_{ij})}.
\]

The new experimental aggregation rule is

\[
 \bar A_{ij}=
 \begin{cases}
 1,&i=j\text{ and }i\text{ is valid},\\
 A_{ij}\left[0.1+0.9r_{ij}\exp(-d_{ij}/2)\right],&i\ne j\text{ and both flows are valid},\\
 0,&\text{otherwise},
 \end{cases}
 \qquad
 P_{ij}=\frac{\bar A_{ij}}{\max(1,\sum_\ell\bar A_{i\ell})}.
 \tag{EDGE.1}
\]

Use \(P\) in the same width-128, three-layer residual graph encoder and reconstruction decoder. The encoder, mask token, readout and decoder retain the comparator's parameter count; there is no fitted gate network. The 0.1 floor retains weak neighbor information when no coordinates overlap, while self loops remain one before normalization. Padded rows and columns contribute nothing. Coordinate discrepancy is computed only where both measurements are available; unknown placeholders do not count as evidence.

Crucially, pretraining recomputes this gate from the **masked input view**. The feature availability of an additionally masked node is cleared for gating, even though the encoder receives a learned mask token. Original reconstruction targets cannot leak through the edge weights. Graph topology itself remains the supplied endpoint relation.

**EDGE_GMAE** receives the existing five masked-reconstruction pretraining epochs and the common supervised BCE-plus-Brier loss. **EDGE_DENSE** uses the identical weighted graph architecture without pretraining. EDGE_GMAE versus GMAE changes aggregation with the pretraining objective held fixed; EDGE_DENSE versus DENSE checks the aggregation change without pretraining; EDGE_GMAE versus EDGE_DENSE checks adding that pretraining under the weighted representation.

This gate is a heuristic about observable compatibility, not a probability of truth or a deception detector. Matching spoofed values may appear compatible. Useful heterophilic edges can be downweighted. The fixed scale mixes standardized numeric coordinates with bounded ports and flags; that design choice is frozen rather than tuned on test outcomes.

## Variant 2: two-view latent and prediction consistency

CONS_GMAE and its control retain the **exact GMAE inference architecture and parameter count**. Their change is in training. For each pretraining batch, independently sample two masks, each selecting 30% of valid nodes. Encode the corresponding views to obtain \(h_i^{(1)}\) and \(h_i^{(2)}\) before decoder remasking. Apply the existing GraphMAE-adapted reconstruction procedure independently to both views, yielding \(L_{\mathrm{rec}}^{(1)}\) and \(L_{\mathrm{rec}}^{(2)}\). Each reconstruction loss uses only originally observed coordinates of additionally masked real nodes.

For valid nodes \(V\), define detached symmetric cosine alignment

\[
 C(h^{(1)},h^{(2)})=
 \frac{1}{2|V|}\sum_{i\in V}
 \left[
 1-\cos(h_i^{(1)},\operatorname{sg}(h_i^{(2)}))
 +1-\cos(h_i^{(2)},\operatorname{sg}(h_i^{(1)}))
 \right],
\]

and use

\[
 L_{\mathrm{CONS,pre}}=
 \tfrac12\left(L_{\mathrm{rec}}^{(1)}+L_{\mathrm{rec}}^{(2)}\right)
 +0.5\,C(h^{(1)},h^{(2)}).
 \tag{CONS.1}
\]

Here \(\operatorname{sg}\) stops gradients through the target argument. Both views still receive gradients through their own argument. Alignment includes all valid nodes, not padding, and uses safe normalization. There is no separate teacher, exponential moving average, extra projector, botnet-family supervision or inaccessible clean feature target.

During supervised training, first compute the standard prediction \(p_i=f_\theta(x,A)_i\). Independently remove 20% of currently observed feature coordinates, clearing both their values and indicators, to form \(\tilde x\). Let \(\tilde p_i=f_\theta(\tilde x,A)_i\). The full variant uses

\[
 L_{\mathrm{CONS,sup}}=L_{\mathrm{BCE+Brier}}(p,y)
 +\frac{0.5}{|V|}\sum_{i\in V}
 \left(\tilde p_i-\operatorname{sg}(p_i)\right)^2.
 \tag{CONS.2}
\]

The original supervised loss still trains the original prediction; the extra term trains the masked-view prediction toward the detached original. The consistency term does not read labels. Both views are derived from the available training observations, which may already contain the fixed missingness or replacement stress. Thus the target is not guaranteed correct or uncorrupted. Endpoint metadata is unchanged.

**CONS_PRE** receives equation CONS.1 but omits the extra supervised term in CONS.2. CONS_GMAE versus CONS_PRE tests the added prediction regularizer at fixed two-view pretraining. CONS_PRE versus GMAE changes both the number of reconstruction views and latent alignment, so it cannot isolate the cosine term alone. There is no two-view, zero-alignment control. Reconstruction and supervised labels help constrain learning, but the design does not mathematically rule out latent collapse or incorrect consistency targets.

## Fixed comparison and training protocol

The authoritative executable specification is [study_real_gmae.json](../configs/study_real_gmae.json); implementation is [real_gmae_variants.py](../src/real_gmae_variants.py) and [run_real_gmae.py](../scripts/run_real_gmae.py).

| Method | Inference representation | Pretraining | Additional supervised term |
|---|---|---|---|
| GMAE, imported baseline | Uniform endpoint graph | Existing single-view reconstruction, 5 epochs | None |
| DENSE, imported control | Uniform endpoint graph | None | None |
| EDGE_GMAE | Equation EDGE.1 | Existing single-view reconstruction, 5 epochs | None |
| EDGE_DENSE | Equation EDGE.1 | None | None |
| CONS_PRE | Uniform endpoint graph | Equation CONS.1, 5 epochs | None |
| CONS_GMAE | Uniform endpoint graph | Equation CONS.1, 5 epochs | Equation CONS.2 |

All four new methods retain 121,643 trainable parameters, including the inactive-at-inference reconstruction decoder. They use seeds 11, 23, 37, 51 and 71, AdamW learning rate 0.001, weight decay 0.0001, 256 windows per batch, at most 40 supervised epochs, and patience 8. All twenty new training runs and their validation-only thresholds finish before any new test inference. Early stopping uses the same base supervised validation objective across clean, missing20 and spoof20, not a test metric or a method-specific regularizer. Thresholds maximize attack F1 on clean validation and are then held fixed across test conditions. Settings and new scientific sources are frozen before executing the full run; development checks use training-only plumbing data.

The nine earlier methods, including conventional Gaussian naive Bayes, remain visible as **imported frozen results**, with their source, data, input, checkpoint and prediction hashes checked. They are not presented as freshly rerun training. This preserves an exact reference instead of selecting new baseline seeds after observing test results. Inference may be replayed for audit and timing, without retraining or threshold changes.

Epoch count and model capacity are controlled where stated; total computation is not equal. CONS pretraining performs two encoder/decoder views and alignment, and full CONS adds a supervised masked-view forward pass. EDGE adds pairwise coordinate comparisons. Report actual training duration, inference latency, incremental memory and bounded host-to-GPU-to-host cost; do not attribute a gain to a single mechanism where the control changes several components.

## Data and uncertainty scope

The [previous real-data protocol](REAL_DATA_METHOD.md) still defines CTU-13 parsing, normalization, masks and input construction: 442,084 training flows from captures 1–8 and 13; 214,880 validation flows from capture 9; 142,813 test flows from captures 10–12. The first six numeric/log features use training-only normalization. Fourteen values plus their fourteen availability indicators enter each model. Fixed windows contain up to 32 retained flows; known-label filtering occurs before window and graph construction. The reused cases are clean, missing20, missing50, spoof20 and spoof40. Exact input hashes must match the first study.

The two planned clean contrasts are **EDGE_GMAE minus GMAE** and **CONS_GMAE minus GMAE** in attack average precision, with equal weight on each of the three fixed test captures before averaging seeds. Use 1,000 shared paired bootstrap draws across the two contrasts: resample five seeds and contiguous 64-window blocks within each capture, recomputing exact tie-aware AP. The earlier test pool contains 60, 6 and 5 blocks in captures 10, 11 and 12, so two block populations are small. Reject and count any common draw whose resampled capture has no positive labels.

Report ordinary 95% intervals and 97.5% marginal percentile intervals (1.25% and 98.75% quantiles), the latter corresponding to a nominal Bonferroni two-comparison target. This adjustment only addresses these two specified contrasts under the bootstrap approximation; it does not account for the earlier adaptive choice of research directions, all historical experiments, uncertain block independence or selection of these captures. It does not turn reused data into an independent confirmation. All remaining method, stress, operating-point and compute comparisons are descriptive. A numerical mean gain, an interval above zero, and a reproducible result on a newly reserved dataset are different levels of evidence.

CTU-13 consists of real recorded university traffic with controlled executions of botnets, dating to 2011. The 799,777 verified-label flows are roughly 4% of the original 19,976,700 rows. Host-associated From-Botnet and From-Normal labels are retained; unknown background is excluded. Hosts can recur across splits. Predictions use completed-flow statistics and all measurements in a window, so they are retrospective rather than packet-causal. Artificial feature removal and benign-centroid replacement are not observed adversarial deception or packet-valid adaptive attacks. These limitations apply to every new variant and prevent generalizing a gain to the unknown background population or unseen organizations.

## Connection to the conference and proposal

The ML investigation is whether representation aggregation and masked learning objectives improve generalization, calibration and robustness at measured compute cost. EDGE.1 adds an explicit availability-aware rule to a prior graph representation; CONS.1–2 add a controlled reconstruction-and-consistency objective. Together with the controls, these equations make the investigation concrete and falsifiable.

The proposal connection remains a perception/representation component for incomplete or corrupted evidence. This extension does not evaluate real courses of action, action execution, mission utility, logical prerequisite consistency, HDC communication compression or distributed replanning. The supplied endpoint graph is relational evidence, not a symbolic implication graph. The earlier planning equations are preserved, not reinterpreted as flow-label guarantees.

Raw arrays, checkpoints, logs and intermediate renders remain under ignored directories. Retain concise manifests, reproducible source, four rich empirical figures, and a three-page supplement with two extra figures. Follow the GPU launcher for all model fitting and inference; CPU is limited to data I/O, statistical analysis and artifact rendering. The results report, not this design document, determines whether either variant actually improves.
