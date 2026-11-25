# Third approach: direct candidate prediction and cheap reconstruction

This prospectively frozen study tests a different output representation. SCR predicts course-of-action candidate success directly from observable evidence and symbolic requirements, bypassing fact reconstruction and products of fact marginals. It uses four latent experts mixed by a shared observational context. These experts have no attack-type labels and must not be identified as recovered attack mechanisms without separate evidence.

Hardware: NVIDIA RTX 5000 Ada Generation; PyTorch 2.9.1+cu128, CUDA 12.8. Twenty-five new models (five variants × five seeds) train on the common 20,000-snapshot mixture and 2,000 validation snapshots. All seventeen previous methods are evaluated from frozen checkpoints or fixed rules. The completed run has 22 methods × 18 conditions × 5 seeds = 1,980 records, with recorded study wall time 19.04 minutes.

## Comparison design

SCR, CAND_CTX and CAND_LOCAL use the same candidate BCE + candidate Brier + 0.25 soft decision-regret objective and smooth group aggregation (beta 2). SCR has four experts; CAND_CTX has one with global context; CAND_LOCAL removes the global-context features from the single-expert design. These comparisons test mixture capacity and context, not a generic first-ever set-network contribution. SCR versus CAND_CTX isolates the expert construction while also changing parameter count. CAND_CTX versus CAND_LOCAL is a descriptive context ablation; its interval is not part of the prespecified SCR contrast family.

LOGIT and DAE predict fact readiness and score candidates by products of those probabilities. DAE first reconstructs pre-corruption observed features from additional feature masking/noise, then fine-tunes with the same supervised fact-and-decision loss as v2. Its autoencoder target is the already noisy observed feature vector, not clean hidden readiness. LOGIT receives no such pretraining. Maximum supervised epochs, patience, batch size and learning rate are shared, but actual training time and model capacity differ. Direct candidate models and fact models have different losses and supervision targets. The frozen v2 models also received five graph-masked pretraining epochs, whereas the new direct scorers receive none. Their comparison is a system-level comparison, not an isolated architecture ablation or an equal-total-optimization comparison.

The new method retains symbolic root/child requirement structure in candidate construction, but produces no reconstructed fact beliefs. Candidate success probabilities are marginal probabilities for each candidate, not a joint distribution over all candidate outcomes. Candidate-specific value/cost tensors enter utility scoring and the decision loss, not the new feasibility heads. However, inherited observable feature x[12] is a known value-weighted candidate-criticality summary used by all methods. The feasibility models therefore retain indirect dependence on the supplied candidate values; changing value/cost tensors while holding x fixed tests only direct tensor blinding, not full value independence.

The fresh evaluation pool uses seed 62027, five training seeds (11, 23, 37, 51, 71), and 4,000 snapshots per condition. Prior pools 32027/42027 and pilots 52027/72027 are development evidence. The four primary conditions pair the same hidden worlds, topologies and candidates. Of 4,000 snapshots, 3,970 have a feasible candidate and 30 do not. All snapshots enter regret and candidate Brier; only feasible snapshots enter conditional Recall@3.

## Primary result

Macro candidate-oracle regret: SCR **0.11152**, ID **0.08644**. SCR−ID: **+0.02508**, 95% crossed-bootstrap interval **[+0.02091, +0.02919]**. Lower is better. The primary interval favors the frozen mixed-data dense independent baseline.

The interval uses 2,000 draws across five training seeds and 250 topology clusters, retaining all sixteen snapshots in each cluster. The same cluster draw is shared across seeds, methods and four fixed, equally weighted conditions. It quantifies training-seed and test-topology variation within this generator; it does not capture uncertainty about the generator, real deployments, or the common training and validation datasets.

## Four-condition macro results

N/A means the method has no fact-belief output; it is not a zero error or perfect semantic score. Recall is conditional on feasibility and is shown as a percentage. Each entry averages the four fixed conditions and five training seeds.

| Method | Recall@3 (%) | Regret ↓ | Candidate Brier ↓ | Fact Brier ↓ |
|---|---:|---:|---:|---:|
| SCR: Context-conditioned four-expert candidate scorer | 99.475 | 0.11152 | 0.06893 | N/A |
| CAND_CTX: Single-expert candidate scorer with global context | 99.414 | 0.12431 | 0.07420 | N/A |
| CAND_LOCAL: Single-expert candidate scorer without global context | 99.370 | 0.12725 | 0.07918 | N/A |
| LOGIT: Cheap logistic fact reconstruction | 98.598 | 0.16455 | 0.10018 | 0.06419 |
| DAE: Cheap denoising-autoencoder fact reconstruction | 99.326 | 0.12873 | 0.07952 | 0.04987 |
| JH: HDC dependency-factorized regeneration | 99.723 | 0.08713 | 0.05967 | 0.03566 |
| JD: Dense dependency-factorized regeneration | 99.727 | 0.08867 | 0.05991 | 0.03574 |
| IH: HDC independent reconstruction | 99.712 | 0.08556 | 0.05935 | 0.03539 |
| ID: Dense independent reconstruction | 99.725 | 0.08644 | 0.05957 | 0.03554 |
| JH_ERM: JH with mean group loss | 99.734 | 0.08732 | 0.05960 | 0.03571 |
| JH_product: JH checkpoint, product-of-marginals scoring | 99.723 | 0.08689 | 0.05970 | 0.03566 |
| B0: Original constrained prior | 97.676 | 0.22293 | 0.14079 | 0.09944 |
| B1: Original GraphMAE-adapted reconstruction | 97.756 | 0.20870 | 0.14031 | 0.07337 |
| B2: Original dense decision-aware reconstruction | 97.832 | 0.21006 | 0.13824 | 0.07242 |
| B3: Original HDC cleanup | 96.801 | 0.24361 | 0.12780 | 0.08137 |
| B4: Original P decoder, full repair | 97.330 | 0.23399 | 0.15104 | 0.08045 |
| B5: Original P decoder, entropy gate | 97.373 | 0.22129 | 0.11962 | 0.07196 |
| P: Original learned selective regeneration | 97.786 | 0.23874 | 0.16557 | 0.09238 |
| random_gate: Original random-gate ablation | 97.141 | 0.23617 | 0.13140 | 0.08036 |
| no_provenance: Original direct-source-cue ablation | 97.733 | 0.24030 | 0.16393 | 0.09141 |
| no_semantic: Original semantic-loss ablation | 97.702 | 0.24665 | 0.16061 | 0.09016 |
| dense_gate: Original dense-gate ablation | 97.738 | 0.24322 | 0.16356 | 0.09113 |

## Prespecified secondary contrasts

All effects are SCR minus comparator macro regret. Secondary intervals are descriptive and unadjusted for multiple comparisons. Negative values favor SCR; intervals containing zero do not establish equivalence.

| Comparator | Effect | 95% interval | Interpretation |
|---|---:|---|---|
| ID | +0.02508 | [+0.02091, +0.02919] | Direct candidate prediction versus frozen mixed-data dense fact reconstruction; architecture, objective, supervision and pretraining differ. |
| CAND_CTX | -0.01279 | [-0.01595, -0.00970] | Four latent experts versus one expert; both use global context. Parameter count also changes. |
| CAND_LOCAL | -0.01573 | [-0.01910, -0.01244] | Latent mixture plus global context versus local single-expert scoring; two components change. |
| DAE | -0.01721 | [-0.02117, -0.01322] | Direct candidate prediction versus a smaller pretrained denoising autoencoder with supervised fact fine-tuning. |
| LOGIT | -0.05304 | [-0.05760, -0.04854] | Direct nonlinear candidate prediction versus a smaller logistic fact model. |
| JH | +0.02439 | [+0.01998, +0.02877] | Direct candidate prediction versus HDC dependency-factorized full regeneration. |
| B2 | -0.09854 | [-0.10738, -0.09066] | New mixed-data design versus original targeted-only dense decision-aware reconstruction; training distributions also differ. |
| P | -0.12722 | [-0.13495, -0.11898] | New direct scoring versus original targeted-only sparse regeneration; training distribution and output scope also differ. |

## Condition-specific comparisons

These descriptive means expose condition-specific regressions that a macro average can hide. All four mechanisms occur in mixed training; they are not previously unseen attack families.

| Condition | SCR | CAND_CTX | CAND_LOCAL | DAE | LOGIT | ID | JH | Original B2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| benign | 0.03150 | 0.03270 | 0.04252 | 0.04690 | 0.04104 | 0.03240 | 0.03258 | 0.24332 |
| targeted | 0.13249 | 0.17685 | 0.17779 | 0.17371 | 0.32564 | 0.10247 | 0.10166 | 0.04013 |
| random | 0.14598 | 0.14809 | 0.14741 | 0.15533 | 0.13010 | 0.11113 | 0.11486 | 0.32649 |
| coherent | 0.13611 | 0.13960 | 0.14127 | 0.13897 | 0.16144 | 0.09975 | 0.09943 | 0.23029 |

## Compute and output scope

The following descriptive means are from the common reference condition (targeted 20%, 64 facts, 2,048 bytes per remote observer). CUDA model-plus-ranking latency excludes host packet processing. Training times are recorded actual work, not a promise that fewer parameters always imply lower latency.

| Method | Trainable parameters | Mean training seconds | Mean seed-median CUDA latency (ms) | Mean additional peak VRAM (MiB) |
|---|---:|---:|---:|---:|
| SCR | 14,888 | 36.89 | 2.216 | 0.104 |
| CAND_CTX | 12,481 | 32.46 | 1.978 | 0.104 |
| CAND_LOCAL | 12,481 | 31.82 | 2.027 | 0.104 |
| DAE | 4,233 | 25.85 | 0.623 | 0.029 |
| LOGIT | 57 | 20.04 | 0.373 | 0.024 |
| ID | 119,966 | 40.75 | 1.367 | 0.172 |
| JH | 120,350 | 46.66 | 2.119 | 0.177 |

Parameter counts include retained pretraining decoders. DAE has 4,233 retained parameters, of which 2,097 participate in readiness inference. Prior-model training durations were recorded in their original runs; they were not retrained for this study. Latency above is the mean of the five seed-specific warm CUDA medians, while memory is additional peak allocation above resident tensors.

DAE and LOGIT trade predictive quality for lower measured cost: their macro regrets are 0.12873 and 0.16455, compared with ID's 0.08644. SCR improves over the smaller new models and its single-expert control, but its fewer parameters do not translate into lower measured inference latency than ID. Architecture size alone is therefore an incomplete efficiency argument.

## Candidate monotonicity diagnostic

A diagnostic adds requirements by taking adjacent candidate unions with observable x held fixed. A valid conjunction probability should not increase when requirements are added. The unions contain up to ten requirements, whereas training candidates have two to five, so this is a query-length stress test. The violation rate is a descriptive logical-consistency check, not an accuracy measure or a training objective. No shadow candidate is inserted into the ranking catalogue, and the inherited catalogue-criticality feature is not recomputed for these shadow queries.

| Method | Macro shadow-union violation fraction ↓ |
|---|---:|
| SCR | 0.41153 |
| CAND_CTX | 0.42845 |
| CAND_LOCAL | 0.45096 |
| DAE | 0.00000 |
| LOGIT | 0.00000 |
| ID | 0.00000 |
| JH | 0.00000 |

SCR increases its success probability after adding requirements in **41.15%** of these shadow-union comparisons. This is a measured shortcoming of the direct scorer: its queried probabilities can violate conjunction monotonicity and do not constitute a guaranteed coherent joint model of candidate outcomes. The reconstruction methods shown above have zero violations in this check. Because the unions extend beyond the training query lengths, this diagnostic establishes a stress-test failure, not the cause of SCR's regret gap on the original catalogue.

## Evidence boundaries

- SCR can compare supplied candidates without regenerating a reusable fact state. It cannot claim fact-level repair, marginal logical coherence or fact-level explanations. Fact and repair metrics are intentionally unavailable.
- Every one of the twenty-five new models reached the fixed forty-epoch supervised cap; none stopped early. Selected checkpoints were at zero-based epochs 36–39 (mostly 39). These results compare methods under that optimization budget and do not establish converged architecture rankings or rule out improvements with longer training.
- The four-expert gate is a latent predictive mechanism. Expert identities are not supervised and have no established one-to-one relation to attack types. A low-entropy gate is not proof of deception detection.
- Global context may identify correlations specific to this simulator. Fresh topology snapshots do not establish robustness to arbitrary adversaries or unseen deception mechanisms.
- Relative gains over original P/B2 combine broader training, different output scope and changed architecture/objectives. The old sparse repair budget is not a constraint on candidate-only scoring.
- Every method uses the same sparse report codec. These comparisons do not establish a hyperdimensional wire-compression advantage. The direct candidate design has no HDC encoder; prior HDC methods remain explicit comparators.
- The benchmark is synthetic, with independent snapshots, a one-level prerequisite forest and fixed candidate conjunctions. It does not establish multistep cyber planning, temporal recovery performance, real cyber-range effectiveness or operational mission utility.
- The equation and methodological composition are proposed contributions. Set aggregation, mixtures of experts, denoising autoencoders and decision-focused learning have established antecedents; novelty beyond this composition requires a broader literature assessment.

## Reproducibility

`results/v3/manifest.json` records executed settings, source identity, data hashes, hardware, checkpoint provenance and training durations. `results/v3/comparisons.json` records the prespecified paired analysis, per-condition/per-seed effects and all raw input hashes. The analysis validates source/checkpoint integrity and raw/summary parity before generating this report. Raw arrays, weights and logs remain ignored by Git. Reproduce the analysis with `.venv/bin/python scripts/analyze_v3.py` only after the complete GPU run.

## Prepared-evidence host-wall timing

This separate bounded measurement includes actual packet packing/unpacking, transfers, feature construction, GPU inference, ranking and candidate IDs returned to the host. It excludes sensing, bandwidth admission, topology/catalogue construction, network delay and cold loading. Percentiles concern repeated executions of the recorded snapshot, not deployment uncertainty.

| Method | Median (ms) | 10th–95th percentiles (ms) |
|---|---:|---|
| SCR | 5.517 | 5.016–6.444 |
| CAND_CTX | 5.193 | 4.809–6.075 |
| CAND_LOCAL | 4.826 | 4.672–5.728 |
| LOGIT | 2.913 | 2.802–3.778 |
| DAE | 3.064 | 2.677–3.608 |
| ID | 3.443 | 3.407–4.105 |
| JH | 4.232 | 4.071–5.195 |

The common measured packet payload is 1244 bytes.
