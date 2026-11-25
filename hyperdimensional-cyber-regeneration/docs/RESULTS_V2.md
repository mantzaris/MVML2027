# Second approach: dependency-factorized regeneration

This report records the second, prospectively frozen experiment. The first study's results motivated its design; they were not a fresh confirmation set. Original reports and measurements are preserved.

Hardware: NVIDIA RTX 5000 Ada Generation; PyTorch 2.9.1+cu128, CUDA 12.8. The run trains 25 new model instances (five variants × five seeds), using 20000 training and 2000 validation snapshots. Recorded study wall time: 23.32 minutes. It evaluates 17 methods across 18 conditions, giving 1,530 seed/method/condition records.

## Changes and comparison logic

The new approach (JH) regenerates the full belief state, predicts root readiness and child readiness conditional on its root, and computes candidate success with each shared root counted once. It replaces the first method's fixed six-fact gate and perfect-truth single-fact teacher. HDC retrieval remains an optional input augmentation, tested against a dense representation.

All five new trained variants receive the same mixed training data and optimization budget. JH, JD, IH and ID use the same smooth maximum over group losses (beta 2); JH_ERM replaces that aggregation with a group mean. JH_product reuses the JH checkpoint but scores candidates from the product of fact marginals. The original eleven methods are reevaluated on the new test pool without retraining. Their comparison combines multiple changes, whereas the new controls separate them.

Fresh evaluation seed: 42027. Five model seeds (11, 23, 37, 51, 71); 4,000 snapshots for each evaluated condition. The primary macro average covers benign observations and targeted, random and coherent interventions at 20%. The same hidden worlds and candidates are paired across these four conditions. 3,978 snapshots per condition have a feasible candidate; 22 do not. No-feasible cases are retained in regret and plan Brier.

## Primary result

Mean candidate-oracle regret: JH **0.08593**, ID **0.08424**. Paired difference JH−ID: **+0.00169**, 95% crossed-bootstrap interval **[-0.00017, +0.00350]**. Lower is better. The primary interval contains zero; the study does not establish an improvement or equivalence.

The interval uses 2,000 draws across five training seeds and 250 topology clusters. Cluster draws are shared across all selected seeds and all four paired conditions; each cluster retains its snapshots. Conditions receive fixed equal weights. This estimates seed/topology variation for the fixed generator, not uncertainty about real cyber environments or the generator design.

## Macro comparison across four conditions

Entries are averages over conditions and training seeds. Recall is conditional on a truly feasible candidate and is reported as a percentage. Fact and plan Brier are distinct scoring rules; plan Brier directly checks the candidate-success probabilities used for decisions.

| Method | Recall@3 (%) | Regret ↓ | Fact Brier ↓ | Plan Brier ↓ |
|---|---:|---:|---:|---:|
| JH: HDC dependency-factorized regeneration | 99.785 | 0.08593 | 0.03621 | 0.06082 |
| JD: Dense dependency-factorized regeneration | 99.779 | 0.08538 | 0.03631 | 0.06107 |
| IH: HDC independent reconstruction | 99.794 | 0.08405 | 0.03590 | 0.06059 |
| ID: Dense independent reconstruction | 99.779 | 0.08424 | 0.03605 | 0.06072 |
| JH_ERM: JH with mean group loss | 99.800 | 0.08564 | 0.03626 | 0.06090 |
| JH_product: JH checkpoint, product-of-marginals scoring | 99.789 | 0.08585 | 0.03621 | 0.06087 |
| B0: Original constrained prior | 97.788 | 0.22527 | 0.09958 | 0.14251 |
| B1: Original GraphMAE-adapted reconstruction | 97.998 | 0.20714 | 0.07396 | 0.14237 |
| B2: Original dense decision-aware reconstruction | 98.069 | 0.20934 | 0.07305 | 0.14057 |
| B3: Original HDC cleanup | 97.059 | 0.23748 | 0.08116 | 0.12846 |
| B4: Original P decoder, full repair | 97.619 | 0.23217 | 0.08121 | 0.15379 |
| B5: Original P decoder, entropy gate | 97.733 | 0.20923 | 0.07201 | 0.12091 |
| P: Original learned selective regeneration | 98.040 | 0.23771 | 0.09285 | 0.16804 |
| random_gate: Original random-gate ablation | 97.504 | 0.23146 | 0.08053 | 0.13308 |
| no_provenance: Original direct-source-cue ablation | 98.000 | 0.23929 | 0.09177 | 0.16587 |
| no_semantic: Original semantic-loss ablation | 98.003 | 0.24502 | 0.09064 | 0.16300 |
| dense_gate: Original dense-gate ablation | 97.998 | 0.24146 | 0.09156 | 0.16578 |

Coherence and prediction quality are separate properties. The mean fraction of facts violating the supplied child≤parent marginal inequality (tolerance 1e-6) is 0.0000% for JH, 0.0000% for JD, 5.3170% for IH and 4.5692% for ID. A zero violation rate does not prove that the inferred facts are true or that candidate ranking is better.

JH changes 99.91% of beliefs numerically from their provisional values, versus 9.37% for original P (absolute-change threshold 1e-4). This measured intervention difference is part of the full-versus-sparse policy change, not a compute-saving result.

## Secondary mechanism comparisons

All effects below are JH minus comparator macro regret. Secondary intervals are descriptive and not corrected for multiple comparisons. A negative effect favors JH.

| Comparator | Effect | 95% interval | What changes |
|---|---:|---|---|
| ID | +0.00169 | [-0.00017, +0.00350] | Representation and factorized inference; primary comparator. |
| IH | +0.00188 | [-0.00008, +0.00381] | Factorized versus independent regeneration with HDC retained. |
| JD | +0.00056 | [-0.00149, +0.00231] | HDC input augmentation within factorized regeneration. |
| JH_ERM | +0.00029 | [-0.00123, +0.00174] | Smooth worst-group emphasis versus mean group loss. |
| JH_product | +0.00009 | [-0.00044, +0.00062] | Joint scoring versus product of marginals using identical weights. |
| P | -0.15177 | [-0.16004, -0.14388] | Full redesign versus original sparse learned gate; data and repair policy also differ. |
| B2 | -0.12341 | [-0.13194, -0.11559] | Mixed-data structured design versus original targeted-only dense baseline. |

## Condition-specific diagnosis

These descriptive means expose regressions that an overall macro average could conceal. Training now covers the listed mechanisms; these are fresh worlds, not unseen attack-family tests.

| Condition | JH regret | ID regret | Original P regret | Original B2 regret | JH − ID | JH − B2 |
|---|---:|---:|---:|---:|---:|---:|
| benign | 0.03215 | 0.03109 | 0.25317 | 0.24411 | +0.00106 | -0.21197 |
| targeted | 0.09968 | 0.10057 | 0.15921 | 0.03726 | -0.00089 | +0.06242 |
| random | 0.11810 | 0.11355 | 0.33592 | 0.32312 | +0.00454 | -0.20502 |
| coherent | 0.09381 | 0.09175 | 0.20254 | 0.23288 | +0.00205 | -0.13907 |

Relative to original B2, JH has lower mean regret in benign, random, coherent and higher mean regret in targeted. These are descriptive condition means. The old specialist was trained only on targeted 20% interventions, whereas new models train on a broader mixture; a macro improvement cannot be interpreted as universal dominance or a benefit of factorization alone.

## Limits and evidence boundaries

- Full regeneration changes all fact beliefs as needed. It does not satisfy the old six-fact intervention cap, and improvements against old P must not be presented as a win at an equal sparse-repair budget.
- The factorization is coherent under the model's assumed root/child conditional structure. It is not an exact posterior for arbitrary correlated deception, arbitrary graphs or multiple causal parents. Corrupted observations can still make the inferred probabilities wrong.
- The smooth group-loss maximum emphasizes predefined training groups. It does not guarantee robustness to every attack distribution or certify worst-case deployment safety.
- HDC benefit is isolated by JH versus JD. A successful joint model alone does not establish a useful HDC contribution or a wire-compression advantage. Every method still receives the common sparse report codec.
- These are independent synthetic snapshots and shared candidate conjunctions. No operational multistep cyber planning, temporal recovery delay or external cyber-range performance is established.
- The old test pool was used to motivate the redesign. The fresh pool is protected from model selection by the frozen second-study protocol; five seeds and a single generator still limit generalization claims.
- New method composition and equations are proposed research contributions. Probabilistic factorization, semantic constraints, masked feature learning and robust group losses have established antecedents; this experiment alone does not establish a first-ever novelty claim.

## Reproducibility

`results/v2/manifest.json` records the executed settings, source identity and environment. `results/v2/comparisons.json` records paired effects, per-seed/per-condition effects, bootstrap design and hashes of every raw input used in the contrasts. Aggregate metrics and curves are retained under `results/v2/`; raw arrays and checkpoints remain ignored. Regenerate this report with `.venv/bin/python scripts/analyze_v2.py` only after the full second run completes.

## Prepared-evidence host-wall timing

A separate audit measures 50 repetitions after five warmups on seed 37 and the first 64-fact fresh-test snapshot. It includes actual packet packing/unpacking, transfers, GPU feature construction, model inference, ranking and candidate IDs returned to the host. Every measured marginal and candidate-probability path is checked against the original inputs before timing.

| Method | Median (ms) | 10th–95th percentiles (ms) |
|---|---:|---|
| JH | 4.985 | 4.850–5.943 |
| JD | 4.942 | 4.514–5.845 |
| IH | 4.580 | 4.417–5.340 |
| ID | 4.223 | 4.075–4.972 |
| JH_ERM | 5.183 | 4.937–6.250 |
| JH_product | 4.645 | 4.460–5.378 |
| P | 5.035 | 4.788–5.871 |
| B2 | 4.213 | 4.013–5.001 |

The shared packet payload contains 1256 bytes. This bounded loop excludes sensing, bandwidth admission, topology/catalogue construction, network delays and cold model loading. These percentiles describe repeated timings on one fixed snapshot, not uncertainty across deployments or training seeds. Main figure CUDA timings exclude this host work and must be interpreted separately.
