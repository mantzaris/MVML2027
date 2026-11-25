# External application to captured CTU-13 traffic

This experiment retrains explicit adaptations on real captured network-flow measurements. CTU-13 mixes normal university traffic with controlled executions of real botnets. It is not a record of arbitrary operational incidents, and its labels do not supply host readiness or course-of-action outcomes.

## Data and task boundaries

Only author-labeled `From-Normal` and `From-Botnet` flows enter the binary task. Background and other uncertain labels are excluded rather than treated as benign. Exclusion makes this a selected verified-label benchmark; success does not establish performance on the omitted background distribution.

Captures 1-8 and 13 (August 10 through August 17 morning, 2011) supply training, capture 9 (August 17 midday/afternoon) supplies validation, and captures 10-12 (August 18-19) supply the held-out test. Scenario numbering is not chronological: capture 13 belongs in the earlier training period. Test captures 10/11 contain Rbot and capture 12 contains NSIS.ay, which is not represented in training. Hosts may recur, so the split is capture/time-disjoint, not host-disjoint.

The selected subset contains 799,777 of 19,976,700 original flows (4.00%). Training/validation/test retain 442,084/214,880/142,813 flows. Their pooled attack prevalences are 0.324, 0.861 and 0.817; this substantial prior shift is part of the held-out challenge. The parser verified that every retained training flow completed before validation starts, and every validation flow completed before test starts.

Models receive fourteen derived flow measurements and fourteen observation indicators. Raw addresses, absolute timestamps, scenario IDs and labels are not input features. Addresses are used only to construct observable within-window endpoint-sharing edges. Windows and graphs are constructed after verified-label filtering; they are not an unfiltered live-stream context. Context is fixed by consecutive 32-flow windows rather than arbitrary minibatch composition. Predictions require all flow statistics in the window, including final durations/byte totals, and are retrospective classifications after those measurements become available, not causal packet-by-packet decisions.

The clean condition uses real recorded measurements. Missing-value and feature-spoofing conditions are explicitly artificial corruptions applied to those measurements with original labels held fixed. The observed endpoint graph is retained during feature corruption. These are stress tests, not evidence of naturally occurring telemetry deception. No synthetic conjunction is presented as a measured cyber course of action.

## Primary held-out result

Clean macro attack average precision: SCR_FLOW **0.96616**, GMAE **0.97830**. Paired difference: **-0.01215**, 95% interval **[-0.04490, +0.01159]**. Higher is better. The interval contains zero and establishes neither an improvement nor equivalence.

The primary average gives each held-out capture equal weight before averaging training seeds. The interval crosses five training seeds and contiguous 64-window blocks within each of three fixed test captures. It recomputes tie-aware attack AP for every weighted bootstrap sample; individual flows are not treated as independent. The interval conditions on these captures and training data and does not quantify uncertainty over organizations or capture selection.

There were 1,000 accepted bootstrap draws and 0 rejected draws lacking any positive flow in a resampled capture. The fixed captures provide 60, 6 and 5 blocks respectively; the last two are small block populations.

## Clean capture-macro comparisons

| Method | Attack AP ↑ | ROC-AUC ↑ | Brier ↓ | Balanced accuracy ↑ | Recall ↑ | FPR ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian naive Bayes (NB) | 0.91784 | 0.85354 | 0.07667 | 0.83413 | 0.95910 | 0.29085 |
| Logistic classifier (LOGIT) | 0.63538 | 0.49517 | 0.30425 | 0.47448 | 0.94267 | 0.99370 |
| Denoising autoencoder (DAE) | 0.86743 | 0.75879 | 0.08797 | 0.78952 | 0.86692 | 0.28788 |
| Dense graph classifier (DENSE) | 0.98151 | 0.98084 | 0.04634 | 0.87259 | 0.81632 | 0.07115 |
| Masked-pretrained graph classifier (GMAE) | 0.97830 | 0.97753 | 0.05973 | 0.87242 | 0.81815 | 0.07330 |
| HDC-augmented masked graph classifier (HDC) | 0.97578 | 0.97603 | 0.04873 | 0.87185 | 0.81210 | 0.06840 |
| Shared-context mixture (SCR_FLOW) | 0.96616 | 0.97284 | 0.06672 | 0.85346 | 0.80520 | 0.09828 |
| Single expert with context (CTX_FLOW) | 0.95602 | 0.94072 | 0.07728 | 0.80825 | 0.82071 | 0.20422 |
| Single expert without context (LOCAL_FLOW) | 0.84756 | 0.73533 | 0.09652 | 0.75162 | 0.79046 | 0.28721 |

Threshold metrics use validation-selected thresholds, frozen before test evaluation and retained across all stress conditions. AP/AUC are threshold-free but depend on the selected data distribution; AP in particular changes with attack prevalence.

Within this recorded test pool, DENSE has the largest descriptive clean macro AP (0.98151); this is not a separate population-level superiority test. Context access is useful in the matched single-expert comparison: CTX_FLOW 0.95602 versus LOCAL_FLOW 0.84756. The conventional NB reference reaches 0.91784, exceeding the small DAE (0.86743) and LOGIT (0.63538) in this dataset. These secondary rankings are descriptive.

## Capture-specific outcomes

| Capture | Flows | Attack prevalence | SCR_FLOW AP | GMAE AP | DAE AP | LOGIT AP |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 122,157 | 0.8706 | 0.99916 | 0.99862 | 0.97509 | 0.84877 |
| 11 | 10,873 | 0.7509 | 0.98175 | 0.95449 | 0.67682 | 0.72075 |
| 12 | 9,783 | 0.2216 | 0.91756 | 0.98181 | 0.95039 | 0.33661 |

## Artificial corruption on captured telemetry

| Condition | SCR_FLOW AP | GMAE AP | HDC AP | DAE AP | LOGIT AP |
|---|---:|---:|---:|---:|---:|
| clean | 0.96616 | 0.97830 | 0.97578 | 0.86743 | 0.63538 |
| missing20 | 0.93361 | 0.96844 | 0.96486 | 0.83781 | 0.62769 |
| missing50 | 0.86606 | 0.93553 | 0.93899 | 0.77497 | 0.63170 |
| spoof20 | 0.95389 | 0.97663 | 0.97511 | 0.84215 | 0.62840 |
| spoof40 | 0.92977 | 0.96716 | 0.96805 | 0.81337 | 0.62318 |

Clean SCR_FLOW-minus-GMAE AP differences are +0.00055, +0.02726 and -0.06424 for captures 10, 11 and 12. The largest observed deficit occurs in capture12 (NSIS.ay, absent from training). Capture identity, traffic mix and family change together, so this is not an isolated causal effect of an unseen malware family.

## Method and compute interpretation

SCR_FLOW retains the shared latent mixture and fixed observation-window context from the third synthetic study, adapting the output to flow attack probabilities. CTX_FLOW and LOCAL_FLOW compare single-expert context and local-only alternatives. They are newly trained adaptations, not a zero-shot transfer of the original synthetic checkpoints. DENSE/GMAE separate graph-masked pretraining; GMAE/HDC match the supervised objective and pretraining while changing input augmentation. The supplied graph expresses shared observed endpoints, not child-implies-parent logic. No conditional-neurosymbolic consistency guarantee is invented for these labels.

LOGIT and DAE provide inexpensive neural alternatives; Gaussian naive Bayes supplies a conventional conditional-independence reference with 58 fitted statistics; it has no gradient-trained parameters. Its deterministic fit is repeated in the five seed slots for paired bookkeeping, not five stochastic training outcomes. Parameter counts, actual training effort and measured inference costs must accompany accuracy. DAE reconstructs observed flow features during denoising pretraining; there is no no-pretraining DAE control, so its comparison does not isolate a causal pretraining benefit.

| Method | Mean training seconds | Mean CUDA window latency (ms) | Incremental peak VRAM (MiB) |
|---|---:|---:|---:|
| NB | 0.004 | 0.470 | 0.011 |
| LOGIT | 2.225 | 0.499 | 0.012 |
| DAE | 7.712 | 0.773 | 0.016 |
| DENSE | 21.304 | 2.020 | 0.093 |
| GMAE | 24.148 | 2.006 | 0.093 |
| HDC | 22.166 | 2.392 | 0.464 |
| SCR_FLOW | 16.752 | 1.489 | 0.036 |
| CTX_FLOW | 12.017 | 1.248 | 0.036 |
| LOCAL_FLOW | 6.922 | 1.272 | 0.036 |

GPU timing is warmed model inference for one fixed observation window; it excludes storage/download and is not an operational network latency measurement. The separate prepared-evidence wall audit records its own included operations. Models retain differing pretraining budgets and capacities.

## Evidence audit and scope

The analysis checked 675 unique records and 675 raw archives, recomputed 4725 metrics, verified paired labels/window blocks, and checked recorded scientific source/checkpoint hashes. The raw archives and downloaded captures remain ignored by Git.

Floating-point audit: 321 model-flow predictions exceeded a probability boundary within the accepted 1e-6 rounding tolerance; maximum excess was 1.19e-07. Raw scores and metric calculations are preserved. Reliability bins are regenerated with display-only clipping to [0,1] so no boundary-roundoff prediction is omitted.

This is external evidence for representation, reconstruction and classification under shifted recorded traffic. It is not a validation of multi-step cyber planning, autonomous execution, hyperdimensional wire compression or operational mission outcomes. Author labels are partly host-derived; excluded identities reduce direct leakage but do not remove all dataset-specific artifacts. Only one dataset, three test captures and five model seeds are covered. Secondary seed-t intervals are descriptive and conditional on the fixed test pool, not the primary crossed-block interval.

## Sources and reproducibility

- García, Grill, Stiborek and Zunino (2014), *An empirical comparison of botnet detection methods*, Computers & Security 45, 100-123. [DOI](https://doi.org/10.1016/j.cose.2014.05.011).
- [Official CTU-13 description](https://www.stratosphereips.org/datasets-ctu13) and [author usage/citation FAQ](https://www.stratosphereips.org/datasets-faq).
- `results/real/manifest.json` records the executed protocol, hardware, source and data/checkpoint hashes. `results/real/comparisons.json` records the analysis design, exact raw hashes and uncertainty scope.
