# Two GraphMAE variants on captured CTU-13 traffic

This is an exploratory follow-up to the completed captured-traffic study. The same test captures were already inspected when choosing to investigate GraphMAE. All four new methods were frozen before their new test predictions and use validation-only checkpoint/threshold selection, but this is not a fresh confirmatory test. Nominal multiplicity adjustment cannot remove this adaptive reuse.

## Variants and matched controls

**EDGE_GMAE** changes message passing using an observed-overlap compatibility weight on existing endpoint-sharing edges. It reduces messages between flows with sparse shared observations or incompatible measured features, keeps self loops, and uses the same trainable parameter count and reconstruction/supervised objectives as GMAE. **EDGE_DENSE** removes its masked pretraining. Gate values use only the inputs available to each encoder view, not hidden masked targets.

**CONS_GMAE** retains GMAE inference architecture while averaging reconstruction over two independently node-masked views and adding symmetric stop-gradient cosine alignment of their latent encodings over all valid nodes, then prediction consistency under additional feature masking during supervised training. The original view is the supplied observation, which can already be corrupted; it is not oracle clean truth. **CONS_PRE** keeps the consistency pretraining but removes the supervised consistency term. Stop-gradient views use the same current model, not a separately trained or EMA teacher.

These are task-specific combinations of established masked reconstruction, input-dependent message weighting and consistency regularization. No first-ever architecture claim is made. The [method and equations](REAL_GMAE_VARIANTS.md) document the lineage, coefficients and limitations.

## Data and evaluation scope

The data, features, 32-flow windows, observed endpoint graphs, preprocessing, train/validation/test captures and artificial test corruptions match the preceding study exactly. There are 799,777 verified-label flows among 19,976,700 original CTU-13 rows (4.00%): 442,084 training, 214,880 validation and 142,813 test. Only From-Normal and From-Botnet labels enter; background and other uncertain labels remain excluded. Windows/graphs are formed after that filtering.

All training flows complete before validation begins, and validation completes before testing begins. Hosts may recur, so this is capture/time-disjoint, not host-disjoint. Capture 12 contains NSIS.ay, absent from training; two test captures contain Rbot. These are 2011 university recordings with controlled botnet executions, not arbitrary operational incidents. The target is retrospective completed-flow classification, not measured course-of-action success.

Nine earlier baselines are imported without changing their predictions, summaries or checkpoints. Four new methods are trained from scratch with the same five seeds and maximum 40 supervised epochs, validation patience 8, AdamW learning rate 0.001 and batch 256. EDGE_GMAE, CONS_GMAE and CONS_PRE receive five pretraining epochs; EDGE_DENSE receives none. All have the same trainable parameter count as GMAE, but consistency introduces additional forward passes and is not equal-FLOP training.

## Two clean-data contrasts against GMAE

Each primary difference gives three test captures equal weight within each seed. Positive differences favor the new variant. The wider interval is the 97.5% marginal percentile interval for a nominal two-comparison Bonferroni adjustment.

| Variant | Clean AP | GMAE AP | Difference | 95% interval | 97.5% interval |
|---|---:|---:|---:|---|---|
| EDGE_GMAE | 0.99411 | 0.97830 | +0.01580 | [+0.00097, +0.03710] | [-0.00036, +0.04321] |
| CONS_GMAE | 0.98348 | 0.97830 | +0.00518 | [-0.00184, +0.01441] | [-0.00257, +0.01639] |

EDGE_GMAE has an adjusted interval containing zero; neither improvement nor equivalence is established.

CONS_GMAE has an adjusted interval containing zero; neither improvement nor equivalence is established.

The paired bootstrap uses 1,000 accepted shared draws and rejects 0 draws without positives. Captures 10/11/12 contain 60/6/5 contiguous 64-window blocks (up to 2,048 flows each). It crosses training seeds with within-capture blocks, not independent individual flows. Captures themselves are fixed and not population-resampled; the last two block populations are small.

## All clean capture-macro comparisons

| Method | Attack AP | ROC-AUC | Brier | Balanced accuracy | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| NB | 0.91784 | 0.85354 | 0.07667 | 0.83413 | 0.95910 | 0.29085 |
| LOGIT | 0.63538 | 0.49517 | 0.30425 | 0.47448 | 0.94267 | 0.99370 |
| DAE | 0.86743 | 0.75879 | 0.08797 | 0.78952 | 0.86692 | 0.28788 |
| DENSE | 0.98151 | 0.98084 | 0.04634 | 0.87259 | 0.81632 | 0.07115 |
| GMAE | 0.97830 | 0.97753 | 0.05973 | 0.87242 | 0.81815 | 0.07330 |
| HDC | 0.97578 | 0.97603 | 0.04873 | 0.87185 | 0.81210 | 0.06840 |
| SCR_FLOW | 0.96616 | 0.97284 | 0.06672 | 0.85346 | 0.80520 | 0.09828 |
| CTX_FLOW | 0.95602 | 0.94072 | 0.07728 | 0.80825 | 0.82071 | 0.20422 |
| LOCAL_FLOW | 0.84756 | 0.73533 | 0.09652 | 0.75162 | 0.79046 | 0.28721 |
| EDGE_GMAE | 0.99411 | 0.99566 | 0.04262 | 0.88667 | 0.81600 | 0.04265 |
| EDGE_DENSE | 0.99500 | 0.99569 | 0.04222 | 0.88092 | 0.80805 | 0.04621 |
| CONS_GMAE | 0.98348 | 0.98391 | 0.04661 | 0.87542 | 0.81717 | 0.06632 |
| CONS_PRE | 0.98272 | 0.98357 | 0.04715 | 0.87822 | 0.82896 | 0.07251 |

The numerically largest clean mean AP is EDGE_DENSE (0.99500). This descriptive ranking is not an additional multiplicity-adjusted superiority test. Recall/FPR use clean-validation F1 thresholds fixed across every test condition. AP depends on the selected class prevalence; fixed thresholds face substantial prior shift.

## Capture-specific paired differences

| Capture | EDGE minus GMAE | CONS minus GMAE |
|---|---:|---:|
| 10 | +0.00079 | +0.00082 |
| 11 | +0.04242 | +0.00917 |
| 12 | +0.00420 | +0.00553 |

## Artificial stress and controls

| Condition | GMAE | DENSE | EDGE | EDGE no pretrain | CONS | CONS pretrain only |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.97830 | 0.98151 | 0.99411 | 0.99500 | 0.98348 | 0.98272 |
| missing20 | 0.96844 | 0.97001 | 0.98840 | 0.98786 | 0.97469 | 0.97487 |
| missing50 | 0.93553 | 0.93888 | 0.95962 | 0.95360 | 0.95249 | 0.95311 |
| spoof20 | 0.97663 | 0.98034 | 0.99221 | 0.99168 | 0.98292 | 0.98237 |
| spoof40 | 0.96716 | 0.97386 | 0.98535 | 0.98432 | 0.97781 | 0.97777 |

EDGE_GMAE improves the clean mean AP from 0.97830 to 0.99411, while the no-pretraining EDGE_DENSE control reaches 0.99500. The gated family's clean gain cannot be attributed to masked pretraining alone. At 50% missingness, EDGE_GMAE reaches 0.95962, compared with GMAE 0.93553 and EDGE_DENSE 0.95360; this is a descriptive severity comparison, not another primary test.

CONS_GMAE reaches clean AP 0.98348, compared with CONS_PRE 0.98272. The small numerical increment does not by itself establish a benefit from the extra supervised consistency term.

Missing20/Missing50 and Spoof20/Spoof40 are artificial feature removal or training-normal-centroid replacement on recorded data. They are not observed adversarial deception. The endpoint graph is retained during corruption; inference-time EDGE reweights its existing edges. Matching the representation and objective does not guarantee that feature similarity equals class similarity.

EDGE_GMAE versus EDGE_DENSE measures the pretraining contribution within the gated architecture; EDGE_DENSE versus DENSE changes the gate at no pretraining. CONS_PRE versus GMAE changes both reconstruction-view count and latent alignment, so it does not isolate the cosine term; CONS_GMAE versus CONS_PRE adds supervised consistency. These secondary contrasts use descriptive seed-t intervals conditional on the same test pool.

## Recorded compute and evidence audit

| Method | Trainable parameters / fitted NB statistics | Mean train seconds | Warm CUDA window ms | Incremental VRAM MiB |
|---|---:|---:|---:|---:|
| NB | 58 | 0.004 | 0.470 | 0.011 |
| LOGIT | 29 | 2.225 | 0.499 | 0.012 |
| DAE | 1951 | 7.712 | 0.773 | 0.016 |
| DENSE | 121643 | 21.304 | 2.020 | 0.093 |
| GMAE | 121643 | 24.148 | 2.006 | 0.093 |
| HDC | 125227 | 22.166 | 2.392 | 0.464 |
| SCR_FLOW | 10664 | 16.752 | 1.489 | 0.036 |
| CTX_FLOW | 8257 | 12.017 | 1.248 | 0.036 |
| LOCAL_FLOW | 8257 | 6.922 | 1.272 | 0.036 |
| EDGE_GMAE | 121643 | 24.665 | 2.389 | 0.187 |
| EDGE_DENSE | 121643 | 25.899 | 2.388 | 0.187 |
| CONS_GMAE | 121643 | 29.994 | 1.867 | 0.093 |
| CONS_PRE | 121643 | 23.711 | 1.878 | 0.093 |

The separate common-session audit holds the timing window and measurement loop fixed across all thirteen methods. For the primary variants:

| Method | Host-wall median ms | Host-wall p95 ms | CUDA median ms |
|---|---:|---:|---:|
| GMAE | 2.374 | 2.805 | 1.848 |
| EDGE_GMAE | 2.929 | 3.714 | 2.390 |
| CONS_GMAE | 2.381 | 2.839 | 1.839 |

EDGE_GMAE lowers descriptive clean FPR from 0.07330 to 0.04265, with recall 0.81600 versus 0.81815. It adds pairwise aggregation work: bounded wall median 2.929 ms versus 2.374 ms. CONS_GMAE retains the original inference architecture and records 2.381 ms. These are observed operating-point/cost tradeoffs, not additional adjusted hypotheses.

Baseline timings were recorded in the prior run and are imported; new-method timings come from the follow-up run. They are measurements on the same Ada device, not simultaneous randomized hardware trials. Training totals include pretraining and supervision. The separate common-session GPU audit remeasures all thirteen inference architectures and is used in the resource figures. Its bounded host-wall path includes decoded-input transfers, normalization, graph construction and prediction return; neither timing is full operational detection latency. NB repeats a deterministic closed-form fit in five seed slots, not five stochastic learning outcomes.

The audit checked 975 summary records, 975 raw archives, 6825 recomputed metrics and 65 checkpoint hashes. Imported baseline records remain unchanged. Reliability bins contain all 46,414,225 model-flow predictions. 321 raw scores exceed a probability boundary within the 1e-6 roundoff allowance (maximum excess 1.19e-07); only reliability binning/display clips them to [0,1]. Raw rankings and metrics remain as executed.

Raw data, predictions, checkpoints, logs and PDF review renders remain ignored by Git. Code, concise measurements, hashes, four main figures and a three-page supplement with two extra figures are retained. This extension does not establish operational planning, actual deception handling, network compression, unseen-organization performance or complete mission-state regeneration.

## Sources and artifacts

- [GraphMAE](https://arxiv.org/abs/2205.10803), Hou et al. (2022); [GraphMAE2](https://arxiv.org/abs/2304.04779), Hou et al. (2023). The variants are adaptations, not reproductions or claims to have originated their general ingredients.
- [Official CTU-13 dataset](https://www.stratosphereips.org/datasets-ctu13); García et al. (2014), [An empirical comparison of botnet detection methods](https://doi.org/10.1016/j.cose.2014.05.011).
- [Four main figures](../figures/real_gmae/index.html); [three-page supplement](../supplemental/real_gmae/Supplement_real_gmae.pdf).
- `results/real_gmae/manifest.json` records source/input/checkpoint hashes and baseline origins; `comparisons.json` records statistics and uncertainty scope; `calibration_audit.json` records every bin count.
