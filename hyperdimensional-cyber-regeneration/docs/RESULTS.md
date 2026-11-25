# Experimental results and scope

This report is generated from completed experiment outputs. It reports observed results, including null or negative comparisons; no missing result is imputed.

Hardware: NVIDIA RTX 5000 Ada Generation; PyTorch 2.9.1+cu128, CUDA 12.8. Five training seeds (11, 23, 37, 51, 71), 20000 training snapshots, 2000 validation snapshots, 4000 shared test snapshots per condition, 5 masked-feature pretraining epochs and at most 40 supervised epochs (patience 8). Recorded study wall time: 0.26 hours.

## Primary paired comparison

At the reference condition, 3,967/4,000 snapshots contain at least one truly feasible candidate. The remaining 33 snapshots (0.825%) have no feasible candidate; they are excluded only from the conditional Recall@3 denominator and retained for regret and other snapshot metrics.

Prespecified reference condition: 64 facts, targeted corruption intervention fraction 0.2, 2,048-byte budget per remote observer and 10% repair budget. P is the proposed decision-gated HDC model; B2 is the dense decision-aware reconstruction baseline.

- Recall@3: P 99.607%, B2 100.000%; paired P−B2 difference **-0.393 percentage points**, 95% interval [-0.590, -0.232]. The interval supports lower recall in this controlled setting.
- Candidate-oracle regret (lower is better): P 0.15732, B2 0.03740; paired P−B2 difference **+0.11992**, 95% interval [+0.11335, +0.12638]. The interval supports higher regret in this controlled setting.

Intervals use 2,000 paired bootstrap draws over the crossed factors of five training seeds and 250 topology clusters. A cluster draw is shared across the resampled seeds because all models use the same test set. All snapshots inside a sampled cluster are retained. Recall is conditional on at least one truly feasible candidate and is recomputed from eligible numerators and denominators. Regret includes all snapshots. These are descriptive 95% percentile intervals, without multiplicity adjustment. The fixed synthetic generator and finite shared training/validation set are outside their uncertainty scope.

## Reference-condition results

Recall entries are the mean ± sample standard deviation over training seeds, not confidence intervals. Other columns are seed means. Confidence error is the mean per-snapshot error rate among beliefs below 0.1 or above 0.9; coverage is the fraction of beliefs meeting that threshold. Zero-coverage snapshots contribute zero to the error statistic.

| Method | Recall@3 (%) | Regret ↓ | Brier ↓ | Confidence error (%) ↓ | Confidence coverage (%) |
|---|---:|---:|---:|---:|---:|
| B0: Symbolic constrained prior | 93.093 ± 0.000 | 0.41978 | 0.16647 | 9.90 | 45.90 |
| B1: GraphMAE-adapted reconstruction | 100.000 ± 0.000 | 0.03545 | 0.01689 | 0.59 | 93.46 |
| B2: Dense decision-aware baseline | 100.000 ± 0.000 | 0.03740 | 0.01775 | 0.57 | 92.79 |
| B3: HDC associative cleanup | 90.345 ± 0.000 | 0.49152 | 0.12767 | 6.94 | 56.32 |
| B4: P decoder, full repair (inference ablation) | 99.985 ± 0.014 | 0.05128 | 0.02199 | 0.76 | 91.43 |
| B5: P decoder, entropy selection (inference ablation) | 92.619 ± 0.133 | 0.44100 | 0.11273 | 6.22 | 63.07 |
| P: Decision-gated HDC regeneration | 99.607 ± 0.052 | 0.15732 | 0.06642 | 1.19 | 62.15 |
| random_gate: P decoder, random selection (inference ablation) | 92.100 ± 0.029 | 0.45220 | 0.11728 | 5.82 | 59.80 |
| no_provenance: Direct source/age-channel ablation | 99.612 ± 0.085 | 0.15896 | 0.06580 | 1.11 | 62.07 |
| no_semantic: Semantic-loss ablation | 99.657 ± 0.014 | 0.16022 | 0.06552 | 1.19 | 62.14 |
| dense_gate: Dense representation with learned gate | 99.561 ± 0.058 | 0.15992 | 0.06605 | 1.16 | 62.07 |

## Secondary comparisons and original hypotheses

The following reference-condition effects are descriptive seed means. All contrasts are P minus the named comparator. A positive recall difference favors P; a positive regret difference favors the comparator. No secondary significance claim is made.

| Comparator | Recall difference (percentage points) | Regret difference | Interpretation |
|---|---:|---:|---|
| B5 | +6.988 | -0.28368 | Same P decoder, entropy selection; same maximum repair budget. |
| random_gate | +7.507 | -0.29488 | Same P decoder, random selection; same maximum repair budget. |
| dense_gate | +0.045 | -0.00260 | Separately trained dense model with the gated objective; representation ablation. |
| B4 | -0.378 | +0.10604 | Same P decoder with full reconstruction; difference quantifies the cost of sparse selection. |

The learned-selection advantage does not extend to all tested observation conditions. Training and validation use 20% targeted interventions. At zero injected attack, mean regret is 0.25363 for P versus 0.04393 for entropy selection; under random attacks it is 0.34795 versus 0.18112. Under coherent attacks the means are 0.20049 versus 0.19980. These descriptive diagnostics show a substantial dependence on the training-matched corruption mechanism; the H1 finding below is confined to the reference condition.

| Hypothesis | Status supported by this experiment |
|---|---|
| H1: selective regeneration improves viable-candidate recovery at fixed repair budget | Narrowly supported descriptively: learned selection has higher mean Recall@3 than entropy and random selection at the same maximum repair budget in every training seed. This is a static candidate-ranking finding; validation thresholds can use fewer than the maximum repairs. |
| H2: decision-impact teacher improves planning beyond generic reconstruction | Not isolated. There is no otherwise identical retrained model without the teacher. P versus B2 changes representation and repair policy, while B4/B5/random share teacher-trained weights. The primary comparison cannot attribute a benefit to teacher supervision. |
| H3: HDC improves quality–bandwidth tradeoffs | Untested as specified: all methods share the symbolic report codec. dense_gate addresses the internal HDC feature contribution, not learned-message or wire-codec superiority. |
| H4: provenance/calibration reduces false certainty at matched coverage | Untested as specified: confidence-error and coverage diagnostics are reported, but coverage is not matched and the source/age ablation retains the common fused prior. |

Full reconstruction and sparse selection answer different budget questions. A useful gate relative to entropy or random selection can coexist with worse regret than full reconstruction. Any small P-versus-dense_gate difference is a descriptive result in this generator, not evidence of universal HDC superiority. The primary comparison and all ablations remain visible regardless of their direction.

## Interpretation boundaries

- B1 adapts GraphMAE-style masked feature reconstruction to this synthetic task; it is not an official benchmark reproduction. B2 adds decision and symbolic training terms. The separately trained dense_gate model is the appropriate representation ablation.
- B4 (full repair), B5 (entropy selection), and random_gate reuse P's weights and change inference selection. They are inference ablations, not independently optimized algorithms.
- The direct source/age-channel ablation removes explicit source distinctions and ages but retains the shared fused provisional belief. It does not remove every indirect provenance influence.
- All methods receive the same sparse symbolic report serialization. HDC is an internal associative memory; bandwidth curves do not demonstrate an HDC wire-compression advantage.
- The benchmark contains independent decision snapshots with a noisy prior-time observation and genuine state changes. Candidates are abstract conjunctions with terminal utilities. It does not establish sequential planning, temporal recovery delay, real cyber effectiveness, or distributed consensus.
- Coherent attack severity counts interventions; some overwritten report values remain correct. The compact summary also records actual changed-report fractions.
- The complete decoder runs before selecting repairs. Fewer changed beliefs do not establish inference compute savings. Preservation in the raw summaries is an unnecessary-change rate and is better when lower.
- CUDA latency is warmed batch-one model inference plus candidate scoring and excludes serialization, transfer, and end-to-end system overhead. Peak VRAM is incremental above already resident models and data, not total deployment memory.
- The experiment tests a particular composition of prior methods and the one-fact regret-benefit gate target. It does not establish a first-ever novelty claim or satisfy the complete proposed DARPA capability.

## Reproducibility and artifacts

The compact summary is in `results/summary.csv`; bin-level calibration and training curves are in `results/calibration.csv` and `results/curves.csv`. `results/study_manifest.json` records actual settings, hardware, source digests and data hashes. `results/paired_comparison.json` records the paired contrasts, bootstrap design, per-seed effects and hashes of raw inputs. The raw per-snapshot arrays and checkpoints remain local and ignored by Git. Regenerate this report with `python scripts/analyze_results.py` after the complete GPU study.

The separate 2–4 page supplemental PDF is prepared as requested. Public conference instructions checked for this project do not verify acceptance of a separate supplement; confirm with the conference before upload.

## Prepared-evidence pipeline timing

A separate audit on training seed 37 and one fixed 64-fact reference snapshot measured 50 repeats after 5 warmups. It includes actual report packing/unpacking, host-to-device evidence and metadata transfer, GPU feature fusion, inference/ranking, and top-three IDs returned to the host. All eleven reconstructed prediction paths were checked against their original CUDA inputs before timing.

P median: 5.247 ms (10th–95th percentiles 5.049–6.208 ms); B2 median: 4.305 ms (10th–95th percentiles 4.043–4.814 ms). The common remote payload was 1340 bytes.

This bounded host-wall measurement excludes sensing, bandwidth admission, graph/catalogue construction, transport delay and cold model loading. Repetition percentiles describe timing variability for that one snapshot, not seed or environment uncertainty. All methods and the exact scope are in `results/pipeline_timing.json`.
