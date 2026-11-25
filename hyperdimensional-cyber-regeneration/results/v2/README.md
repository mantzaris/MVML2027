# Second-study empirical record

This folder contains the new study's compact outputs. The first study's outputs remain in the parent directory. Its test results motivated the redesign; second-study efficacy is measured on fresh test seed 42027.

The complete grid contains five training seeds × seventeen methods × eighteen conditions = 1,530 rows. Five variants are independently trained per seed: JH (HDC conditional posterior), JD (dense conditional posterior), IH (HDC independent marginals), ID (dense independent marginals), and JH_ERM (JH with mean group risk). JH_product changes candidate scoring using the same JH checkpoint. The eleven original methods reuse their frozen checkpoints and validation thresholds, evaluated anew on the same fresh worlds.

| File | Purpose |
| --- | --- |
| `manifest.json` | Frozen settings, environment, source/configuration hashes, checkpoint hashes, data hashes, test fingerprints and runtime. |
| `summary.csv` | Seed/method/condition metrics, including fact and plan Brier, logical violation, CUDA timing and incremental allocation. |
| `calibration.csv` | Ten plan-probability reliability bins at the targeted reference condition. JH uses its actual joint candidate probabilities. |
| `dependency_bins.csv` | Candidate Brier and mean probabilities/truth by repeated-root count, identifying shared-prerequisite effects. |
| `curves.csv` | Compact epoch-level pretraining/training/validation summaries. |
| `comparisons.json` | Primary and secondary crossed seed/topology bootstrap comparisons; input hashes and per-seed effects. |
| `replay.json` / `example_trace.json` | Restored-checkpoint verification over eight method/condition combinations and an inspectable held-out example with evaluator truth separated. |
| `pipeline_timing.json` | Separate bounded host-wall audit including real packet serialization, transfer, feature fusion, inference and ranking. |

The primary contrast is JH minus ID in equally weighted mean regret over benign0 and targeted/random/coherent0.2, at 64 facts, 2,048 bytes per remote observer, and 20% missing observations. The dense ID control shares the new training data, loss terms, validation rule, and full-regeneration policy. Original P comparisons also change training exposure and repair policy.

`repair_fraction=0.1` is the scenario's legacy sparse budget, applied only to methods labeled `repair_policy=budgeted`. Full models may change all facts. `preservation` is the unnecessary-change rate, not the amount preserved. `logical_violation` is the fraction of child marginals exceeding parent marginals; zero does not imply factual correctness. Candidate probabilities are exact under the assumed conditional forest model, not guaranteed true posteriors under arbitrary correlated deception.

The primary four conditions share the same worlds and topology clusters across every method and seed. Bootstrap cluster draws are therefore shared across seeds and conditions. Macro recall averages condition-specific recall among eligible snapshots; regret and Brier retain no-feasible-candidate snapshots.

All methods use the same sparse report codec. `latency_ms` and `latency_p95_ms` are warmed batch-one CUDA model plus candidate-ranking times. `peak_vram_mb` measures additional allocation above resident models/data. These do not measure a deployed operational planner. `train_seconds` for old methods reports their original reused training cost, and JH_product reuses JH's cost.

Raw arrays reside in ignored `../runs/v2/`; checkpoints are in ignored `../checkpoints/v2/`. Publication figures and the supplemental PDF have their own script, input and output hashes. See [RESULTS_V2.md](../../docs/RESULTS_V2.md) for measured effects and limitations.
