# Experiment artifacts

The run manifest and compact tables describe actual CUDA experiments. See [the results report](../docs/RESULTS.md) for interpretation and [the implemented protocol](../docs/METHOD_IMPLEMENTATION.md) for scope.

| Artifact | Contents |
| --- | --- |
| `study_manifest.json` | Frozen run arguments, hardware/runtime, source hashes and revision, train/validation fingerprints, measured elapsed time, and protocol deviations. |
| `summary.csv` | One row for each training seed × method × condition; 770 rows in the complete study. |
| `calibration.csv` | Reference-condition plan-success reliability bins for every method and seed. |
| `curves.csv` | Epoch-level pretraining, training, and validation summaries, not per-step logs. |
| `paired_comparison.json` | Primary P-minus-B2 contrast with 2,000 paired crossed seed/topology-cluster bootstrap draws and raw-input hashes. |
| `selected_thresholds.json` | Gate thresholds recovered from validation only, candidate scores, and checkpoint hashes. |
| `example_trace.json` | Deterministically selected seed-37 median-regret reference snapshot, recommendations, evidence changes, and separately labelled evaluator-only truth. |
| `pipeline_timing.json` | Post-training host-wall timing audit of the prepared-evidence processing loop; separate from CUDA kernel timings. |
| `runs/` (ignored) | Raw per-snapshot arrays for paired statistics and local auditing. |
| `checkpoints/` (ignored) | Best validation-selected model weights. |

The reference condition is 64 facts, a 20% requested targeted report-intervention rate, 2,048 bytes allowed per remote observer, and a 10% repair budget. All methods share the same sparse symbolic report codec; no HDC wire-compression improvement is measured. Coherent overwrites can leave an already-true report unchanged, so delivered interventions and actual changed reports are recorded separately.

Recall@3 conditions on at least one truly feasible shared candidate. Regret uses the best shared candidate under evaluator truth. Fact Brier, false-repair, confident-error, and stable-fact modification rates have distinct denominators. False-repair and confident-error fractions are averaged per snapshot with zero for empty denominators; confidence coverage is reported separately. `preservation` stores the *rate of unnecessary changes*, so lower is better. `peak_vram_mb` is incremental allocated GPU memory above the resident baseline, not total memory. `latency_ms` is warmed batch-one CUDA model plus ranking time.

The train/test topology generators are independently seeded. Each test topology cluster contains up to 16 snapshots. The same test pool is reused across model seeds, so the primary bootstrap resamples clusters jointly across selected seeds. Figure intervals describe training-seed variation conditional on that pool; they do not represent the primary crossed-bootstrap intervals.

Large raw arrays, checkpoints, logs, environments, and caches remain local. Source, seeds, package versions, reconstruction commands, and small provenance records are committed so they can be regenerated.
