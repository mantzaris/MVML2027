# Third-study measurement record

The completed run compares five new learning variants (SCR, CAND_CTX, CAND_LOCAL, LOGIT and DAE), five frozen study-2 models, their scoring ablation, and all eleven original methods. All training and model computation use the attached NVIDIA RTX 5000 Ada Generation. The first two study records remain unchanged.

- `summary.csv`: 1,980 seed/method/condition rows (22 methods × 18 conditions × 5 seeds), from fresh generator seed 62027 and 4,000 snapshots per condition. CUDA latency is warmed batch-one inference plus ranking. Incremental GPU memory excludes resident models/data. Fact-level metrics are NaN for candidate-only models, meaning unavailable.
- `calibration.csv`: ten candidate-success reliability bins for each of four core conditions, method and seed. Counts are candidate instances, not independent worlds. Empty bins carry NaN accuracy.
- `dependency_bins.csv`: candidate Brier stratified by repeated prerequisite roots, with support counts. Same four core conditions. Rare/empty bins cannot support strong conclusions.
- `curves.csv`: new-model optimization traces. Candidate and fact losses differ and are not directly comparable. Only DAE has observed-feature denoising pretraining. All 25 new models reached the 40-epoch cap, so every supervised epoch averages five seeds; these are fixed-budget results, not a convergence claim.
- `manifest.json`: frozen protocol, source commit and hashes, exact train/validation observation hashes, checkpoint provenance, test-input hashes, hardware and actual per-model training cost. Old checkpoints were reused, not retrained in this run.
- `comparisons.json`: SCR minus ID primary macro-regret effect, descriptive secondary effects, 2,000 paired crossed seed/topology bootstrap draws, raw archive hashes and data-integrity checks.
- `pipeline_timing.json`: seven models timed through actual prepared-evidence codec processing, transfers, GPU feature construction, inference/ranking and returned IDs. This is not network transit or complete operational planning latency.
- `replay.json`: seed-37 SCR/DAE/LOGIT checkpoint replay on four core conditions, checking 48,000 model/snapshot results against raw arrays.

Core-condition diagnostic `shadow_union_violation` counts candidate pairs for which the predicted probability of their prerequisite union exceeds the smaller original probability by more than 1e-4. These queries can contain up to ten required facts (training: two to five) and use fixed observation features; they are a structural stress test. Non-core rows are NaN. `gate_entropy` and `gateweight_0..3` describe SCR's latent mixture only. Expert indices are exchangeable across seeds and do not identify deception classes.

Raw per-snapshot arrays are local under ignored `results/runs/v3/`; new weights under ignored `results/checkpoints/v3/`; pilot and logs under ignored `work/`. These are not committed. Regenerate them using the documented GPU runner. Committed source, settings and compact summaries support inspection; reproducing checkpoint/raw-hash audits from a clean clone requires rerunning the studies to restore their local artifacts.

See [method and equations](../../docs/METHOD_V3.md), [results](../../docs/RESULTS_V3.md), [audit](../../docs/V3_AUDIT.md), [four main figures](../../figures/v3/index.html), and [three-page supplement](../../supplemental/v3/Supplement_v3.pdf).
