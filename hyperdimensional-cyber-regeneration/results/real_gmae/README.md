# Exploratory GraphMAE follow-up artifacts

This study tests two changes to the captured-traffic GraphMAE adaptation: **EDGE_GMAE** uses available-coordinate overlap and discrepancy to weight observed endpoint edges; **CONS_GMAE** uses two-view reconstruction/latent consistency and an additional supervised masked-view prediction regularizer. EDGE_DENSE removes masked pretraining from the gated architecture; CONS_PRE removes the additional supervised consistency term. The method names denote specific experimental adaptations of established ideas, not first-ever general architectures.

The same CTU-13 test captures were already inspected in the preceding study. This is an **exploratory reused-test follow-up** even though every new setting is frozen and all twenty new training runs plus validation threshold selections finish before new test inference. Multiplicity-adjusted intervals cannot remove adaptive reuse. New independent captures or datasets are needed for confirmation.

## Layout and provenance

- `manifest.json` records the source freeze, executed settings, exact training/validation/test input fingerprints, training-only preprocessing, method origins, source/data hashes and all fitted-model records. It contains **20 new checkpoints** and **45 imported reference checkpoints**, not 65 newly trained models. Baseline manifest, summary, training-curve and calibration hashes are retained.
- `summary.csv` contains **975 evaluation rows**: 675 unchanged prior rows plus 300 new rows. Each of thirteen methods is evaluated in five seed slots, five observation conditions and three fixed test captures. The Gaussian naive-Bayes fit is deterministic; five seed slots do not create five stochastic fits.
- `curves.csv` preserves imported histories and records new pretraining, total supervised training, base supervised loss, extra regularizer and common validation loss. CONS has an additional training objective, so total training curves are not identical-loss comparisons. Actual stopping/checkpoint metadata determine the number of contributing epochs.
- `comparisons.json` records the two planned clean macro-AP contrasts against GMAE, 1,000 shared crossed seed/time-block bootstrap draws, ordinary 95% intervals and 97.5% marginal percentile intervals for a nominal two-comparison adjustment. Captures are equally weighted and fixed; the 60/6/5 block populations do not represent independent organizations. Remaining contrasts are descriptive.
- `calibration.csv` contains 9,750 reliability-bin records. `calibration_audit.json` verifies every case contributes every flow exactly once, for **46,414,225 model/flow predictions**. This is repeated model evaluation of 142,813 unique held-out flows, not millions of independent test samples. Raw ranking scores and metrics remain unchanged; only bin membership/display clips permitted one-ulp probability roundoff to [0,1].
- `gpu_audit.json` records a common-session comparison of all thirteen architectures: 50 bounded host-wall repetitions and 30 warm CUDA repetitions. The wall path includes decoded-input transfers, normalization, endpoint-graph construction, inference and return of probabilities. It excludes CSV/flow extraction, waiting for completion, cold loading and network transit. Actual transfers are 2,784 input bytes and 128 output bytes per 32-flow window; this is not a network codec/compression result.
- The same GPU audit replays seed-37 EDGE_GMAE, EDGE_DENSE, CONS_GMAE, CONS_PRE and GMAE across all fifteen test cases: **75 replays and 3,570,325 model/flow evaluations**. It checks inputs, checkpoint/raw hashes, paired metadata, predictions, AP and Brier without refitting.

Raw original data stays in ignored `data/external/ctu13/`; prior raw predictions/checkpoints stay in ignored `results/runs/real/` and `results/checkpoints/real/`; new arrays/weights stay in ignored `results/runs/real_gmae/` and `results/checkpoints/real_gmae/`. Logs, numerical-development files and PDF renders stay in ignored `work/`. Do not force-add any of these paths. The tracked JSON/CSV files are compact measurements and provenance, not large raw logs.

## Reproduction and verification

A clean clone contains compact results but not the downloaded tables, prepared arrays, predictions or weights required for replay. First reproduce the preceding [captured-data study](../real/README.md) with its documented environment to create those reference artifacts. The follow-up imports their exact evidence and fails if its data/source/input checks disagree. Source and device differences can produce a distinct replication record; do not mislabel a new replication as the original executed run.

The stage order for a new follow-up run, from the project directory, is:

```bash
python3 scripts/run_gpu.py -- -m unittest discover -s tests -p test_real_gmae_variants.py -v
python3 scripts/run_gpu.py -- scripts/run_real_gmae.py
python3 scripts/run_gpu.py -- scripts/audit_real_gmae.py
.venv/bin/python scripts/analyze_real_gmae.py
.venv/bin/python scripts/make_real_gmae_results.py
.venv/bin/python scripts/make_real_gmae_supplement.py
.venv/bin/python scripts/verify_real_gmae_artifacts.py
```

The runner intentionally refuses to overwrite an existing completed `results/real_gmae/manifest.json`. For a fresh replication, use a separate copy of the project and archive that copy's existing compact `results/real_gmae/` directory under its ignored `work/` before running. Keep the authoritative prior results intact. The final verifier is read-only and needs the local ignored arrays/checkpoints; it does not train models, perform inference or rerun the bootstrap.

All learning, graph weighting, reconstruction and inference use the RTX 5000 Ada launcher, with no CPU fallback. CPU handles parsing, serialization, statistical analysis and rendering. Equal maximum epochs and matched parameter counts do not mean equal FLOPs: the consistency variant uses more forward passes, while the edge variant computes pairwise feature compatibility.

The task remains retrospective classification of the 799,777 verified-label flows, approximately 4% of 19,976,700 original CTU-13 rows. Unknown background is excluded before constructing windows and graphs; hosts can recur across the chronological splits. The data contains controlled botnet executions in recorded university traffic, not measured course-of-action success. Feature masking and centroid replacement are artificial telemetry stresses, not observed real deception.

See the [method and equations](../../docs/REAL_GMAE_VARIANTS.md), [results report](../../docs/RESULTS_REAL_GMAE.md), [four main figures](../../figures/real_gmae/index.html), and [three-page supplement with two extra figures](../../supplemental/real_gmae/Supplement_real_gmae.pdf). The results report and verification outputs establish execution status and measured conclusions; this artifact guide does not assume either new variant improves.
