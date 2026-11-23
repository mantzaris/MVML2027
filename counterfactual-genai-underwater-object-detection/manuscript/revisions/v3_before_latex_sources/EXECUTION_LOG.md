# Research execution log

## 23 November 2026 start

The user authorized executing the full planned study and requested persistent checkpoints for recovery after power loss. All new work stays inside this submission directory. The previous submission is read-only and unchanged.

The project-local CUDA environment is being installed. The local RTX 5000 Ada is available and selected by UUID. In parallel, the implementation is developing resumable storage, CUDA signal representations and controlled synthesis, and a real-data provenance/access audit.

The initial UniqueShip small-release archive returned a public download-quota error; alternate access is being investigated. No research training or test evaluation has run yet. Model development must not consume final test outcomes. Machine-readable status is in manifests/research_status.json; subsequent durable stage events and checkpoints will be recorded under logs/ and checkpoints/.

## Environment and numerical pilot completed

Installed a dedicated CUDA-enabled .venv and pinned dependency/model revisions. The real CUDA restart test passed across separate processes, including fallback from a damaged latest checkpoint. Fourier/STFT/wavelet numerical checks passed; bounded HHT convergence was inadequate on noisy fixtures and is now exploratory. A64-step conditional-flow functionality pilot completed and saved a durable checkpoint. It is not a scientific performance result.

The accessible PORTIA v2 cohort was chosen before any real model evaluation. Conflicting cross-task labels were excluded, and vessel identities plus UTC days are disjoint between partitions. Native audio acquisition is in progress. The execution amendment records these decisions and the limited number of independent test days.


## Full-horizon supervisor and pre-outcome audit

Background-flow training, all3synthetic temporal detector runs and all12supplemental representation heads completed on the RTX5000 Ada. Synthetic train/development feature caches are durable. Agent single and batched protocol fixtures passed; controlled waveform checks found strong overlapping-source collateral suppression, so no source-selective counterfactual claim is made.

`bash scripts/resume_research.sh` is the project-local continuation command. The active supervisor reuses the existing download and will advance as verified split manifests appear. Completed stages are recorded in `runs/stages/`; current work appears in `runs/horizon_status.json`. The current session is still supervising execution, not merely handing off a background job.

Metadata-only dependence auditing found one connected day–vessel component across five test days. The analysis now reserves population inference: primary AP difference is descriptive; the day bootstrap is a separate sensitivity analysis. Comparator selection uses development AP of mean calibrated seed probabilities, matching the final endpoint. These decisions precede real model outcomes.


### Global manifest freeze binding — 2026-11-23T18:22:19.002766+00:00

The test gate now pins `data/manifests/real.jsonl`, the row source preferred by `load_rows` after acquisition, and asserts exact content and ordering equality against each of the four split manifests. The future freeze-stage input signature also includes the global manifest. Metadata-only validation passed for all 6,270 rows (3,735 train, 735 development, 600 calibration, 1,200 test); all existing completed stage signatures remain unchanged. Receipt: `manifests/freeze_manifest_validation.json`. No audio, test outcomes, models, data decoding code, or training sources were accessed or changed, and no test freeze was created.

Four metadata-only stage-provenance regression tests also passed; 14 completed horizon stage signatures were unchanged.


### First-freeze policy and test-artifact guards — 2026-11-23T18:24:57.959506+00:00

`freeze_protocol.py` now compares every policy source fingerprint and central-intervention index fingerprint to the exact provenance construction in `run_policies.py`. The first-freeze guard covers real/foundation/synthetic test feature directories, central/full intervention directories, test policy/prediction namespaces, supplemental and robustness outputs, and final tables, including partial shards without indices. Metadata-only validation checked 6 completed policy summaries and the available central evidence indices; 6 calibration policy summaries remain pending and must pass at freeze. Changed source/evidence fingerprints were rejected, and seven isolated project-temporary artifact fixtures were detected. Canonical test-derived namespaces remain absent. Receipt: `manifests/freeze_policy_binding_validation.json`. No method sources, models, GPU jobs, test inference, or test-unlock state changed.


Before runtime or final test outcomes, added a first-freeze requirement that all15 measured runtime passes match their archived calibrated predictions/actions under existing benchmark tolerances. A mismatch stops freeze for an audit rather than pairing inconsistent cost and accuracy. Experimental inference and tolerance definitions are unchanged.


## Full-horizon completion — 2026-11-23T20:04:29.237400+00:00

All38 execution stages completed. All185 frozen artifacts and38stage receipts passed final integrity verification; all12final PDF pages and5main figures passed visual review. Main result is negative (agent ensemble AP0.573788, lockedwavelet0.590610). Independent audits passed real, robustness, synthetic, supplemental and semi-synthetic results. Supplemental seed-field metadata is corrected in a separate traceable export; original measurements remain unchanged. Presentation-only revision2 corrects two-source/family labels and preserves revision1. Completion receipt and recovery instructions are saved locally; no submission, Git commit or push was performed.
