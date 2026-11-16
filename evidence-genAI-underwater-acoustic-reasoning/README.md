# Evidence-Grounded Generative AI for Underwater Acoustic Reasoning

An anonymous MVML 2027 research manuscript and its reproducibility project. All work is contained here. Commit on **main only**; the owner performs pushes manually.

## Execution status

The real-recording experiments and anonymous eight-page manuscript are complete:
[PDF](manuscript/manuscript.pdf), [editable DOCX](manuscript/manuscript.docx), and [plain text](manuscript/manuscript.txt).
The predefined adaptive-selection success criteria were **not met**. On 100 held-out assessment clips from 93 recorded source groups, the learned policy achieved 13.03% supported physical coverage versus 19.49% for the development-locked entropy comparator: **−6.46 percentage points**, paired source-bootstrap 95% interval **[−11.05, −1.96]**, Holm-adjusted p=0.018.

| Method | Reports retained | Supported coverage | Wrong/unsupported assertions | Whole-test useful coverage |
|---|---:|---:|---:|---:|
| Learned sequential policy | 71% | 13.03% | 2.63% | 9.25% |
| Locked entropy comparator | 68% | 19.49% | 14.52% | 13.25% |
| Deterministic renderer | 65% | 55.00% | 14.37% | 35.75% |

These are automated consistency proxies, with different realized retention rates. Lower composite loss and fewer assertions did not establish greater useful information. The full [main results](results/tables/main_results.json) include every baseline, ablation, classifier seed, confidence interval and cost measure.

Completed work includes three classifier seeds on 2,240 test recordings, 4,590 main terminal reports, 360 transformed clips with 720 fresh AI assessments, 200 paraphrase reports, 175 evidence interventions and 120 AI prose audits. Clean-calibration rejection changes sharply under waveform shifts. Both paraphrases collapse to all-uncertain slots, while saved prose examples still contain physical details. The AI prose auditor confuses vessel-class assertions with mechanical claims in deterministic reports; its flags are retained as unvalidated diagnostics, not a method ranking. See [stress results](results/tables/robustness_results.json), [report audits](results/tables/report_audits.json), and [audit limitations](results/manifests/prose_audit_diagnostics.json).

The scientific source, models, thresholds and main references match the prospective **2026-11-16 00:06:30 UTC** test freeze. Verification passed for 56 portable tests, 58 worker tests, 500 tool fixtures and 100 structural interventions. All 18,581 indexed worker artifacts (7.32 GB), including audio, predictions, failures and 16 checkpoints, match their local copies. They remain outside Git. The [execution ledger](results/manifests/execution_ledger.json) records a conservative 4.98-hour research window within the ten-hour ceiling; this is not a measurement of prior billed allocation. The Pod was left running and idle after the jobs finished. **No push or submission was performed.**

The user’s constraints are a hard ten-hour research window, approximately 50 GB remote storage, no human reviewers, a small Git repository, and **leave the Pod running when work finishes**. Dollar rate and earlier billed allocation are unknown. No additional service, paid API, repository push, submission or publication is part of this execution.

## Figure for an operator audience

[View the figure (PNG)](results/figures/operator_context.png) or [download the vector PDF](results/figures/operator_context.pdf). It shows full vessel-category names, correct classification percentages, sample counts and common confusions. A separate panel translates report quality into supported and wrong/unsupported details across the same 100 clips, with the observed tool-call and completeness trade-offs. Category percentages are recall within each dataset category; they are distinct from the overall 50.0% classifier accuracy.

The [caption](results/figures/operator_context_caption.txt) and [underlying values](results/tables/operator_context.json) specify denominators and limitations. This additional figure uses the saved frozen results and automated reference checks; it is not an operator-performance study. Rebuild it locally with `.venv/bin/python scripts/make_operator_figure.py`.

## Study design and limitations

The original execution prompt and specification are preserved unchanged. The implemented design is defined by the [prospective execution amendment](protocol/execution_amendment.md), [automated scoring rules](protocol/automated_scoring.md), [development decoding correction](protocol/decoding_amendment.md), [reference correction](protocol/reference_correction.md), [statistical analysis](protocol/statistical_analysis.md), and [robustness plan](protocol/robustness_plan.md). The earlier [preparation protocol](protocol/preregistered_plan.md) is historical. These timestamped internal records are not an external preregistration.

UniqueShip fold 1 supplies five civilian classes. Recorded vessels/background days, whole UTC days, a 60-second boundary guard and decoded-waveform hashes are separated across partitions. Parent-recording/deployment identifiers are unavailable, and background device identifiers are missing; conservative recorded-group separation does not establish complete physical independence. Native audio is 32 kHz. Explicit anti-aliased derivatives feed the 20 kHz classifier and 16 kHz generator.

| Partition | Eligible recordings | Assessment clips |
|---|---:|---:|
| Acoustic-fit | 2,459 | — |
| Policy-fit | 1,827 | 100 |
| Development | 384 | 33 |
| Calibration | 508 | 37 |
| Test | 2,240 | 100 |

Assessment sampling permits at most three clips per recorded source, at least ten minutes apart. The test assessment has 93 source groups and only 13 passenger vessels. These limits replace the infeasible original 600-reference design before model outcomes.

Independent numerical estimators and two blinded roles of the pinned Omni model provide **automated proxies**, not human validation or measured hypothetical-human performance. Shared model ancestry and diagnostic anchoring limit reviewer independence. Swapped or internally consistent evidence can still be wrong against the separate reference. No submarine-specific, localization, source-motion, mechanical-speed or calibrated sound-pressure conclusion is supported.

## Reproduction

CPU tooling uses Python 3.10 and `requirements.lock`:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_synthetic.py
```

The remote environment uses Python 3.12, torch 2.8.0+cu128, torchvision 0.23.0+cu128 and torchaudio 2.8.0+cu128. These base packages must already be installed because setup inherits system packages. Additional versions are pinned in [setup_remote.sh](scripts/setup_remote.sh). The GPU is an RTX PRO 4500 Blackwell with approximately 32 GB VRAM. The CPU lock is not a GPU environment lock. The exact deadline in `configs/execution.json` is an execution safeguard; a new run needs its own authorized accounting window.

The execution sequence is:

1. Probe the public release, build the conservative split with `prepare_experiment.py`, and acquire selected members with `download_experiment.py`. Keep private manifests and audio outside Git.
2. Train the anchor/search/seeds using `train_classifier.py`; compute independent references and cards with `prepare_evidence.py`.
3. Validate AI reviewer controls and the development terminal pilot. The corrected terminal decoder uses XGrammar 0.2.9, a common schema and a 1,024-token cap. Original failed pilot outputs remain in ignored raw artifacts.
4. `run_core_after_training.sh` builds the policy/development/calibration subset matrices, runs the bounded planner, and fits the controllers. Content keys prevent incompatible cache reuse.
5. Inspect `pilot_gate.json`, `controller_training.json` and `pretest_design_audit.json`. Run `audit_repeat.py` on the completed grammar smoke; then `freeze_test.py --unlock` verifies the pre-test caches and freezes software, protocol, prompt, split, model, reference and reviewer hashes. `audit_freeze_integrity.py` checks them again after evaluation.
6. `run_test_and_stress.sh` executes the clean test matrix, frozen analyses, 360 waveform transformations, evidence interventions, prompt paraphrases and blinded prose audits. Every job is deadline guarded.
7. Run `audit_cached_outputs.py` and `audit_extended_outputs.py`. On the idle worker, run `collect_worker_finish.py` and `verify_preservation.py --write-worker-index`; sync artifacts locally, then run `verify_preservation.py --verify-local`. Build the figures and manuscript, convert with LibreOffice, and inspect every page. Finish with `summarize_execution.py --finished`, `audit_freeze_integrity.py`, `build_artifact_manifest.py` and `check_readiness.py` after recording the actual manuscript inspection.

Inspect [check_readiness.py](scripts/check_readiness.py) for current completion checks. The trained checkpoints, original data and detailed outputs are copied to this local directory and ignored by Git. Model download revisions are pinned; pretrained weight caches remain on the worker. The remote overlay is ephemeral, so it is not the sole artifact copy.

For a fresh authorized worker run, the concrete commands below reproduce the
stages from the project directory. Use a separate artifact copy with a new
authorized deadline and an initially locked test record; preserve this run's
frozen record and outputs. The shell runners expect the documented worker path.

```bash
bash scripts/setup_remote.sh
.venv/bin/python scripts/probe_release.py
.venv/bin/python scripts/audit_release_metadata.py
.venv/bin/python scripts/prepare_experiment.py
.venv/bin/python scripts/download_experiment.py
.venv/bin/python scripts/download_models.py omni
.venv/bin/python scripts/download_models.py planner
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_guarded.py --name classifier .venv/bin/python scripts/train_classifier.py
.venv/bin/python scripts/prepare_evidence.py --stage references
.venv/bin/python scripts/run_reviewer_controls.py
bash scripts/run_core_after_training.sh
.venv/bin/python scripts/audit_repeat.py
.venv/bin/python scripts/freeze_test.py --unlock
bash scripts/run_test_and_stress.sh
.venv/bin/python scripts/audit_cached_outputs.py
.venv/bin/python scripts/audit_freeze_integrity.py
```

With this run's saved predictions and checkpoints, `analyze_experiment.py`
recomputes the tables in the worker environment without new model generation.
Figures and DOCX construction use the local CPU environment; see
[manuscript build commands](manuscript/FORMAT_NOTES.md). The guarded runners
enforce the configured deadline; direct helper commands do not grant a new
compute allowance.

## Artifact layout

| Location | Contents |
|---|---|
| `protocol/` | Prospective amendments, evaluation rules and test freeze |
| `configs/` | Execution limits, label mapping and locked controller choices |
| `src/underwater_reasoning/` | DSP, separate references, generation, scoring and controller |
| `scripts/` | Acquisition, actual training/evaluation, analysis and document builders |
| `results/manifests/` | Compact integrity, training, pilot and execution summaries |
| `results/tables/`, `results/figures/` | Aggregate empirical outputs |
| `manuscript/` | Official template, verified references and manuscript artifacts |
| `sources/` | Verified bibliography and public-access/release audits |
| `data/`, `checkpoints/`, `results/raw/` | Local private/large artifacts excluded from Git |

The historical `preparation_artifacts.json` describes preparation only. The superseded draft remains in Git history and an ignored local archive. The [completed artifact manifest](results/manifests/research_artifacts.json) identifies the finished deliverables; [preservation](results/manifests/final_preservation.json) and [manuscript inspection](results/manifests/manuscript_inspection.json) record the final checks. The manuscript is anonymous by the user’s instruction; author affiliation and contact details are omitted. The official DOCX’s embedded horizontal margins differ from its prose guidance, as recorded in the conference audit; its working geometry is preserved.
