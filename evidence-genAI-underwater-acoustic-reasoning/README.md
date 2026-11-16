# Evidence-Grounded Generative AI for Underwater Acoustic Reasoning

Alexander V. Mantzaris’s MVML 2027 research project. All work is contained here. Commit on **main only**; the owner performs pushes manually.

## Execution status

Real GPU experiments are running on the authorized existing RunPod. Classifier training is complete for seeds 17/29/43; all 7,418 selected recordings passed acquisition and decoded-waveform duplicate checks. The corrected development generation passed its feasibility gate. All 540 main AI assessments and 2,890 pre-test terminal reports are saved; four terminal truncations remain as penalized failures. The controller, calibration thresholds and analysis were frozen at **2026-11-16 00:06:30 UTC**, before test generation, with the entropy heuristic as the locked comparator. Held-out test and stress evaluation are now running.

The user’s constraints are a hard ten-hour research window, approximately 50 GB remote storage, no human reviewers, a small Git repository, and **leave the Pod running when work finishes**. Dollar rate and earlier billed allocation are unknown. No additional service, paid API, repository push, submission or publication is part of this execution.

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

The remote environment uses Python 3.12, torch 2.8.0+cu128, torchvision 0.23.0+cu128 and the pinned packages in [setup_remote.sh](scripts/setup_remote.sh). The GPU is an RTX PRO 4500 Blackwell with approximately 32 GB VRAM. The CPU lock is not a GPU environment lock. The exact deadline in `configs/execution.json` is an execution safeguard; a new run needs its own authorized accounting window.

The execution sequence is:

1. Probe the public release, build the conservative split with `prepare_experiment.py`, and acquire selected members with `download_experiment.py`. Keep private manifests and audio outside Git.
2. Train the anchor/search/seeds using `train_classifier.py`; compute independent references and cards with `prepare_evidence.py`.
3. Validate AI reviewer controls and the development terminal pilot. The corrected terminal decoder uses XGrammar 0.2.9, a common schema and a 1,024-token cap. Original failed pilot outputs remain in ignored raw artifacts.
4. `run_core_after_training.sh` builds the policy/development/calibration subset matrices, runs the bounded planner, and fits the controllers. Content keys prevent incompatible cache reuse.
5. Inspect `pilot_gate.json`, `controller_training.json` and `pretest_design_audit.json`. Run `audit_repeat.py` on the completed grammar smoke; then `freeze_test.py --unlock` verifies the pre-test caches and freezes software, protocol, prompt, split, model, reference and reviewer hashes. `audit_freeze_integrity.py` checks them again after evaluation.
6. `run_test_and_stress.sh` executes the clean test matrix, frozen analyses, 360 waveform transformations, evidence interventions, prompt paraphrases and blinded prose audits. Every job is deadline guarded.
7. Run `audit_cached_outputs.py`, `make_figures.py`, `make_real_examples.py`, and `build_manuscript.py` on completed results. Convert the DOCX to PDF with LibreOffice and inspect all pages before finalization.

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

The historical `preparation_artifacts.json` and initial draft describe preparation only; they are not evidence of completed experiments. The final artifact manifest will identify the finished deliverables. The manuscript is anonymous by the user’s instruction; author affiliation and contact details are omitted. The official DOCX’s embedded horizontal margins differ from its prose guidance, as recorded in the conference audit; its working geometry is preserved.
