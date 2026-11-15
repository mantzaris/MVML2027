# Evidence-Grounded Generative AI for Underwater Acoustic Reasoning

Alexander V. Mantzaris’s MVML 2027 research project. All work is contained here. Commit on **main only**; the owner performs pushes manually.

## Execution status

Real GPU experiments are running on the authorized existing RunPod. Classifier training is complete for seeds 17/29/43; all 7,418 selected recordings passed acquisition checks and decoded-waveform duplicate checks. Both blinded AI reviewer roles have run on all 270 assessment clips (539/540 valid responses). The corrected development comparison is running; test terminal outcomes remain locked until policy settings, calibration thresholds and analysis rules are frozen.

The user’s constraints are a hard ten-hour research window, approximately 50 GB remote storage, no human reviewers, a small Git repository, and **leave the Pod running when work finishes**. Dollar rate and earlier billed allocation are unknown. No additional service, paid API, repository push, submission or publication is part of this execution.

## Study design and limitations

The original execution prompt and specification are preserved unchanged. The implemented design is defined by the [prospective execution amendment](protocol/execution_amendment.md), [automated scoring rules](protocol/automated_scoring.md), [development decoding correction](protocol/decoding_amendment.md), [statistical analysis](protocol/statistical_analysis.md), and [robustness plan](protocol/robustness_plan.md). The earlier [preparation protocol](protocol/preregistered_plan.md) is historical. These timestamped internal records are not an external preregistration.

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
5. Inspect `pilot_gate.json` and `controller_training.json`; `freeze_test.py --unlock` verifies the complete pre-test caches and freezes software, protocol, prompt, split, reference and reviewer hashes.
6. `run_test_and_stress.sh` executes the clean test matrix, frozen analyses, 360 waveform transformations, evidence interventions, prompt paraphrases and blinded prose audits. Every job is deadline guarded.
7. `make_figures.py` and `build_manuscript.py` render actual aggregate results. Convert the DOCX to PDF with LibreOffice and inspect all pages before finalization.

Inspect [check_readiness.py](scripts/check_readiness.py) for current completion checks. The trained checkpoints, original data and detailed outputs are copied to this local directory and ignored by Git. Model download revisions are pinned; pretrained weight caches remain on the worker. The remote overlay is ephemeral, so it is not the sole artifact copy.

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

The historical `preparation_artifacts.json` and initial draft describe preparation only; they are not evidence of completed experiments. The final artifact manifest will identify the finished deliverables. Author affiliation/contact details were not supplied; they must be added before any eventual submission. The official DOCX’s embedded horizontal margins differ from its prose guidance, as recorded in the conference audit; its working geometry is preserved.
