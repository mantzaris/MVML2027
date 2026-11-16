# Evidence-Grounded Generative AI for Underwater Acoustic Reasoning

An anonymous MVML 2027 research manuscript and its reproducibility project. All work is contained here. Commit on **main only**; the owner performs pushes manually.

## Identification study and eight-page paper

The active manuscript addresses a practical acoustic-assistance question: **which vessel category is present, when should the system refer the recording for review, and does a generative answer preserve the identification?** It uses the full conference allowance of **eight pages, including references**: [PDF](manuscript/manuscript.pdf), [editable DOCX](manuscript/manuscript.docx), [plain text](manuscript/manuscript.txt).

The proposed method combines a frozen acoustic representation, measured spectral/modulation features and source-balanced training prototypes. Its fresh evaluation contains 235 recordings from 218 previously unused source groups, six transformations of each recording, 705 generative category answers and a supplementary 72-recording challenge from untrained vessel categories. Models and thresholds were frozen before fresh test inference. The supplementary challenge was added after the primary results were known and has its own pre-inference freeze.

| Method | Correct category, all 235 | Macro-F1 | Correct labels issued | Wrong labels issued | Referred |
|---|---:|---:|---:|---:|---:|
| Single acoustic model | 55.3% | 0.523 | 107 | 61 | 67 |
| Three-model ensemble | 62.1% | 0.586 | 102 | 51 | 82 |
| Matched acoustic-only refit | 56.2% | 0.546 | 105 | 65 | 65 |
| Proposed evidence fusion | 60.4% | 0.583 | 96 | 36 | 103 |

The primary success criterion was **not met**: fusion minus the development-selected ensemble is −0.003 macro-F1, with paired source-bootstrap 95% interval [−0.073, 0.066]. This establishes neither superiority nor equivalence. The first two columns score every recording before referral; the final three count outcomes at each method's separately frozen reporting threshold.

The category tradeoff matters: fusion identifies 62% of tankers versus the ensemble's 18%, but only 50% of cargo vessels versus 80%. The matched acoustic-only refit also identifies 62% of tankers, so that gain cannot be attributed specifically to the added physical evidence. Removing physical features reduces fusion macro-F1 from 0.583 to 0.517; this is an exploratory ablation, not a replacement primary endpoint. Removing prototypes makes little difference (0.582).

Clean calibration fails under severe noise: at 0 dB SNR fusion issues 205 labels, of which 155 are wrong. On 72 recordings from six untrained categories, it still assigns 35 trained-category labels. Generative fused-evidence prompting identifies 99/235 recordings, changes seven correct numeric labels into wrong labels, withholds 39 additional correct labels, and corrects three prior errors. These outcomes distinguish incorrect overrides from cautious withholding.

The fresh cohorts share the original collection and test recording days, while excluding every earlier source group and all fitting/selection/calibration days. They are **not independent deployments or the official fold test set**. Category scoring uses public dataset labels, with no human reviewers or measured human-operator benefit. See [full results](results/tables/identification_results.json), [exploratory diagnostics](results/tables/identification_diagnostics.json), [untrained categories](results/tables/untrained_categories_results.json), and the [prospective follow-up](protocol/identification_followup.md).

### Figures for an operator audience

The [operator overview (PNG)](results/figures/identification_operator_results.png) and [vector PDF](results/figures/identification_operator_results.pdf) show named categories, correct-category percentages, sample counts, and correct/wrong/referred outcomes per 100 recordings. The manuscript includes [category comparisons](results/figures/identification_class_accuracy.pdf), [recording-degradation results](results/figures/identification_robustness.pdf), and the new [agentic contribution figure (PNG)](results/figures/agentic_reasoning_contribution.png) / [vector PDF](results/figures/agentic_reasoning_contribution.pdf).

The agentic figure shows the implemented **observe → select a tool → update evidence → query again or stop → generate a cited report** loop. The earlier learned controller makes 101 logical queries versus the heuristic's 200, with one wrong/unsupported detail versus nine, but only 37 supported details versus 53. Its LLM-planner and deterministic controls are also shown. These are acquisitions in cached replay, not measured runtime savings. A separate panel shows the fresh fixed-input generative experiment: fused evidence yields 24 fewer wrong labels and 24 more withheld answers than the acoustic prior, with 99 correct in both. The two cohorts are kept separate; the conventional fusion classifier is not relabeled as an adaptive agent. See the [caption](results/figures/agentic_reasoning_contribution_caption.txt) and [auditable aggregate values](results/tables/agentic_reasoning_contribution.json). Rebuild with `.venv/bin/python scripts/make_agentic_figure.py`; it makes no model calls.

The [two opposing real spectrogram examples](results/figures/identification_examples.pdf) remain in the artifact bundle. They were selected after evaluation and are illustrative, not representative estimates; the agentic figure replaces them in the eight-page manuscript.

### Reproducing the follow-up

The original fitted backbones, partitions, arrays and labels are prerequisites. Use Python 3.10 with `requirements-identification.lock` for portable analysis, tests and manuscript building; the worker retains the original Python 3.12 / torch 2.8.0+cu128 environment. Do not overwrite this run's frozen records. A fresh execution requires a separate artifact copy and a newly authorized deadline.

```bash
.venv/bin/python -m pip install -r requirements-identification.lock
.venv/bin/python -m pytest -q
# Before fresh evaluation, on the documented worker:
.venv/bin/python scripts/prepare_identification_cohort.py
.venv/bin/python scripts/run_guarded.py --name identification_train_features .venv/bin/python scripts/extract_identification_features.py fitting
.venv/bin/python scripts/run_guarded.py --name identification_training .venv/bin/python scripts/train_identification.py
.venv/bin/python scripts/run_guarded.py --name identification_omni_pilot .venv/bin/python scripts/run_identification_omni.py --pilot
.venv/bin/python scripts/freeze_identification.py
bash scripts/run_identification_followup.sh
# Separate exploratory challenge, with unchanged models and thresholds:
.venv/bin/python scripts/prepare_untrained_categories.py
.venv/bin/python scripts/run_guarded.py --name untrained_categories .venv/bin/python scripts/evaluate_untrained_categories.py
```

`summarize_identification_diagnostics.py` computes explicitly post-hoc comparisons from saved outputs. `make_identification_figures.py`, `make_agentic_figure.py` and `build_identification_manuscript.py` rebuild the active paper; see [format and conversion instructions](manuscript/FORMAT_NOTES.md). The original `build_manuscript.py` rebuilds the **superseded report-support manuscript**, so use the identification builder for the current paper.

The [follow-up audit](results/manifests/identification_audit.json) checks source boundaries, frozen hashes, raw prediction accounting and preservation. The [completion record](results/manifests/identification_completion.json) records final validation and timing. Original and new audio, checkpoints, detailed outputs and logs are preserved locally outside Git. The ten-hour deadline applies to this existing research window; earlier billed allocation and dollar cost are unknown. The Pod is left running by instruction, and no push or submission is performed.

## Original report-support study (retained)

The original report-support experiment is retained unchanged in its frozen source and tables. Its former manuscript is preserved in Git history and an ignored local archive; the active manuscript above incorporates it as motivation.
The predefined adaptive-selection success criteria were **not met**. On 100 held-out assessment clips from 93 recorded source groups, the learned policy achieved 13.03% supported physical coverage versus 19.49% for the development-locked entropy comparator: **−6.46 percentage points**, paired source-bootstrap 95% interval **[−11.05, −1.96]**, Holm-adjusted p=0.018.

| Method | Reports retained | Supported coverage | Wrong/unsupported assertions | Whole-test useful coverage |
|---|---:|---:|---:|---:|
| Learned sequential policy | 71% | 13.03% | 2.63% | 9.25% |
| Locked entropy comparator | 68% | 19.49% | 14.52% | 13.25% |
| Deterministic renderer | 65% | 55.00% | 14.37% | 35.75% |

These are automated consistency proxies, with different realized retention rates. Lower composite loss and fewer assertions did not establish greater useful information. The full [main results](results/tables/main_results.json) include every baseline, ablation, classifier seed, confidence interval and cost measure.

Completed work includes three classifier seeds on 2,240 test recordings, 4,590 main terminal reports, 360 transformed clips with 720 fresh AI assessments, 200 paraphrase reports, 175 evidence interventions and 120 AI prose audits. Clean-calibration rejection changes sharply under waveform shifts. Both paraphrases collapse to all-uncertain slots, while saved prose examples still contain physical details. The AI prose auditor confuses vessel-class assertions with mechanical claims in deterministic reports; its flags are retained as unvalidated diagnostics, not a method ranking. See [stress results](results/tables/robustness_results.json), [report audits](results/tables/report_audits.json), and [audit limitations](results/manifests/prose_audit_diagnostics.json).

The original scientific source, models, thresholds and main references match the prospective **2026-11-16 00:06:30 UTC** test freeze. Its verification passed for 56 portable tests, 58 worker tests, 500 tool fixtures and 100 structural interventions. All 18,581 indexed original worker artifacts (7.32 GB), including audio, predictions, failures and 16 checkpoints, match their local copies. They remain outside Git. The historical [execution ledger](results/manifests/execution_ledger.json) records the original stage's conservative 4.98-hour window; follow-up timing is recorded separately in `identification_completion.json`. These are not measurements of billed allocation. **No push or submission was performed.**

The user’s constraints are a hard ten-hour research window, approximately 50 GB remote storage, no human reviewers, a small Git repository, and **leave the Pod running when work finishes**. Dollar rate and earlier billed allocation are unknown. No additional service, paid API, repository push, submission or publication is part of this execution.

## Original report-support figure

[View the figure (PNG)](results/figures/operator_context.png) or [download the vector PDF](results/figures/operator_context.pdf). It shows full vessel-category names, correct classification percentages, sample counts and common confusions. A separate panel translates report quality into supported and wrong/unsupported details across the same 100 clips, with the observed tool-call and completeness trade-offs. Category percentages are recall within each dataset category; they are distinct from the overall 50.0% classifier accuracy.

The [caption](results/figures/operator_context_caption.txt) and [underlying values](results/tables/operator_context.json) specify denominators and limitations. This additional figure uses the saved frozen results and automated reference checks; it is not an operator-performance study. Rebuild it locally with `.venv/bin/python scripts/make_operator_figure.py`.

## Original study design and limitations

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

## Original study reproduction

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
