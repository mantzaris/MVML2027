# Hyperdimensional cyber regeneration

CUDA research project for MVML 2027, informed by DARPA HYPER-CAP (DPA27BZ02-DV001).

**Manuscript title:** Toward Vector-Symbolic and Neurosymbolic Cyber Planning under Adversarial Uncertainty.

The implemented study learns a decision-impact gate for repairing beliefs in a synthetic prerequisite-planning task. It builds on GraphMAE-style feature pretraining, decision-focused learning, semantic loss, and fixed-codebook hyperdimensional retrieval. The experiment compares seven main methods and four ablations on the attached NVIDIA RTX 5000 Ada Generation.

This is a controlled snapshot study with noisy prior-time evidence, three partial observers, and 32 shared candidate conjunctions. It does not implement a full sequential cyber range. The original research plan includes extensions beyond this experiment; the frozen executable protocol and implementation notes define what was actually measured. Favorable performance and methodological novelty are hypotheses, not assumptions.

## Anonymous manuscript draft

The [consolidated anonymous paper](manuscript/anonymous/main.pdf) is **eight pages with four figures**; the [anonymous supplement](manuscript/anonymous/supplement.pdf) is **four pages with four additional figures**. Both preserve the exploratory interpretation, negative findings and matched controls. The main focuses on captured-telemetry graph learning; the supplement retains synthetic regeneration and candidate-scoring evidence separately.

[Editable anonymous source archive](manuscript/anonymous-source.zip), [rebuild instructions and provenance](manuscript/README.md), and [conference format notes](manuscript/FORMAT_NOTES.md) accompany the PDFs. Public guidelines confirm the eight-page main limit but do not confirm a separate supplemental-upload allowance. The prior study reports and figures below remain unchanged.

## GraphMAE follow-up: two variants and matched controls

The [GraphMAE variant study](docs/REAL_GMAE_VARIANTS.md) tests **EDGE_GMAE**, which gates graph messages using shared observable-feature coverage and compatibility, and **CONS_GMAE**, which combines two-view masked reconstruction, latent consistency and prediction consistency under additional feature hiding. Both retain the baseline's 121,643 parameters. **EDGE_DENSE** removes gated-model pretraining; **CONS_PRE** removes supervised prediction consistency.

This is an exploratory follow-up on the already examined CTU-13 test captures. The implementation and constants were frozen before new test inference, and all 20 new training/checkpoint/threshold selections finish before that inference. Nine earlier baselines are imported with exact input, source and checkpoint provenance. New runs use the same five seeds, observations and validation protocol; extra forward passes are accounted for rather than described as equal compute.

The completed Ada follow-up gives clean macro attack AP **0.99411 for EDGE_GMAE**, **0.98348 for CONS_GMAE**, and **0.97830 for GMAE**. Their gains are **+0.01580** and **+0.00518**; nominal two-comparison 97.5% intervals are **[−0.00036, +0.04321]** and **[−0.00257, +0.01639]**, respectively. Both include zero. EDGE has a positive unadjusted 95% interval, but no adjusted superiority claim follows. **EDGE_DENSE reaches 0.99500**, so the clean result does not credit masked pretraining for the gain. The replay audit exactly reproduces 3,570,325 model/flow evaluations.

- [Protocol, equations and related work](docs/REAL_GMAE_VARIANTS.md), [configuration](configs/study_real_gmae.json), and [results](docs/RESULTS_REAL_GMAE.md).
- [Four main figures](figures/real_gmae/index.html), [three-page supplement](supplemental/real_gmae/Supplement_real_gmae.pdf), and [artifact record](results/real_gmae/README.md).

```bash
python3 scripts/run_gpu.py -- -m unittest discover -s tests -p test_real_gmae_variants.py -v
python3 scripts/run_gpu.py -- scripts/run_real_gmae.py
python3 scripts/run_gpu.py -- scripts/audit_real_gmae.py
.venv/bin/python scripts/analyze_real_gmae.py
.venv/bin/python scripts/make_real_gmae_results.py
.venv/bin/python scripts/make_real_gmae_supplement.py
.venv/bin/python scripts/verify_real_gmae_artifacts.py
```

The runner requires the preceding CTU-13 study and its ignored raw predictions/checkpoints, and refuses to overwrite a completed follow-up. Large arrays, weights and logs stay ignored. The two main variant-minus-GMAE comparisons use shared crossed seed/time-block bootstrap draws with 95% and nominal two-comparison 97.5% marginal intervals. These conditional intervals do not remove the limitation of reusing test captures; independent data is needed for confirmation.

## External application: captured CTU-13 telemetry

The [real-data study](docs/REAL_DATA_METHOD.md) applies nine methods to actual CTU-13 university network measurements captured during controlled botnet executions. Its target is retrospective completed-flow classification: **From-Botnet versus From-Normal**, with all background and incoming-only labels excluded. The 799,777 retained flows are approximately 4% of 19,976,700 rows across all 13 official tables. Results describe this verified-label subset; they do not establish operational course-of-action outcomes or performance on the full traffic population.

The split follows actual capture timestamps and flow completion: **442,084 training flows** from captures 1–8 and 13, **214,880 validation flows** from capture 9, and **142,813 test flows** from captures 10–12. All training records complete before validation starts, and validation completes before test traffic starts. Hosts may recur across captures, so this is a future-capture split, not a host-disjoint split. Fourteen observable features and their masks enter the models; endpoint identities are used only to construct genuine shared-endpoint graphs within fixed 32-flow windows.

Eight neural adaptations—**SCR_FLOW, CTX_FLOW, LOCAL_FLOW, LOGIT, DAE, DENSE, GMAE and HDC**—are freshly trained on captured data, alongside CUDA-fitted Gaussian naive Bayes (**NB**). No synthetic-study weights transfer. SCR_FLOW adapts the shared-context mixture to flow classification; the graph models use observed endpoint relationships without inventing prerequisite implications. The clean-data primary contrast is SCR_FLOW minus GMAE in macro attack average precision across the three held-out captures. Artificial feature masking and benign-value replacement separately test telemetry corruption; they are not observed real-world deception.

The completed Ada run covers five training seeds and 675 method/condition/capture records. Clean macro attack AP is **0.96616 for SCR_FLOW versus 0.97830 for GMAE**: paired difference **−0.01215**, 95% crossed seed/time-block interval **[−0.04490, +0.01159]**. This establishes neither an improvement nor equivalence. DENSE has the numerically highest clean mean AP (**0.98151**); Gaussian naive Bayes reaches **0.91784** and the small DAE **0.86743**. The separate checkpoint audit exactly reproduces 2,142,195 model/flow evaluations.

- [Method, adaptation equation and limitations](docs/REAL_DATA_METHOD.md), [frozen protocol](configs/study_real.json), and [results and interpretation](docs/RESULTS_REAL.md).
- [Four main comparison figures](figures/real/index.html) and [real-study supplemental PDF](supplemental/real/Supplement_real.pdf).
- [Artifact and execution record](results/real/README.md), [official sources and attribution](data/ctu13_sources.json), [download hashes](data/ctu13_downloads.json), and [data audit and temporal split](results/real/ctu13_data_manifest.json).

The method and configuration define the external experiment; the results and artifact record establish its measured outcomes and completion status. This experiment can be reproduced independently of earlier synthetic checkpoints:

```bash
# Download only official text flow tables, verify hashes, and parse all 13 captures.
.venv/bin/python scripts/prepare_ctu13.py all --workers 2
python3 scripts/run_gpu.py -- -m unittest discover -s tests -p 'test_real*.py' -v
python3 scripts/run_gpu.py -- scripts/run_real.py
python3 scripts/run_gpu.py -- scripts/measure_real_pipeline.py
python3 scripts/run_gpu.py -- scripts/replay_real.py
.venv/bin/python scripts/analyze_real.py
.venv/bin/python scripts/make_real_results.py
.venv/bin/python scripts/make_real_supplement.py
.venv/bin/python scripts/verify_real_artifacts.py
```

Scaling, model fitting, inference and metric tensor calculations run on the attached Ada GPU. CPU handles text parsing, serialization, bootstrap statistics, plotting and PDF rendering. The source tables (approximately 2.73 GB) and prepared arrays remain in ignored `data/external/ctu13/`; raw experiment arrays, checkpoints and logs remain in ignored run/checkpoint/work directories. Only compact provenance, measurements, code and selected final figures/documents belong in Git.

## Third approach: direct candidate scoring and inexpensive ML

The [third method](docs/METHOD_V3.md) tests **SCR**, a shared-context candidate scorer that predicts each candidate's feasibility directly. It combines canonical prerequisite sets with learned set embeddings and a four-expert mixture. It does not reconstruct a complete fact state. Single-expert controls test the effects of global context and the mixture; **LOGIT** and a pretrained **denoising autoencoder (DAE)** provide inexpensive fact-reconstruction comparisons.

The frozen comparison uses the same 20,000 mixed training observations and 2,000 validation observations as study 2. Five training seeds and fresh test seed **62027** compare all 22 methods over 18 conditions. The primary endpoint is SCR minus the dense mixed-data model ID in macro candidate-oracle regret across four reference conditions. The completed study gives SCR regret **0.11152** versus ID **0.08644**; the paired difference is **+0.02508 [0.02091, 0.02919]**, favoring ID. SCR improves over its single-expert control (0.12431), while DAE (0.12873) and LOGIT (0.16455) trade accuracy for lower measured latency. Different prediction targets and pretraining make the primary comparison one of complete pipelines. Fact-level repair and consistency metrics are unavailable for direct candidate models.

- [Third-study results and interpretation](docs/RESULTS_V3.md)
- [Four third-study empirical figures](figures/v3/index.html)
- [Third-study supplemental PDF](supplemental/v3/Supplement_v3.pdf), with two additional figures
- [Frozen protocol](configs/study_v3.json), [audit](docs/V3_AUDIT.md), and [artifact record](results/v3/README.md)

These paths hold the third-study record; the manifest and audit identify the completed execution and verification. All models run on the attached Ada GPU, including LOGIT and DAE. Plotting and document generation consume recorded experiment outputs.

```bash
python3 scripts/run_gpu.py -- -m unittest discover -s tests -p test_v3.py -v
python3 scripts/run_gpu.py -- scripts/run_v3.py
python3 scripts/run_gpu.py -- scripts/measure_v3_pipeline.py
python3 scripts/run_gpu.py -- scripts/replay_v3.py
.venv/bin/python scripts/analyze_v3.py
.venv/bin/python scripts/make_v3_results.py
.venv/bin/python scripts/make_v3_supplement.py
.venv/bin/python scripts/verify_v3_artifacts.py
```

The third study reuses frozen earlier checkpoints for its historical comparisons. Reproduction therefore requires the original and second-study checkpoints locally, or reproducing those studies first. Third-study raw arrays and checkpoints remain in ignored `results/runs/v3/` and `results/checkpoints/v3/`; source, compact measurements and selected final artifacts are tracked.

## Second approach and fresh comparison

[Dependency-aware full regeneration](docs/METHOD_V2.md) addresses the first gate's sparse-repair regret and mechanism dependence. It predicts a coherent conditional state, counts shared prerequisite uncertainty once when scoring candidates, and trains across four observation groups. Its proposed composition builds on established probabilistic logic and group-risk learning; novelty and improvements remain empirical questions.

The [second results report](docs/RESULTS_V2.md) compares five newly trained variants, one scoring ablation, and all eleven original methods on fresh test seed 42027. Its fair primary comparator is a full-reconstruction dense model trained on the same mixed data. The original study below remains unchanged.

- [Four second-study comparison figures](figures/v2/index.html)
- [Second-study three-page supplement](supplemental/v2/Supplement_v2.pdf), with two additional figures
- [Frozen protocol](configs/study_v2.json), [audit](docs/V2_AUDIT.md), and [artifact record](results/v2/README.md)

```bash
python3 scripts/run_gpu.py -- -m unittest discover -s tests -p test_v2.py -v
python3 scripts/run_gpu.py -- scripts/run_v2.py
python3 scripts/run_gpu.py -- scripts/measure_v2_pipeline.py
python3 scripts/run_gpu.py -- scripts/replay_v2.py
.venv/bin/python scripts/analyze_v2.py
.venv/bin/python scripts/make_v2_results.py
.venv/bin/python scripts/make_v2_supplement.py
.venv/bin/python scripts/verify_v2_artifacts.py
```

Second-study raw arrays and checkpoints use ignored `results/runs/v2/` and `results/checkpoints/v2/`. Generated data, raw logs, and environments are excluded from Git.

## Research record

- [Measured results and interpretation](docs/RESULTS.md), including the primary paired comparison and limitations.
- [Four empirical paper figures](figures/paper/index.html), with vector PDFs, PNG previews, and provenance sidecars.
- [Three-page supplemental PDF](supplemental/Supplement.pdf), containing two additional empirical figures.
- [Implemented method and equations](docs/METHOD_IMPLEMENTATION.md), including the proposed decision-impact repair teacher.
- [Implementation audit](docs/IMPLEMENTATION_AUDIT.md) and [original research plan](RESEARCH_PLAN.md).
- [Frozen executable settings](configs/executed_study.json), [run manifest](results/study_manifest.json), and [compact measurement tables](results/README.md).
- [References and inherited methods](docs/REFERENCES.md), [GPU setup](docs/GPU_SETUP.md), and [project instructions](AGENTS.md).

## Reproduce on the Ada GPU

The launcher selects the attached Ada by UUID and refuses CPU fallback. Training uses BF16 autocast with FP32 losses and HDC accumulators; measured model inference uses FP32 with TF32 allowed. See the setup document for recreating the pinned environment.

```bash
# From this project directory; requires the attached Ada GPU and project .venv.
python3 scripts/run_gpu.py -- -m unittest discover -s tests -v
python3 scripts/run_gpu.py -- scripts/run_study.py
python3 scripts/run_gpu.py -- scripts/explain_example.py
python3 scripts/run_gpu.py -- scripts/measure_pipeline.py

# CPU statistics and document rendering use recorded CUDA measurements.
.venv/bin/python scripts/analyze_results.py
.venv/bin/python scripts/make_results.py
.venv/bin/python scripts/make_supplement.py
.venv/bin/python scripts/verify_artifacts.py
```

The full run trains six models for each of five seeds and evaluates 11 methods across 14 conditions, with 20,000 training, 2,000 validation, and 4,000 test snapshots per condition. The test scenarios are paired across methods and training seeds; they are not 3.08 million independent worlds. All generated research tensors and model computation stay on CUDA. Plotting, statistics, serialization, and PDF rendering use the CPU.

`results/pipeline_timing.json` separately measures a bounded host processing loop including packet serialization/deserialization, transfers, feature construction, inference, and candidate ranking. The figures' CUDA latency excludes those host contributions. Neither timing is an operational network deployment measurement.

## Artifact policy

All project work belongs in this directory. Source, compact CSV/JSON summaries, selected figures, and the supplemental PDF are tracked. The local `.venv/`, `work/`, raw `results/runs/`, and `results/checkpoints/` are ignored. Re-running overwrites aggregate results and retrains checkpoints; use a local backup before preserving multiple studies. Do not force-add ignored artifacts.

The existing [design diagrams](figures/design/index.html) explain the proposal and are not empirical result figures. The requested supplemental length is implemented, but the public MVML guidelines inspected do not explicitly confirm a separate supplemental upload allowance. Confirm that category before submission.
