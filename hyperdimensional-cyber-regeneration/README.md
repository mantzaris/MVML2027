# Hyperdimensional cyber regeneration

CUDA research project for MVML 2027, informed by DARPA HYPER-CAP (DPA27BZ02-DV001).

**Working title:** Learning to Regenerate Cyber Planning States under Deception with Hyperdimensional Neurosymbolic Models.

The implemented study learns a decision-impact gate for repairing beliefs in a synthetic prerequisite-planning task. It builds on GraphMAE-style feature pretraining, decision-focused learning, semantic loss, and fixed-codebook hyperdimensional retrieval. The experiment compares seven main methods and four ablations on the attached NVIDIA RTX 5000 Ada Generation.

This is a controlled snapshot study with noisy prior-time evidence, three partial observers, and 32 shared candidate conjunctions. It does not implement a full sequential cyber range. The original research plan includes extensions beyond this experiment; the frozen executable protocol and implementation notes define what was actually measured. Favorable performance and methodological novelty are hypotheses, not assumptions.

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
