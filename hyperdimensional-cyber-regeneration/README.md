# Hyperdimensional cyber regeneration

GPU research project for MVML 2027, informed by DARPA HYPER-CAP, topic DPA27BZ02-DV001.

**Working paper title:** Learning to Regenerate Cyber Planning States under Deception with Hyperdimensional Neurosymbolic Models.

The study develops a learned, budgeted regeneration operator that repairs decision-relevant beliefs and evaluates the resulting cyber courses of action using a common symbolic planner. It builds on GraphMAE, decision-focused learning, semantic loss, and vector-symbolic computing.

- [Research plan](RESEARCH_PLAN.md): problem, proposed equations, prior work, baselines, GPU execution, evaluation, milestones, and bid alignment.
- [Figure plan](docs/FIGURE_PLAN.md): six multipanel paper figures, including four mandatory empirical result figures.
- [References](docs/REFERENCES.md): primary sources and exactly what is inherited.
- [Study configuration](configs/study.yaml): proposed experiment settings, not a training implementation.
- [Design figure gallery](figures/design/index.html): four original vector diagrams of the research design, with no simulated performance claims.
- [GPU readiness check](scripts/check_gpu.py): refuses to substitute CPU computation for CUDA research runs.
- [Project instructions](AGENTS.md): scope, GPU requirement, evidence standards, and direct-to-main workflow.

All project code, data, logs, manuscripts, figures, and temporary work belong in this directory. Use `work/` for scratch files and `results/` for actual experiment outputs. This initial package contains the research design; no model has been trained and no empirical result figures have been generated.

Run the GPU readiness check from this directory:

```bash
python3 scripts/check_gpu.py
```

The diagram generator uses Python's standard library and can run without a GPU:

```bash
python3 scripts/render_design_figures.py
```

These are documentation utilities. Future training and model inference must require CUDA. Keep commits on the repository's existing `main` branch.
