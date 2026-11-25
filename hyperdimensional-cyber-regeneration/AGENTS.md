# Project instructions

These instructions apply to this project directory and its descendants.

- Keep all work for this research project inside this directory, including scratch work under `work/`.
- Use the existing repository's `main` branch. The user explicitly requested direct commits and pushes to main, without new branches or worktrees.
- Preserve unrelated research projects and user changes elsewhere in the repository.
- Design and execute model training, HDC tensor operations, counterfactual batches, and model inference on a CUDA GPU. Fail clearly when CUDA is unavailable; never silently substitute CPU research runs.
- CPU orchestration, file I/O, symbolic audit checks, and document rendering are allowed. Include their contribution in end-to-end latency.
- Maintain the required comparison with the GraphMAE-adapted baseline and the conventional belief-update baseline. Include the matched objective and matched representation ablations.
- Preserve the proposed decision-impact regeneration equations and document revisions to their assumptions.
- Produce at least four rich empirical paper figures from real recorded experiments. Follow `docs/FIGURE_PLAN.md`.
- The four existing design diagrams are schematics, not empirical evidence. Never fill result plots with invented observations or present planned targets as measured results.
- Maintain topology-disjoint data splits, hidden evaluation truth, paired scenarios, seed-level uncertainty, and complete communication accounting.
- Record model settings, environment, dataset hashes, code commit, raw metrics, and figure provenance.
- Treat novelty and expected improvements as hypotheses until literature review and experiments justify stronger claims.
- Keep published material based on public sources and the project's synthetic scenarios.
