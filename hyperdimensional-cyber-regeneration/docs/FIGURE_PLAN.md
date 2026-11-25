# Multipanel figures for the research paper

The paper must contain at least four empirical figures generated from recorded GPU experiments. Six figures are planned; Figures 2–5 are mandatory empirical figures. Four separate design diagrams already explain the protocol and must never be presented as measured results.

Use a consistent method palette, line styles and marker shapes, readable panel labels, and accessible colors. Preserve a vector PDF/SVG master and a 300 dpi PNG preview. Use Matplotlib or another standard scientific plotting library for measured plots. At final manuscript size use approximately 8–10 pt labels and avoid legends obscuring data.

Every empirical figure requires a sidecar JSON record with the plot script commit, input file hashes, run IDs, seeds, aggregation unit, filtering rules, confidence-interval procedure, and output hash. Regenerate figures from committed scripts and actual data; do not manually edit plotted values.

## Figure 1 Architecture and regeneration mechanism

**Scientific purpose:** explain where the learning contribution enters the planning loop.

- **A:** three partial observers, separate evidence provenance, and communication budget.
- **B:** GPU HDC encoding and graph reconstruction with the shared codebook.
- **C:** equations (1) and (3), showing the decision-impact teacher available only during training and the learned gate used at inference.
- **D:** a single evidence-to-prerequisite-to-COA trace with retained, repaired, and unresolved facts distinguished.

Use an actual held-out trace for panel D once results exist. Until then label every example as a constructed schematic. Show the symbolic planner shared across methods and the boundary preventing evaluator truth from entering inference.

**Inputs:** architecture/configuration, frozen rule catalogue, trace JSON.

**Output stem:** `figures/paper/fig01_architecture`.

**Draft caption:** “Decision-impact selective regeneration combines vector-symbolic evidence memory with a learned repair gate. Hidden simulator state supervises training and scores evaluation; it is excluded from inference.”

## Figure 2 Planning recovery under deception

**Scientific purpose:** test the primary hypothesis with several views of the same paired experiment.

- **A:** feasible-COA recall at three versus deception fraction, with seed/scenario uncertainty bands.
- **B:** decision regret versus the same fractions, sharing the method palette.
- **C:** recovery curves over elapsed steps after injection; show unrecovered fraction and censoring.
- **D:** paired method differences by corruption family, including a zero-deception control.

Distinguish coherent deception from random report replacement. Define the denominator for feasible-COA recall and show the number of eligible episodes. Compare P, B0, B1, B2, B3, and B5; use small multiples if a combined legend becomes unreadable.

**Inputs:** `results/runs/*/planning_metrics.jsonl` and split/scenario manifests.

**Output stem:** `figures/paper/fig02_deception_recovery`.

**Required claim:** an interval-supported effect against B2, or an explicit null/negative finding.

## Figure 3 Belief repair and calibrated uncertainty

**Scientific purpose:** test whether plan recovery comes from justified reconstruction.

- **A:** recovery precision–recall for corrupted facts, with random and targeted corruption separated.
- **B:** plan-success reliability diagram with a prediction-count histogram.
- **C:** false-certainty versus recommendation coverage under coherent deception and zero-independent-evidence conditions.
- **D:** fact-by-time heatmaps from one preselected held-out episode: hidden truth, observations, and predicted beliefs; overlay selected repairs.

Choose the qualitative episode by a predefined rule, such as the median regret episode in a named stratum. Add the same episode for B2. Do not select only a visually successful case. Missing observations must be visibly distinct from negative facts.

**Inputs:** belief probabilities, ground truth held by evaluator, gate masks, confidence thresholds, episode-selection record.

**Output stem:** `figures/paper/fig03_repair_calibration`.

## Figure 4 Communication and GPU resource tradeoffs

**Scientific purpose:** determine whether HDC provides practical value at matched information budgets.

- **A:** feasible-COA recall at three versus serialized bytes per observer-round; mark HDC, dense embeddings, and symbolic deltas.
- **B:** quality versus measured end-to-end batch-one latency, with points labeled by repair fraction.
- **C:** stacked memory accounting for neural weights, HDC buffers, evidence ledger, and symbolic structures.
- **D:** latency breakdown for transfers, HDC retrieval, gate/decoder, serialization, and planning; add p95 markers.

Report warm-up, precision, device, batch size, and sample count in the caption. Separate GPU kernel time from complete loop time. Include setup/amortized codebook cost and burst-outage cases. Pareto fronts must be calculated from measured points, not interpolated into unsupported claims.

**Inputs:** serialized payload logs, CUDA event measurements, host timings, memory snapshots.

**Output stem:** `figures/paper/fig04_gpu_communication`.

## Figure 5 Ablations and repair selection

**Scientific purpose:** isolate the proposed equation, representation, and symbolic constraints.

- **A:** paired effect sizes on the primary endpoint for removal of decision-impact supervision, semantic loss, and provenance.
- **B:** regret versus repair fraction for learned, entropy, random, and oracle-training-reference gates.
- **C:** same objective/gate with HDC and dense representations, using matched budgets.
- **D:** selection analysis stratified by fact confidence and decision criticality; count confidently wrong prerequisites explicitly.

The oracle gate is an analysis ceiling using hidden truth and must be visibly marked as unavailable at inference. Plot actual decoder work and latency alongside K in supplementary data so mask sparsity is not confused with compute savings.

**Inputs:** ablation manifests and per-fact priority, entropy, teacher gain, gate decisions, and timing.

**Output stem:** `figures/paper/fig05_ablation_selection`.

## Figure 6 Generalization and failure boundaries

**Scientific purpose:** identify where the method transfers and where the additive teacher fails.

- **A:** heatmap of performance across unseen topology families and deception mechanisms.
- **B:** performance versus graph size, including the 96-host extrapolation set.
- **C:** paired joint-repair stress test: two individually ineffective repairs whose combined correction restores a COA.
- **D:** a documented failure trace and the competing hypotheses retained by the model.

Report uncertainty over independent topology clusters. Distinguish an insufficient-evidence failure from a model error. If page limits require consolidation, move A–B into Figure 5 and retain C–D as supplementary analysis.

**Inputs:** extrapolation and interaction-test manifests, evaluator labels, selected failure trace.

**Output stem:** `figures/paper/fig06_generalization`.

## Four existing design diagrams

| File | Contents | Evidence status |
| --- | --- | --- |
| `design01_architecture.svg` | Distributed observations, GPU model, selective repair, shared planner | Proposed architecture |
| `design02_equation.svg` | Prior-method lineage, counterfactual teacher, learned gate, limiting cases | Mathematical design |
| `design03_benchmark.svg` | Synthetic world, partial observers, corruption conditions, split and truth boundaries | Experimental design |
| `design04_evidence.svg` | Comparison ladder, paper-figure map, GPU measurement contract, conference/bid evidence | Evaluation plan |

The gallery is `figures/design/index.html`. These figures intentionally contain no fabricated performance curves.
