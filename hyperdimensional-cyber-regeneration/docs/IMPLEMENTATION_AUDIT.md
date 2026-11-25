# Independent implementation and scientific audit

Audit date: November 25, 2026. Reviewed the research plan, `src/regen.py`, `scripts/run_study.py` and `tests/test_research.py`. This is an implementation and inference-scope audit, not external peer review or proof of novelty.

## Verified properties

- Inference accepts observed feature tensors and adjacency. Hidden current truth, previous truth, attack labels and transition labels are generator/evaluator outputs; they are used for training targets and evaluation, not test-time repair selection.
- An independent CUDA mutation check covered all eleven deployed prediction paths, with nonzero randomly initialized belief-head weights. Replacing current/previous truth and attack masks while holding observable inputs fixed left every prediction exactly unchanged. The persistent core test also checks the restricted model input interface.
- Hidden worlds, candidate preconditions, utilities and topology are drawn before corruption or visibility branches. The paired-condition test verifies identical worlds and candidate sets when corruption, missingness, bandwidth and attack family change.
- Candidate generation does not consult hidden truth. Known candidate criticality is a public catalogue feature; it is legitimate input even though the targeted corruption generator also uses it.
- The analytic teacher implements the positive decrease in soft candidate regret after substituting one fact's training truth. The core test compares it with explicit one-fact interventions at temperature 0.1. These single-coordinate gains ignore joint repair interactions and the probability that a fact is reconstructable; they are not an optimal-subset guarantee. The runner passes the same temperature used in its decision loss.
- Core tests cover repair-budget limiting cases, generated symbolic consistency, hard-state utility semantics, finite gradients through HDC and masked feature reconstruction, packet budget limits, and byte counts against actual serialization.
- Masked pretraining includes a learned input mask token, latent remasking and neighbor decoding with scaled cosine reconstruction. “GraphMAE-adapted” is an accurate label; an exact upstream GraphMAE reproduction would not be.
- The direct source/age-channel ablation uses the intended feature channels. It retains the common provisional fused belief, so it does not remove all indirect provenance information.
- dense_gate is trained independently with the same gated objective as P. The full-repair, entropy-selection and random-selection variants share P's learned weights and are explicitly inference ablations.
- Checkpoint selection evaluates the deployed prediction path. Gate thresholds are selected on validation data. Test labels are confined to metrics after predictions are generated.
- The benchmark enforces CUDA tensor execution and the runner asserts the RTX 5000 Ada device. Mixed precision applies to training; inference timing is not automatically mixed precision.
- Source digests, actual run settings, hardware, framework version and dataset hashes are recorded in the study manifest. Per-snapshot arrays and checkpoints remain ignored by Git.

## Issues corrected during review

| Issue | Resolution |
|---|---|
| Incorrect feature indices in the source/age ablation | Replaced with the core helper that removes explicit source distinctions and age channels. |
| Teacher temperature different from training decision surrogate | Runner now explicitly supplies 0.1 to the teacher. |
| Dense gate ablation reused a network trained for full reconstruction | A separate dense_gate model now receives the gated training objective. |
| Gated checkpoints ranked using full reconstruction | Validation now calls the deployed prediction function. |
| CUDA deterministic GEMM configuration | GPU launcher supplies the required deterministic cuBLAS workspace environment. |
| Confidence error lacked coverage context | The output now also reports the fraction of confident beliefs. |
| Byte accounting did not materialize a packet | A concrete little-endian header/record serializer and byte-equality test were added. |
| Snapshot bootstrap would ignore shared topology and reused test set | Paired analysis resamples both training seeds and topology clusters, with shared cluster draws across resampled seeds. |
| A preimplementation Git revision alone would not identify executed code | Manifest records source SHA256 digests in addition to the Git revision. |

## Material limits and required wording

1. **This is a controlled belief-repair and candidate-ranking study.** A sample is an independent decision snapshot with a noisy prior-time observation. A candidate is a conjunction of initial facts and a terminal utility. There is no executed multistep course of action, planning search, temporal recovery curve or closed-loop cyber range. Describe relevance to course-of-action replanning; do not claim a complete planner.
2. **Facts are not hosts.** The 64/128/256 settings count Boolean prerequisites. The graph generator consists of parent/root dependency motifs, with a changed root proportion in the larger test family. Size extrapolation is not evidence of arbitrary network-topology generalization.
3. **HDC is internal memory.** Source-separated bipolar associative retrieval supplies three features to the graph network. All methods use the same sparse symbolic report transport. The bandwidth sweep cannot establish the plan's hypothesized HDC wire-compression advantage.
4. **Ages are synthetic covariates.** Report ages are independently sampled; they do not index a physically delayed observation process. Prior-time observations do reflect an independently generated previous state with genuine changes.
5. **Attack intensity needs precise labeling.** The coherent attack writes an all-ready narrative; some targeted reports already have that value. Nominal intervention intensity differs from the changed-report fraction and final erroneous-report fraction. Conditions do not independently sweep correlated packet loss, bursts or authenticated source compromise.
6. **The core comparison does not isolate one factor.** P versus B2 changes the representation and the deployment repair policy. B2 also trains an auxiliary gate head that is unused in its full-reconstruction inference. The independently trained dense_gate comparison is the appropriate matched representation ablation. B1/B2 are task-specific adaptations, not official published benchmark implementations.
7. **Sparse repair does not imply cheap inference.** The complete graph decoder executes before top-K selection. The experiment can measure belief-change sparsity and observed latency, not claim decoder work is avoided.
8. **Preservation is an error-style metric.** The field named preservation is the rate of unnecessary changes to stable, correctly classified beliefs. Lower is better. It is not the fraction of correct facts preserved.
9. **Probability products are an approximation for uncertain states.** Conjunction semantics are exact for hard truth, but multiplying marginal probabilities ignores dependencies induced by the symbolic graph. Calibration plots evaluate the resulting scores; they do not turn the product into an exact joint probability.
10. **Timing and memory have narrow scope.** Warmed batch-one CUDA events cover model inference plus candidate ranking. They exclude serialization, device transfers and broader application overhead. Incremental peak allocation above resident models/data is not total deployment memory. Training wall time includes orchestration. A separate prepared-evidence pipeline audit (`scripts/measure_pipeline.py`) includes actual host packing/unpacking, transfers, feature fusion, ranking and returned IDs. It validates all eleven round-trip prediction paths before timing; when generated, `results/pipeline_timing.json` reports 50 warmed repetitions on one fixed snapshot, excluding sensing, admission, transport delay and cold loading.
11. **Statistics remain generator-specific.** Five training seeds share finite train/validation sets and one paired test set per condition. Crossed seed/cluster bootstrap respects the shared test set, but cannot account for uncertainty in the synthetic generator's design or establish external validity. Zero-containing intervals do not prove equivalence. Seed error bars on exploratory plots are not paired primary-effect intervals.
12. **Validation uses means of batch means.** The final incomplete validation batch receives the same weight as a full batch. This is a minor reproducible selection choice, not a sample-weighted objective estimate.
13. **Novelty is a hypothesis.** The proposed one-fact regret-benefit teacher and gated composition are investigable extensions of prior reconstruction, decision-focused learning, symbolic loss and HDC approaches. The implementation does not establish a first-ever invention or a broad literature novelty claim. Negative or null results must remain in the report.
14. **Bid and conference scope remain limited.** This synthetic experiment addresses a defensible ML subproblem within the proposed DARPA topic. It does not establish a distributed deployed mission system. The requested separate 2–4 page supplemental PDF can be prepared, but the public instructions reviewed do not verify that MVML accepts a separate supplement.

## Results interpretation

The authoritative completed comparison is generated in `docs/RESULTS.md` from `results/summary.csv` and the ignored per-snapshot run arrays. The paired effect estimates and raw-input hashes are in `results/paired_comparison.json`. The analysis must be run only after all five training seeds complete. The script intentionally refuses a partial run.

The primary comparison is declared as P minus B2 at 20% targeted report interventions, 64 facts, a 2,048-byte per-remote-observer budget and a 10% repair budget. The bootstrap retains all snapshots within topology clusters and resamples the same cluster indices across all selected seeds. Recall@3 is conditioned on at least one feasible candidate; regret includes all snapshots.

If full reconstruction wins, the appropriate conclusion is that the learned gate has not overcome the restriction imposed by the sparse repair budget on this benchmark. A comparison with entropy and random selection can still characterize which facts the gate chooses, but cannot overturn a negative primary result.

## Completed-run findings

The final study contains 770 unique method/seed/condition rows: five seeds, eleven deployed variants and fourteen conditions. The executed core and runner source SHA256 digests still match the manifest; all ten primary raw-input hashes match the paired analysis. Threshold recovery and example replay reproduce all 4,000 saved P regrets at seed 37 with maximum absolute error zero.

The primary result is negative for the proposed sparse repair model: P minus B2 is −0.393 percentage points in conditional Recall@3 (paired crossed-bootstrap 95% interval −0.590 to −0.232) and +0.119920 in regret (interval +0.113353 to +0.126383; lower is better). The learned gate improves on entropy and random selection at the targeted reference condition, but its advantage does not extend to the zero-attack or random-attack diagnostics. The matched dense gate has a small descriptive mean difference; this does not establish broad HDC superiority.

The prepared-evidence timing audit checks all eleven reconstructed prediction paths before measuring 50 repeats. For one fixed seed-37 snapshot, P's median is 5.247 ms and B2's is 4.305 ms, including the explicitly bounded host and device pipeline. The paired analysis file remained byte-identical when the report was regenerated to include these timing results.

The implementation is suitable for reporting this controlled experiment and its negative/conditional findings. It is not evidence that all original hypotheses were established or that the full proposed cyber-planning capability was implemented.
