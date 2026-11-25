# Independent audit of the second approach

This is an implementation and inference-scope review, not external peer review or proof of novelty. It concerns the frozen second protocol in `configs/study_v2.json`, `src/regen_v2.py`, and `scripts/run_v2.py`. The original sources and published first-study results remain unchanged.

## Why the design changes are warranted

The first study exposed three distinct issues: the learned repair gate fit targeted deception poorly outside its training distribution; restricting repair to six facts increased regret relative to full reconstruction; and scoring a conjunction by multiplying fact marginals counted a shared root repeatedly. The second approach addresses these through mixed observation training, full reconstruction, and an explicit conditional root/child probability model. These are separate interventions, so an improvement over old P alone cannot identify which one helped.

The primary comparator ID is a dense independent model trained on exactly the same mixed data with the same objective and optimization budget. IH holds HDC retrieval fixed while changing factorization; JD tests the HDC contribution within the factorized model. JH_ERM changes the smooth group-risk aggregation to a mean. JH_product keeps the JH checkpoint and marginal beliefs but changes only candidate scoring. The original eleven methods retain their original checkpoints and thresholds when reevaluated on fresh worlds.

## Verified code and data properties

- A direct CUDA audit on 256 samples confirms that mixed training preserves the original hidden current/previous worlds, parent graph, adjacency, candidates, values, costs, and topology-group IDs. Each of the four observation groups has 64 samples. Every targeted/random/coherent severity assignment exactly matches the corresponding generator output.
- The full runner also checks the mixed train/validation hidden-world and candidate fingerprints against the first run. It records legacy checkpoint hashes and the original validation-threshold file hash, as well as new checkpoints and fresh input fingerprints.
- Each training world receives one observation realization; the mixture does not silently quadruple training examples. Index-based group assignment is used only to construct and aggregate training data. Inference reads observed features, adjacency, parents, candidate requirements, and their public values/costs; it does not read an attack-group label.
- The condition catalogue contains 18 settings and 17 evaluated methods, giving 1,530 seed/method/condition records for five seeds. The legacy `repair_fraction` scenario field remains 0.1, but new methods use full regeneration; the repair policy and measured changed-fact fraction must be read alongside it.
- BF16 autocast is limited to the training network pass. The factorized root-incidence computation explicitly disables autocast, and probabilities and loss reductions use FP32. Model inference and candidate scoring require CUDA.
- Checkpoint selection uses the same sample-correct mean of four validation-group losses for all new variants. Training uses a smooth log-mean-exp over available group means, with beta 2, except JH_ERM. There is no test-based threshold search for the new full-reconstruction models.
- The original threshold mapping is preserved for old methods. B4/B5/random reuse the original P checkpoint; thresholded old variants use their original validation-selected thresholds. They are not retuned on the fresh pool.
- The original six-fact perfect-truth teacher is absent from the new training objective. Hidden truth supplies ordinary supervised training and evaluation targets, not inference-time repair instructions.

## Semantic issue found and corrected before freezing

The first draft reused the old independent-probability semantic term `-log(1 - p_child*(1-p_parent))`. For the factorized posterior this is inappropriate: a root with probability 0.5 and child-given-root probability 0.8 has child marginal 0.4 and zero logical-violation probability, yet that old expression incurs a positive penalty.

The new implementation instead applies the same marginal-consistency penalty to all variants: the average squared positive part of `p_child - p_parent`. It is zero for the coherent structured posterior and positive when an independent prediction violates the implication inequality. It is a consistency penalty, not the probability of a violated clause. This correction was made before the fresh confirmatory evaluation.

## Scientific checks and statistical accounting

The seven dedicated CUDA tests passed on the attached Ada GPU. They cover enumeration of tiny valid worlds, shared-root deduplication, hard-state conjunction semantics, coherent marginals, finite nonzero training gradients for all five variants, hidden-label mutation with a nonzero belief head, and identical marginals for the product-scoring ablation. In particular, adding an explicitly required parent to a candidate that already requires its child cannot multiply that root probability a second time.

The analysis function passed separate checks that identical paired outputs yield exactly zero effect, a constant contrast is recovered exactly, and fixed condition means remain equally weighted in every bootstrap draw. It uses a crossed bootstrap over five training seeds and 250 topology clusters. The same selected clusters are used across every selected seed, comparator and all four primary observation conditions. Conditions themselves are fixed, not resampled. Recall denominators are recomputed within each condition before macro averaging.

The analysis refuses incomplete manifests, mismatched test seeds, duplicate summaries, missing methods/conditions, changed executed-source digests, unpaired group/eligibility arrays, and nonfinite metric arrays. It retains raw-input digests in `results/v2/comparisons.json`; raw arrays remain ignored by Git.

## Claims that remain limited

1. The first pool (seed 32027) is development evidence because its results informed this redesign. The fresh pool (seed 42027) must remain outside model selection. The pilot uses seed 52027 and is not counted as confirmation.
2. The new mixture covers the previously tested attack families. Fresh-world improvement is not evidence of generalization to an unseen deception mechanism. The missingness and larger-graph conditions provide separate stress tests, with their scope explicitly stated.
3. The root/child formula is exact only under the chosen conditional star-forest posterior family. Observation manipulation can create dependencies outside that family. Logical consistency does not make a deceptive observation true.
4. Full regeneration abandons the old sparse intervention cap. It is compared primarily with a full-reconstruction control and cannot be described as a gain at an equal six-fact budget.
5. The smooth group-risk objective emphasizes difficult predefined groups; it is not an adversarial robustness certificate. A finite five-seed interval cannot establish universal superiority or equivalence.
6. HDC gains must be assessed using JH versus JD, not inferred from the combined approach. The shared symbolic wire codec still prevents any claim of an HDC communication-compression advantage.
7. The design remains a synthetic snapshot/candidate-ranking study, without operational sequential planning or external cyber-range evidence.
8. The new equation is an investigated composition of known probabilistic and learning ideas. [DeepProbLog](https://proceedings.neurips.cc/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html) establishes neural probabilistic reasoning, and [Sagawa et al.](https://arxiv.org/abs/1911.08731) study group-robust neural training. Neither Bayesian factorization nor group robustness is claimed as newly invented here.

Final empirical conclusions are generated separately in `docs/RESULTS_V2.md` after all five seeds complete. Both favorable and unfavorable component comparisons must remain visible.

## Completed-run audit and interpretation

The full Ada run completed in 1,399.27 seconds with all 1,530 planned records. Final analysis verified the complete condition catalogue for every seed and method, fresh test seed 42027, source digests, paired topology ordering, and common feasibility labels across the four primary conditions. A separate file audit verified all 25 new checkpoint hashes, all 30 reused legacy checkpoint hashes, the unchanged legacy threshold file, and all 160 raw inputs used in the paired contrasts.

The post-run CUDA replay recomputed JH and ID for training seed 37 under all four core conditions: 32,000 model/snapshot evaluations. Candidate probabilities, regret, and plan Brier matched the saved arrays exactly (maximum absolute error 0.0). The illustrative trace preserves the original inference batch context and verifies that the displayed candidate has the reported regret. It separates observed evidence and model factors from evaluator-only truth.

The primary JH-minus-ID macro-regret estimate is +0.001693, with a 95% crossed-bootstrap interval [-0.000169, +0.003503]. The data do not establish a gain or equivalence against the matched dense baseline. Comparisons isolating factorization, HDC features, group-risk aggregation, and joint scoring also have secondary regret intervals containing zero. These component findings must not be concealed by the much larger improvement over the old restricted models.

JH reduces macro regret from old P's 0.237706 to 0.085933 (approximately 63.85%) on the fresh four-condition pool, but that comparison changes training coverage and repair policy. The original B2 specialist remains better in its targeted reference condition (0.03726 versus JH's 0.09968). JH has zero measured marginal implication violations, compared with 4.5692% for the matched independent dense model across the primary conditions. This is a structural coherence advantage, not evidence of more accurate hidden-state recovery or better decision regret.

The bounded host processing-loop audit measured a JH median of 4.985 ms and an ID median of 4.223 ms on one fixed snapshot. No latency advantage is established. The main CUDA timings and this host-loop measurement retain their distinct scopes; neither is a full cyber-operation runtime.
