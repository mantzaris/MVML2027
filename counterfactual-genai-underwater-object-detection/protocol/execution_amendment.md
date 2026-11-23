# Execution decisions before model evaluation

Recorded 23 November 2026. These decisions adapt the prospective plan to actual data access and numerical checks. No real-data model predictions or final test results had been produced when these choices were made.

## Data source

The requested UniqueShip public archives returned download-quota pages. The study instead uses the public PORTIA v2 release from Valencia: https://zenodo.org/records/21409381, with AIS-associated vessel-presence labels. This is a geographically and temporally different collection from the sibling submission's ONC data.

The primary cohort combines single-vessel task positives and binary-task backgrounds, excluding every window with conflicting cross-task labels. The metadata audit found 2,388 conflicting window IDs; neither copy is used. It is a custom cohort, not a reproduction of the official PORTIA split. Day assignment uses a predetermined seed17 hash allocation, followed by purging vessel identities across partitions, prioritizing test, calibration, development, then training. Selection and class/source caps use metadata only, before audio-model outcomes.

Final selected counts are 3,735 training clips, 735 development, 600 calibration and 1,200 locked test clips. Vessel identities and UTC days are disjoint across partitions. The test set includes42 vessel identities and five days; the effective independent-day sample is small. The original goal of100 background-day groups is not achievable in this source. Paired uncertainty is computed at recording-day level, with source-group sensitivity and an explicit five-day limitation; clip counts do not substitute for independent groups. AIS presence is not human-verified audibility. No real event boundaries or object range/bearing labels are invented.

Native audio has three valid hydrophone channels in a four-channel file, with the unused channel excluded. Select hydrophone channel0 before a GPU anti-aliased32kHz→16kHz derivative. The primary unit is the native three-second recording, with one recording-level candidate for presence verification. Controlled longer scenes separately evaluate temporal event activity; the manuscript must not imply real temporal localization or individual resolution of overlapping objects.

All acquisition receipts, conflicting-label IDs, split rules and hashes are preserved in data/manifests/. Downloaded original four-channel files remain unchanged. This source change is driven by accessibility and independent provenance, not performance.

## Representations

CUDA Fourier/STFT/wavelet numerical fixtures passed. The bounded piecewise-linear EMD implementation reconstructed inputs and recovered a pure tone, but approximately96–100% of modes in noisy fixtures reached the ten-sift/non-IMF limit. Therefore Fourier, STFT and wavelets are the primary validated representations. The bounded HHT approximation is retained only as an explicitly exploratory branch in the supplemental study; it is not represented as converged classical EMD. See manifests/dsp_validation.json and protocol/numerical_amendment.md.

## Feasible model sizes and interventions

The initial32-channel flow U-Net has1,997,121 parameters. It completed a64-step training/replacement smoke pilot within the GPU memory limit. The smaller model replaces the prospective10–30million-parameter estimate; model size is not a novelty claim. The pilot is a functionality check, not evidence of detection benefit.

Primary real verification uses a whole-recording candidate, so no fabricated within-recording real labels are needed. The class-free generator is trained with both partial and full masks; whole-recording generation uses no target class or label. Supplementary controlled event tests use masks or proposed intervals whose label provenance is stated separately. A generated replacement may remove all sources in its support, so source-selective claims remain unsupported unless preservation controls demonstrate them.

The same-support nuisance edit uses a background-noise phase with a spectrum shaped to the actual replacement perturbation, then the identical support/taper and RMS matching. Its exact target preservation and spectral mismatch are evaluated on controlled stems. A nuisance edit can still obscure a source; it is a confound control, not a physically identified counterfactual.

## Checkpointing and frozen evaluation

The project-local environment is locked in requirements-research.lock. All substantial tensors execute on the UUID-pinned RTX5000 Ada. CPU decoding, tokenization, serialization, statistical summaries and plotting are incidental host work.

Checkpoint storage preserves model, optimizer, scheduler/scaler when used, random states, next-batch cursor, configuration, data/code fingerprints and two integrity-checked generations. A separate-process CUDA test reproduced the next optimizer step exactly and recovered from a damaged latest checkpoint using the prior generation. Validation artifacts are in manifests/cuda_resume_validation.json.

Development may choose hyperparameters and numerical validity thresholds. Calibration fits score mappings and operating thresholds. Freeze configurations, weights, splits, prompts, baseline selection and statistical endpoints before producing final test predictions. Stages are marked complete only after their outputs are durable and checksummed; a resumed run validates existing artifacts and recomputes unfinished shards.

## Fixed conditional probe adopted before real outcomes

A design review refined the primary operation before real detector fitting or inference. The primary mask is the fixed central1.5 seconds of the native3-second recording, [.75,2.25]s, with a10ms cosine taper. It retains observed context and is covered by the generator's partial-mask training distribution. The scoring target remains whole-record AIS-associated presence. This is a fixed verification probe, not a claimed event boundary or demonstrated target-specific removal. An event outside the probe can remain detectable; verification is optional evidence and the base detector is retained.

Whole-record replacement remains a separately named unconditional-null ablation. The full generator was trained with70%partial and30%whole-record masks before either evaluation; no mask placement is optimized against real performance.

The primary validity penalty is the sum of normalized exterior-change RMS and boundary-perturbation derivative RMS, averaged across generated draws. Both are normalized by original waveform RMS with epsilon1e-8. Background-distribution distance over the entire replacement is deliberately excluded, because retained context can correctly contain vessel sound. Reject a verification if any draw has exterior-change ratio≥1e-5, perturbation-RMS mismatch≥.05, eight-band normalized-power L1 mismatch≥.2, nonfinite features or generated/original whole-wave RMS outside(.01,10). Spectral matching is checked after the sham's mask/taper. These tolerances are engineering validity criteria, not probabilities of physical correctness.

The median convention for Equation7 and all ablations is torch.median's lower central order statistic for an even number of draws/views. It is used consistently in cached, live and calibrated computation; NumPy midpoint medians are not substituted. The controller's candidate uncertainty is updated from the numeric detector for the actually acquired subset after each action. Neither unacquired view scores nor labels enter the controller state. Verifier-internal waveform validity checks are mandatory and their cost belongs to verification.


## Crossed day–vessel dependence audited before real model outcomes

The immutable metadata reveal repeated vessel identities across the five test days. Joining days that share any positive vessel identity produces **one connected component**, not five independently sampled vessel populations. `data/manifests/dependence_audit.json` records the exact graph and all split components. Disjoint train/development/calibration/test days and vessel identities still prevent cross-partition leakage; within-test dependence is a separate issue.

The primary comparison will therefore report the descriptive AP difference on this locked cohort, three-seed variation, and **no estimable independent-component confidence interval**. The prespecified 2,000 paired day resamples remain a separately labeled, nonconfirmatory sensitivity analysis that conditions on this small reused-vessel collection. They cannot support a population superiority, significance or equivalence claim. Clip-level resampling is not substituted. This limitation also applies to semi-synthetic and noise-corruption studies derived from these parents. A further independent collection is required for confirmatory generalization evidence.


## Comparator selection aligned with the primary endpoint

Before fitting real models, the final comparator selection rule was clarified: rank all eligible baselines by development AP of the **mean of three calibrated seed probabilities**, matching the primary test endpoint exactly. Mean per-seed AP is a separately reported stability summary. Calibration still fits only the calibration split, and no test outcomes enter selection. Lexical method order breaks exact ties.

Audio features retain native digital waveform amplitude; no pressure-unit calibration is asserted. The matched-volume synthetic-augmentation baseline does not add a learned real/synthetic amplitude alignment. The synthetic temporal CNN likewise uses its frozen log-mel scale, so semi-synthetic transfer may expose both content and amplitude mismatch. These are declared transfer limitations, not corrected with test-driven normalization.


## Reasoning-agent batch size selected from protocol fixtures

Before real model fitting or outcomes, 128 repeated numeric protocol fixtures were evaluated at batch sizes8,32,64. All384actions parsed validly. Measured amortized seconds/state were0.18145,0.07140,0.06398; peakallocated GPU memory was8334,10210,12713MiB respectively. Batch32 is selected prospectively for the main cached-evidence policy evaluation, with more context-memory headroom than64. These fixture timings do not establish acoustic performance or end-to-end savings. `manifests/agent_batch_scaling.json` retains traces. Production-shaped later-round numeric fixtures separately validate the selected batch.

The isolated live benchmark uses32 calibration recordings at batch32 with three timed repetitions after warmup. The smaller120-record robustness study retains its prospective batch8 and reports that geometry explicitly. Generated-evidence caching remains batch8 with fixed seeds so all matched policies receive identical samples. Acoustic generation, feature extraction and numeric validity are charged in live end-to-end cost.


## Strict stop-schema prompt correction before acoustic outcomes

The first production-shaped state stress test rejected32of128actions: every rejected response selected stop but wrote the string `"null"` instead of the JSON literal `null`. The generic prompt's quoted schema placeholder was ambiguous. The parser correctly rejected these actions; its rules were not relaxed. The prompt now includes separate concrete examples for a tool call and a stop call, explicitly forbidding the string form. Failed traces and original prompt source are preserved in `manifests/agent_policy_state_validation_failed_v1.json` and `manifests/source_snapshots/agent_prompt_v1.py`. The revised prompt must pass a fresh production-shaped fixture gate and the actual development/calibration acoustic-action gate before freezing. No real acoustic outcomes informed this correction.


## Descriptive mechanism and policy diagnostics

The real-data mechanism panel will summarize nuisance-adjusted logit contrast separately for AIS-background and AIS-vessel-present records, using only verification draws that pass all numerical gates. It reports the mean across detector seeds, accepted counts and validity fractions; no physical causal interpretation or independent-sample confidence interval is assigned. If a stratum has no valid verification, it has no measured contrast and is not plotted as a fabricated zero. Raw per-seed records remain available. Tool-use counts, verification rate, acquired-view count and action-format validity are also reported for both agent modes. These diagnostics are specified before real model outcomes and do not replace the primary AP endpoint.


## Additional controls before calibration and final test

Real-development intervention coverage was around40–45% under the existing numerical validity gates. Individual detector development APs had been observed; proposed-policy calibration, runtime measurements and final test outcomes had not. This motivated two continuous-score ablations, `fixed_validity_only` and `agent_validity_only`: preserve the exact original base logits, acquired views, query decisions, validity masks, fitting rows, minimum-count/two-class rules and positive two-input calibration procedure, while setting the continuous verification score V to zero. The second control retains the original verification-informed policy path; it is an output-score control, not an agent deprived of generated evidence. V=0 removes contrast, dispersion and boundary penalty together, so it is not described as a delta-only ablation.

These comparisons isolate any continuous-score contribution beyond validity-dependent recalibration. They are component ablations excluded from the primary comparator-selection family, consistent with the other component-removal controls, and will be reported even if unfavorable. No numerical validity threshold or trained detector is changed.

The live runtime benchmark adds the already-defined heuristic-verification comparator to the prior four methods, for five methods×three timed passes. This closes a cost-coverage omission in the planned agent-versus-rule comparison. The original four-method configuration and the scope amendment are preserved. No cost or final-test outcome informed the addition; the cost figure does not claim to measure an overall frontier across methods that were not timed.


## Constrained action decoding after a failed development integration gate

Despite passing numeric fixtures, the unconstrained controller failed on actual development states: after acquiring wavelet evidence it repeatedly requested STFT, which was already the initial observation and no longer an available tool. The parser rejected these requests and returned a stop fallback. The measured development action-validity gate failed. All completed and partial replay batches, receipts and the log are preserved under `results/raw/archive/unconstrained_policy_v2/` and in `manifests/agent_unconstrained_development_failure.json`. No held-out test outcome was produced or used.

The controller is revised to constrained greedy generative decoding over a per-state token trie of legal JSON completions. Allowed tools/candidate IDs are derived only from the current declared tool set and observed/history entries; all enumerated reason codes remain available. Stop requires the literal JSON null. The same pretrained Qwen weights choose tokens among legal continuations; no labels, accuracy-based retries or hand-coded action ranking are introduced. Raw generated JSON remains auditable. Valid format and tool legality under this decoder are an enforced integration property, not evidence that the language model independently learned to follow constraints or improved its reasoning. Token likelihoods, if logged, describe the constrained generation distribution and are not acoustic detection confidence.

All development and calibration policies must be regenerated under the same new controller before model selection/calibration/freeze. Previously trained numeric detector and generator weights, source splits, intervention caches and numerical gates remain intact. The original failed replay is a reported engineering failure, not silently discarded favorable-outcome selection.
