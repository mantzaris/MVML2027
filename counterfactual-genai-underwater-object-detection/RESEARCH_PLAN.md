# Generative AI Reasoning Agents for Underwater Acoustic Object Detection through Counterfactual Verification

Research design for a new MVML 2027 submission. Prepared 23 November 2026.

The proposed study tests whether a generative reasoning agent can improve underwater acoustic source detection by checking what happens to a candidate detection when plausible background replaces its acoustic evidence. It combines Fourier analysis, spectrograms, Hilbert–Huang analysis and wavelets, with real recordings, mixtures of real recordings, and controlled synthetic signals. The central contribution is a **nuisance-adjusted generative verification method**, tested against strong detectors and equally resourced non-agent systems.

This is a prospective research plan. Performance thresholds below are hypotheses and decision criteria, not measured results. The project is separate from the completed evidence-genAI-underwater-acoustic-reasoning submission; its files, models, frozen protocols and results remain unchanged.

## 1 Scientific scope and paper identity

**Preferred title:** Generative AI Reasoning Agents for Underwater Acoustic Object Detection through Counterfactual Verification.

**Alternative title:** Generative AI Reasoning Agents for Underwater Acoustic Object Detection with Fourier, Hilbert–Huang and Wavelet Evidence.

The preferred title emphasizes the methodological contribution; the alternative makes the signal-processing expertise explicit. Suggested keywords: underwater acoustics; acoustic object detection; generative AI; reasoning agents; counterfactual verification; Hilbert–Huang transform; wavelets; synthetic-to-real generalization.

An acoustic object is an audible source event, initially a civilian vessel source. The output is a target-presence probability and, where temporal supervision exists, an event tuple (onset, offset, class, confidence). Vessel class recognition is secondary. Spectrogram regions describe acoustic support, not physical object outlines. Single-channel recordings do not establish range, bearing, depth, identity, or detection of silent submerged objects.

Use two explicitly separated evaluation tracks:

| Track | Data and labels | Supported conclusion |
|---|---|---|
| Real detection | Original real vessel/background recordings, source and recording groups preserved | Generalization of target-presence detection against the dataset's labels |
| Event localization | Fully synthetic signals and semi-synthetic mixtures with known insertion intervals; separately annotated real continuous recordings if acquired | Temporal event detection on the particular data origin evaluated |

Real continuous-event annotation is a planned extension and a prerequisite for claims of real temporal localization. Clip class labels must never be converted into invented onset/offset labels.

## 2 Prior work and the novelty boundary

The closest recent work already connects acoustic physics, complementary representations, synthetic data and language models. Zhang et al.'s [Part I](https://arxiv.org/abs/2610.22064) develops a physics-aware recognition framework; Li et al.'s [Part II](https://arxiv.org/html/2610.31346v1) extends it to passive detection with array processing and structured reports. [UA-Bench](https://arxiv.org/abs/2610.16942) tests underwater audio-language models, including a silent-audio control. [Physics and metadata-guided generative learning](https://doi.org/10.1121/10.0046886) already studies generative augmentation for underwater recognition. Adding an LLM, wavelets or synthetic examples alone is therefore insufficient.

The sibling MVML submission already studies sequential evidence selection, supported generated reports, and real-vessel category identification. The new paper's contribution must instead rest on **waveform interventions that change a detection decision**, their validity controls, and their measured benefit. Existing evidence-selection code may inform a separately implemented comparator, with attribution; previously reported results must not be presented as new evidence.

Three candidate contributions are proposed:

1. A paired generative intervention protocol that contrasts candidate-interval replacement with nuisance-matched sham edits across acoustic representations.
2. A calibrated detection score that discounts generator variability, disagreement between representations and unintended waveform changes.
3. A bounded generative agent that chooses candidate intervals and verification tools, with experiments separating its contribution from generation, representation fusion and extra computation.

These are candidate contributions, not an established first-in-literature claim. The [literature and novelty review](protocol/literature_and_novelty.md) records primary sources and competing approaches. Refresh it before protocol freeze and submission.

## 3 Questions and prespecified hypotheses

**Primary question:** On untouched real recordings, does generative verification improve vessel-presence average precision over the strongest comparator selected on development data?

**H1 detection:** Seek an absolute average-precision improvement of at least 0.03, with a paired group-bootstrap 95% interval excluding zero. Report the full estimate and interval even when this target is missed.

**H2 mechanism:** On controlled mixtures with known target contributions, target-focused edits should reduce target evidence more than matched nuisance edits, without equivalently destroying unrelated events. Test whether the proposed adjustment improves detection beyond ordinary masking or interpolation.

**H3 agent value:** At a matched measured inference budget, adaptive verification should improve accuracy over fixed verification and a non-language policy, or attain comparable accuracy with at least 20% lower latency. Treat equivalence as requiring a prespecified margin and interval; a nonsignificant difference is not equivalence.

**H4 robustness:** Test whether gains persist under unseen backgrounds, propagation families, low SNR, and representation failure. Keep naturally occurring domain shift separate from artificial corruptions of real audio.

A positive headline requires H1 and no material loss of recall at the selected operating point. A generative-AI-agent headline additionally requires evidence that the language agent contributes beyond a generator-assisted non-language detector. Otherwise narrow the title and contribution to the component the experiment supports.

## 4 Data design and independence

### Real recordings

Use the [UniqueShip release](https://uniqueshipdata.org/) as the first feasible real source because it includes vessel identities and background examples. Its [paper](https://arxiv.org/html/2609.13659v1) motivates identity-disjoint evaluation. Begin with a manageable released subset; acquisition size is chosen after manifest inspection, not by assuming all published hours are available locally.

UniqueShip labels rely on AIS proximity. An AIS-defined background is not guaranteed absence of every audible vessel. Report this distinction and audit a stratified real subset independently. If acoustic-audibility labels cannot be validated, describe the main endpoint as detection against AIS-derived vessel-presence labels.

Use [ShipsEar](https://underwaternoise.atlanttic.uvigo.es/) as a candidate external corpus, subject to access and suitable background labels. [DeepShip](https://github.com/irfankamboh/DeepShip) is useful for recognition transfer but does not by itself provide a complete background-only detection set. Do not treat missing classes as negatives. Verify source-specific terms and available metadata before downloading or redistributing data; full access to these resources may require a request.

Target at least 100 independent vessel/encounter groups and 100 background day/deployment groups in the real test assessment, subject to actual availability. These are design targets, not a power calculation or a guarantee. Use development-only resampling to estimate precision for a 0.03 AP effect before freezing sample size. If the available independent groups are too few, report the attainable interval and limit the claim.

For a real event extension, target 2–4 hours of continuous audio, at least 100 audible events and several recording days. Two independent annotators mark audibility and interval uncertainty, with adjudication of disagreements; AIS is supporting metadata, not the sole audible-event reference. If this effort is unavailable, retain the two-track design and label real localization unvalidated. Do not substitute LLM judgments as temporal ground truth.

### Controlled data

Keep three origins distinct in manifests and every result table:

- **Natural real:** unmodified original recordings.
- **Semi-synthetic:** independently grouped real source excerpts inserted into real backgrounds; insertion boundaries are construction labels. A source excerpt can contain its own background, so it is not a clean isolated physical source.
- **Fully synthetic:** harmonic and amplitude-modulated sources, chirps and transients, with known source components and GPU-generated propagation/noise.

Generate an initial 20,000 training scenes of 10 seconds, with online regeneration if useful, and 2,000 validation and 2,000 test scenes. Record their much smaller number of independent parents. Include zero, one and two overlapping sources; durations 0.5–8 seconds; controlled received SNRs of −15, −10, −5, 0, 5, 10 and 20 dB. Fix final ranges using development data and physical plausibility checks.

Generate multipath delays and gains, frequency-dependent attenuation, smooth Doppler/time variation, colored ambient noise and impulsive interference. This is a bounded acoustic simulation, not a validated full-ocean propagation solver. Hold out complete background recordings and propagation parameter families, not merely new random seeds. Test with a second synthesis implementation or alternative family to expose simulator shortcuts.

### Split and contamination rules

Split **before** segmentation, synthetic mixing, generator training, normalization, retrieval or augmentation. Use vessel identity, encounter/parent recording, device/deployment and day as grouping constraints. Merge linked records into leakage components; keep each component in one partition. If these constraints leave insufficient data, document a narrower supported split instead of silently weakening independence.

Use approximate group allocations of 60% training, 15% development, 10% calibration and 15% locked test, subject to stratification and any official split requirements. Separate policy-training folds within training. Generator, detector and policy fitting may never use development/calibration/test waveforms as training targets.

All derivatives inherit every parent group, including background, source excerpt and impulse response. Use disjoint pools for each partition so a mixture cannot bridge train and test. Hash exact audio and audit near-duplicates, timestamps and metadata. Compare proposed identities and timestamps against the sibling submission's manifests **read-only**. Previously inspected groups may support development, but not a new untouched confirmatory test claim. If an uninspected cohort cannot be established, obtain another source or state that the evaluation is a reused benchmark.

Do not assume two dataset names imply independent collection: ONC-derived sources may overlap. Freeze label mapping, exclusions, provenance, split hashes and source counts before test inference.

## 5 Signal representations and mathematical foundations

Preserve original audio. Start with an anti-aliased 16 kHz analysis derivative and a 20–7,500 Hz band where acquisition supports it; retain native-rate material for sensitivity tests. This is a proposed engineering choice to validate, not the native rate of every corpus. Avoid per-clip normalization that erases received-energy evidence; use train-derived scaling and document any gain normalization.

For a length-N frame, use the discrete Fourier representation:

$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N},\qquad
x[n]=\frac{1}{N}\sum_{k=0}^{N-1}X[k]e^{j2\pi kn/N}.
\tag{1}
$$

This is the finite-data counterpart of Fourier-series reasoning. Include power, harmonic spacing, spectral concentration and phase-sensitive information where justified. A spectrum alone loses event timing.

Compute spectrograms at two resolutions:

$$
S(m,k)=\left|\sum_n x[n]w[n-mH]e^{-j2\pi kn/N}\right|^2 .
\tag{2}
$$

Initial FFT sizes are 512 and 2,048, with quarter-window hops and Hann windows. Align outputs in physical seconds and hertz. Log compression floors, scaling and image interpolation must be fixed; do not infer frequencies from unlabeled image pixels.

For Hilbert–Huang analysis, empirical mode decomposition gives intrinsic mode functions c_q and residual r:

$$
x(t)=\sum_{q=1}^{Q}c_q(t)+r(t),\quad
z_q(t)=c_q(t)+j\mathcal H[c_q](t)=a_q(t)e^{j\phi_q(t)},\quad
f_q(t)=\frac{1}{2\pi}\frac{d\phi_q(t)}{dt}.
\tag{3}
$$

This follows [Huang et al.](https://doi.org/10.1098/rspa.1998.0193). Mask low-amplitude and edge regions before interpreting instantaneous frequency. Record mode mixing, extrema failures, sift stopping and reconstruction error. HHT is adaptive decomposition, not simply applying a Hilbert transform to the original signal.

The continuous wavelet transform is:

$$
W_x(a,b)=\frac{1}{\sqrt{|a|}}\int x(t)\psi^*\!\left(\frac{t-b}{a}\right)dt .
\tag{4}
$$

Use analytic Morlet wavelets and 48 initial logarithmic scales, with scale-to-frequency conversion and edge masks recorded. A learnable wavelet front end is a comparator inspired by [AGNet](https://arxiv.org/abs/2306.01002), not an independent novelty claim.

Implement FFT/STFT, Hilbert analytic signals, convolutions and wavelets as CUDA tensor operations. EMD is the difficult branch: implement bounded batched extrema finding, envelope interpolation and sifting on CUDA, initially at most six IMFs and ten sifts per IMF. A piecewise-linear envelope variant must be called an EMD approximation and compared with a classical spline reference on small correctness fixtures. Ordinary CPU PyEMD must not become the main preprocessing path. If GPU HHT is numerically inadequate, keep validated Fourier/STFT/wavelet inference primary and report HHT as a failed or limited experimental branch.

## 6 Proposed generative verification method

### Detection and intervention model

A compact temporal convolutional detector proposes intervals using STFT or raw-audio features. Per-representation heads provide aligned class/presence logits for each candidate interval. Use a compact foundation-audio baseline as well as a learned detector so that gains cannot be explained merely by a weak encoder.

Freeze candidate-to-record aggregation before testing. A permissive development-selected proposal threshold creates common candidates for all matched methods. Verify at most the two candidates with highest base uncertainty, using a frozen tie-break; other candidates retain base scores. The record score is the maximum candidate score, followed by the same calibration opportunity for each method. If no candidate exists, use the base record-level head instead of dropping the record or assigning perfect certainty. Report proposal recall separately because verification cannot recover an event the proposal stage misses.

Process native five-second examples as five-second examples. Use windows of at most ten seconds within a single parent recording; zero-pad only for batching and mask padding from transforms, losses and pooling. Never concatenate unrelated clips to create context. Longer recordings use a frozen overlap/merge rule, and predictions touching window boundaries are merged before scoring.

Train a small conditional waveform generator to replace candidate content with plausible background while preserving context. A 1D U-Net of approximately 10–30 million parameters with conditional flow matching is the initial choice. [Flow Matching](https://arxiv.org/abs/2210.02747) supplies the generative training framework; it is not claimed as new.

For a known clean background target b, Gaussian latent z, observed waveform u and binary editable mask M, use the clamped training path y_t=M⊙[(1−t)z+tb]+(1−M)⊙u. A basic conditional objective is:

$$
\mathcal L_{\mathrm{FM}}=
\mathbb E_{t,z,b,u,M}\left[
\left\|M\odot\left(v_\theta(y_t,t,u,M)-(b-z)\right)\right\|_2^2
\right],
\tag{5}
$$

The conditioning input supplies the mask and unedited context, with the editable region hidden. Train on controlled pairs whose background target is known; clamp context outside M identically during training and sampling. In implementation normalize by the number of masked samples. Begin with eight sampling steps and K=4 generated replacements; assess 4/8/16 steps and K=1/4/8 on development.

Crucially, the primary background generator receives no true target class or test label. It should learn background completion, not manufacture evidence for a desired class. Train/validate it independently of the detector's test decisions. Source-conditioned or class-conditioned generation belongs only in explicitly labeled ablations.

### Proposed equation for nuisance-adjusted verification

For candidate i and representation v, let L_iv(x) be the clipped detector logit averaged over the candidate's temporal support. Let x_i^−k be replacement k of the candidate support and x_i^0k a nuisance edit on the **same support**, matched for duration, perturbation RMS, affected band and boundary taper. Construct the sham by adding a background-model nuisance residual while retaining the original waveform, scaling its perturbation to the replacement's RMS. Freeze this construction without test labels. On controlled stems, quantify whether shams preserve target detectability and whether energy/band matching succeeds. If no valid matched sham exists, mark verification unavailable and return the base score. Nearby-region edits are a separate locality diagnostic, not the primary matched null.

Define a paired contrast:

$$
\delta_{ivk}=
\left[L_{iv}(x)-L_{iv}(x_i^{-k})\right]
-\left[L_{iv}(x)-L_{iv}(x_i^{0k})\right].
\tag{6}
$$

The candidate statistic is:

$$
\boxed{
V_i=
\operatorname{median}_{v,k}(\delta_{ivk})
-\lambda\,\operatorname{MAD}_{v,k}(\delta_{ivk})
-\gamma R_i,\qquad
p_i=\sigma\!\left(a\,L_i^{\mathrm{base}}+b\,V_i+c\right)
}
\tag{7}
$$

MAD is the median absolute deviation around the median. R_i is a dimensionless validity penalty comprising off-mask normalized RMS change and standardized boundary discontinuity; standardization constants come from training, and the exact formula is frozen before testing. The base logit is retained so the intervention statistic cannot be mistaken for a complete generative likelihood model. Require a,b≥0 for interpretable score direction. Fit a,b,c on calibration only; choose lambda and gamma on development. Give comparators equivalent calibration access.

Equation 7 is a **proposed acoustic formulation and research hypothesis**. Median aggregation, dispersion penalties and calibration are established ingredients; the contribution must be demonstrated through the paired acoustic intervention protocol and controlled detection improvement. These deterministic representations are correlated views of one waveform, not independent witnesses. Interval replacement is a model-based sensitivity experiment, not proof of a physical causal effect. Same-support nuisance edits reduce a confound but do not establish a perfect counterfactual, particularly on real recordings without stems.

Background completion can remove every source inside the interval, including an overlapping unrelated event. Therefore the implemented operation is **candidate-interval replacement**, not demonstrated source-selective deletion. Evaluate single-source and overlapping-source scenes separately, and quantify collateral suppression. A future component/time-frequency mask may support a stronger source-selectivity claim only after validating preservation of known other sources.

Count failed, mismatched and invalid edits. If all edits fail a validation gate, return the base detector score with a verification-unavailable flag; never quietly drop the difficult sample. On controlled stems, measure residual target energy, background preservation and unintended changes to overlapping sources. Also test deterministic interpolation, energy-matched noise, phase randomization and sham generation so ordinary energy destruction cannot masquerade as useful verification.

### Observable reasoning agent

Use a frozen [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) controller initially. It is a text tool-use model, not a raw-audio model; provide structured numerical summaries and evidence identifiers from the transforms. The model's non-thinking mode is compatible with an agent defined by observable decisions, not claims about hidden reasoning.

At each step the controller receives candidate uncertainty, acquired measurements, intervention validity and remaining budget. It emits schema-constrained actions: inspect Fourier; inspect STFT; inspect HHT; inspect wavelet; verify a named candidate; stop. Allow at most three additional actions, one verification batch per candidate and two candidates per 10-second window. Cap context at 4,096 tokens and each action at 128 generated tokens; no unconstrained numerical diagnosis.

The numeric head computes probabilities and intervals. The LLM selects tools and can generate a short summary quoting returned measurements; it cannot overwrite calibrated probabilities, invent measurements or access labels. Log invalid actions, exhausted budgets, fallbacks and trace hashes. Malformed actions deterministically stop and return current numeric predictions.

A frozen prompted controller is the first experiment. A small policy model trained on training-only tool outcomes is the key comparator. Any later LoRA policy adaptation requires a separate development decision, training-only trajectories and a new freeze; it is not necessary to make the core study feasible.

## 7 Experimental matrix and fair comparisons

| ID | Method | What it isolates |
|---|---|---|
| B0 | Spectral energy/tonality detector, calibrated on development | Classical detection reference |
| B1 | Strong single-view temporal detector | Learned acoustic baseline |
| B2 | Foundation-audio encoder plus matched detection head | Pretrained encoder contribution |
| B3 | Four-view numeric fusion without language or generation | Representation contribution |
| B4 | B3 plus matched-volume synthetic augmentation only | Training-data benefit |
| B5 | Generator verification with fixed action order and no language model | Generator benefit |
| B6 | Generator verification with an uncertainty heuristic or learned small policy | Need for language reasoning |
| B7 | Bounded language agent with transforms but no generative interventions | Need for generation |
| P | Full method using Equation 7 and bounded agent | Proposed system |
| U | All-view/all-verification system with recorded higher cost | Accuracy ceiling, not a budget-matched claim |

Use the same detector initialization, training examples, candidate proposals and calibration opportunity for matched component comparisons. Include a separately trained stronger baseline if it wins on development. Select the strongest eligible primary comparator before opening the test set.

Give methods the same development tuning budget and report total fitting cost. Compare policies at measured latency budgets, not only action counts: language decoding and HHT can cost more than fixed fusion. A separate cost-matched baseline may spend its saved compute on a wider detector or more samples.

Train the core detector/full system and strongest comparator with three seeds. Use a development screening study to retain only essential confirmatory ablations: no generator, no null adjustment, no language policy, and each representation removed. Representation substitutions and generator hyperparameter sweeps remain supplemental/exploratory. Share frozen detector checkpoints between policy variants where scientifically appropriate.

## 8 Outcomes, analysis and decision rules

**Primary endpoint:** target-presence AP on untouched natural-real recordings. Use the same evaluated records and class prevalence for every method; disclose the assessment prevalence because AP depends on it. Record the score for every sample before any referral or abstention.

**Secondary detection outcomes:** recall at a calibration-selected operating threshold targeting at most 1% negative-clip false-positive rate; precision/recall curves; AUROC; Brier score; calibration plots; per-class AP when valid. Report the actual test false-positive rate and its interval. False alarms per hour require contiguous monitored background duration and a defined rule merging adjacent alarms; disconnected clips support false positives per clip, not an invented hourly rate.

**Temporal outcomes:** class-averaged temporal AP at IoU 0.3, 0.5 and 0.7, onset/offset error and event F1 on interval-labeled sets. Define one-to-one matching, minimum duration, merge gap and duplicate handling before test. Score predicted intervals in seconds. Keep natural real, semi-synthetic and fully synthetic outcomes in separate panels.

**Mechanism and cost:** target-removal and sham-response distributions; validity failures; preserved unrelated-event recall; schema failures; acquired views; all-sample performance and coverage; wall time; synchronized CUDA timing; peak allocated and device memory; host-to-device transfer time; GPU-hours. Record eGPU transfer overhead and first-call compilation separately from steady-state inference.

Use 2,000 paired bootstrap resamples at independent leakage-component level, with all descendants and methods carried together. Where vessel and day dependence cross, use the merged grouping or a justified multiway procedure; do not treat adjacent clips as independent. Report seed variation separately from sampling uncertainty. Predefine the primary contrast P versus the locked strongest comparator; use a small Holm-adjusted family for confirmatory mechanism contrasts when valid cluster-based tests are used.

Do not optimize thresholds, plot choices or stopping rules against test outcomes. Archive raw predictions for all methods. A lack of gain is a valid finding about when generative reasoning is unnecessary or unreliable; do not replace a failed primary endpoint with a favorable exploratory subgroup.

## 9 Main paper figures and eight-page allocation

Plan **five main figures, four containing experimental results**. “Impressive” means interpretable, convincing evidence with uncertainty and strong comparators, rather than promised favorable numbers.

| Figure | Content and intended insight | Evidence required |
|---|---|---|
| 1 Method | Real waveform and four aligned representations; candidate, target/sham edits, numeric score and action trace | Clearly labeled illustrative example, units and mask boundaries |
| 2 Main detection results | Real AP and operating-point recall/FPR; separate external-real and temporal panels | Group-bootstrap intervals, source counts, strongest comparator, every core seed |
| 3 Robustness | AP/temporal AP versus SNR, held-out background and propagation; natural-real shift separate from corruptions | Frozen severities, uncertainty bands, clean reference and failure regime |
| 4 Mechanism and ablation | Target-versus-sham score change, component removals, intervention failures and unrelated-source preservation | Known-stem controls plus separate real sensitivity results; no causal overclaim |
| 5 Accuracy and compute | Accuracy versus measured milliseconds/window and GPU-hours; fixed, small-policy and LLM agents | End-to-end eGPU measurements, peak VRAM, matching budget and all-call costs |

Select examples using a fixed rule, such as the median-error example plus the largest-confidence failure, rather than only favorable spectrograms. Use vector PDF plots, readable units, colorblind-safe colors and consistent axes. Generate figures directly from frozen prediction tables and keep failed cases visible.

The [official guidelines](https://mvml.org/author-guidelines/) limit a full paper to eight pages. Budget approximately one page for abstract/introduction, one for related work and task/data, two for methodology/equations, one for experiments, two for results/figures, and one for limitations/conclusion/references, adjusting with the official template. Keep the core novelty, primary result and reproducibility essentials in the main paper.

## 10 Focused four-page supplemental study

**Supplement title:** When Do Hilbert–Huang and Wavelet Evidence Improve Generative Underwater Acoustic Detection?

This study asks whether adaptive nonstationary representations add useful information beyond Fourier/STFT once generator artifacts and compute are controlled.

| Page | Study content | Planned artifact |
|---|---|---|
| 1 | Harmonic, chirped, amplitude-modulated and impulsive regimes; overlap and propagation equations; independent source/background partitions | Simulator parameters and exact interval-label rules |
| 2 | GPU EMD reconstruction, mode mixing and edge effects; wavelet family/scales; known instantaneous-frequency error with amplitude masks | Figure S1 numerical validity and regime-dependent errors |
| 3 | Factorial STFT versus +HHT versus +wavelet versus all views, at fixed detector/agent budget; missing-view and sham-edit tests | Figure S2 performance-versus-cost by nonstationarity regime |
| 4 | Generator leakage/failure cases, per-class outcomes, seed intervals, runtime/VRAM breakdown and implementation hashes | Tables S1–S2 and representative failure examples |

Use at least two source-generator families, held-out parameter combinations and a predeclared small matrix. Avoid a full combinatorial sweep on test data. The supplement's mechanistic conclusions must be supported by its own controlled experiments, not only more plots of the main result.

The public conference guidelines inspected on 23 November do not specify a supplemental submission channel or separate allowance. Prepare a standalone 3–4 page PDF; verify portal or organizer policy before submission. If unsupported, retain it as a permitted linked companion artifact, and keep every essential claim within the eight-page paper.

## 11 Local 32 GB Ada GPU execution

The requested local eGPU is now identified independently of the old remote worker:

- Device: NVIDIA RTX 5000 Ada Generation.
- GPU UUID: GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf.
- Reported total memory: 32,760 MiB; driver: 590.48.01.
- PCIe device observed at 09:00.0, connected through authorized Sonnet Thunderbolt 3 hardware reporting 40 Gb/s.
- Internal Quadro T2000: 4,096 MiB; excluded from this project's workloads.

Use the included UUID-checking launcher for every training, transform and inference process. The device index may change after reboot; UUID selection is stable for the physical card. Inside the restricted process this eGPU becomes cuda:0. The launcher must fail if the UUID, model or minimum 30,000 MiB total memory check does not match. No automatic CPU or smaller-GPU fallback is allowed. [NVIDIA documents UUID-based device selection](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html).

All substantial signal transforms, generation, training and inference run on the Ada GPU. CPU use is limited to orchestration, decoding/I/O, small reference correctness checks, statistical reporting and manuscript plotting. This distinction avoids claiming that a computer can operate with literally no CPU activity.

Use BF16 where numerically valid, FP32 for EMD/Hilbert phase calculations and critical reductions. Start with microbatch 4–8 and gradient accumulation for effective batch 32; use checkpointing if needed. A 4B model in BF16 has roughly 8 GB of raw parameter storage, with additional runtime memory. Budget approximately 12 GB for the controller, 8 GB for detector/generator/tensors and at least 4 GB headroom, then measure actual peaks. Never promise fit from parameter counts alone.

Train major components in separate processes; load the controller only for policy inference. Inference co-residency must pass a peak-memory test. If needed, shrink batches/context or use an explicitly validated GPU quantization before freeze; do not spill model layers to CPU through an automatic device map. Cache immutable transform tensors with content hashes, while measuring uncached acquisition cost separately.

Thunderbolt bandwidth can dominate small jobs. Transfer batches once, retain tensors on GPU through multiple views, and log transfer-inclusive latency. Do not claim the same throughput as a directly attached workstation card. Before detaching, finish jobs, synchronize CUDA, save checkpoints and verify no project process uses the device; follow the established power-off attach/detach workflow when stability requires it.

## 12 Work packages and resource gates

The following GPU-hour ranges are planning estimates, not measurements or an instruction to begin a full study. This task creates the plan and performs a brief connection/smoke check; it does not train research models. The previous project's expired ten-hour execution window does not apply to this separate submission.

| Stage | Deliverable and gate | Estimated Ada GPU-hours |
|---|---|---:|
| A | Device and tensor smoke check; new project environment; manifests and licensing | Under 1 |
| B | Four transforms, GPU HHT validity, throughput/memory pilot, leakage audit | 2–4 |
| C | Detector baselines and generator fitting, development-only screening | 8–12 |
| D | Paired intervention validity and small-policy/LLM comparisons | 4–6 |
| E | Frozen core methods across three seeds and real/controlled tests | 10–16 |
| F | Focused supplement, final cost profiling and rerun reserve | 5–9 |
| Total | Re-estimate after the pilot; record actual active/elapsed time separately | Approximately 30–48 |

Require a credible generator validity result, usable GPU HHT or a documented scope reduction, sufficient independent real groups and feasible measured throughput before spending the full budget. If source annotations or external access are delayed, keep the primary real-presence track feasible while making localization limitations explicit.

Proposed calendar:

- **23–29 November 2026:** dataset/access and contamination audit, numerical/GPU pilot, finalize scope.
- **30 November–13 December:** detector and generator development, null controls and calibration design.
- **14–27 December:** agent comparisons, focused ablations and protocol freeze.
- **28 December–10 January 2027:** locked evaluation, supplemental experiments and figures.
- **11–18 January:** eight-page manuscript, 3–4 page supplement, independent methods/results audit.
- **19–20 January:** format and citation verification, final author review.

The currently listed [paper deadline is 21 January 2027](https://mvml.org/important-dates/); the conference is 15–17 August 2027. Recheck dates and supplement policy before submission.

## 13 Reproducibility and completion criteria

The project should ultimately contain immutable source/split manifests, approved acquisition terms, model revision hashes, environment lock, training and generator seeds, prompts/tool schemas, calibration parameters, raw predictions, actual GPU records, source code, figure builders and manuscript sources. Put large data/checkpoints in ignored local storage and retain at least one verified backup. Keep credentials out of version control.

Before a positive paper claim, verify:

1. The previous submission remains intact and any reused ideas/data are attributed.
2. GPU jobs use the verified RTX 5000 Ada UUID and fail closed otherwise.
3. Training/test provenance and all parent relationships pass the frozen audit.
4. Natural-real detection is evaluated separately from controlled temporal localization.
5. Strong, cost-matched non-agent and no-generation controls are included.
6. The proposed equation's benefit survives sham-edit and generator-validity controls.
7. At least four result figures show actual estimates and uncertainty, including failures.
8. The supplement answers its stated nonstationarity question in 3–4 pages.
9. Every number in the manuscript is traceable to saved predictions; no planned target is presented as a result.

The first executable milestone is the GPU and data-validity pilot. The decisive scientific milestone is a positive or negative answer to whether controlled generative verification improves untouched real detection beyond a well-tuned acoustic detector.
