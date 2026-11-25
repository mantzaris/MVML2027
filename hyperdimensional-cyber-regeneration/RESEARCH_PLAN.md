# GPU research plan for cyber planning regeneration

**Working title:** Learning to Regenerate Cyber Planning States under Deception with Hyperdimensional Neurosymbolic Models

**Prepared:** November 25, 2026

**Targets:** MVML 2027 and research evidence relevant to DARPA HYPER-CAP, DPA27BZ02-DV001

**Project root:** `~/Documents/repos/MVML2027/hyperdimensional-cyber-regeneration/`

**Research stage:** Original proposal; a first CUDA snapshot study is now implemented.

**Execution record:** The [frozen executable protocol](configs/executed_study.json), [method mapping](docs/METHOD_IMPLEMENTATION.md), and [measured results](docs/RESULTS.md) define the completed experiment. The sections below preserve the original broader plan. Sequential episodes, multiple wire codecs, link-burst tests, and external cyber-range validation remain extensions; they are not claimed as completed.

## Research contribution

Develop a CUDA-based model that learns which uncertain facts to regenerate, using their potential effect on the feasibility and ranking of cyber courses of action (COAs). Regeneration reconstructs selected beliefs and updates their dependent plan branches while preserving supported information. It includes retaining ambiguity when the observations do not identify a unique state.

The intended contribution is a **decision-impact supervised regeneration gate**, coupled to hyperdimensional evidence memory and constrained graph reconstruction. The key question is whether this targeted repair improves viable-plan recovery at fixed information, computation, and communication budgets.

The model learns from structured observations and simulated decision outcomes. A shared symbolic planner provides the downstream task and inspectable action dependencies. The scientific contribution is in representation learning, reconstruction, calibration, and generalization.

Proposed hypotheses:

| ID | Hypothesis | Required evidence |
| --- | --- | --- |
| H1 | Selective regeneration improves recovery of viable COAs at a fixed repair budget. | Higher feasible-COA recall than uncertainty-only and random gates on paired held-out episodes. |
| H2 | Supervision by counterfactual decision impact improves planning beyond generic reconstruction. | Improvement against an otherwise identical model without the decision-impact teacher; report reconstruction accuracy separately. |
| H3 | Hyperdimensional memory provides a useful quality–bandwidth tradeoff. | Comparison against dense learned messages and compact symbolic deltas with all metadata counted. |
| H4 | Provenance and calibrated uncertainty reduce confident false repair. | Lower false-certainty rate at matched recommendation coverage, including coherent deception. |

Neither HDC superiority nor the uniqueness of the proposed equation is assumed. A targeted literature check must examine closely related budgeted feature acquisition, decision-focused imputation, learned belief revision, and plan repair before the manuscript makes a priority claim.

## Prior approaches and their role

| Prior approach | What this study inherits | What this study changes |
| --- | --- | --- |
| GraphMAE, Hou et al., KDD 2022 [R1] | Masked graph-feature pretraining and an encoder–decoder representation. | Temporal provenance, deceptive feature replacements, and selective inference-time reconstruction trained for COA consequences. |
| Decision-focused graph learning, Wilder et al., NeurIPS 2019 [R2] | Optimizing a prediction model for downstream decision quality. | A supervision signal for where to regenerate, obtained from the plan regret reduced by repairing individual facts. |
| Semantic loss, Xu et al., ICML 2018 [R3] | Learning with logical constraints over probabilistic outputs. | Apply small local constraint factors to planning beliefs; retain a hard symbolic verifier for final candidates. |
| NeuralHD, Zou et al., SC21 [R4] | The precedent for regeneration within HDC. | Here the regenerated units are semantic facts and dependency-linked beliefs; the shared vector codebook stays fixed. |
| Resonator networks, Frady et al., 2020 [R5] | Compositional vector representations and associative recovery as related foundations. | Address decision-relevant false or missing evidence rather than claiming vector factorization itself is new. |
| Neurosymbolic cyber policy learning, Samaddar et al., 2026 [R6] | A relevant cyber learning comparison under partial observability. | Predict and repair planning beliefs; attacker-policy imitation is not the principal learning target. |

GraphMAE originally reconstructs node features. The primary reproduction must preserve that feature-reconstruction task before adding our downstream belief head. A cyber adaptation must be called **GraphMAE-adapted**, not presented as an unchanged reproduction of the original benchmark.

See [References](docs/REFERENCES.md) for primary papers, code, and source-specific qualifications.

## Problem and information boundary

An episode contains a known action schema, a service-dependency graph, a mission objective, and a hidden dynamic state `y_t`. Three observers receive overlapping but incomplete evidence. A source may be delayed, missing, or manipulated. The adversarial intervention changes observation reports; genuine service failures change the hidden state. These interventions must be separately labeled in the generator.

A fact can describe service readiness, resource availability, credential validity, a dependency, or a mission prerequisite. Observations include source and time metadata. The base study assumes that report transport identity is authenticated, but an authenticated source can lie. Forged source identities and compromised rule specifications are separate extension conditions.

The model may access only its local history, received messages, the shared action schema, and previously inferred beliefs. It cannot access injection masks, the hidden state, source-compromise labels, or evaluator utilities at test time. Topology generation seeds, episode identifiers, and fact ordering must not leak split membership or corruption identity.

A frozen candidate generator enumerates at most 32 COAs from the known action catalogue, with at most eight abstract steps per COA. It uses the same expansion budget and observable topology for every method, never evaluator-only state. The planner evaluates those candidates under each method's beliefs. In the first benchmark, actions have deterministic symbolic effects and candidate utility is evaluated by simulated rollout. This establishes a controlled starting point before adding stochastic effects.

The reference COA is the best member of this shared candidate set under hidden ground truth. It is a candidate-set reference, not a claim of globally optimal planning. Empty or infeasible candidate sets are retained and reported.

## GPU architecture

1. **Common evidence update.** Produce a provisional fact-belief vector `b_t^-` from prior beliefs and currently available reports. Use the same preliminary update across representation ablations. Legitimate changes may invalidate earlier beliefs.
2. **Vector-symbolic encoding.** Encode role-specific entity, relation, value, source, and time features into bipolar hypervectors. Use a fixed shared codebook with reproducible seeds and role permutations to distinguish ordered relations. Bundle source/time windows separately to preserve provenance structure.
3. **Neural inference.** A three-layer graph network with 128 hidden channels consumes retrieved HDC similarities, source/time features, prior probabilities, and dependency edges. It produces candidate regenerated probabilities `r_theta` and repair-priority logits `a_phi`.
4. **Budgeted repair.** Select at most `K` facts using the learned gate. Decode only the selected neighborhoods in the efficiency implementation. A full-decoder implementation is a correctness reference; selection alone does not establish a computational saving.
5. **Planning and explanation.** A common planner re-evaluates prerequisites, generates the top three distinct candidates, and attaches evidence and changed-assumption traces. Candidate diversity is measured using action-set distance.
6. **Communication.** Compare a quantized HDC summary, a learned dense summary, and sparse symbolic deltas at equal byte budgets. Include source identifiers, timestamps, indices, scales, headers, and codebook setup costs. Deduplicate shared observations by origin IDs.

Keep a bounded local evidence ledger and unquantized accumulators where needed for deletion and revision. Do not claim that an arbitrary individual fact can be exactly removed from a lossy binary bundle. Count the ledger and accumulators in total memory.

For ambiguous states, evaluate a fixed number of sampled or explicitly retained hypotheses and report conditional feasibility and calibrated success estimates. The symbolic verifier proves compliance with the supplied state and rules; it does not prove that a deceptive report is true.

## Proposed equations and derivation

Let `F` denote the number of facts, `P_t` the shared candidate set, `C` the supplied rules, and `K` the maximum number of facts allowed to change at one update. Let `x_t` contain observable history and received messages. A reconstruction model returns `r_theta(x_t) in (0,1)^F`.

The selective regeneration operator is

$$
b_t^g=(\mathbf{1}-g_t)\odot b_t^-+g_t\odot r_\theta(x_t),
\qquad
g_t\in\{0,1\}^{F},\quad \|g_t\|_0\le K.
\tag{1}
$$

Equation (1) is a masking construction, not the novelty claim by itself. It extends full masked reconstruction with an explicit inference-time budget.

For training, define a differentiable candidate-regret surrogate:

$$
\mathcal R_\tau(b;U_t^*)=
\sum_{\pi\in\mathcal P_t}
\frac{\exp(\widehat U(\pi;b)/\tau)}
{\sum_{\pi'\in\mathcal P_t}\exp(\widehat U(\pi';b)/\tau)}
\left[U_t^*(\pi_t^*)-U_t^*(\pi)\right],
\qquad
\pi_t^*=\arg\max_{\pi\in\mathcal P_t}U_t^*(\pi).
\tag{2}
$$

Here `U_t^*` is utility from training-simulator truth, and `Uhat` is a fixed differentiable utility surrogate shared by all learned models. It evaluates relaxed prerequisites through the same action sequence and subtracts fixed costs. Fuzzy conjunction and marginal-probability products are approximations, not an assumption that real facts are independent. Final evaluation uses hard simulator rollouts and the same reference candidates.

**Proposed new decision-impact regeneration equation.** Replace ordinary random or uncertainty-based mask supervision with the reduction in decision regret obtainable by repairing one fact:

$$
\Delta_{tf}=
\left[
\mathcal R_\tau(b_t^-;U_t^*)-
\mathcal R_\tau\!\left(
b_t^-+(y_{tf}-b_{tf}^-)\,e_f;\,U_t^*
\right)
\right]_+,
\quad
g_{t,K}^{*}=
\underset{
z\in\{0,1\}^{F},\,\|z\|_0\le K
}{
\arg\max}
\sum_{f=1}^{F}z_f\,\Delta_{tf}.
\tag{3}
$$

`e_f` is the unit vector for fact `f` and `[v]_+=max(v,0)`. Choose only strictly positive scores, break ties deterministically, and use the sparsest maximizing mask; all-zero scores therefore produce no repairs. With unit fact costs, the teacher mask is computed by a positive-score top-K operation. This targets *recoverable planning loss*, including low-entropy beliefs that are confidently wrong.

Equation (3) is the proposed methodological extension: decision-focused supervision is used to teach the location of regeneration in a vector-symbolic belief model. The constituent ideas are inherited from [R1–R3]. Its originality must be established against relevant prior methods, not merely by giving the expression a name.

Train the gate probabilities `q_phi=sigmoid(a_phi(x_t))` and decoder jointly:

$$
\begin{aligned}
\mathcal L_{\rm regen}
=\mathbb E\Bigg[
&\mathcal R_\tau(b_t^g;U_t^*)+
\frac{\lambda_{\rm rec}}{F}
\sum_f {\rm BCE}(r_{\theta f},y_{tf})\\
&+\frac{\lambda_{\rm gate}}{F}
\sum_f {\rm BCE}(q_{\phi f},g_{t,K,f}^{*})
+\lambda_{\rm sem}\mathcal L_{\rm sem}(b_t^g;\mathcal C)\\
&+\lambda_{\rm keep}
\frac{\sum_f m_{tf}(b_{tf}^g-b_{tf}^-)^2}
{\max(1,\sum_f m_{tf})}
\Bigg].
\end{aligned}
\tag{4}
$$

`m_tf` marks facts whose true values did not change and whose provisional probabilities were already within a validation-fixed tolerance of truth. It is a training-only mask; it must not penalize legitimate state transitions. `L_sem` is the average negative log probability of satisfying small local logical factors. Compute exact factor probabilities for the bounded factors used in the base benchmark; do not claim exact inference over an unrestricted global formula.

Use a hard top-K forward gate with a straight-through continuous relaxation during training, and record this biased-gradient approximation. At inference, select up to K logits above a validation-chosen threshold. No oracle quantities from equations (2–4) are available to that inference gate. Reconstruction BCE trains the decoder on all facts even when a gate is closed.

Compute teacher deltas in GPU chunks across facts and candidate plans. Cache labels generated with a frozen surrogate so the target does not move with gate parameters. Independent one-fact gains ignore interaction effects: explicitly evaluate cases where two individually useless repairs are jointly necessary. If this limits the method, an exploratory pairwise teacher may be studied separately with its extra GPU cost reported.

Limiting checks:

- `K=0` reproduces the provisional belief state exactly.
- Selecting every fact yields full reconstruction through the same decoder.
- Removing the regret and gate-supervision terms gives a reconstruction-and-constraint baseline.
- Replacing `Delta` with entropy yields the uncertainty-gated baseline.
- Changing HDC to dense embeddings while retaining equations (1–4) tests the representation contribution.
- Indistinguishable true and deceptive histories cannot support guaranteed truth recovery; measure calibrated ambiguity rather than inventing a recovery guarantee.

## Required comparisons

| ID | Method | Purpose |
| --- | --- | --- |
| B0 | Probabilistic evidence update plus symbolic truth maintenance and shared planner | Conventional non-neural reference. |
| B1 | GraphMAE-adapted pretraining plus a supervised belief decoder and shared planner | Required comparison to an established ML approach. |
| B2 | GraphMAE-adapted model with the same decision-aware and semantic losses as the proposed model, dense messages, and full reconstruction | Strong control for gains caused only by a better loss. |
| B3 | Fixed-codebook HDC, associative cleanup, no learned selective repair | Isolate what HDC retrieval alone provides. |
| B4 | Proposed model with full reconstruction and otherwise matched training | Test selectivity and actual end-to-end compute savings. |
| B5 | Proposed model with an entropy-based gate at the same K | Test the value of the new equation. |
| P | HDC memory with the decision-impact learned gate and constrained decoder | Proposed complete method. |

Mandatory ablations add a random gate, no provenance, no semantic term, and a dense representation with the proposed gate and objective. Equalize observations, candidate sets, training examples, tuning budgets, and allowable memory/communication; report parameter counts and measured GPU costs rather than asserting perfect compute equality. Tune each trainable baseline on validation data with the same maximum number of trials.

NeuralHD is a conceptual predecessor, not automatically a fair planning baseline. A supervised NeuralHD feasibility-classifier adaptation may be added as an auxiliary comparison if sufficient time remains. It does not replace B1 or B2.

## Dataset and experiments

Build a reproducible synthetic service-dependency benchmark first. Use 16, 32, and 64 hosts for training/validation and a separately generated 96-host extrapolation set. Hidden facts are grouped into service readiness, resources, access prerequisites, and dependencies. Cap the base study at 512 fact slots; use padding masks and consistent capacity across methods.

The pilot uses 2,000 training episodes. The main target is 20,000 training, 2,000 validation, and 4,000 test episodes, each with up to 64 steps. These are proposed dataset sizes, to be finalized after GPU profiling and before test inspection. Split by topology family and generation seed; never distribute adjacent windows of one episode across splits. Reserve different dependency motifs and mission compositions for out-of-distribution tests.

The three observers have overlapping partial visibility. Sweep:

- Report corruption: 0%, 10%, 20%, and 40% of eligible reports.
- Missing observations: 0%, 20%, and 50%.
- Deception: random replacements, targeted critical-prerequisite reports, and coherent correlated false reports.
- Benign dynamics: service failures, recovery, and changed mission priorities.
- Message budget: 0, 512, 2,048, and 8,192 bytes per observer per communication round, measured on serialized payloads.
- Link loss: 0%, 25%, and 50%; bursts of 0, 4, and 16 steps without exchange.
- Repair fraction: 5%, 10%, 25%, and 100% of active facts.
- HDC dimension: 2,048, 4,096, and 8,192.

Use planned one-factor sweeps around a frozen reference setting, plus a small declared interaction grid; do not run an uncontrolled full Cartesian product. Separate attack sophistication from the amount of corrupted information. Match random and targeted corruption budgets.

At test time include held-out corruption-generation mechanisms, coordinated stories that satisfy local logic, zero-independent-evidence cases, and the joint-repair stress case for equation (3). Maintain a single list of paired scenario IDs and corruption realizations across methods.

CybORG/CAGE Challenge 4 [R7] provides a possible second public simulation setting. Export observable traces, attach the project's own corruption wrapper, map the action/observation schema explicitly, and document that this wrapper is new. Use prescribed policies to collect data; autonomous-policy learning is outside the first paper's scope. Make this external validation an extension if integration would displace the mandatory comparisons.

## GPU compute and reproducibility

**Verified local platform:** the attached NVIDIA RTX 5000 Ada Generation, with 32 GB nominal memory and 31.48 GiB reported by CUDA, driver 590.48.01, Python 3.12.3, and PyTorch 2.9.1+cu128. CUDA, HDC algebra, and BF16 forward/backward operations passed the hardware smoke check on November 25, 2026. Use the [Ada launcher](docs/GPU_SETUP.md) and retain a 24 GiB memory envelope for the planned constrained-memory comparison. Record the device, power limit, runtime, and usable memory again in each research-run manifest.

- Place HDC encoding, similarity search, graph neural layers, counterfactual teacher batches, loss computation, and learned inference on CUDA.
- Use BF16 autocast where supported; retain FP32 probabilities, logarithms, HDC accumulation, losses, and calibration calculations. Validate mixed-precision output against an FP32 GPU reference on a fixed batch. Follow the installed PyTorch release's AMP documentation [R10].
- Start with microbatch 16, gradient accumulation 4, and effective batch 64. If VRAM is insufficient, reduce microbatch and increase accumulation while preserving the effective batch and data order.
- Start at 4,096 HDC dimensions and a 128-channel, three-layer graph network. Chunk fact counterfactuals and candidate scoring to avoid allocating the entire batch × fact × plan × hypervector tensor.
- Use AdamW, a maximum of 100 epochs, early stopping patience 10, and five independent training seeds. Learning rate, loss weights, and gate threshold are selected only on validation data.
- Precompute simulation traces and teacher labels where appropriate. CPU file handling, benchmark generation, and symbolic bookkeeping are permitted; model computation must not silently fall back to CPU.
- Run B0's numeric evidence updates as CUDA tensor operations as well; time the shared host-side symbolic routines separately.
- Compare every learned method on the same GPU and precision policy. Measure batch-one latency for the decision-support setting as well as batched throughput.
- Warm up kernels and report median/p95 model latency using synchronized CUDA timing [R11]. Also report end-to-end wall time including decoding, message serialization, transfers, and symbolic planning.
- Measure peak allocated/reserved VRAM, host RAM, and all state retained for evidence provenance. Record run duration and GPU-hours. Power or energy claims require measured samples and a specified integration method.
- Freeze package versions and a container digest only after a working CUDA pilot. Do not label an untested environment file as reproducible.
- Keep raw seed-level metrics and failed runs. Record dataset hashes, settings, commit, environment, and candidate-generator version in each run manifest.

Use a pilot of at most four GPU-hours to measure feasible batch sizes and throughput. A provisional total cap of 200 GPU-hours covers the main comparison and core ablations; revise the experiment grid using measured pilot cost before launching it. This is a planning budget, not authorization to provision paid infrastructure.

The GPU readiness utility exits with a clear failure if CUDA or PyTorch is missing. The hardware smoke check alone is not a research result. The subsequent research run is documented separately in the execution record above.

## Metrics and statistical protocol

**Primary endpoint:** feasible-COA recall at three, computed as the fraction of eligible post-deception decision points where at least one of the three recommendations is truly feasible. Report results conditional on a feasible candidate existing and separately report the frequency of no-feasible-candidate episodes.

Secondary endpoints:

| Measure | Definition or reporting rule |
| --- | --- |
| Decision regret | Ground-truth utility of the best shared candidate minus utility of the selected candidate; include defined failure penalties. |
| Recovery delay | Steps and wall-clock time to the first truly viable recommendation after intervention; treat unrecovered episodes as censored and report their rate. |
| Belief accuracy | Fact-level Brier score and class-balanced reconstruction accuracy. |
| False repair | Fraction of changed facts that move farther from the hidden truth. |
| False certainty | Wrong fact predictions above a validation-fixed confidence threshold; report coverage alongside error. |
| Preservation | Rate of changes to facts that were correct and remained true. |
| COA diversity | Pairwise action-set distance among the recommended candidates, alongside their feasibility. |
| Calibration | Reliability diagrams and Brier score for plan success; ECE as a secondary bin-sensitive summary. |
| Resource cost | Serialized bytes, total state memory, batch-one median/p95 latency, throughput, and GPU-hours. |

Use five training seeds and paired held-out episodes. Compute paired effect sizes and a hierarchical bootstrap over training seeds and scenario/topology clusters; never treat correlated time steps as independent samples. Use 95% intervals and report each seed. Predeclare P versus B2 as the main learned comparison; report other tests as secondary and apply a stated multiplicity adjustment when presenting p-values.

Freeze utility weights, penalties, confidence thresholds, reference conditions, splits, stopping criteria, and the primary analysis before final test runs. If planning gains disappear against B2, narrow the claim accordingly. If symbolic delta messages dominate HDC, report that outcome and the operating conditions rather than selecting a favorable bandwidth point.

## Figures and paper structure

Produce **six rich multipanel paper figures**, including at least four empirical result figures. Exact panels, inputs, captions, and provenance rules are in [Figure plan](docs/FIGURE_PLAN.md).

1. Architecture, mathematical mechanism, and a traceable plan-repair example.
2. Feasible-COA recovery and decision regret across deception severity.
3. Belief regeneration, calibration, and the boundary of recoverability.
4. Communication, memory, and GPU latency tradeoffs.
5. Ablations, representation comparisons, and gate behavior.
6. Generalization and the interaction failures of the one-fact teacher.

Four initial design diagrams accompany this plan in `figures/design/`. They explain the proposed experiment; they do not count as empirical evidence.

A proposed eight-page allocation is 0.75 page for introduction/contributions, 0.75 related work/problem, 1.5 method/equations, 1.0 experimental protocol, 2.5 results with figures, and 1.5 discussion/conclusion/references. Adjust this allocation against the official template. Use concise multipanel figures sized for legibility; if only four empirical figures fit, retain Figures 2–5 and combine generalization into Figure 5.

## MVML and DARPA alignment

MVML includes adaptive systems, machine learning, neurocomputing, and statistical learning. Its author guidance currently limits full papers to eight pages, and its posted submission deadline is January 21, 2027 [R8]. Frame the manuscript around the learned operator, comparison, and measured uncertainty.

DARPA's topic seeks structured, explainable COA generation and revision under incomplete or manipulated observations and degraded communications [R9]. The study supports that larger system through:

| Topic need | Research evidence to produce |
| --- | --- |
| Generate and compare alternatives | Shared generator, top-three quality, diversity, and ground-truth utility. |
| Replan after deception | Recovery curves and linked changes to facts, prerequisites, and branches. |
| Hyperdimensional or neurosymbolic representation | HDC-versus-dense and HDC-versus-symbolic-message comparisons. |
| Distributed knowledge under degraded links | Three partial observers, serialized communication budgets, and burst outages. |
| Explain recommendations | Deterministic traces connecting actions to evidence, rules, confidence, and changed assumptions. |
| Constrained deployment | GPU resource profiles and an offline, GPU-equipped tactical-server demonstration target. |

The FAQ allows hybrid architectures, requires offeror-supplied testbeds, and describes a mobile tactical server without real-time cloud connectivity [R9]. Research training may use a larger GPU; deployment evidence must separately measure a stated GPU-equipped server without cloud dependence. A workstation speedup alone does not establish tactical suitability.

The bid is Direct to Phase II and requires prior feasibility evidence; this plan is not that evidence. Actual prototype artifacts and measured results can support the technical package. The complete bid also needs its required integration, commercialization, cost, and administrative material. Map the public recovery benchmark to the bid's named operational user and mission objectives; do not treat a recovery-only simulation as proof of complete operational coverage. The current DARPA page lists January 6, 2027 at noon Eastern as the close [R9].

Keep publication experiments public and synthetic. Treat any later handling of controlled mission information as a separate workstream with the applicable institutional process.

## Schedule and deliverables

| Window | Deliverable | Acceptance condition |
| --- | --- | --- |
| Nov 25–Dec 2 | Final problem definition, prior-work matrix, CUDA pilot, split manifest | Candidate-set and truth boundaries explicit; GPU execution verified. |
| Dec 3–Dec 12 | Generator, B0/B1/B2/B3, initial architecture | Same observations and planner across methods; baseline reproduction checks documented. |
| Dec 13–Dec 23 | Equations (1–4), proposed model, B4/B5, unit and integration checks | Gate budget, no-truth inference, constraint factors, and limiting cases verified. |
| Dec 24–Jan 4 | Core comparison, seed runs, communication sweep | Real logs and preliminary Figures 2–4; sufficient evidence to assess a bid contribution. |
| Jan 5–Jan 12 | Ablations, generalization, failure cases | Figures 5–6; main claims supported or narrowed. |
| Jan 13–Jan 20 | Manuscript, final figures, citation and reproducibility review | Template compliant, figures readable, all claims traceable. |
| Jan 21 | MVML submission target | Final author-approved submission package. |

Maintain `docs/` for protocol decisions, `configs/` for settings, `scripts/` and future `src/` for code, `data/` for manifests/data, `results/` for measurements, `figures/` for visual outputs, future `manuscript/` for the paper, and `work/` for temporary files.

Do not expand into a full autonomous cyber platform before the primary comparison is complete. The deliverable is a defensible learning contribution and a reusable module for the broader planning bid.


## Third-study update: direct feasibility prediction and cheap reconstruction

This addendum defines a third investigation alongside the original proposal preserved above and the [second study](docs/METHOD_V2.md). Its question is whether bypassing fact reconstruction improves candidate selection, and whether inexpensive ML can match more elaborate representations. The [third method specification](docs/METHOD_V3.md) and [frozen executable configuration](configs/study_v3.json) define this experiment. No improvement or broad novelty claim is assumed.

**SCR (shared-context candidate ranking)** encodes each observed fact together with its known parent's features. It pools required nonroot children and unique touched roots, so an explicitly repeated ancestor does not change the representation when the received feature tensor is fixed. Global embedding means and standard deviations drive a four-expert mixture shared across the candidates:

$$
\widehat s_j=\sum_{k=1}^{4}\operatorname{softmax}_k(a_\theta(g))\,
\sigma\!\left(\rho_{\theta,k}([u_j,g])\right),
\qquad \widehat U_j=v_j\widehat s_j-c_j.
\tag{V3.2}
$$

The [method document](docs/METHOD_V3.md) defines the canonical representation, objective and numerical conventions, and attributes set pooling to Deep Sets and gating to mixtures of experts. This task-specific composition is an investigation building on previous approaches. The latent experts are not identified attack classes. Predicted candidate probabilities need not define a coherent joint world model, and SCR supplies no regenerated fact state or fact-level explanation. Its contribution to the bid is a possible candidate-ranking component, with these limitations explicit.

The new comparisons are:

| Method | Role in the third investigation |
| --- | --- |
| SCR | Four-expert direct candidate predictor with shared global context. |
| CAND_CTX | One-expert candidate predictor with the same global context; tests the additional mixture, with parameter count also changing. |
| CAND_LOCAL | The same single-expert architecture with global inputs zeroed; tests global context access against CAND_CTX. |
| LOGIT | Shared 56-feature linear sigmoid fact predictor followed by marginal-product candidate scoring. |
| DAE | Small feature-denoising autoencoder, supervised fact decoder and marginal-product candidate scoring. |
| Earlier ID/JH and all original methods | Frozen mixed-data and original references, retaining the conventional and GraphMAE-adapted comparisons. |

DAE uses a 56→32→8 encoder and 8→32→56 decoder. Five pretraining epochs reconstruct the original observed feature vector after 25% masking and Gaussian noise with standard deviation 0.05; the target can itself contain manipulated evidence. A supervised 8→1 head then predicts fact readiness. LOGIT has no hidden layers or prior-logit shortcut. Both use the second study's fact-based objective. The three direct models instead use candidate feasibility BCE, candidate Brier loss and a soft decision-regret term. This difference in targets, together with pretraining differences, must accompany comparisons across model families.

Inference consumes only received observations, known prerequisites and candidate requirements. Candidate-specific value and cost tensors do not enter feasibility heads, but the inherited observed feature vector includes value-weighted candidate criticality. Training truth supplies supervision only. Direct-model fact Brier, repair rates and fact consistency are unavailable, not zero.

The frozen protocol uses exactly the second study's 20,000 mixed training observations and 2,000 validation observations, with five training seeds, batch 256, a maximum of 40 supervised epochs, patience 8 and AdamW learning rate 0.001. New models train on CUDA; the launcher selects the attached NVIDIA RTX 5000 Ada Generation and rejects CPU fallback. Earlier checkpoints remain unchanged. Fresh test seed **62027** contains 4,000 snapshots for each of 18 conditions, shared across all 22 methods. Pilot seed **72027** is separate; earlier test results are development evidence.

The primary endpoint is SCR minus ID in macro candidate-oracle regret over benign and targeted/random/coherent 20% intervention conditions. Secondary comparisons separate the mixture, global context and cheap reconstruction. Report Recall@3 with its eligibility definition, candidate Brier/calibration, parameter count, inference latency, bounded host-loop timing, memory and the unchanged report bytes. Paired uncertainty crosses training seeds and topology clusters; it does not treat all model evaluations as independent worlds. A shadow-union stress diagnostic probes whether longer conjunctions receive probabilities larger than their subsets. It is not used to tune on the fresh test pool.

Four rich empirical main figures and two additional supplemental figures will use only recorded results. The supplemental target remains within the requested 2–4 pages; the separate-upload allowance must still be checked against the conference's submission system. This investigation fits MVML through learned representations, prediction targets, calibration and measured computational comparisons. The original HDC/neurosymbolic baselines remain necessary for assessing the proposal's explanatory and distributed-reasoning tradeoffs. Static synthetic candidate ranking does not demonstrate a complete operational cyber planner.

Third-study record:

- [Results and interpretation](docs/RESULTS_V3.md), [four main figures](figures/v3/index.html), and [supplemental PDF](supplemental/v3/Supplement_v3.pdf).
- [Method and equations](docs/METHOD_V3.md), [configuration](configs/study_v3.json), [audit](docs/V3_AUDIT.md), and [artifact record](results/v3/README.md).

Reproduce from this project directory after the prior studies' frozen checkpoints are available:

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

Training and inference stay on CUDA. Statistical analysis, plotting, PDF rendering and file handling may use CPU. Keep raw arrays, logs and checkpoints under ignored `results/runs/v3/`, `results/checkpoints/v3/` and `work/`; commit concise summaries, provenance, source and selected final artifacts. The manifest and audit establish execution status, and the results report establishes measured conclusions.
