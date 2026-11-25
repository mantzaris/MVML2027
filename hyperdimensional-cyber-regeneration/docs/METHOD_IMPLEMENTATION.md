# Executed method and scope

This document maps the frozen CUDA experiment to the research plan. The contribution investigated is **decision-impact supervision for selecting belief repairs** (plan Equation 3), building on masked graph reconstruction [R1], decision-focused learning [R2], and local semantic losses [R3]. Novelty and performance advantages remain hypotheses. The implementation is a controlled synthetic decision study, not an operational cyber planner.

## Data and shared decision problem

Each example contains 64 Boolean facts, three partially observing sources, a previous-state noisy observation, and 32 fixed candidate prerequisite sets. Facts are not hosts. Each candidate requires 2–5 facts and has a known value and cost sampled independently of hidden state. Candidate utility is

\[
\widehat U_j(b)=v_j\prod_f[(1-r_{jf})+r_{jf}b_f]-c_j.
\]

On binary simulator truth this is exact deterministic terminal utility. On marginal beliefs the product is a relaxed conjunction, not a calibrated joint probability or a sequential rollout. Every method receives the same candidates, observations, and graph.

Graphs are collections of root–child prerequisites, enforcing child \(\Rightarrow\) parent in both previous and current truth. Up to 16 snapshots share a topology. Separate generator seeds create training, validation, and test splits; generation samples worlds before condition-dependent corruption, preserving paired worlds across compatible attack and bandwidth settings. Training/validation/test sizes are 20,000/2,000/4,000 snapshots. Extrapolation changes both fact count (128/256) and root density, so it is not an isolated size-only experiment.

Sensors have 4% natural flip noise and 20% missingness. Genuine latent transitions occur with probability 6% before semantic projection. Attacks intervene on a fraction of available reports: random flips, criticality-targeted flips, or correlated all-ready replacements. Coherent replacements may leave already-positive reports unchanged; summaries report both intervention and changed-report fractions. Equal intervention budgets therefore do not imply equal corruption severity.

Two remote sources send freshest-first sparse records; source zero is local. A record is 12 bytes and a nonempty source packet has a 16-byte header. The serialization test checks actual payload lengths. All methods share this codec; HDC is internal representation, not a measured communication-compression advantage.

## Representation and masked pretraining

The input has 14 channels: three report values, three availability indicators, three ages, noisy history, provisional current belief, normalized degree, candidate criticality, and current/history disagreement. Provisional beliefs combine reports with age weights \(1.5-0.5a\) and history weight 0.45.

For P, fixed bipolar fact keys of dimension 4,096 bundle signed reports into three source-specific memories. Dot-product retrieval supplies three additional features. Keys are reproducible and shared across methods, seeds, and size prefixes. This is associative retrieval; it is not resonator factorization, role-binding relation composition, or NeuralHD dimension regeneration [R4–R5]. Original features remain available, so this is HDC-augmented graph learning rather than a purely hyperdimensional network.

Both dense and HDC variants use width 128 and three residual graph layers, with row-normalized neighbor aggregation, GELU, and LayerNorm. Separate heads predict belief residual logits and gate logits. The belief head starts at zero, making its initial prediction the provisional belief.

Dense/HDC models contain 119,966/120,350 trainable parameters (479,864/481,400 bytes as FP32). Of these, 101,634/102,018 participate in belief-and-gate inference; the remainder belong to the pretraining mask token and feature decoder. These counts exclude fixed codebooks, resident observations/graphs/candidates, intermediate activations, and optimizer state. HDC therefore does not have exactly matched parameter or retained-state costs. The current codebook cache retains a 512-by-4,096 FP32 allocation (8 MiB) for each requested fact-count key; distinct size keys remain cached during the process.

Five pretraining epochs mask 30% of fact feature vectors, encode them using a learned mask token, remask corresponding latent vectors, and decode through neighbor aggregation. Scaled cosine feature error \((1-\cos)^2\) is computed only at masked nodes. This preserves the feature-reconstruction principle of GraphMAE [R1], but uses a project-specific architecture and downstream task: **GraphMAE-adapted**, not an unchanged upstream reproduction.

## Equations 1–4 in executable form

Equation 1 interpolates the provisional belief and reconstruction using a binary gate. At 64 facts, the standard budget is six repairs. Equation 2 weights true candidate utility gaps by a softmax of predicted utilities, with \(\tau=0.1\).

Equation 3 measures each single-fact truth substitution's nonnegative reduction in that regret. The positive top-six deltas form a frozen teacher mask; stable sorting resolves ties, and zero gains are excluded. GPU evaluation exploits the conjunction ratio \(y_f/b_f\) for candidates requiring fact \(f\), exactly matching explicit substitution. This teacher uses privileged **training truth only**. It ignores repair interactions and does not establish that a fact can be reconstructed from available observations.

Equation 4 adds reconstruction BCE, regret, gate BCE, semantic loss, and preservation loss with weights \(1,1,1,0.1,0.1\). The local semantic term is

\[
-\operatorname{mean}_{f\ne p(f)}
\log[1-b_f(1-b_{p(f)})].
\]

This is exact satisfaction probability for each binary implication under product marginals, not global logical inference. Preservation applies only to unchanged true facts whose initial belief is within 0.15 of truth. Its mask is training-only.

Training uses hard top-K forward decisions with the biased straight-through gradient \(g_{\rm hard}+q-\operatorname{stopgrad}(q)\). Inference additionally removes gates below a validation-selected probability threshold from \(\{0,0.1,0.3,0.5\}\). Model inference accepts features and adjacency only; neither evaluation truth nor injected-corruption labels enters its input.

## Comparisons and optimization

| Variant | Executed comparison |
| --- | --- |
| B0 | Age-weighted provisional belief, then child/parent minimum. |
| B1 | Dense GraphMAE-adapted pretraining and supervised reconstruction BCE; full reconstruction. |
| B2 | Dense full reconstruction with the decision, gate, semantic, and preservation objectives. |
| B3 | Fixed HDC associative cleanup, without learned repair. |
| P | HDC-augmented model, Equation 3 gate, full objective. |
| B4 / B5 / random gate | P checkpoint evaluated with full reconstruction, entropy selection, or random selection. |
| Dense gate | Separately trained dense model with P's gate and objective. |
| No semantic | Separately trained P without the semantic loss. |
| Direct source/age ablation | Separately trained P with source distinctions averaged and ages zeroed. |

B4/B5/random are inference interventions, not separately retrained models. The source/age ablation retains the common age-weighted provisional belief; it is not complete removal of provenance.

Six learned models are trained for each of five seeds, using AdamW, learning rate 0.001, weight decay \(10^{-4}\), batch 256, gradient clipping at 5, and up to 40 supervised epochs. Validation BCE plus regret selects checkpoints, with patience eight. Graph training uses BF16 autocast; HDC retrieval and losses use FP32. CUDA is mandatory; CPU is reserved for orchestration and artifact handling.

## Limits and reporting

The executable protocol supersedes the plan's preliminary temporal episodes, batch/epoch counts, and tuning search. It has no sequential recovery-delay estimate, external CybORG validation, pair-repair interaction study, separate link-loss/burst sweep, or explicit multi-hypothesis inference. Top-three feasibility is conditional on at least one feasible candidate.

The full decoder executes before selection, so sparse repair does not establish compute savings. The main sweep reports warmed batch-one CUDA inference and candidate scoring, excluding serialization and host transfers. A separate post-training host-wall audit (`results/pipeline_timing.json`) measures real packet packing/unpacking, host/device transfers, CUDA feature reconstruction, model inference, ranking, and returning candidate IDs. It uses five warmups and 50 measurements on the first reference snapshot, with seed-37 checkpoints; all 11 methods reproduce original features and predictions within absolute tolerance `1e-6` after the 1,340-byte packet round trip. This is a bounded prepared-evidence-to-ranking measurement: sensing, bandwidth admission, graph/catalogue construction, network transport delay, and cold loading remain excluded. It does not establish full operational latency. Incremental peak VRAM excludes resident data and model memory. Threshold metadata omitted by the initial runner is reconstructed from frozen checkpoints using the identical validation-only procedure; test outcomes do not select thresholds.

Sources [R1–R5] and their primary links are catalogued in [REFERENCES.md](REFERENCES.md). Exact configuration and recorded limitations accompany the run in `configs/executed_study.json` and `results/study_manifest.json`.
