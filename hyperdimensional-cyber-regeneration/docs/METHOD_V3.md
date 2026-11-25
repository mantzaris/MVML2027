# Third investigation: shared-context candidate ranking and cheap reconstruction

The first two studies rebuild fact beliefs before ranking candidate cyber courses of action. This third study asks whether predicting the candidate's feasibility directly is more accurate or cheaper. It evaluates a genuinely different prediction target and model family, retaining symbolic prerequisite closure but removing graph message passing and hyperdimensional retrieval. It also tests a linear fact predictor and a small pretrained denoising autoencoder. These are GPU implementations, including the inexpensive models.

## Proposed methodological composition

Let observed fact features be $x_f\in\mathbb R^{14}$, with known parent $p(f)$ in the benchmark's one-level prerequisite forest. Define $h_f=\phi_\theta([x_f,x_{p(f)}])\in\mathbb R^{32}$ and global context $g=[\operatorname{mean}_f h_f,\operatorname{std}_f h_f]$. A candidate requirement set $R_j$ has required nonroot children $C_j=\{f\in R_j:p(f)\ne f\}$ and distinct touched roots $A_j=\{p(f):f\in R_j\}$. The canonical candidate representation is

$$u_j=[\operatorname{mean}_{C_j}h,\operatorname{std}_{C_j}h,\operatorname{mean}_{A_j}h,\log(1+|C_j|),\log(1+|A_j|)].\tag{V3.1}$$

Empty pools contribute zeros. Moments use population normalization and numerical variance stabilization as implemented in `src/regen_v3.py`. With received x held fixed, adding a root already implied by a required child leaves this representation unchanged. Rebuilding catalogue-derived criticality is a separate input change. This symbolic canonicalization is valid for the supplied one-level implication forest, not arbitrary prerequisites.

The proposed **SCR (shared-context candidate ranking)** probability and ranking equation is

$$\widehat s_j=\sum_{k=1}^{4}\underbrace{\operatorname{softmax}_k(a_\theta(g))}_{\pi_k(g)}\,\sigma\!\left(\rho_{\theta,k}([u_j,g])\right),\qquad \widehat U_j=v_j\widehat s_j-c_j.\tag{V3.2}$$

This composes prerequisite-aware set pooling with a shared latent mixture gate and decision-focused supervision. Shared weights allow all candidates to respond to the same observational context. The experts are not supervised attack labels, are exchangeable across seeds, and need not specialize or avoid collapse. Their mixture is a predictor of each candidate's marginal success probability; it does **not** define a coherent joint distribution over all candidates. It provides no regenerated fact beliefs or fact-level explanations.

The architecture uses a 28→32→32 fact encoder, a 162→64→4 expert head, and a 64→32→4 context gate. **CAND_CTX** has one expert with the same global context. **CAND_LOCAL** has the same 162→64→1 head but zeros its 64 global inputs. Identical random seeds yield identical initialization for those two controls; their parameter counts include the inactive global input columns in LOCAL. No candidate-specific value or cost tensor enters the feasibility heads. The inherited shared feature vector does contain normalized value-weighted criticality (channel 12), which also influences where the synthetic adversary intervenes. Thus the model is not completely independent of catalogue values; the invariance test holds x fixed when changing value/cost tensors. Identical requirements within a fixed catalogue receive identical probabilities. Values and costs also enter utility and the training decision term. Candidate order is equivariant; fact renumbering and redundant ancestors are invariant up to floating-point tolerance.

For each training snapshot with evaluator feasibility $z_j=\prod_{f\in R_j}y_f$, SCR and its controls minimize

$$\ell=\operatorname{mean}_j\operatorname{BCE}(\widehat s_j,z_j)+\operatorname{mean}_j(\widehat s_j-z_j)^2+0.25\left[U^*-\sum_j\operatorname{softmax}(\widehat U/0.1)_jU_j\right].\tag{V3.3}$$

The four observed training groups receive the same smooth maximum of group losses as study 2, with beta 2. Group labels are training-only. Early stopping uses the mean of validation group means. The labels are synthetic training supervision; test truth, corruption masks and observation groups are never prediction inputs.

## Inexpensive ML controls

Both controls receive 56 observable features per fact: its 14 features, its parent's 14 features, and the global mean and population standard deviation of the 14 raw features. **LOGIT** is a shared 56→1 linear sigmoid fact predictor. It has no hidden layer or residual prior shortcut. **DAE** is a 56→32→8 encoder, 8→32→56 decoder, and 8→1 sigmoid belief head. Five pretraining epochs reconstruct original *observed* features after independent 25% masking and Gaussian noise with standard deviation 0.05. Pretraining therefore does not recover an oracle clean state and can learn features of manipulated evidence. Supervised fine-tuning uses the same fact BCE + candidate Brier + 0.25 soft regret + 0.1 squared marginal-consistency penalty as study 2. Candidate scoring multiplies predicted fact marginals.

DAE is a task-adapted denoising autoencoder control, not an exact reproduction of Vincent et al. (2008). Its decoder is only active during pretraining. Retained parameter counts include that decoder; training cost and inference latency are measured separately. Without a no-pretraining DAE control, its comparison does not isolate the causal benefit of pretraining.

## Frozen comparison and shortcomings

The primary comparison is SCR minus the prior dense independent mixed-data model ID, for macro candidate-oracle regret over benign and targeted/random/coherent 20% intervention conditions. This compares complete prediction pipelines: architecture, targets and pretraining differ. SCR versus CAND_CTX is the secondary, closer comparison of the shared four-expert mixture. CAND_CTX versus CAND_LOCAL diagnoses access to global context. Cheap controls test whether extra complexity is worthwhile; superiority is not assumed.

Use five seeds, 20,000 shared mixed training observations, 2,000 shared validation observations, max 40 supervised epochs, patience 8, batch 256 and AdamW lr 0.001. Prior checkpoints remain frozen. All 22 methods are evaluated on fresh generator seed **62027**, 4,000 snapshots per each of 18 conditions. Prior test seeds 32027/42027 and pilot52027 informed development; v3 pilot72027 is separate. The 4 core conditions share worlds, candidates and 250 topology clusters. Confidence intervals cross training seeds and paired topology clusters, with 2,000 bootstrap draws; they do not quantify uncertainty about the synthetic environment's design.

Direct models lose complete state regeneration, while gaining freedom from factorized probability assumptions. A prespecified diagnostic adds unions of adjacent candidate prerequisite sets and measures $\widehat s(R_j\cup R_{j+1})>\min(\widehat s_j,\widehat s_{j+1})+10^{-4}$. Union sets can contain up to 10 facts versus training sets of 2–5, so this is a structural stress test. No diagnostic is used to tune the fresh-test models. Fact/repair metrics are unavailable for direct models, never assigned artificial zeros.

All methods share the same sparse report codec. HDC remains an internal representation of earlier models; no communication compression gain is claimed. The benchmark is a static synthetic ranking/replanning proxy, not an executed multi-step cyber campaign. This investigation fits an ML paper through representation, supervised learning, calibration and computational comparisons. The bid's compact distributed reasoning and explanatory requirements remain only partially addressed by SCR; preserving earlier neurosymbolic/HDC baselines makes those tradeoffs visible.

## Prior approaches and novelty scope

- Zaheer et al., **Deep Sets**, NeurIPS 2017. [Primary paper](https://papers.neurips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf). Permutation-invariant set aggregation is inherited.
- Jacobs, Jordan, Nowlan and Hinton, **Adaptive Mixtures of Local Experts**, Neural Computation 1991. [Primary paper](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf). Input-dependent mixture gating is inherited.
- Vincent, Larochelle, Bengio and Manzagol, **Extracting and Composing Robust Features with Denoising Autoencoders**, ICML 2008. [Primary paper](https://icml.cc/2008/papers/592.pdf). Corrupt-and-reconstruct pretraining is inherited.
- Earlier project [references](REFERENCES.md) document GraphMAE, decision-focused learning, semantic losses and hyperdimensional representations.

Equations V3.1–V3.3 describe a proposed task-specific combination and investigation, not a claim to have invented sets, mixtures, autoencoders or decision-focused learning. A comprehensive novelty review and validation in a richer external benchmark remain necessary before submission.
