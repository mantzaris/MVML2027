# Second approach: dependency-aware full regeneration

This second investigation responds to the first study's observed weaknesses: fixed sparse repair left substantial decision regret, the learned gate transferred poorly away from its targeted-corruption training condition, and multiplying marginal beliefs ignored shared prerequisites. It is an adaptive follow-up: the old test results informed its design. A fresh held-out test seed is therefore required, with the old results retained as a historical development record.

The new method regenerates a conditional probability model over all facts, scores candidates using dependency-aware conjunction probabilities, and trains across multiple observation-corruption groups. It removes both the fixed six-fact gate and its perfect-truth single-fact teacher. Full regeneration is a different repair-budget choice, so comparison with the original sparse method alone cannot identify the benefit of the new probability model.

## Conditional regeneration model

The existing benchmark has one-level root–child prerequisite forests. Let \(p(f)\) denote a fact's parent, with \(p(r)=r\) for a root. A neural encoder predicts one parameter \(a_f\in(0,1)\) per fact. At roots this parameter is \(P(Y_r=1\mid x)\); at children it is \(P(Y_f=1\mid Y_{p(f)}=1,x)\). A child is deterministically false when its parent is false. Roots are independent under this conditional model, and children are conditionally independent given their roots and the observed feature vector \(x\).

The corresponding marginal probabilities are

\[
 b_r=a_r,\qquad b_f=a_f a_{p(f)}\quad(f\ne p(f)).
\]

Thus \(b_f\le b_{p(f)}\), and every world supported by the conditional model satisfies the supplied child-implies-parent rules. This is exact coherence for the benchmark's forest schema. It does not establish that the true posterior factorizes: correlated deception can leave dependencies between roots or siblings even after conditioning on observations.

Let \(R_j\) be candidate \(j\)'s required facts, and \(A_j=\{p(f):f\in R_j\}\) the distinct roots touched by those requirements. The second approach's proposed task-specific regeneration and scoring construction is

\[
\boxed{
 S_j^{\rm dep}(x)
 =\prod_{r\in A_j}a_r(x)
  \prod_{\substack{f\in R_j\\f\ne p(f)}}a_f(x),
 \qquad
 \widehat U_j=v_j S_j^{\rm dep}-c_j.
}
\tag{5}
\]

Equation (5) counts shared prerequisite uncertainty once. In contrast, the original marginal product repeats a root's probability for every required child. If a root has probability 0.6 and two child conditional probabilities are 0.8 and 0.7, the joint candidate probability is \(0.6\times0.8\times0.7=0.336\). Multiplication of the child marginals gives \((0.6\times0.8)(0.6\times0.7)=0.2016\). Explicitly requiring that same root does not reduce the dependency-aware probability again.

Bayesian factorization and exact probabilistic logic inference are established ideas. The research extension is their specific coupling to hyperdimensional evidence retrieval, full belief regeneration, a common decision objective, and controlled corruption-group evaluation. Equation (5) is not presented as the first discovery of conditional probability or shared-event deduplication. Its scientific value must be established by the matched comparisons below.

## Implementation and common objective

`src/regen_v2.py` inherits the first study's encoder and masked-feature pretraining implementation without changing `src/regen.py`. Dense and HDC encoders retain the same width, depth, pretraining decoder, and parameter accounting as their original counterparts. The original gate head remains in the architecture for accounting consistency but is unused by second-study inference and supervised loss.

The full decoder predicts residual logits. For independent models, the residual starts at the common provisional marginal belief. For conditional models, the initial child factor is the provisional child belief divided by its parent belief, clipped to [0.02, 0.98]; roots retain their provisional probabilities. Consequently, a child initially judged more likely than its parent is reduced to a coherent marginal. Conditional initialization is an explicit inductive-bias difference; it is not an exact copy of every original marginal.

Every independently trained second-study variant receives the same per-example loss:

\[
\ell_i=
\operatorname{BCE}_{\rm facts}(b_i,y_i)
+\operatorname{Brier}_{\rm candidates}(S_i,s_i^*)
+0.25\,\mathcal R_{0.1}(S_i;U_i^*)
+0.1\,\mathcal L_{\rm cons}(b_i).
\tag{6}
\]

Here \(s_i^*\) is hard candidate feasibility and the regret term uses softmax weights from predicted candidate utilities. Hidden truth supplies training targets and evaluation only. Neither a structured negative-log-likelihood objective nor an oracle repair teacher is added to JH/JD, avoiding an extra objective difference relative to the independent controls.

Every second-study variant uses the same consistency penalty: the mean squared positive violation of child marginal less than or equal to parent marginal. It is zero for coherent marginals, including uncertain conditional posteriors. This avoids incorrectly treating an independent-product implication penalty as the structured model's violation probability. The penalty is a consistency regularizer, not a probability of logical violation. Candidate scoring uses the model's joint semantics unless the explicit product ablation is selected.

Training assigns each of the same 20,000 hidden worlds to one observed condition: benign or one of three attack mechanisms, with equal group counts. Attack severities cycle through 0.1, 0.2 and 0.4. Group labels supervise loss aggregation and never enter model input. Validation uses the same balanced construction on 2,000 independent worlds. All variants select checkpoints using the mean of the four validation group losses, with the same epoch and patience budget.

For group-average losses $L_g$, the new robust variants minimize the entropy-regularized group risk

$$
\mathcal L_{\rm group}=\frac{1}{\beta}\log\left(\frac{1}{G}\sum_{g=1}^{G}e^{\beta L_g}\right)
=\max_{w\in\Delta_G}\left[\sum_gw_g L_g-\frac{1}{\beta}D_{\rm KL}(w\Vert\operatorname{Uniform}(G))\right],\qquad G=4,\ \beta=2.
\tag{7}
$$

This established variational identity explains the weighting: higher-loss groups receive more weight, constrained by a penalty for departing from uniform group weights. It is a smooth emphasis on the four modeled groups, not an unconstrained worst-case adversary or a guarantee for unseen mechanisms. JH_ERM replaces Equation (7) with the ordinary group mean. In minibatches the means are estimated from the available members of each group.

The [frozen configuration](../configs/study_v2.json) was committed before evaluation on new test seed 42027. The primary endpoint averages regret equally over benign0 and targeted/random/coherent0.2; it compares JH with full-reconstruction ID. Tests reuse the same worlds across compatible conditions and use five independent training seeds. The old test seed 32027 is explicitly development evidence for this redesign.

## Comparisons that separate the changes

| ID | Representation | Probability model | Training aggregation |
| --- | --- | --- | --- |
| JH | Fixed-codebook HDC features plus graph encoder | Conditional forest; dependency-aware candidate score | Robust group risk |
| JD | Dense graph encoder | Same conditional forest and candidate score | Same robust group risk |
| IH | Same HDC features and graph encoder | Independent fact marginals and their product | Same robust group risk |
| ID | Dense graph encoder | Independent fact marginals and their product | Same robust group risk |
| JH_ERM | Same architecture and inference as JH | Same conditional forest | Mean risk |
| JH product override | Frozen JH checkpoint | Same coherent marginals, but candidate probability is their product | No additional training |

JH versus JD tests the HDC feature contribution. JH versus IH tests the conditional-model package at fixed HDC representation and loss weights. The product override isolates candidate scoring without changing learned beliefs. JH versus JH_ERM tests robust aggregation. ID provides a full-regeneration dense baseline trained on the same mixed observations; comparison only with old targeted-only models would confound method improvements with broader training exposure.

All variants may change every fact. They do not claim sparse-decoder computation savings or preservation of a fixed six-fact budget. The wire protocol remains the common sparse report codec, so any HDC communication-compression claim is still untested. The benchmark remains a synthetic decision snapshot rather than a full sequential cyber-operation planner.

## CUDA checks and information boundary

The seven second-study correctness tests passed on the attached NVIDIA RTX 5000 Ada GPU. They enumerate tiny valid worlds to check normalization and exact candidate probabilities; verify shared-root deduplication; verify coherent marginals, empty conjunctions, hard 0/1 semantics, and zero consistency penalty for valid uncertain posteriors; check finite nonzero gradients for all five trainable variants; mutate hidden labels, previous truth, attack masks, and transitions without changing inference; and confirm that the product override retains exactly the same marginals.

The public inference API is `infer(model, batch, product_override=False)`, returning marginal beliefs and candidate probabilities. It reads only observable features, adjacency, parent relations, and candidate requirements. Truth enters `loss_terms` or external evaluation after inference. The new source enforces CUDA tensors and has no CPU research fallback.

## Primary-source lineage

- Manhaeve et al. (2018), [DeepProbLog: Neural Probabilistic Logic Programming](https://proceedings.neurips.cc/paper/2018/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html), provides a primary precedent for integrating neural predictions with probabilistic symbolic semantics. This project implements a narrow forest-specific computation, not DeepProbLog itself.
- Ahmed, Chang, and Van den Broeck (2023), [Semantic Strengthening of Neuro-Symbolic Learning](https://proceedings.mlr.press/v206/ahmed23a.html), studies the loss of dependency semantics in approximated neuro-symbolic objectives. The second study uses known forest dependencies rather than learning which constraints to strengthen.
- Sagawa et al. (2020), [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731), motivates robustness across predefined groups and emphasizes regularization and validation. Its results do not guarantee robustness to unseen attacks here; mean versus robust aggregation is therefore controlled empirically.
- GraphMAE, decision-focused learning, and the original semantic-loss lineage remain documented in [REFERENCES.md](REFERENCES.md).
