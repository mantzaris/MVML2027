# Third-study evidence audit

This document is generated from the completed third-study manifest and validated raw inputs. It records checks actually performed by `scripts/analyze_v3.py`; model-behavior tests, timing and replay are separate evidence when available.

## Completed statistical and provenance checks

- Complete catalogue: 1,980 unique records, 22 methods, five seeds and eighteen conditions per method/seed.
- Confirmatory pool: generator seed 62027, 4,000 snapshots per condition, with the prescribed 20,000 training/2,000 validation counts and optimization settings. Prior test and pilot pools are disclosed as development evidence.
- Matched data: the prior v2 manifest hash matches, and complete train/validation observation-plus-truth/candidate hashes equal the v2 mixture. New-model pretraining epochs, best/stopping epochs and recorded training durations satisfy the protocol.
- Executed source identity: 7 recorded source hashes match the current files; source commit `daa9e0f82be0725376f1923a697c1bbd0e96dba6` is recorded.
- Weights: all 25 new checkpoint hashes and 55 recorded reused-checkpoint hashes match disk. Recorded original thresholds are checked when present.
- Candidate-only output scope: all 270 SCR/CAND_CTX/CAND_LOCAL rows keep fact Brier, false-repair, false-certainty, coverage, preservation, marginal logical violations and changed-fact rates unavailable.
- Optimization-budget audit: 25/25 new models reached all forty supervised epochs; selected checkpoints span zero-based epochs 36–39. None stopped early. The ranking is a fixed-budget result, not evidence that every architecture converged.
- Communication: reported mean bytes are nonnegative and do not exceed the two remote-observer budgets.
- Paired raw inputs: 180 archives have SHA-256 records; topology groups, evaluator candidate truth and eligibility match across methods, seeds and four conditions.
- Raw candidate probabilities are finite and lie in [0,1]; evaluator labels are Boolean; per-snapshot candidate Brier is recomputed and checked.
- Raw/summary agreement: 540 comparisons of regret, candidate Brier and eligible Recall@3 agree within floating-point tolerance.
- Uncertainty: exactly 2,000 paired crossed training-seed/topology bootstrap draws; 250 clusters of sixteen snapshots; fixed equal condition weights; seed and cluster draws shared across paired methods/conditions.

## Primary evidence

SCR−ID macro regret = +0.02508099; 95% percentile interval [+0.02091362, +0.02919276]. The primary outcome and comparator were fixed before the fresh test. Secondary intervals are descriptive and not multiplicity-adjusted.

## Separately recorded artifacts

- GPU replay verified 48,000 model-snapshot evaluations: SCR, DAE and LOGIT at seed 37 across all four core conditions. Candidate probabilities, regret and candidate Brier have maximum absolute error 0. All twelve replay input/checkpoint hashes, manifest identity and replay-script hash match the recorded files.
- Bounded prepared-evidence timing verified seven methods, fifty repetitions after five warmups, seed 37/snapshot 0 and a 1244-byte payload. Source/checkpoint/manifest hashes and timing quantiles match; the timing script records successful probability and available-marginal roundtrip assertions. This is a fixed-snapshot wall-time audit, not full operational latency.
- `results/v3/replay.json` exists with SHA-256 `101dcb91f145a955766cc1eb2528832f5202ba46879ac557f28d5c33c3379872`. The validation scope is stated above and in that artifact.
- `results/v3/pipeline_timing.json` exists with SHA-256 `26d6b42c784b41b5fd76ea1b0ec09b7ac056481b5490bf407bc39a6433ffce38`. The validation scope is stated above and in that artifact.

## Interpretation limits

The audit establishes consistency of the recorded synthetic experiment, not operational cyber effectiveness or first-ever methodological novelty. Candidate models supply no fact beliefs. A latent expert is not an identified attack class. New feasibility heads exclude candidate-specific value/cost tensors, but inherited x[12] retains a known value-weighted criticality summary; tensor-blinding tests with x fixed do not imply complete value independence. Full-versus-sparse output scope and objective/supervision differences remain material in cross-family comparisons. Fresh-test isolation is the prospective study procedure recorded by the protocol and source commit; file hashes alone cannot prove that no human viewed a test pool early.
