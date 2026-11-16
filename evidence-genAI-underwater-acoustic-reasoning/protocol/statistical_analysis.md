# Locked analysis rules, fixed before test outcomes

Select the fixed pair, retrieval-plus-DSP pair, and deterministic pair by minimum
mean development composite loss. The entropy comparator chooses a pair within
three development entropy bins; bin edges and choices are fixed thereafter.
The strongest B3/B9 comparator is the one with lower mean development loss.
Tune the claim-risk policy's beta/lambda on that same development loss, breaking
ties by fewer tools and then smaller beta/lambda. The class-only policy tunes
the same grid against class loss. Its terminal reports and rejection risk head
are shared with the other selectors. Ablations retain the proposed policy's
settings, changing only the declared component.

For every method, predict risk using the full claim-risk ensemble at the actual
acquired state. Set a separate threshold from the calibration scores at upper
order statistic ceil(.70 n). Keep ties and report actual test coverage; the
calibration set has 37 clips, so 70% is a target, not a distribution-free promise.
The raw-audio method uses the initial-state risk estimate; B8 uses the same head
even though its deterministic renderer differs. These are limitations of the
matched rejection procedure. All reports are generated before rejection.

Primary: correct-and-supported physical assertions divided by four times the
number of retained reports. Mandatory guardrail: wrong-or-unsupported assertions
divided by all definite assertions, with NA when none exist. Also report retained
reports/test reports, useful assertions/(4 total test reports), assertion rate,
factual mismatch on identifiable slots, format/schema validity, class accuracy,
five-class macro-F1 (abstention is an error), and actual tool/call counts.
Reference identifiability and uncertainty stay in the fixed four-slot denominator.

Paired bootstrap resamples the 93 recorded source groups with replacement 2,000
times (seed 17), keeping both methods and every selected clip from a sampled
source together. Fixed calibration thresholds are not re-fit on test. Recompute
each conditional metric's numerator and denominator within each resample;
different selectors may retain different reports. Use percentile 95% intervals.
The two-sided approximate bootstrap p-value uses the centered difference
distribution: (1 + count(|delta* - delta| >= |delta|))/(B + 1). Apply Holm to the
three declared primary contrasts: P versus the locked B3/B9 comparator, B7,
and A1. Report effect sizes even when p-values are not significant. These are
small-sample approximate inferences, not guarantees of physical independence.
Repeat source-group comparisons with UTC-day clusters as a dependence sensitivity.

Classifier metrics use all eligible test clips for each of seeds 17, 29, 43;
controller-seed variation is separate and reuses the same fixed greedy reports.
Only genuine temperature-scaled classifier scores receive multiclass Brier and
10-bin equal-frequency ECE. Risk/coverage curves ranked on test risk are explicitly
descriptive, separate from the calibration-locked operating point.
Report descriptive 50/70/90% ranked operating points and the area under the
composite-risk/coverage curve, defined as the mean cumulative risk across all
ranked prefixes (uniform increments from 1/n through 1). Citation support
precision uses assertions with at least one citation as its denominator;
also give the stricter numerator requiring independent correctness. These
diagnostics do not change controller targets or the confirmatory endpoint.
Single-method intervals at calibrated points and budget-sensitivity points use
the same source-cluster resampling and conditional numerator/denominator rules.
Risk-coverage curves remain descriptive; the displayed intervals apply only to
their calibrated dots, not to a simultaneous confidence band over the curve.

Success requires a >=.05 supported-coverage gain over the locked comparator,
a positive paired lower confidence bound, no higher wrong/unsupported rate,
class macro-F1 no more than .02 worse, and supportive B7/A1 contrasts. Otherwise
report a negative or inconclusive adaptive-selection result. All-tools and
deterministic baselines remain visible even if they outperform the learned policy.
Latency reconstructed from cached tools and batched generation is identified as
such, and never substituted for a measured online speedup.

Before test unlock, report a development-only precision diagnostic: aggregate
unfiltered supported-coverage differences within recorded source groups and
estimate normal-approximation 95% half-width and 80% detectable effect at the
planned 93 test groups. This optimistic calculation does not estimate power for
the calibrated conditional endpoint or the Holm family. Selection on development
and unmeasured dependence may invalidate its transfer. Keep the fixed assessment
size and report actual intervals; never expand it after seeing test outcomes.
