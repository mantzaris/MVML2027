# Prospective robustness and prose audits

Select 60 test assessment clips (12 per class) by seed-17 hash order, preserving
their original source clusters. For each, create Gaussian-noise variants at
20/10/0 dB waveform SNR, a fourth-order zero-phase 2 kHz low-pass variant, a .25
amplitude-gain variant, and a clipping variant formed by scaling to RMS .4 then
clipping to [-1,1]. Recompute native-rate references, both blinded AI assessments,
model features, priors, and all analytical cards after every transformation.
Reuse only the fitted classifier temperature/PCA/controller and frozen prompts.
Evaluate P, the locked fixed/entropy comparator, all tools, and the deterministic
renderer; no robustness outcomes select a policy or threshold. Reuse clean
calibration thresholds and report realized coverage. Label all stress analyses
exploratory and carry transformed copies in the original source cluster.

On 25 test assessment clips selected by the same pre-outcome hash rule, assess
four evidence interventions using the same audio/prior and a fixed A+B+C card
set: remove A, replace A with an anonymous other-clip card, replace a tone/band
number with a false value in an otherwise valid A card, and remove B. These
100 generator calls test missing and misleading evidence. Preserve both the
presented card and the clean reference. Do not equate internal card consistency
with independent correctness, and do not grant a swapped card the query's
provenance. Report abstention and erroneous assertion changes, not hidden repairs.

Prompt paraphrase: regenerate P's selected evidence on all 100 clean assessment
test clips with one fixed paraphrase of the user instructions, retaining the
same schema, token budget and decoder. Report paired change descriptively.

Prose audit: select 30 clean test clips (six per class) and four report methods
(P, locked comparator, all tools, deterministic), giving 120 reports. A separate
blinded AI audit call receives the anonymous audio, independent reference
diagnostics, and report text. It does not receive method identity, the other
reviewer outputs, source identities or dataset class. Evaluate unsupported
mechanical/causal claims, new unsupported physical claims, slot/rationale
inconsistency and overstated certainty. Retain failures as missing audits and
report their denominator. This shared-checkpoint audit cannot establish human
interpretability or unbiased self-evaluation.

These components are prioritized after essential clean comparisons and share
the hard ten-hour ceiling. If the deadline prevents a component, explicitly
report its missing scope and do not present it as completed. No synthetic-only
robustness result is substituted for an unrun real-audio transformation.
