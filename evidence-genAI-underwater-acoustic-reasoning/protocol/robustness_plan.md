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

Final scope check before test unlock: add three controls to match the execution
brief. On the same 25 clips, cyclically change every retrieved class label to
the next of the five labels in the all-tools report (125 interventions in total
so far). Separately, remove a cited and an uncited card from all-tools reports:
eligible clips must contain a cited/uncited A-D family respectively. Choose up
to 25 clips per condition in seed-17 hash order and remove the first eligible
family in A-D order. Report eligible/selected counts; if fewer than 25 qualify,
use all eligible clips without replacement. Compare each intervention with its
matching unmodified subset report. This yields at most 175 intervention calls.
These fixed-evidence interventions do not rerun the acquisition policy and
therefore make no claim about post-corruption action changes.

Prompt paraphrases: regenerate P's selected evidence on all 100 clean assessment
test clips with two fixed paraphrases of the user instructions (200 calls),
retaining the same schema, token budget and decoder. Both prompt files and this
scope extension are included in the prospective test freeze. Report paired
changes separately and descriptively; do not select a preferred paraphrase.

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
