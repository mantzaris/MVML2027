# Automated scoring and reviewer interpretation

Initial rules were fixed before main development reports. The [constructed-signal correction](reference_correction.md) updates modulation estimation and AI references before controller fitting and test outcomes. The original
human-reviewed endpoint cannot be produced under the user's no-human design.
Use independent numerical estimates as the primary **proxy** reference, with
two blinded AI assessments and a conservative agreement sensitivity analysis.
This retains disagreement rather than declaring model consensus to be truth.
Both AI roles use the pinned Omni checkpoint, receive original anonymous audio
and independent diagnostic estimates, and see no dataset class, source identity,
generated report, proposed card, or selected tool. Shared model ancestry and
reference-algorithm anchoring limit the independence of these assessments.

The primary numerical reference uses full-window DPSS (NW=3, five tapers) band
powers and local peak contrast. A dominant band is identifiable when its power
exceeds the runner-up by at least 1 dB. Narrowband candidates require 12 dB
full-window contrast and a matching peak at 9 dB in at least three consecutive
non-overlapping one-second multitaper frames. Modulation uses squared-amplitude
low-pass envelopes and an autocorrelation peak >=.30 with prominence >=.08 and
a repeated-period correlation >=.15. Version 1.2 uses 1 kHz envelope sampling, interpolates the earliest qualifying autocorrelation peak, requires at least 1% carrier-passband energy, and rejects later-peak aliases; no qualifying peak is ambiguous, not a
verified absence. First/last-second received-energy change is compared with
3 dB; values within .5 dB of either boundary are ambiguous. Silence is ambiguous.
No estimator certifies physical source mechanisms or motion.

For physical assertions, correct means the reference identifies the same value;
tone tolerance is max(3 Hz, 3%), modulation max(1 Hz, 10%). A localized tone must
persist >=2.5 s and overlap the independent persistent span for >=80% of its
claimed interval. Modulation intervals must lie in the measured central 4.5 s.
Supported additionally requires every cited ID to be supplied for the query and
compatible with the slot and claimed value. Numeric tone/rate claims require the
corresponding A0/B0 measurement; D0 never supports physical attributes. W0 can
support categorical band/trend only when the independent estimate and both AI
reviewers agree. P0 supports class only. No citation receives credit solely for
existing. Incorrect numbers in a valid card can pass the internal card check
but still fail independent correctness, which is reported separately.

The primary loss and coverage formulas remain those in the draft. Invalid JSON
or malformed slots are failures, not confident predictions. A rationale length
violation is separately reported as format noncompliance; it does not erase an
otherwise parseable physical claim or convert it to uncertainty. Report JSON
parse, slot schema, and 60-100-word compliance separately. Prose is assessed in a
separate blinded AI audit, not silently folded into the structured endpoint.

The conservative AI-agreement sensitivity counts support only if the numeric
reference and both reviewers identify the same categorical attribute and match
the stated tone/rate tolerance. Unparseable reviews and disagreements withhold
consensus credit. Report raw agreement and validity. This is an automated
consistency sensitivity, never a measure of hypothetical-human performance.
