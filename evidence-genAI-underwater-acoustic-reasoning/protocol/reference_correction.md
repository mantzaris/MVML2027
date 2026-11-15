# Development-phase reference correction — 15 November 2026

Code review and constructed amplitude-modulated signals exposed a bug in the
initial independent modulation proxy before controller fitting or test terminal
outcomes. At a 250 Hz envelope rate, finding peaks after slicing the allowed lag
interval discarded boundary peaks: a 2 Hz control was missed, while 70/80 Hz
controls selected approximately half-rate peaks. Exact original and corrected
values are retained in `reference_boundary_pilot.json`.

Intermediate version 1.1 used 1 kHz envelope sampling and quadratic interpolation.
Its known-signal pilot exposed a related failure: attenuated out-of-band pure
tones could create periodic residuals in the squared-amplitude envelope, and a
later autocorrelation peak could falsely label them as in-range modulation.
Both AI roles repeated the false modulation diagnostic for a pure 60 Hz tone.
This is direct evidence of correlated diagnostic anchoring, not independent
perceptual validation.

The final pre-test reference version is `independent-proxy-1.2`. It samples the
separately low-pass-filtered squared-amplitude envelope at 1 kHz, requires at
least 1% of total waveform energy in the carrier passband, locates peaks before
lag filtering, and considers only the earliest qualifying peak. A faster
out-of-range period is not relabeled using a later peak. Quadratic interpolation
uses a tiny endpoint acceptance interval of 1.99–80.1 Hz, clipping accepted
boundary estimates to 2–80 Hz. Original contrast, prominence, relative-envelope
variation and repeated-period criteria remain fixed.

Regression cases cover 2/6/30/70/80 Hz modulation and unmodulated 20/60/90/250/1500
Hz tones at both 16 and 32 kHz, plus an out-of-range 120 Hz modulation control.
These ideal positive controls estimate rate within 0.2 Hz; negative controls
remain unidentifiable. They do not establish accuracy on all ocean recordings.

Original and intermediate references, both AI roles, known-signal review outputs
and development gates are archived separately. All 270 main references and 540
AI assessments are recomputed with version 1.2, and the development gate is rerun.
The 20-signal reviewer control set includes 2/6/70/80 Hz modulation. Invalid
responses receive no agreement credit. AM-tone truth includes the carrier and
both sidebands; this scoring clarification reuses saved outputs without extra
AI calls. No repeated querying selects a preferred answer.

Terminal outputs never receive references or AI reviewer judgments, so completed
terminal caches remain valid. All controller targets and final scores use
version 1.2 consistently. The corrections use constructed-signal failures, not
test-model outcomes; no test terminal generation has been unlocked. Final freeze
verifies each AI assessment against its actual reference content hash.
