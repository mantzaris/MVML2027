# Automated reference estimates and AI reviewers

The user's instruction on 2026-11-15 replaces the original two-human-reviewer design. **No human reviewers are used.** All future reviewer outputs are AI-generated assessments, not human annotations or independently verified perceptual truth. “Hypothetical human” means a simulated reviewer role; do not invent people, credentials, participation, agreement rates, or human validation.

## Reference pipeline

1. After source-integrity approval, sample source-diverse records using the protocol's partition rules. Class labels and source identifiers remain in a separate evaluation manifest.
2. Compute reference views using `annotation.py`: DPSS multitaper PSD and rectified-envelope autocorrelation. These are separate from the agent's Welch/Hilbert implementations but still fallible algorithms.
3. Run two blinded AI review roles on anonymous original audio and these reference views/numeric estimates. Role A assesses measurable attributes; role B challenges over-specific or unidentifiable claims. Neither receives dataset class, MMSI, date, deployment, generator outputs, proposed tool cards, or policy choices.
4. Retain both raw outputs, model/revision/precision, prompt hashes, token counts, runtime, and disagreements. Prefer different model families if a pilot fits the budget; two prompts on the same model are correlated assessments, not independent reviewers. The exact reviewer checkpoint(s) must be fixed before evaluation.
5. Disagreement yields uncertainty unless a predefined numeric rule resolves it. Do not repeatedly query until agreement or select whichever judgment helps the proposed method. Report AI agreement descriptively, with explicit model provenance.
6. Compute structured numerical consistency against separate reference estimators. AI prose audits are secondary: assess at least 120 reports, blinded to method identity, and score unsupported causal/mechanical assertions. Use known-signal fixtures to characterize estimator/reviewer errors before real-data scoring.

## Required fields

For each of dominant band, persistent narrowband, envelope modulation, and energy trend: categorical status; uncertainty/identifiability; inspected time interval; frequency/rate interval and units when relevant; tolerance; concise ambiguity note. Preserve ambiguous and failed assessments. Never infer ship identity from these fields alone.

Expected tone persistence is at least half the clip; initial frequency tolerance max(3 Hz, 3%), modulation tolerance max(1 Hz, 10%). Modulation is not shaft rate or blade count. Relative digital amplitude is not calibrated sound-pressure level. Received energy change is not approach, distance, or acceleration.

Recompute references and AI assessments after waveform transformations. Clean reference labels cannot validate an attribute deliberately removed by a corruption. Keep all variants in the original source cluster. Include AI review costs within the **10-hour GPU ceiling**; reduce optional scope before extending resource use.

## Local inspection tooling

`create_annotation_pack.py` and `review.html` remain optional inspection/debugging tools. Their availability does not imply a human study. They render anonymous audio and reference plots, keep browser drafts keyed by reviewer ID, and export incomplete/complete records without inventing missing fields. The identity key stays outside the served directory. No external hosting or messages are required.

The standalone synthetic demo is not an actual annotation exercise. Formal pack creation currently refuses missing source metadata or an infeasible sampling request. Reviewer-model inference and final semantic scoring are not yet implemented or executed.

## Reporting limits

Use “automated reference estimate,” “AI reviewer assessment,” “proxy supported coverage,” and “cross-estimator consistency.” Do not use “human validated,” “expert agreement,” or “independent perceptual ground truth.” Separate synthetic tests, real-data numerical proxies, AI prose assessment, and class-label accuracy. A shared model between generator and reviewer creates a self-evaluation limitation that must be reported.
