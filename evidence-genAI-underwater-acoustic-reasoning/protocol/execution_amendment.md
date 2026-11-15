# Prospective execution amendment — 15 November 2026

This amendment is fixed before classifier or generator outcomes. It supersedes
infeasible portions of the draft, while retaining the original scientific
question. The user requested completion under a ten-hour ceiling with AI
reviewers, no human participants, a small Git repository, and the existing Pod
left running. There is no external preregistration.

## Data and provenance

Use UniqueShip v1.0 fold 1 and the five declared classes. Normalize MMSIs stored
as integer-valued decimal strings. Exclude 42 metadata/filename date mismatches
and five non-five-second recordings rather than infer their provenance. Split
official training vessels/background days 75:25, and validation groups 50:50,
using the declared deterministic seed-17 hash ordering. Whole UTC days across
all classes and hydrophones belong to only one partition, with priority test,
calibration, development, policy-fit, acoustic-fit. Enforce a global 60-second
guard between five-second clips across partitions. Cap non-test source counts
at 100. Verify decoded waveform hashes before fitting. This loses many training
clips deliberately. It proves the recorded vessel/day/time/hash separations;
it does **not** prove unavailable parent-recording or deployment independence.
Missing parent identities remain missing and are a reported limitation.

The resulting counts are 2,459 acoustic-fit, 1,827 policy-fit, 384 development,
508 calibration, and 2,240 test clips. Assessment subsets contain 100 policy-fit,
33 development, 37 calibration, and 100 test clips. Use at most three clips per
source, at least ten minutes apart. Class shortfalls in development/calibration
remain shortfalls. The 100 assessment test clips include 93 independent recorded
vessel/background-day groups (13 passenger vessels); inference is clustered by
these recorded groups, with whole-day sensitivity where feasible. Full test
classifier performance uses all 2,240 eligible clips.

## Automated references and claims

Separate signal-processing implementations supply physical reference proxies;
two blinded AI reviewer roles assess uncertainty and evidence consistency. No
human ground truth or hypothetical-human performance is claimed. Correlated AI
errors and thresholds defining acoustic identifiability are explicit limitations.
Known-signal controls and development-only checks precede final prompt freeze.

## Budget and order

Use the existing 32 GB GPU. Conservatively start the ten-hour wall-clock allowance
at the recorded authorization time, 2026-11-15 20:40:57 UTC; all research GPU
jobs end by **2026-11-16 06:40:57 UTC**. Prior billing and dollar rate are unknown.
The Pod remains running after jobs end. Its 50 GB overlay is temporary: original
data and code are local, and checkpoints/results are copied back with hashes.
No new service, paid API, model training endpoint, or push is authorized.

Training, generation, automated reviews, and stress tests share this allowance.
The all-subset policy/development/calibration cache requires 2,720 terminal calls
at the reduced sample sizes. Benchmark actual throughput before full expansion.
Record any budget-driven truncation as an incomplete experimental component,
never as a completed result. The maximum remains 30 classifier epochs, with
development early stopping. Save exact actual epochs and seed counts. A failed
pilot gate yields a completed feasibility/negative study with its measured
failure, rather than unsupported adaptive-performance claims.

Fix the terminal prompt and scoring definitions after development checks; freeze
controller settings and the strongest fixed/heuristic comparator before unlocking
test reports. Keep the original planned confirmatory family if evaluated, label
additional diagnostics exploratory, and report all departures. Synthetic tests
alone are insufficient for a real-data result. All generated data, checkpoints,
audio and detailed logs remain outside Git; only compact audit and result tables,
source code, figures, and the final manuscript are committed to main.
