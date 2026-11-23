# Supplemental comparison metadata erratum

Recorded 23 November 2026. The original results/tables/supplemental_views.json is preserved unchanged. In its 18 comparisons records only, dictionary expansion allowed the bootstrap seed to replace the detector seed. Raw values 1746, 1758 and 1772 denote bootstrap RNG seeds; the corresponding detector seeds are 17, 29 and 43 (subtract the frozen bootstrap base 1729). The 24 view, 96 regime and 54 missing-view result rows retain correct detector seeds.

The corrected derived export reports/supplemental_comparisons_corrected.json separates detector_seed and bootstrap_seed and records the source output SHA256. Its mapping was verified against all prediction payload paths and independently recomputed AP contrasts. No statistic, source file, trained model, inference output or test configuration was changed. The manuscript builder does not consume these comparison records.

A separate terminology clarification applies to the frozen regime key overlapping_sources: its selection predicate is source_count equal to two, so it includes both overlapping and nonoverlapping intervals. Present this stratum as two-source scenes. Preserve the original key and numerical grouping.
