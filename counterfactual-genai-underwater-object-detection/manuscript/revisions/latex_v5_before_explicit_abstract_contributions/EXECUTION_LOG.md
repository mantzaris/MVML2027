# Research execution log

## 23 November 2026 start

The user authorized executing the full planned study and requested persistent checkpoints for recovery after power loss. All new work stays inside this submission directory. The previous submission is read-only and unchanged.

The project-local CUDA environment is being installed. The local RTX 5000 Ada is available and selected by UUID. In parallel, the implementation is developing resumable storage, CUDA signal representations and controlled synthesis, and a real-data provenance/access audit.

The initial UniqueShip small-release archive returned a public download-quota error; alternate access is being investigated. No research training or test evaluation has run yet. Model development must not consume final test outcomes. Machine-readable status is in manifests/research_status.json; subsequent durable stage events and checkpoints will be recorded under logs/ and checkpoints/.

## Environment and numerical pilot completed

Installed a dedicated CUDA-enabled .venv and pinned dependency/model revisions. The real CUDA restart test passed across separate processes, including fallback from a damaged latest checkpoint. Fourier/STFT/wavelet numerical checks passed; bounded HHT convergence was inadequate on noisy fixtures and is now exploratory. A64-step conditional-flow functionality pilot completed and saved a durable checkpoint. It is not a scientific performance result.

The accessible PORTIA v2 cohort was chosen before any real model evaluation. Conflicting cross-task labels were excluded, and vessel identities plus UTC days are disjoint between partitions. Native audio acquisition is in progress. The execution amendment records these decisions and the limited number of independent test days.


## Full-horizon supervisor and pre-outcome audit

Background-flow training, all3synthetic temporal detector runs and all12supplemental representation heads completed on the RTX5000 Ada. Synthetic train/development feature caches are durable. Agent single and batched protocol fixtures passed; controlled waveform checks found strong overlapping-source collateral suppression, so no source-selective counterfactual claim is made.

`bash scripts/resume_research.sh` is the project-local continuation command. The active supervisor reuses the existing download and will advance as verified split manifests appear. Completed stages are recorded in `runs/stages/`; current work appears in `runs/horizon_status.json`. The current session is still supervising execution, not merely handing off a background job.

Metadata-only dependence auditing found one connected day–vessel component across five test days. The analysis now reserves population inference: primary AP difference is descriptive; the day bootstrap is a separate sensitivity analysis. Comparator selection uses development AP of mean calibrated seed probabilities, matching the final endpoint. These decisions precede real model outcomes.


### Global manifest freeze binding — 2026-11-23T18:22:19.002766+00:00

The test gate now pins `data/manifests/real.jsonl`, the row source preferred by `load_rows` after acquisition, and asserts exact content and ordering equality against each of the four split manifests. The future freeze-stage input signature also includes the global manifest. Metadata-only validation passed for all 6,270 rows (3,735 train, 735 development, 600 calibration, 1,200 test); all existing completed stage signatures remain unchanged. Receipt: `manifests/freeze_manifest_validation.json`. No audio, test outcomes, models, data decoding code, or training sources were accessed or changed, and no test freeze was created.

Four metadata-only stage-provenance regression tests also passed; 14 completed horizon stage signatures were unchanged.


### First-freeze policy and test-artifact guards — 2026-11-23T18:24:57.959506+00:00

`freeze_protocol.py` now compares every policy source fingerprint and central-intervention index fingerprint to the exact provenance construction in `run_policies.py`. The first-freeze guard covers real/foundation/synthetic test feature directories, central/full intervention directories, test policy/prediction namespaces, supplemental and robustness outputs, and final tables, including partial shards without indices. Metadata-only validation checked 6 completed policy summaries and the available central evidence indices; 6 calibration policy summaries remain pending and must pass at freeze. Changed source/evidence fingerprints were rejected, and seven isolated project-temporary artifact fixtures were detected. Canonical test-derived namespaces remain absent. Receipt: `manifests/freeze_policy_binding_validation.json`. No method sources, models, GPU jobs, test inference, or test-unlock state changed.


Before runtime or final test outcomes, added a first-freeze requirement that all15 measured runtime passes match their archived calibrated predictions/actions under existing benchmark tolerances. A mismatch stops freeze for an audit rather than pairing inconsistent cost and accuracy. Experimental inference and tolerance definitions are unchanged.


## Full-horizon completion — 2026-11-23T20:04:29.237400+00:00

All38 execution stages completed. All185 frozen artifacts and38stage receipts passed final integrity verification; all12final PDF pages and5main figures passed visual review. Main result is negative (agent ensemble AP0.573788, lockedwavelet0.590610). Independent audits passed real, robustness, synthetic, supplemental and semi-synthetic results. Supplemental seed-field metadata is corrected in a separate traceable export; original measurements remain unchanged. Presentation-only revision2 corrects two-source/family labels and preserves revision1. Completion receipt and recovery instructions are saved locally; no submission, Git commit or push was performed.

## Standalone LaTeX conversion — 2026-11-23T20:40:19.096487+00:00

Created manuscript/main.tex + main.bib and supplementary.tex + supplementary.bib. Each document keeps all prose, native equations and tables in one source file. The final pdfLaTeX/BibTeX build has eight main pages and four supplementary pages with no compilation warnings, undefined references/citations or overfull boxes. All pages passed visual review. Scientific prose, values, tables, five display equations and seven embedded figures were checked against the reviewed report-builder content. Build command: bash scripts/build_latex.sh. Checksum-bound evidence: reports/latex_conversion_audit.json. All 185 frozen artifacts and prior completed outputs were verified unchanged, except explicitly archived documentation updates. No experiments rerun and no prior submission modified.

## Accessible abstract and conclusions — 2026-11-23T21:15:40.904549+00:00

Revised the main abstract (243 words), conclusion (206 words) and supplement closing paragraph. Added intuitive harbour-listening and fixed-checklist explanations, clarified methodological contribution and practical usefulness, and retained negative AP results, measured compute cost, nonadaptive traces, exploratory bounded HHT, label limitations and cohort dependence. Exact source comparison confirms other prose, all equations and tables, figures and bibliographies unchanged. Final LaTeX build: manuscript/latex_build/runs/20261123T211309Z-498013; main8/supplement4pages, resolved references, no compiler warnings or overfull boxes. All changed pages individually reviewed; unchanged pages matched archived renders. Original version and completion receipt retained in manuscript/revisions/latex_v1_before_accessible_abstract. No research experiments rerun. Final review: reports/editorial_revision_2_review.json.

## Success-led abstract and conclusions — 2026-11-23T21:47:26.354513+00:00

Revised only the main abstract (201 words), conclusion (195 words) and supplementary closing paragraph. Explicit successes: implemented controlled multi-representation GPU workflow; observed real spectrogram-CNN ensemble AP0.6692 versus locked wavelet0.5906; separate controlled temporalCNN meanAP0.847 at IoU0.5 across24 in-family seed/condition rows; fixed verification attaining the same observed agentAP with about one-seventh its measured warm-batch time. Supplemental STFT AP0.995/0.880 across in-family/held-out test families verified from frozen results. The agent accuracy failure, matched-score harm, label/dependence limitations and exploratory HHT remain explicit. No novel CNN, causal source removal, population equivalence or agent gain claimed. All other paper content, equations, tables, bibliography and figure assets unchanged. Final build manuscript/latex_build/runs/20261123T214419Z-542082 passed at8+4pages; all changed pages reviewed. Prior revision archived at manuscript/revisions/latex_v2_before_success_framing; review reports/editorial_revision_3_review.json. No experiments rerun.

## Qualitative abstract for nonexpert readers — 2026-11-23T22:05:09.028728+00:00

Revised only the main abstract (216 words), removing numerical results and explaining tones, timing, sound images and practical computing value. The distinctive contribution is controlled integration of representations, generated/control edits, score-removal tests, observable agent choices and processing cost. Explicit successes are the higher-scoring real spectrogram-network configuration, separate synthetic activity-interval detection, a substantially faster fixed-verification option with the agent's observed score, and diagnosis of ineffective/unreliable components. The abstract retains no agent gain and bounded real/synthetic interpretation. Independent scientific review passed. Source comparison confirms every byte outside the main abstract is unchanged, including the conclusion; supplement, equations, tables, bibliographies and figures are unchanged. The main-only build at manuscript/latex_build/runs/20261123T215938Z-565958 passed at eight pages with no compiler warnings, unresolved references or overfull boxes. Root and an independent reviewer visually inspected page 1; pages 2–8 exactly match prior reviewed renders. All 185 frozen artifacts and all 345 prior completion-bound artifacts were checked; changes were confined to explicitly allowed documentation. Prior reviewed version and recovery records are preserved in manuscript/revisions/latex_v3_before_qualitative_abstract. Current review: reports/editorial_revision_4_review.json. No research experiments, commit, push or submission performed.

## Contribution-first framing throughout both manuscripts — 2026-11-23T23:24:02.175870+00:00

Applied the user's standing editorial direction to lead with methodological distinction, practical importance and value gained. Main and supplementary prose and captions now explain stronger reference configurations, fixed orchestration retaining the observed agent score with substantially lower measured processing cost, scoring improvements identified by matched score-removal controls, and numerical checks that guide representation readiness. The abstract remains qualitative (209 words); the conclusion is 201 words. Exact observed contrasts, metric definitions, seed-specific direction, label provenance and descriptive real-cohort scope are preserved. The bootstrap interval explicitly names the agent-minus-Wavelet comparison. Tables, equations, figure assets and dimensions, bibliographies, citations, preambles and page breaks are unchanged. A preliminary nine-page attempt is retained; concise mechanism prose restored the main paper to eight pages without changing layout settings. Final builds: manuscript/latex_build/runs/20261123T232051Z-681503/main and manuscript/latex_build/runs/20261123T231923Z-679204/supplementary. All twelve page contents passed individual visual review; only page 6 changed during the final wording rebuild and was freshly inspected. Independent scientific review passed. All 353 prior completion-bound artifacts and 185 frozen experimental artifacts were checked; differences were confined to the documented prose/build/recovery scope. Previous reviewed revision 4 is archived under manuscript/revisions/latex_v4_before_contribution_framing. Standing editorial guidance is in manuscript/README.md; source snapshot and review are manuscript/latex_migration/editorial_revision_5.json and reports/editorial_revision_5_review.json. No research experiments, commit, push or submission performed.
