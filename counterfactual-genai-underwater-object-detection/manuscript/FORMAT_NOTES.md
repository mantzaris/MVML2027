# Manuscript generation and format notes

The report builder is scripts/build_results_report.py. It reads measured artifacts
and refuses to produce the research PDFs when required real, synthetic or
supplemental result arrays are missing. It never fills missing values with
illustrative or predicted performance.

Run from the submission directory after the experiment reports are complete:

    source scripts/project_env.sh
    .venv/bin/python scripts/build_results_report.py

The default inputs are the assembled results/tables/final_real_results.json,
reports/synthetic_results.json, results/tables/semisynthetic_results.json and
results/tables/supplemental_views.json. Command-line --real, --synthetic,
--semisynthetic and --views accept explicit project-local paths. The core
real_results.json remains an intermediate result and is not the final build input.

The manuscript includes five main figures, four driven by experimental result
arrays. The four-page supplement reports controlled representation, numerical
validity and missing-view evidence. Each figure is saved as vector PDF and 300 dpi
PNG; the corresponding numeric rows are exported under manuscript/figure_data/.
The build receipt records source/output hashes and page counts. Pages render under
manuscript/renders/ for visual inspection. A completed render is not visual signoff:
manuscript/build_manifest.json must retain visual_inspection_complete=false until
every final page is inspected and any layout issue is corrected.

## Conference source and geometry

The official guidelines at https://mvml.org/author-guidelines/ were checked on
23 November 2026. Full papers are limited to eight pages; LaTeX is also accepted.
The official template is saved as template/MVML27_Paper_Template.docx, downloaded
from https://mvml.org/documents/MVML27_Paper_Template.docx.
Its SHA-256 is 86364438af0f7090bc905442437f88c8f72412770d040e885415402eb2c19a35.

The template prose specifies US letter, 25 mm margins, single-spaced 11 pt Times New
Roman, 16 pt Arial bold title, 12 pt Arial bold headings, IEEE-style references and
an abstract of at most 300 words. Its embedded XML instead uses 680-twip left/right
margins (about 12 mm). This draft follows the written 25 mm margin instructions.
The installed Tinos and Arimo fonts are metrically compatible substitutes for
Times New Roman and Arial and are embedded in the PDF. Font substitution and the
margin conflict should be resolved during final author/template review.

The draft is anonymous and uses no invented paper acceptance code. The public
guidelines do not specify a separate supplemental-material allowance; the standalone
supplement is prepared for author review, with submission-channel policy still to
be confirmed. No submission or publication is performed by this script.

## Scientific distinctions

- Natural-real outcomes are three-second AIS-window presence, not human audibility
  or natural event-boundary truth. The custom test cohort deliberately balances
  600 single-vessel positives and 600 binary-task backgrounds; AP, precision and
  probability calibration are not natural-population prevalence estimates.
- Five test collection days are connected by repeated vessel IDs into one dependence
  component. Independent-component uncertainty is not estimable; paired day
  resampling is explicitly nonconfirmatory sensitivity analysis.
- Mean seed-specific AP is distinct from AP of averaged per-record probabilities.
- Generator seed 17 is shared across three detector seeds, rather than three
  independently trained generators.
- The primary probe replaces the central 1.5 seconds of a three-second record. The
  full-record-null ablation lacks waveform context but retains observed RMS scaling.
  Neither is source-selective removal or a physical counterfactual.
- Controlled temporal predictions describe the union of acoustic activity;
  overlapping source intervals are merged.
- HHT is an exploratory bounded approximation after a measured convergence failure.
- Tensor-resident pilot timings and cached replay cost estimates remain explicitly
  distinct from measured end-to-end runtime.

- The final non-thinking Qwen controller uses constrained greedy legal-JSON token
  tries; schema validity is enforced by construction. Its archived unconstrained
  development failure remains reported. All 2,205 final development clip-seed
  traces took wavelet, Fourier, then verify; these traces do not demonstrate
  evidence-dependent adaptation or reasoning benefit.
- Fixed/agent V=0 controls retain queries, views, validity masks and the same
  two-input calibration procedure. The agent control retains generated-evidence
  access and its original verification-enabled actions, rather than rerunning the
  no-generation controller. In the observed paths verification was the final
  action, so generated evidence informed no subsequent tool choice.
- Fully synthetic and semi-synthetic temporal results concern a separately trained
  spectral CNN, not a demonstrated benefit from the Qwen/flow method.
- Live cost covers five measured methods including heuristic verification, not an
  overall frontier over untimed stronger baselines.

## Layout and final visual review

The main display set is result-independent: numeric fusion, CNN, frozen wav2vec2,
fixed/heuristic verification, both V=0 controls, agent and agent without generation,
plus the dynamically locked development comparator if not already included. All
methods remain in the complete machine-readable CSV/JSON results and component
ablations. Membership is never selected according to favorable test outcomes.

A separate watermarked rehearsal is kept under tmp/layout_review/. It shows measured core real results and cost where available, while reserving
remaining chart space and using literal pending cells for unfinished studies.
It contains no invented empirical scores and is not a research deliverable. The rehearsal currently fits eight main
pages and four supplemental pages. It does not replace final chart/page inspection.

The final build must be rerendered with Poppler after the actual artifacts arrive.
Inspect all eight main and four supplemental pages at readable resolution, check
figure labels, equation aspect ratios, tables, references, margins and page breaks,
and only then mark manuscript/build_manifest.json as visually inspected. The
builder checks the page counts but cannot substitute for visual review. Tinos and
Arimo remain declared metric-compatible font substitutions; anonymity is a draft
convention, not a verified conference requirement.

The final report derives acquired-view and verification usage from the assembled
real table's policy_usage rows. It cross-checks order against the existing
reports/test_agent_behavior_audit.json and original trace checksums. The completed
3,600 test clip-seed paths all took wavelet, Fourier, then verify. This is reported
as a lack of adaptive tool selection; generation occurred last and informed no
later action. These behavior statements are separate from accuracy, which remains
gated on the final result tables.

The final results text separately reports AP of averaged seed probabilities and
mean seed-specific AP. The passed independent real-test audit and every source
checksum it records are bound into the manuscript build manifest. The locked
wavelet comparator is explicitly selected on development data; it is not relabelled
as the strongest observed test method. Negative continuous-score contrasts and
actual measured cost remain visible, alongside the absence of adaptive selection.
