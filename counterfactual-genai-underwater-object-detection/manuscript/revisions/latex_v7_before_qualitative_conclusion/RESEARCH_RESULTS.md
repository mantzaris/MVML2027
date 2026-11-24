# Completed research results

The full frozen research horizon is complete: all 38 execution stages, independent numerical audits, five main figures, an eight-page paper draft and a four-page companion study. All 12 final PDF pages were visually reviewed. All substantial signal processing, model fitting and inference used the local 32 GB NVIDIA RTX 5000 Ada; the smaller Quadro T2000 was excluded.

## Acoustic representations in the completed study

Current paper title: **Multi-Representation Underwater Acoustic Object Detection with Generative AI Reasoning Agents**.

| Representation | Implemented use | Scope |
|---|---|---|
| Fourier/FFT spectrum | 32 spectral descriptors; individual detector and fusion combinations | Primary real-data study |
| STFT spectrogram | 64 descriptors from two STFT resolutions | Primary real-data study |
| Wavelets | 56 descriptors from analytic Morlet transforms | Primary real-data study |
| 2D log-mel spectrogram | 96-bin time-frequency tensors processed by Conv2d blocks | Real CNN comparator and separate synthetic/semi-synthetic temporal CNN |
| EMD/Hilbert–Huang | 32 descriptors from bounded piecewise-linear EMD and Hilbert instantaneous frequency | Exploratory; failed numerical convergence gates; excluded from the primary agent |

The real-data comparison includes all seven nonempty Fourier/STFT/wavelet subsets and an all-four-view exploratory head. The supplement adds representation, regime and missing-view ablations. The Qwen controller sees numerical head outputs and bounded verification summaries; it does not inspect spectrogram images or perform vision-language image reasoning. The background generator operates on waveforms. The new title and opening foreground experiments already performed; they do not add an image-reasoning experiment or change any measured result.

## Main finding

**This generative reasoning-agent design did not improve the prespecified real-data detection endpoint.** On the balanced 1,200-record PORTIA test cohort, AP of the mean of three detector-seed probabilities was 0.573788 for the agent and 0.590610 for the development-selected wavelet comparator: a difference of −0.016823. The five collection days form one connected day–vessel component, so these are descriptive estimates without an independent-component confidence interval. AIS-linked three-second window presence is a proxy label, not human acoustic audibility or real event boundaries.

All 3,600 generative-agent evaluations followed wavelet → Fourier → verification. Verification was the last action, so generated evidence informed no subsequent tool choice. Agent and fixed verification had the same observed ensemble AP, with probabilities differing by at most 1.19e−7. This does not establish population equivalence, but it provides no observed benefit from language-model orchestration in this run.

The continuous verification score lowered ensemble AP by 0.025721 relative to matched V=0 controls that preserve queries, views, validity masks and calibration routing. The edit/control eligibility gate accepted 345 of 1,200 test records (28.75%); invalid cases retained the calibrated base score. Mean seed-specific recall at the calibration-frozen operating point was 1.06% for the agent, with 0.17% false-positive rate. It is not a validated deployment detector.

Numeric fusion and the real-data CNN had higher mean seed-specific test AP (0.616046 and 0.665354 respectively); these averages differ from the primary ensemble-AP estimand. The exploratory HHT-augmented model is not promoted to a validated Hilbert–Huang method because the bounded implementation failed numerical convergence gates. The proposed paired verification equation was implemented and tested, without a claim that it produced a gain or is the first such formulation.

## Completed studies

| Study | Completed scope | Result interpretation |
|---|---|---|
| Native real data | 6,270 verified recordings; 33 trained detector runs; 22 evaluated methods × 3 seeds = 66 metric rows | Selected cohort, AIS proxy labels; no deployment-prevalence claim |
| Agent/control inference | 7,200 test clip-seed-variant evaluations over 1,200 distinct recordings | No adaptive acquisition or added agent performance demonstrated |
| Noise robustness | 4 methods × 5 conditions on 120 shared records; 20 aggregate and 100 day rows | Agent/fixed AP matched; clean 0.6667 to −5 dB 0.4981; distinct subset from primary test |
| Controlled temporal detection | 48 conditions, 144 temporal-metric rows, 96,000 scene-condition-seed evaluations | Separate spectral CNN, not evidence of agent localization |
| Representation supplement | 24 view rows, 96 regime rows, 54 missing-view rows and 18 paired comparisons | Synthetic presence classification; HHT remains exploratory |
| Semi-synthetic transfer | 330 base scenes × 3 SNRs; 9 model-seed/SNR conditions and 27 temporal-metric rows | Real noisy excerpt insertion, not naturally annotated object events |
| Live runtime | 5 methods × 3 passes over 32 calibration records at batch 32 | All 15 passes matched archived inference within fixed tolerances |

The separate temporal CNN achieved mean AP at temporal IoU 0.5 of 0.8400 on mixed-SNR in-family synthetic test scenes and 0.8976 on the shifted family. The family change jointly alters sources, backgrounds and propagation and need not make the task harder. Semi-synthetic temporal AP at IoU 0.5 was 0.0396, 0.3792 and 0.7301 at component SNR −10, 0 and +10 dB. Inserted excerpts include their own noise; this is not clean-vessel SNR. These results do not rescue the negative real-data agent finding.

## Measured compute

| Method | Warm batch-32 ms per clip |
|---|---:|
| Numeric fusion | 7.124 |
| Fixed verification | 49.420 |
| Heuristic verification | 38.989 |
| Generative agent | 346.869 |
| Agent without generation | 299.377 |

Timings include waveform decoding, transfer and live requested inference, exclude model loading, and measure throughput rather than individual-request latency. The agent cost 7.02 times fixed verification while matching its observed ensemble AP. Agent peak allocated CUDA memory was 11,644 MiB. Only these five methods were timed; no overall cost-frontier claim is made.

## Deliverables and audit notes

- `manuscript/main.tex` and `manuscript/main.bib`: single-file editable main manuscript and separate bibliography; compiled to `manuscript/latex_build/main.pdf` (eight pages, five figures, five native display equations).
- `manuscript/supplementary.tex` and `manuscript/supplementary.bib`: single-file companion study and separate bibliography; compiled to `manuscript/latex_build/supplementary.pdf` (four pages, two embedded figures). A third supplemental figure is exported separately in `manuscript/figures/`.
- `manuscript/figures/` and `manuscript/figure_data/`: exported figures and their numeric data.
- `results/tables/final_real_results.json`: assembled final result table; source reports and raw predictions remain available.
- `reports/*independent_audit.json`, the visual-review receipts and `reports/final_integrity_audit.json`: completed checks.
- `reports/supplemental_comparisons_corrected.json`: explicit detector_seed/bootstrap_seed fields. A dictionary-key collision in the original comparison metadata is documented in `reports/supplemental_metadata_erratum.md`; all numeric values and frozen originals are retained unchanged. Primary representation rows identify detector seeds correctly.
- The supplemental display label is “two-source scenes”; its frozen raw key overlapping_sources does not require temporal overlap. The first PDF build and its receipts are preserved under `manuscript/revisions/`.

The drafts have not been submitted. Author metadata, final conference formatting and the venue's supplemental-material channel still need author review. This is distinct from completion of the research execution.

## Interpretation and recovery

A new improvement claim would require a separately frozen follow-up study and untouched evaluation evidence. Real-background generator adaptation, reliable edit matching and source-selective validation are future directions, not completed experiments or promised gains. The present test cohort has already been observed.

Completion and recovery records are `runs/research_complete.json`, `runs/horizon_status.json`, `runs/events.jsonl`, `runs/stages/` and `checkpoints/`. From this directory run `bash scripts/resume_research.sh` after an interruption. It verifies completed work, resumes unfinished work and preserves completed status only while the final hash-bound review remains valid. No automatic operating-system startup service is installed. Atomic checkpoints cannot protect against destruction of the storage device.

All saved project work is in /home/meow/Documents/repos/MVML2027/counterfactual-genai-underwater-object-detection. This work did not modify the prior evidence-genAI-underwater-acoustic-reasoning submission.

The LaTeX conversion preserves the reviewed findings and numerical tables. Current reading files are `manuscript/main_paper.pdf` and `manuscript/supplement.pdf`; the LaTeX build synchronizes them with the validated build copies. Original ReportLab PDFs and provenance are archived under `manuscript/revisions/latex_v6_before_current_pdf_names/`. Compile with `bash scripts/build_latex.sh`; publication-migration review: `reports/current_pdf_publication_review.json`. The user opens the files directly, without UI previews.

The main abstract now explicitly separates comparisons of sound representations, controlled tests of generated evidence, and measured comparisons of agent-selected and fixed procedures. It connects results to a tested reference detector, simpler processing at the same observed score, repeatable synthetic activity-timing tests and reliability-aware representation choices. The first page defines short-time Fourier transform (STFT), the bounded Hilbert-Huang transform (HHT) approximation and its decomposition role, convolutional neural network (CNN), and automatic identification system (AIS). Section 7 now explains the figures' practical value: the accuracy/time and memory comparisons, distinct scoring and orchestration choices, and the fixed/agent match in the separate noise study. The conclusion, other main-body content and complete supplement retain their prior reviewed text. Numerical evidence, tables, equations, figures and bibliography are unchanged. Standing editorial guidance is recorded in `manuscript/README.md`. Current review: `reports/editorial_revision_7_review.json`; main and supplement remain eight and four pages.
