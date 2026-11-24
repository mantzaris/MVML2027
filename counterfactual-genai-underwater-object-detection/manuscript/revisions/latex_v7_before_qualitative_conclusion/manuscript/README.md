# Editable LaTeX manuscripts

The current editable sources are:

- `main.tex` with `main.bib`: main paper, eight compiled pages.
- `supplementary.tex` with `supplementary.bib`: supplementary study, four compiled pages.

Each TeX file contains its complete preamble, prose, equations, tables, figure captions and bibliography commands. There are no external section, equation or table source files. References are maintained in the matching BibTeX file and rendered with the installed `IEEEtran` bibliography style. Existing vector figures are loaded from `figures/`; these assets are required when copying the sources to another TeX environment.

From the submission directory, compile and render both documents:

```bash
bash scripts/build_latex.sh
```

**Required after every manuscript edit:** compile each affected document with this script, check the build diagnostics and rendered pages, and ensure `main_paper.pdf` and `supplement.pdf` contain the current LaTeX outputs before reporting completion. The user reads these files directly. Do not open Codex, browser or other UI previews unless explicitly requested. Saving TeX alone does not complete an edit.

Use `bash scripts/build_latex.sh main` or `bash scripts/build_latex.sh supplementary` to build one. The current reading files are `main_paper.pdf` and `supplement.pdf`, kept byte-identical to `latex_build/main.pdf` and `latex_build/supplementary.pdf` by every successful build. Page renders are under `latex_renders/`. Timestamped build attempts, auxiliary files and logs are retained in `latex_build/runs/`. All tool caches and temporary files stay within this project.

The build uses pdfLaTeX, BibTeX, Poppler and the project Python environment. Standard TeX Live packages include newtx, helvet, geometry, amsmath, graphicx, booktabs, array, tabularx, xcolor, caption, titlesec, fancyhdr, microtype, hyperref and cite. Shell escape is disabled. The script rejects unresolved citations/references, overfull boxes and page counts outside the required limits. It publishes compiled PDFs only after these checks; a new build still needs visual review.

Edit the TeX and BibTeX files directly. `latex_migration/content.json` is a provenance snapshot of the reviewed report-builder content, not an input required for LaTeX compilation. The original ReportLab PDFs and their provenance are preserved under `revisions/latex_v6_before_current_pdf_names/`. The root `main_paper.pdf` and `supplement.pdf` now serve as current LaTeX reading copies. Use `scripts/build_latex.sh` for editorial builds; the old ReportLab builder remains a historical source.

The original conversion audit is in `../reports/latex_conversion_audit.json`. The current abstract explicitly identifies the contributions and the practical value of the results; first-page acronyms are defined. Its review is `../reports/editorial_revision_6_review.json`. Section 7 now interprets Figures 2–5 to explain detector, scoring and resource choices; the current document review is `../reports/editorial_revision_7_review.json`. Other main text and the complete supplement are unchanged. The reading-file mapping and original publication migration are recorded in `../reports/current_pdf_publication_review.json`. The prior reviewed sources, PDFs and completion records are preserved under `revisions/latex_v6_before_figure_insights/`. The research completion record binds the reviewed TeX, BibTeX, figures and PDFs by checksum. After further edits, rebuild and renew the document review before claiming that the changed draft matches that completion record. No experiment rerun is needed for document-only changes.

The historical manuscript-stage command/input record is preserved with an explicit document-output supersession: current reading copies are produced by LaTeX, while the original ReportLab PDFs and receipt remain archived. After later editorial builds, renew the document manifest, reading-PDF reviews, manuscript-stage output supersession and completion hashes together. All experimental stage receipts remain unchanged.

Formatting follows the previously recorded written template guidance with Times-like and Helvetica-like fonts. Author metadata and final submission-format checks remain outstanding; these drafts have not been submitted.

## Editorial approach

For every manuscript edit, lead with the research contribution, methodological distinction, practical importance and insight gained. Explain what each comparison establishes for representation choice, verification design, reusable baselines and computational demand. Present fixed verification as retaining the agent's observed score on the evaluated cohort while reducing measured processing time; present matched score-removal tests as identifying a stronger scoring configuration. Treat numerical checks as evidence for representation readiness and further development. Tie substitution claims to the actual metric and conditions tested. Preserve measured values, label provenance and statistical scope, stated affirmatively and concisely. Keep the abstract qualitative and accessible to nonexpert readers.

State each contribution concretely and connect each result to what readers can reuse or decide. Explain generative edits in ordinary language before naming their scores or controls. Expand acronyms at first use and give unfamiliar methods a brief functional explanation.
