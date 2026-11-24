# Editable LaTeX manuscripts

The current editable sources are:

- `main.tex` with `main.bib`: main paper, eight compiled pages.
- `supplementary.tex` with `supplementary.bib`: supplementary study, four compiled pages.

Each TeX file contains its complete preamble, prose, equations, tables, figure captions and bibliography commands. There are no external section, equation or table source files. References are maintained in the matching BibTeX file and rendered with the installed `IEEEtran` bibliography style. Existing vector figures are loaded from `figures/`; these assets are required when copying the sources to another TeX environment.

From the submission directory, compile and render both documents:

```bash
bash scripts/build_latex.sh
```

**Required after every manuscript edit:** compile each affected document with this script, check the build diagnostics and rendered pages, and refresh the visible PDF before reporting completion. Open the newly generated PDF from that build's `latex_build/runs/` directory when a viewer may have cached the older file; the canonical `latex_build/main.pdf` and `latex_build/supplementary.pdf` are updated by the same successful build. Use the generated PDF directly if the built-in editor's preview compiler is unavailable. Saving TeX alone does not complete an edit.

Use `bash scripts/build_latex.sh main` or `bash scripts/build_latex.sh supplementary` to build one. Outputs are `latex_build/main.pdf` and `latex_build/supplementary.pdf`; page renders are under `latex_renders/`. Timestamped build attempts, auxiliary files and logs are retained in `latex_build/runs/`. All tool caches and temporary files stay within this project.

The build uses pdfLaTeX, BibTeX, Poppler and the project Python environment. Standard TeX Live packages include newtx, helvet, geometry, amsmath, graphicx, booktabs, array, tabularx, xcolor, caption, titlesec, fancyhdr, microtype, hyperref and cite. Shell escape is disabled. The script rejects unresolved citations/references, overfull boxes and page counts outside the required limits. It publishes compiled PDFs only after these checks; a new build still needs visual review.

Edit the TeX and BibTeX files directly. `latex_migration/content.json` is a provenance snapshot of the reviewed report-builder content, not an input required for LaTeX compilation. The old ReportLab builder and its `main_paper.pdf`/`supplement.pdf` outputs remain historical snapshots and do not update the editable TeX sources.

The original conversion audit is in `../reports/latex_conversion_audit.json`. The current abstract explicitly identifies the contributions and the practical value of the results; first-page acronyms are defined. Its review is `../reports/editorial_revision_6_review.json`, with unchanged main-body and supplementary content retaining revision 5 review. The latest compiled-PDF refresh is verified in `../reports/latex_refresh_20261124T002709Z_review.json`. The prior reviewed sources, PDFs and completion records are preserved under `revisions/latex_v5_before_explicit_abstract_contributions/`. The research completion record binds the reviewed TeX, BibTeX, figures and PDFs by checksum. After further edits, rebuild and renew the document review before claiming that the changed draft matches that completion record. No experiment rerun is needed for document-only changes.

Formatting follows the previously recorded written template guidance with Times-like and Helvetica-like fonts. Author metadata and final submission-format checks remain outstanding; these drafts have not been submitted.

## Editorial approach

For every manuscript edit, lead with the research contribution, methodological distinction, practical importance and insight gained. Explain what each comparison establishes for representation choice, verification design, reusable baselines and computational demand. Present fixed verification as retaining the agent's observed score on the evaluated cohort while reducing measured processing time; present matched score-removal tests as identifying a stronger scoring configuration. Treat numerical checks as evidence for representation readiness and further development. Tie substitution claims to the actual metric and conditions tested. Preserve measured values, label provenance and statistical scope, stated affirmatively and concisely. Keep the abstract qualitative and accessible to nonexpert readers.

State each contribution concretely and connect each result to what readers can reuse or decide. Explain generative edits in ordinary language before naming their scores or controls. Expand acronyms at first use and give unfamiliar methods a brief functional explanation.
