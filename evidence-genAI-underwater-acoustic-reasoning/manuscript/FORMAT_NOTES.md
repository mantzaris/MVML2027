# Draft format and readiness

The working source is the official [MVML 2027 DOCX](https://mvml.org/documents/MVML27_Paper_Template.docx), downloaded and retained under `template/`. The generated manuscript is explicitly an **incomplete preparation draft**, not a submission-ready results paper. Author affiliation/contact details and all model-result sections remain unresolved. No fabricated result tables are included.

Current [author guidelines](https://mvml.org/author-guidelines/) limit full papers to eight pages and accept LaTeX while providing a DOCX template. Plan for eight total pages including references; an explicit reference-page exemption and review anonymity policy were not established. [Important dates](https://mvml.org/important-dates/) list January 21, 2027 submission, February 25 notification, March 18 final version, and August 15–17 conference dates. Recheck before submission.

The downloaded DOCX is US Letter (215.9 × 279.4 mm), with left/right margins 11.994 mm and top/bottom 25.4 mm. Its prose instead prescribes 25 mm margins on all sides. The draft preserves the actual DOCX section geometry; the conflict remains to be resolved before final submission. It is not silently corrected or called an official LaTeX recreation. Body font is Times New Roman 11 pt and headings Arial; verify font availability in the final publication renderer.

To rebuild the editable draft:

```bash
.venv/bin/python scripts/build_bibliography.py
.venv/bin/python scripts/build_manuscript_draft.py
libreoffice --headless --convert-to pdf --outdir manuscript manuscript/manuscript_draft.docx
pdfinfo manuscript/manuscript_draft.pdf
```

Use a writable temporary LibreOffice profile if the normal application profile is unavailable. Re-render and inspect every page after substantive edits. The draft reports actual data-audit and synthetic/software checks only; AI reviewers and model experiments have not yet run.
