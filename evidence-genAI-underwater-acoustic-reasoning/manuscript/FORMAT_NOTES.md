# Manuscript format and author review

The working source is the official [MVML 2027 DOCX](https://mvml.org/documents/MVML27_Paper_Template.docx), retained under `template/`. The empirical builder reads completed experiment tables and cannot substitute placeholder findings. The manuscript is **anonymous by the user's instruction**, including its author metadata; affiliation and contact details are intentionally omitted. This is a user preference, not a claim about the conference's review policy.

The verified [author guidelines](https://mvml.org/author-guidelines/) impose an eight-page limit. This project counts references within those eight pages. The original source audit retains the access dates, template hash and conference dates; recheck submission requirements before any eventual submission. No submission or publication is authorized here.

The DOCX is US Letter, with left/right margins 11.994 mm and top/bottom 25.4 mm. Its prose instead prescribes 25 mm margins on all sides. The working document preserves the downloaded section geometry. The organizer must resolve that conflict before submission. Body text uses Times New Roman 11 pt and headings Arial; tables/captions/references use 10 pt, following the template. Fonts and every rendered page are checked in the final inspection record.

The current identification-focused manuscript contains **exactly eight total pages, including references**, and a 186-word abstract. This uses the maximum allowance requested by the user. All pages were visually inspected; the final inspection record pins the PDF and DOCX hashes. The prior report-support paper remains in Git history and an ignored local archive. To rebuild the current paper from the saved completed outputs:

```bash
.venv/bin/python scripts/make_identification_figures.py
.venv/bin/python scripts/make_agentic_figure.py
.venv/bin/python scripts/build_identification_manuscript.py
libreoffice -env:UserInstallation=file:///tmp/mvml2027-identification-lo-profile --headless --convert-to pdf --outdir manuscript manuscript/manuscript.docx
pdfinfo manuscript/manuscript.pdf
pdffonts manuscript/manuscript.pdf
```

Inspect every page after substantive edits. `results/manifests/manuscript_inspection.json` records the final PDF hash and visual inspection once completed. The original `build_manuscript.py` builds the superseded report-support paper and should not be used for the active manuscript. The initial preparation draft and prior completed paper are retained in Git history and ignored local archives.

Figure 3 now shows the implemented agentic acquisition loop, the earlier reporting controls, and the separate fresh-cohort generative outcomes. Its cached query counts do not imply measured runtime savings. It replaces the illustrative spectrogram figure and consolidates the previous generative table; those examples and all numerical results remain in the artifact bundle. No scientific model, protocol, threshold or test outcome changed.

The organizer's [FAQ](https://www.international-aset.com/FAQ/) requires all authors to be listed in references instead of `et al.`. The current builder lists every author from the verified bibliography, using initials and surnames; the source records retain the full names. This requirement was checked on November 16, 2026.
