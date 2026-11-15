# Manuscript format and author review

The working source is the official [MVML 2027 DOCX](https://mvml.org/documents/MVML27_Paper_Template.docx), retained under `template/`. The empirical builder reads completed experiment tables and cannot substitute placeholder findings. The manuscript is **anonymous by the user's instruction**, including its author metadata; affiliation and contact details are intentionally omitted. This is a user preference, not a claim about the conference's review policy.

The verified [author guidelines](https://mvml.org/author-guidelines/) impose an eight-page limit. This project counts references within those eight pages. The original source audit retains the access dates, template hash and conference dates; recheck submission requirements before any eventual submission. No submission or publication is authorized here.

The DOCX is US Letter, with left/right margins 11.994 mm and top/bottom 25.4 mm. Its prose instead prescribes 25 mm margins on all sides. The working document preserves the downloaded section geometry. The organizer must resolve that conflict before submission. Body text uses Times New Roman 11 pt and headings Arial; tables/captions/references use 10 pt, following the template. Fonts and every rendered page are checked in the final inspection record.

After the clean analysis, stress tests and report audits finish:

```bash
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_real_examples.py
.venv/bin/python scripts/build_manuscript.py
libreoffice -env:UserInstallation=file:///tmp/mvml2027-lo-profile --headless --convert-to pdf --outdir manuscript manuscript/manuscript.docx
pdfinfo manuscript/manuscript.pdf
pdffonts manuscript/manuscript.pdf
```

Inspect every page after substantive edits. `results/manifests/manuscript_inspection.json` records the final PDF hash and visual inspection once completed. The initial two-page preparation draft is historical and will be removed from the working tree when the empirical manuscript is verified; Git retains its history.
