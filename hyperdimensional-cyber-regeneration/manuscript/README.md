# Anonymous manuscript package

**Title:** Observation-Aware Graph Learning for Corrupted Cyber Telemetry.

The anonymous draft consolidates the completed synthetic regeneration studies, inexpensive ML comparisons, captured CTU-13 application, and two GraphMAE variants. The main paper centers on the captured-flow experiment. Earlier synthetic decision-regret findings remain separate in the supplement; they are not evidence of real-world course-of-action performance.

| Deliverable | Content |
| --- | --- |
| [Main paper](anonymous/main.pdf) | Eight pages including references; four empirical figures, four numbered equations, and a thirteen-method results table |
| [Supplement](anonymous/supplement.pdf) | Four pages; four additional figures covering synthetic comparisons, optimization and paired capture/seed effects |
| [Anonymous editable-source archive](anonymous-source.zip) | Markdown, generated LaTeX, shared style, eight vector figures and both PDFs; excludes internal provenance |
| [Main narrative](anonymous/main.md) / [supplement narrative](anonymous/supplement.md) | Editable text used by the build script |
| [Format notes](FORMAT_NOTES.md) | Official template interpretation, local font substitutions and unconfirmed supplemental-upload policy |
| [Figure provenance](figure_provenance.json) / [build manifest](build_manifest.json) | Internal hashes, source checks, PDF page/figure counts and metadata checks |

The measured EDGE_GMAE gain is promising but exploratory. Both nominal two-comparison adjusted intervals include zero, and EDGE_DENSE has the largest descriptive clean mean, so the paper does not credit masked pretraining with that gain. It explicitly discloses prior test inspection, the limited known-label subset, artificial feature stresses and the absence of operational planning outcomes. It includes negative findings and matched controls alongside the improvements.

## Rebuild

Run from the project root with the existing Python environment, XeLaTeX, Tinos and Arimo installed:

```bash
.venv/bin/python scripts/make_anonymous_figures.py
.venv/bin/python scripts/build_anonymous_manuscript.py
```

These scripts consume compact, recorded result tables and audits. They perform no model fitting or inference. Frozen research source and measurements remain unchanged. Compile logs, downloaded template, PNG previews and page renders stay under ignored `work/manuscript/`.

The generated `.tex` files can also be edited or compiled directly with XeLaTeX inside the extracted anonymous archive. Regenerating them from Markdown overwrites direct TeX edits. The package follows the template's written Letter/25 mm/11 pt settings using metric-compatible local fonts. It is a draft rather than an assertion of final camera-ready certification.

The build enforces an eight-page main limit, a four-page supplement limit and exactly four numbered figure captions in each PDF. It checks page dimensions, author metadata and direct identifying strings. All twelve pages were visually inspected after rendering; source-derived figure values and scientific claims were independently reviewed. Anonymous files contain no author, affiliation, acknowledgements, identifying repository URLs or local filesystem paths. Public literature/dataset citations are retained. Internal notes/manifests are deliberately outside the archive.

Only the anonymous PDFs/source archive are intended as review copies. The rest of this public repository is not anonymized. No conference upload or submission has been performed. The public author guidelines verify the main-paper limit but do not confirm a separate supplemental-upload allowance; see the format notes.
