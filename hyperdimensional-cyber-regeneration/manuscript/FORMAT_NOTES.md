# Internal format and submission-scope notes

This file is an internal preparation record, outside `manuscript/anonymous/`. It is not part of the anonymous paper or supplemental PDF and should not be bundled into an anonymous review upload.

## Verified conference instructions

The official [MVML 2027 author guidelines](https://mvml.org/author-guidelines/) state that full papers are limited to **eight pages**, short papers to **four pages**, and submissions must follow the conference template. LaTeX submissions are accepted. The instructions do not identify an extra reference-page allowance; the main draft therefore targets eight pages including figures and references.

The written instructions in the [official MVML27 Word template](https://mvml.org/documents/MVML27_Paper_Template.docx) specify the following. The downloaded reference copy is retained only in ignored `work/manuscript/MVML27_Paper_Template.docx`.

| Element | Written template instruction |
| --- | --- |
| Page | US Letter, one column, 25 mm margins on all sides |
| Body | Times New Roman, 11 pt, single spacing; first-line indent 7 mm |
| Title | Arial, bold, 16 pt, centered |
| Author information | Names Times New Roman bold 12 pt; affiliations 11 pt; centered |
| Major headings | Arial bold 12 pt |
| Subheadings | Arial bold 11 pt |
| Abstract | One paragraph, at most 300 words, no references; 10 pt Times New Roman; abstract label Arial bold italic 10 pt |
| Keywords | Four to eight |
| Section spacing | One empty 11 pt line between sections; two empty lines before and after the abstract |
| Figures and tables | Numbered and cited in text; centered figure captions below, table captions above |
| Figure lettering | At least 10 pt is recommended, rather than stated as a mandatory minimum |
| Equations | Separate displayed lines, consecutive numbering at the outer right margin |
| References | Numbered IEEE style |

## Template inconsistency and chosen interpretation

The template's prose says **25 mm left/right margins and 11 pt body text**. Its DOCX XML instead stores page margins of 1,440 twips top/bottom (25.4 mm) and 680 twips left/right (approximately 12.0 mm). Its Normal style includes a 24-half-point (12 pt) size. Direct paragraph formatting and style inheritance can further affect individual sample paragraphs.

The draft follows the **written 25 mm margin and 11 pt body instructions**, rather than reproducing those conflicting XML defaults. It does not use the wider text block implied by the 680-twip margins to fit more content. US Letter with 25 mm margins yields an approximately 6.53-inch text width; manuscript figure PDFs were generated at that width.

## Local typography and explicit draft choices

The shared LaTeX style uses **Tinos** for body text and **Arimo** for sans-serif headings. They are metrically compatible substitutes, respectively, for Times New Roman and Arial according to the font packages' primary documentation: [Tinos](https://ctan.org/pkg/tinos) and [Arimo](https://ctan.org/pkg/arimo). Metric compatibility does not make their outlines identical to the named template fonts or establish organizer approval of substitution.

The style preserves the principal requested sizes: **11 pt body**, **16 pt bold main title**, **12 pt bold section headings**, **11 pt bold subsections**, and **10 pt abstract**. The body uses ordinary single-spaced leading rather than double spacing. Mathematical notation uses a Times-compatible math font. The supplemental heading uses the shared section/title hierarchy for a companion document.

This is an anonymous research draft, not a claim of exact camera-ready template certification. Compact float/equation spacing, justified captions, and compact tables are draft layout choices. Empirical figures use a fixed 6.53-inch canvas and readable 7.5–9 pt lettering after placement, which is below the template's recommended 10 pt figure lettering. Their minimum visible font size, dimensions, source hashes, and output hashes are recorded in `figure_provenance.json`; internal preview PNGs remain under ignored `work/manuscript/figure_review/`. These choices should remain visible in any final formatting review.

## Anonymity and separate supplement

The author-guideline page inspected during this preparation does **not** state an anonymous-review or double-blind-review requirement. Anonymity is an explicit **user-requested draft choice**, not a verified conference policy. Author names, affiliations, acknowledgements, identifying repository links, local filesystem paths, and identifying PDF metadata are excluded from the anonymous deliverables. Public research references and dataset citations remain necessary scientific attribution.

The project targets an **eight-page main manuscript with four empirical figures** and a **separate four-page supplemental PDF with four additional empirical figures**. The public author-guideline page inspected does **not** describe a supplemental upload category, supplemental page limit, or permission to exceed the main-paper limit through a supplement. Therefore, the separately prepared supplemental PDF must not be described as an officially confirmed MVML allowance. Producing this requested companion draft does not assert that the submission system will accept it. Actual PDF page counts are verified in the final build, independently of these packaging targets.

## Evidence and administrative scope

The paper centers on machine learning for captured cyber telemetry: GraphMAE-adapted baselines, two observation-aware variants, matched controls, calibration, artificial corruption, and measured Ada-GPU costs. Synthetic planning and replanning studies provide supplemental context. Captured-flow classification is not presented as demonstrated operational cyber course-of-action capability, and artificial feature corruption is not presented as observed adversarial deception.

The CTU-13 follow-up reuses previously inspected test captures and remains exploratory. Descriptive training-seed intervals in figures are distinguished from recorded paired seed/time-block or seed/topology bootstrap intervals. No funding, sponsor endorsement, export-control certification, award eligibility, conference acceptance, or approval of supplemental material is asserted by formatting this draft. No organizer contact, conference upload, or submission has been performed as part of this preparation.
