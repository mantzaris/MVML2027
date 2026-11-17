MANUSCRIPT 2 — REVISED NARRATIVE AND MVML 2027 SUBMISSION FORMAT

This sibling revision explains the motivation, approach, findings and practical
implications of the underwater acoustic reasoning study more directly. The
abstract leads with the problem and qualitative findings. The introduction
connects prior work to this study, and the conclusion identifies what the
evaluation reveals and how it could inform an acoustic assistant.

The original manuscript directory is unchanged. All three tables retain their
values, all three figure image files are unchanged, and all thirteen reference
entries are retained. Narrative and captions were condensed to fit the required
eight pages after correcting margins, author information and reference size.
Implementation details remain in the linked repository. The primary result,
uncertainty and material limitations remain explicit. No experiments were rerun.

FILES

manuscript.docx        Editable Word manuscript.
manuscript.pdf         Checked eight-page PDF, including references.
manuscript.txt         Plain text, including tables and equation text.
prose.json             Authoritative replacement prose.
submission.json        Supplied authors, affiliations, emails and repository URL.
submission_format.py  Applies the conference layout and clickable repository link.
build_manuscript.py    Rebuilds the Word file and plain text.
references.bib         Bibliography retained from the original manuscript.
revision_check.json    Content, package, font and layout verification record.
template/             Retained source manuscript and current official template.

EDITING AND REBUILDING

Edit prose.json for wording and submission.json for author details, then run:

    python -B build_manuscript.py

The builder requires Python 3.10+ and lxml. It checks the retained source hash
and paragraph hashes before editing, then applies the layout rules. It changes
table/figure geometry to fit the text area, but retains table values and image
data. It does not run models or access the network.

Rebuilding replaces manuscript.docx and manuscript.txt. Transfer direct Word
edits into the source files before rebuilding or they will be lost. The builder
does not update the PDF. Export again after every revision, with Arial and Times
New Roman available, and inspect all pages. The checked PDF uses those actual
embedded fonts for headings and text, plus embedded mathematical/symbol fonts.
The current export used the Codex document renderer and bundled LibreOffice.
Pagination can vary with another Word processor; use the checked PDF to assess
the eight-page limit. The conclusion is kept together.

FORMAT CHECKED 16 NOVEMBER 2026

Source: https://mvml.org/documents/MVML27_Paper_Template.docx
Rules: https://mvml.org/author-guidelines/
Organizer FAQ: https://www.international-aset.com/FAQ/

US Letter; 25 mm margins on every side; single-spaced Times New Roman 11 pt text;
7 mm first-line paragraph indents; centred Arial bold 16 pt title; centred bold
12 pt author names and 11 pt affiliations; Arial bold 12 pt section headings;
10 pt abstract and captions; 11 pt tables and references. Displayed equations
are editable Word equations, numbered at the right margin. Captions remain
above tables and below figures. Two 11 pt lines of space surround the abstract.
The abstract has 203 words, within the 300-word limit.

The downloaded template's actual side margins disagree with its written
instruction. This version follows the explicit 25 mm instruction, rather than
its approximately 12 mm side-margin setting. The current official template is
retained locally so this choice can be reviewed.

The first-page header now names EECSS 2027, August 15–17, Prague. The draft header
is removed. Ordinary page numbers are used for the initial submission. The
template says the paper code is assigned after acceptance. No code or DOI has
been invented. Enter the assigned code in submission.json and add the assigned
DOI before preparing the accepted camera-ready version.

AUTHORSHIP AND ANONYMITY

The official submission template includes author names, affiliations and email
addresses. The organizer FAQ requires the corresponding author's institutional
email in the paper. No anonymous or double-blind requirement was found in the
public MVML guidelines or organizer FAQ. The manuscript therefore includes the
author details supplied by the user. This conclusion concerns the published
instructions; private submission-form instructions were not inspected.

Authors: Alexander V. Mantzaris and James H. Korris.
Corresponding institutional contact: alexander.mantzaris@ucf.edu.
Affiliations and contact emails are recorded in submission.json. The supplied
street abbreviation was normalized to “Blvd”; no company address was invented.

CODE AVAILABILITY

https://github.com/mantzaris/MVML2027

The manuscript includes this clickable public repository URL. The repository
contains the project in evidence-genAI-underwater-acoustic-reasoning. The
availability statement distinguishes code, protocols and result records from
large audio and model artifacts stored separately.

This work prepares local submission files. No paper was submitted and no GitHub
commit or push was made.
