"""Prepare an explicitly incomplete draft within the downloaded official DOCX."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
template = ROOT / ".cache/sources/mvml_template.docx"
if not template.exists():
    raise SystemExit("Run inspect_sources.py to obtain the official template first")
out = ROOT / "manuscript"
out.mkdir(exist_ok=True)
(out / "template").mkdir(exist_ok=True)
shutil.copyfile(template, out / "template/MVML27_Paper_Template.docx")
doc = Document(template)
prototype = {"title": deepcopy(doc.paragraphs[1]._p), "author": deepcopy(doc.paragraphs[3]._p),
             "body": deepcopy(doc.paragraphs[17]._p), "heading": deepcopy(doc.paragraphs[16]._p)}
body = doc._element.body
for node in list(body):
    if node.tag.endswith("}sectPr"):
        continue
    body.remove(node)


def add(text, kind="body"):
    node = deepcopy(prototype[kind])
    body.insert(len(body) - 1, node)
    p = Paragraph(node, doc._body)
    p.text = text
    for run in p.runs:
        run.font.name = "Arial" if kind in {"title", "heading"} else "Times New Roman"
        run.font.size = Pt({"title": 16, "heading": 12, "author": 12, "body": 11}[kind])
        run.font.bold = kind in {"title", "heading", "author"}
    if kind in {"title", "author"}:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if text.startswith("Abstract"):
        for run in p.runs: run.font.size = Pt(10)
    return p


add("Evidence-Grounded Generative AI for Underwater Acoustic Reasoning", "title")
add("Alexander V. Mantzaris", "author")
add("Affiliation and contact details pending. INCOMPLETE RESEARCH DRAFT — 15 November 2026.")
add("Abstract — This prospective study asks whether a small learned controller can select acoustic analyses that improve the consistency of a frozen audio-language model's bounded reports. The proposed task covers civilian vessel class, received energy bands, persistent narrowband components, envelope modulation, and received energy trends. Following the project owner's instruction, reference assessment will use separate signal-processing estimates and AI reviewer roles, with no human reviewers. These scores will be reported as automated proxies. The preparation stage verified access to metadata and five training examples from UniqueShip and identified discrepancies in sample-rate documentation, missing recording provenance, and an infeasible class-balanced reference-sampling target. No acoustic classifier, controller, or generative comparison has been run. This draft specifies the intended comparison and records the unresolved prerequisites; it does not establish improved reasoning, human-validated explanations, or operational underwater sensing performance.")
add("Keywords: underwater acoustics; evidence acquisition; generative audio; automated assessment; uncertainty; reproducibility")
add("1. Research Question and Related Work", "heading")
add("The learning question is whether predicting the value of additional measured evidence can improve useful reporting under a fixed analysis budget. Learned audio-tool selection and differential reward are already present in AuTAgent [1]. Audio-Mind [2] records conditional evidence acquisition; SEAR/BAEA [3] uses fixed and adaptive measured evidence with a frozen model; ACE-Cap [4] includes active acquisition and counterfactual credit. Our intended distinction is a controlled study of bounded acoustic claims and a small observation-dependent risk estimator. Removing human reviewers narrows the claim to automated evidence consistency, with substantial overlap that must be addressed empirically.")
add("2. Planned Method", "heading")
add("An acoustic classifier will supply an embedding and class prior. Four available actions measure spectral structure, envelope modulation, temporal energy, or retrieve training-only exemplars. A three-member small MLP ensemble will predict current terminal loss and the change after each unobserved action. The selector will choose the highest predicted gain after uncertainty and cost penalties, then update its state with only acquired observations. The primary budget is two actions; the terminal Qwen2.5-Omni model remains frozen and is called once per report.")
add("The proposed loss combines class error with four physical-field consistency losses. Correct and supported assertions receive zero loss, incorrect or unsupported assertions receive one, and abstentions receive 0.25; explicit uncertainty is appropriate for ambiguous references. Under the revised AI-only assessment design, these are proxy labels rather than independently established perceptual truth. Separate reference estimators and blinded AI reviewer prompts will be checked on known signals. AI agreement will not be described as human or expert agreement.")
add("3. Data Audit and Protocol Amendments", "heading")
add("UniqueShip [5] provides public civilian hydrophone recordings and vessel-grouped folds. The linked small-release directory and multipart archive were accessible. CRC-checked range extraction obtained fold-1 metadata and five training samples, not the full corpus. The selected classes contain 14,372 training, 1,381 validation, and 2,250 test records before exclusions. All inspected metadata and sample headers indicate 32 kHz mono audio, contrary to the landing page's 20 kHz description. Native analysis must preserve 32 kHz, with explicitly resampled 20 kHz encoder inputs.")
add("Official vessel and background-date identifiers are disjoint across fold 1, but required parent recording and deployment identifiers are absent. All background hydrophone fields are missing. These fields must be recovered or an adequate conservative protocol amendment validated before the main evaluation. The planned balanced 240-clip assessment set also fails the three-clips-per-source cap: folds contain only 5–13 passenger test vessels, below the 16 required for 48 passenger clips. No model outcomes were examined to discover these constraints.")
add("4. Preparation Checks and Planned Evaluation", "heading")
add("CPU preparation includes strict source/time/hash checks, masked observed-tool states, training-only distinct-source retrieval, claim-metric denominators, paired source-cluster resampling, and atomic result reuse. Twenty-four focused software tests passed. Five hundred synthetic tone, amplitude-modulation, chirp, noise, and silence fixtures passed their declared tool checks, together with 100 structural evidence checks. These are implementation checks only. A valid evidence ID with a false numeric value intentionally passes the structural checker, demonstrating why it cannot establish semantic truth.")
add("Empirical result tables and risk/coverage figures remain unavailable. Planned comparisons include the strongest development-selected fixed/heuristic selector, an accuracy-only learned selector, a one-shot ablation, all tools, raw audio, and a deterministic report. The primary automated proxy endpoint will be supported coverage at a calibration-targeted 70% report coverage and budget two, with unsupported-claim rate as a guardrail. Source-cluster paired intervals and separate controller/classifier seed variation will be reported. Citation completeness and support are distinct from correctness, consistent with ALCE's evaluation motivation [6].")
add("5. Limitations and Execution Status", "heading")
add("The study has no human validation. AI reviewer agreement, especially with shared model families, may reflect common biases; numeric estimator agreement may reflect common signal-processing errors. Civilian AIS-associated labels, one geographic region, five-second excerpts, sparse passenger sources, and unresolved provenance limit conclusions. No submarine identity, localization, propulsion parameters, or source-motion claims are supported. Low-cost DSP selection does not itself establish GPU savings. The current user authorization is at most 10 GPU-hours on an existing 32 GB GPU with approximately 50 GB ephemeral container disk. Artifact preservation must be verified before research jobs. This draft is incomplete and not ready for submission.")
add("References", "heading")
records = {r["key"]: r["fields"] for r in json.loads((ROOT / "sources/bibliography.json").read_text())}
for i, key in enumerate(["autagent", "audio_mind", "sear", "ace_cap", "uniqueship", "alce"], 1):
    f = records[key]
    authors = f["author"].replace(" and ", "; ")
    venue = f.get("booktitle", f.get("journal", "arXiv preprint " + f.get("eprint", "")))
    p = add(f"[{i}] {authors}, “{f['title']},” {venue}, {f['year']}. {f['url']}")
    for run in p.runs: run.font.size = Pt(10)
for section in doc.sections:
    for container in (section.header, section.first_page_header, section.footer, section.first_page_footer):
        for element in container._element.iter():
            if element.tag.endswith("}t") and element.text:
                element.text = element.text.replace("XXX", "DRAFT").replace("DOI: TBA", "DOI: not assigned (draft)")
    # Rebuild draft footers: the supplied template includes a rendered field/shape
    # that some Word-compatible PDF renderers retain with the original XXX label.
    for footer in (section.footer, section.first_page_footer, section.even_page_footer):
        for element in list(footer._element): footer._element.remove(element)
        p = footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run("MVML DRAFT-")
        run.font.name = "Times New Roman"; run.font.size = Pt(10)
        page = OxmlElement("w:fldSimple"); page.set(qn("w:instr"), "PAGE")
        p._p.append(page)
doc.core_properties.title = "Evidence-Grounded Generative AI for Underwater Acoustic Reasoning"
doc.core_properties.author = "Alexander V. Mantzaris"
doc.core_properties.subject = "Incomplete preparation draft; no model experiment results or human review"
doc.save(out / "manuscript_draft.docx")
print("Created incomplete manuscript draft in the official DOCX geometry")
