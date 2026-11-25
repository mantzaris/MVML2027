#!/usr/bin/env python3
"""Create a three-page supplement from actual aggregate measurements."""
import hashlib
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted
from pypdf import PdfReader

sys.path.insert(0,str(Path(__file__).resolve().parent))
from make_results import ROOT, SUPP, CORE, LABEL, load, subset, reference, estimate, paired, hashfile

def main():
    rows=load("summary.csv"); ref=reference(rows)
    manifest_path=ROOT/"results/study_manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError("A completed study manifest is required")
    manifest=json.loads(manifest_path.read_text())
    settings=manifest.get("settings",{})
    for name in ["figS1_training.png","figS2_seed_variation.png"]:
        if not (SUPP/name).is_file(): raise FileNotFoundError(name)
    SUPP.mkdir(exist_ok=True)
    fonts=Path("/usr/share/fonts/truetype/dejavu")
    if (fonts/"DejaVuSans.ttf").is_file():
        pdfmetrics.registerFont(TTFont("Study",str(fonts/"DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("StudyBold",str(fonts/"DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("Study",normal="Study",bold="StudyBold",italic="Study",boldItalic="StudyBold")
        normal,bold="Study","StudyBold"
    else: normal,bold="Helvetica","Helvetica-Bold"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleStudy",fontName=bold,fontSize=19,leading=23,textColor=colors.HexColor("#17283f"),spaceAfter=8))
    styles.add(ParagraphStyle("SubtitleStudy",fontName=normal,fontSize=10,leading=14,textColor=colors.HexColor("#536779"),spaceAfter=12))
    styles.add(ParagraphStyle("HeadStudy",fontName=bold,fontSize=11.5,leading=15,textColor=colors.HexColor("#087f8c"),spaceBefore=10,spaceAfter=6))
    styles.add(ParagraphStyle("BodyStudy",fontName=normal,fontSize=9.2,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle("CaptionStudy",fontName=normal,fontSize=8.4,leading=11.5,textColor=colors.HexColor("#42576c"),spaceAfter=10))
    styles.add(ParagraphStyle("SmallStudy",fontName=normal,fontSize=7.6,leading=10.3,spaceAfter=5))
    story=[]
    def text(t,style="BodyStudy"): story.append(Paragraph(t,styles[style]))
    def head(t): text(t,"HeadStudy")
    def image(name):
        from PIL import Image as PILImage
        with PILImage.open(SUPP/name) as im: w,h=im.size
        story.append(Image(str(SUPP/name),width=499,height=499*h/w))
    def table(data,widths):
        t=Table(data,colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,-1),normal),("FONTNAME",(0,0),(-1,0),bold),
            ("FONTSIZE",(0,0),(-1,-1),8),("LEADING",(0,0),(-1,-1),11),
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8f0f8")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f8fb")]),
            ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#17283f")),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#b6c8d9")),
        ]))
        story.append(t)
    text("Supplementary methods and results","TitleStudy")
    text("GPU study of selective regeneration for cyber planning under deception","SubtitleStudy")
    train=settings.get("train","not recorded");val=settings.get("val","not recorded");test=settings.get("test","not recorded")
    seeds=sorted({int(r["seed"]) for r in rows})
    text(f"This supplement documents a controlled synthetic study with {train:,} training, {val:,} validation, and {test:,} test snapshots per condition, "
         f"and {len(seeds)} independent training seeds ({', '.join(map(str,seeds))})."
         if all(isinstance(n,int) for n in [train,val,test]) else
         f"This supplement documents a controlled synthetic study with {len(seeds)} independently trained seeds. Exact sample counts are recorded in results/study_manifest.json.")
    head("S1  Experimental unit and information boundary")
    text("A sample is a belief-repair snapshot with three partial observers, prior-time evidence, and a hidden current state. "
         "It is not a 64-step operational episode. Deceptive reports and genuine state changes are distinct generator mechanisms. "
         "The same candidate prerequisites, paired test samples, and utility rule are used for every method. Hidden truth supervises training and scores evaluation; it is excluded from the model input.")
    text("B0 is the probabilistic evidence baseline; B1 adapts GraphMAE feature reconstruction [1]; B2 adds decision-aware supervision [2]. "
         "B3 uses HDC cleanup, B4 reconstructs all beliefs, and B5 uses an entropy gate. P combines HDC with the learned decision-impact gate. "
         "Additional controls remove provenance or semantic constraints [3], randomize the gate, or replace HDC with a dense representation.")
    head("S2  Reference-condition measurements")
    data=[["Method","Recall@3","Regret","Brier","GPU ms"]]
    for m in CORE:
        mr=subset(ref,method=m)
        if mr:
            cells=[m]
            for metric in ["recall3","regret","brier","latency_ms"]:
                mean,ci=estimate(mr,metric)
                cells.append(f"{mean:.3f} ± {ci:.3f}")
            data.append(cells)
    table(data,[55,111,111,111,111])
    text("Table S1. Mean and 95% Student-t intervals across training seeds at 20% targeted corruption and the reference communication allowance. "
         "Recall@3 is conditional on a feasible candidate existing. Brier is fact-level. GPU milliseconds measure model inference plus candidate ranking, excluding host transfers and serialization.","CaptionStudy")
    head("S3  Decision-impact teacher")
    story.append(Preformatted("Delta_f = max(0, R(b; U*) - R(b[f <- y_f]; U*))\ng* = top-K strictly positive single-fact regret reductions",
                              ParagraphStyle("EquationStudy",fontName="Courier",fontSize=9,leading=14,backColor=colors.HexColor("#f1f5f9"),borderPadding=8)))
    text("The teacher extends decision-focused prediction by supervising which beliefs to regenerate. It uses simulator truth only during training. "
         "Independent one-fact improvements are an additive approximation; they do not solve interacting repair selection exactly.","CaptionStudy")
    story.append(PageBreak())

    text("Training behavior and implementation","TitleStudy")
    image("figS1_training.png")
    text("Figure S1. Additional empirical training and validation curves. Each line averages available seed records at that epoch; "
         "early stopping can change the number of contributing seeds. Losses contain method-specific terms, so their absolute levels are not cross-method measures of predictive quality.","CaptionStudy")
    head("S4  Optimization and checkpoint selection")
    text(f"The recorded configuration uses a maximum of {settings.get('epochs','unrecorded')} epochs, "
         f"early-stopping patience {settings.get('patience','unrecorded')}, batch size {settings.get('batch','unrecorded')}, "
         f"and learning rate {settings.get('lr','unrecorded')}. The checkpoint is selected on validation data. "
         "Masked-feature pretraining adapts the GraphMAE mechanism; it is not a reproduction of the original published graph benchmark.")
    text("The run metadata stores the actual settings, source revision, input hashes, and observed GPU. "
         "Every figure is generated from aggregate CSV measurements and carries input/output hashes in a provenance sidecar. "
         "Raw traces, generated data, environments, and checkpoints remain local and ignored by Git.")
    head("S5  Timing and memory interpretation")
    text("Research tensor operations execute on the NVIDIA RTX 5000 Ada GPU. Batch-one latency covers CUDA model inference and utility ranking. "
         "These measurements do not include host-side message serialization, device transfer, or an operational planner's full loop. "
         "Reported peak VRAM is the incremental CUDA allocation above the already resident model/data baseline, not total model or machine memory. Training time includes the implemented optimization stage; "
         "a mask that changes fewer facts does not by itself establish sparse-decoder compute savings.")
    text("The main paper figures separate planning quality, uncertainty, communication allowance, and GPU resource measurements. "
         "The measured synthetic costs must not be extrapolated directly to disconnected tactical hardware or realistic cyber missions.")
    story.append(PageBreak())

    text("Seed variability and limits of inference","TitleStudy")
    image("figS2_seed_variation.png")
    text("Figure S2. Additional paired seed-level contrasts between P and B2 by deception mechanism, with feasible-candidate availability shown separately. "
         "Positive recall differences favor P. Availability describes the denominator of recall@3 and is not a model achievement.","CaptionStudy")
    contrasts=np.array(list(paired(ref).values()))
    if len(contrasts):
        mu=contrasts.mean();ci=2.776*contrasts.std(ddof=1)/np.sqrt(len(contrasts)) if len(contrasts)>1 else 0
        head("S6  Primary comparison and uncertainty")
        text(f"At the reference condition, the paired P-minus-B2 recall difference is {mu:+.4f}, with a 95% training-seed interval "
             f"[{mu-ci:+.4f}, {mu+ci:+.4f}]. "
             +("This interval crosses zero; it does not establish a positive recall advantage." if mu-ci<=0<=mu+ci else
               "The observed difference favors P within this fixed synthetic evaluation." if mu-ci>0 else
               "The observed difference favors B2 within this fixed synthetic evaluation."))
    text("Intervals summarize variation over independent model-training seeds conditional on the fixed test pool. "
         "They do not include a hierarchical bootstrap over new topology clusters. The study is a controlled mechanism test, not a claim of deployment readiness, "
         "globally optimal planning, or robustness to arbitrary deception.")
    deviations=manifest.get("deviations",manifest.get("protocol_deviations",[]))
    if deviations:
        head("S7  Recorded scope differences")
        text(" ".join(escape(str(d)) for d in deviations),"SmallStudy")
    else:
        head("S7  Scope")
        text("The empirical study implements the recorded snapshot protocol. Its evidence does not establish longitudinal recovery delays, "
             "full mission performance, or gains in a public cyber range.","SmallStudy")
    head("References")
    text("[1] Hou et al. GraphMAE: Self-Supervised Masked Graph Autoencoders. KDD, 2022. arXiv:2205.10803. "
         "[2] Wilder et al. End to end learning and optimization on graphs. NeurIPS, 2019. "
         "[3] Xu et al. A Semantic Loss Function for Deep Learning with Symbolic Knowledge. ICML, 2018.","SmallStudy")
    text("Submission note. This 3-page supplement is prepared at the requested length. The public MVML author guidelines inspected on November 25, 2026 "
         "state an 8-page full-paper limit but do not explicitly confirm a separate 2-4 page supplement allowance. Confirm the supplement upload category before submission: "
         '<link href="https://mvml.org/author-guidelines/" color="#355bd6">mvml.org/author-guidelines</link>.',"SmallStudy")
    path=SUPP/"Supplement.pdf"
    def footer(c,doc):
        c.saveState();c.setStrokeColor(colors.HexColor("#cad6e2"));c.line(48,40,547,40)
        c.setFont(normal,7.5);c.setFillColor(colors.HexColor("#536779"))
        c.drawString(48,27,"Hyperdimensional cyber regeneration | Supplemental empirical record")
        c.drawRightString(547,27,f"{doc.page}")
        c.restoreState()
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=52,
                          title="Supplementary methods and results",author="Research project")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(str(path))
    if len(reader.pages)!=3: raise RuntimeError(f"Expected 3 pages; rendered {len(reader.pages)}")
    prov={"inputs":{str(p.relative_to(ROOT)):hashfile(p) for p in [ROOT/"results/summary.csv",ROOT/"results/curves.csv",manifest_path]},
          "script_sha256":hashfile(Path(__file__)),"pdf_sha256":hashfile(path),"pages":len(reader.pages),
          "figures":["figS1_training","figS2_seed_variation"],"supplement_policy":"Not explicitly confirmed in public author guidelines"}
    (SUPP/"Supplement.provenance.json").write_text(json.dumps(prov,indent=2)+"\n")
    print(f"Rendered {path} ({len(reader.pages)} pages, {path.stat().st_size} bytes).")

if __name__=="__main__":
    main()
