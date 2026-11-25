#!/usr/bin/env python3
"""Create a separate three-page empirical supplement for the second study."""
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Image, Spacer
from pypdf import PdfReader

sys.path.insert(0,str(Path(__file__).resolve().parent))
from make_v2_results import ROOT, DATA, SUPP, load, core, subset, estimate, sha


def main():
    manifest_path=DATA/"manifest.json"
    comparison_path=DATA/"comparisons.json"
    manifest=json.loads(manifest_path.read_text())
    comparison=json.loads(comparison_path.read_text())
    settings=manifest["settings"]
    rows=load("summary.csv"); cc=core(rows)
    if sorted({int(r["seed"]) for r in rows}) != [11,23,37,51,71]:
        raise ValueError("Complete five-seed measurements required")
    primary=comparison["primary"]
    if primary["method"]!="JH" or primary["comparator"]!="ID" or primary["metric"]!="macro_regret":
        raise ValueError("Unexpected primary comparison")
    for stem in ["v2_figS1_training","v2_figS2_seed_effects"]:
        prov=json.loads((SUPP/f"{stem}.provenance.json").read_text())
        if sha(ROOT/prov["script"]) != prov["script_sha256"]:
            raise ValueError("Supplementary figure source changed; regenerate plots")
        for relative,digest in prov["inputs"].items():
            if sha(ROOT/relative)!=digest: raise ValueError("Stale supplementary figure inputs")
        if sha(SUPP/f"{stem}.png")!=prov["output_sha256"]["png"]:
            raise ValueError("Supplementary image hash mismatch")
    SUPP.mkdir(parents=True,exist_ok=True)
    fonts=Path("/usr/share/fonts/truetype/dejavu")
    if (fonts/"DejaVuSans.ttf").exists():
        pdfmetrics.registerFont(TTFont("StudyV2",str(fonts/"DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("StudyV2Bold",str(fonts/"DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("StudyV2",normal="StudyV2",bold="StudyV2Bold",italic="StudyV2",boldItalic="StudyV2Bold")
        normal,bold="StudyV2","StudyV2Bold"
    else:
        normal,bold="Helvetica","Helvetica-Bold"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleV2",fontName=bold,fontSize=18,leading=22,textColor=colors.HexColor("#17283f"),spaceAfter=8))
    styles.add(ParagraphStyle("SubtitleV2",fontName=normal,fontSize=10,leading=14,textColor=colors.HexColor("#536779"),spaceAfter=11))
    styles.add(ParagraphStyle("HeadV2",fontName=bold,fontSize=11.2,leading=15,textColor=colors.HexColor("#007f72"),spaceBefore=10,spaceAfter=6))
    styles.add(ParagraphStyle("BodyV2",fontName=normal,fontSize=9,leading=12.8,spaceAfter=7))
    styles.add(ParagraphStyle("CaptionV2",fontName=normal,fontSize=8.2,leading=11.3,textColor=colors.HexColor("#42576c"),spaceAfter=8))
    styles.add(ParagraphStyle("SmallV2",fontName=normal,fontSize=7.5,leading=10.2,spaceAfter=5))
    styles.add(ParagraphStyle("EquationV2",fontName=normal,fontSize=10.5,leading=17,backColor=colors.HexColor("#eef5f5"),borderPadding=8,spaceBefore=8,spaceAfter=13))
    story=[]
    def text(value,style="BodyV2"): story.append(Paragraph(value,styles[style]))
    def head(value): text(value,"HeadV2")
    def image(stem):
        from PIL import Image as PILImage
        path=SUPP/f"{stem}.png"
        with PILImage.open(path) as im: width,height=im.size
        story.append(Image(str(path),width=499,height=499*height/width))
    def equation():
        import matplotlib.pyplot as plt
        path=ROOT/"work/v2-equation.png"
        fig=plt.figure(figsize=(6.9,.78),facecolor="#eef5f5")
        fig.text(.025,.62,r"$S_j^{\mathrm{dep}}(x)=\prod_{r\in A_j}a_r(x)\,\prod_{f\in R_j,\ f\ne p(f)}a_f(x)$",fontsize=12)
        fig.text(.025,.12,r"$\widehat U_j=v_j S_j^{\mathrm{dep}}-c_j$",fontsize=12)
        fig.savefig(path,dpi=400,facecolor=fig.get_facecolor())
        plt.close(fig)
        story.append(Image(str(path),width=499,height=499*.78/6.9))
        story.append(Spacer(1,8))
    def table(data,widths):
        t=Table(data,colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,-1),normal),("FONTNAME",(0,0),(-1,0),bold),
            ("FONTSIZE",(0,0),(-1,-1),8),("LEADING",(0,0),(-1,-1),11),
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5f0ee")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f8f7")]),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#b4cbc6")),
        ]));story.append(t)

    text("Second-study supplementary record","TitleV2")
    text("Dependency-factored robust regeneration under deceptive observations","SubtitleV2")
    text("The first study exposed poor transfer of selective repair and substantial regret from its fixed repair budget. "
         "This follow-up regenerates all beliefs, trains across four observation groups, and scores shared prerequisites using a conditional probability model. "
         "The old test results informed the design; all methods are evaluated here on a fresh held-out pool (generator seed 42027).")
    head("S1  Conditional regeneration and candidate scoring")
    text("For each root, the network predicts its readiness probability; for each child, it predicts readiness conditional on a ready parent. "
         "A child is false when its parent is false. For candidate j, R<sub>j</sub> contains required facts and A<sub>j</sub> contains the distinct roots touched by those facts. "
         "Each shared root is counted once:")
    equation()
    text("This is exact under the modeled root-child factorization, not a guarantee that the true posterior factorizes under deception. "
         "Probabilistic symbolic semantics and group robustness build on established work [1-3]; the task-specific composition and its empirical effects are investigated here.","CaptionV2")
    head("S2  Four-condition macro measurements")
    data=[["Method","Regret","Recall@3","Plan Brier","Fact Brier"]]
    for m in ["B2","P","ID","IH","JD","JH_ERM","JH"]:
        r=subset(cc,method=m)
        values=[m]+[f"{estimate(r,k)[0]:.4f}" for k in ["regret","recall3","plan_brier","brier"]]
        data.append(values)
    table(data,[77,105.5,105.5,105.5,105.5])
    text("Table S1. Means over five training seeds and four equally weighted conditions: benign observations plus targeted, random, and coherent interventions at requested rate 0.2. "
         "Recall is conditional on a feasible candidate. All conditions use 64 facts and the same sparse report codec.","CaptionV2")
    text("ID/IH are dense/HDC independent reconstructions; JD/JH use the conditional model. JH_ERM changes robust group aggregation to a mean. "
         "B2 and P are the original full and selective models, reevaluated with their frozen checkpoints; their training exposure differs from the new models.","SmallV2")
    story.append(PageBreak())

    text("Optimization and fair comparisons","TitleV2")
    image("v2_figS1_training")
    text("Figure S1. Additional training and validation optimization traces. Epoch-wise curves average available seeds; early stopping changes the contributing cohort. "
         "Training objectives differ in group aggregation and are not direct cross-method quality measures. Old checkpoints are reused, not retrained in the second study.","CaptionV2")
    head("S3  Frozen data and optimization protocol")
    text(f"All five new variants use {settings['train']:,} training and {settings['val']:,} validation worlds, "
         f"{settings['pretrain']} masked-feature pretraining epochs, at most {settings['epochs']} supervised epochs, patience {settings['patience']}, "
         f"batch size {settings['batch']}, and learning rate {settings['lr']}. "
         "The same worlds receive balanced benign, targeted, random, and coherent observation conditions. Attack levels 0.1, 0.2, and 0.4 are represented in training. "
         "These familiar attack families are not unseen-mechanism tests.")
    text("The common loss combines fact BCE, candidate Brier, soft decision regret (weight 0.25), and squared positive child-parent inconsistency (weight 0.1). "
         "JH, JD, IH, and ID use smooth maximum group risk with β=2; JH_ERM uses the mean. The same mean validation-group loss selects all checkpoints. "
         "Group labels and hidden truth are used for supervision only.")
    head("S4  Controls that isolate the changes")
    text("The primary comparison is JH versus ID: both regenerate all facts and use identical mixed observations, training budget, and loss weights. "
         "JH versus JD isolates HDC features; JH versus IH tests the conditional-model package; JH versus JH_ERM tests group aggregation. "
         "JH_product keeps the JH checkpoint and coherent fact marginals but replaces dependency-aware candidate scoring with their product. "
         "Comparing only JH with the original sparse P would confound architecture, training exposure, and repair allowance.")
    head("S5  Runtime and retained-state scope")
    text(f"Training and inference execute on {escape(str(manifest['hardware']))}. The recorded study took {manifest['wall_seconds']/60:.1f} minutes. "
         "CUDA latency covers warmed batch-one inference plus candidate ranking. Incremental peak allocation excludes resident model/data and cached codebooks; it is not total deployment memory. "
         "Full regeneration does not claim sparse-decoder savings or a fixed six-fact repair budget.")
    pipeline_path=DATA/"pipeline_timing.json"
    if pipeline_path.is_file():
        text("A separate prepared-evidence wall-clock audit records serialization, host/device transfers, feature reconstruction, inference, and returned rankings in results/v2/pipeline_timing.json. "
             "It remains distinct from sensing, network transit, or a full operational planning loop.","CaptionV2")
    text("Raw arrays, checkpoints, logs, and environment files remain local and ignored. Committed manifests record seeds, settings, source and data hashes; every figure carries input and output digests.","SmallV2")
    story.append(PageBreak())

    text("Paired effects and inference limits","TitleV2")
    image("v2_figS2_seed_effects")
    text("Figure S2. Additional per-seed JH-minus-ID regret differences by condition and incremental GPU allocation. Negative regret differences favor JH. "
         "The same synthetic worlds are paired across methods, seeds, and the four core observation conditions.","CaptionV2")
    head("S6  Prespecified primary contrast")
    lo,hi=primary["percentile_95_ci"]
    conclusion=("The interval favors JH in this controlled setting." if hi<0 else
                "The interval favors ID in this controlled setting." if lo>0 else
                "The interval includes zero; it establishes neither a gain nor equivalence.")
    text(f"JH mean macro regret is {primary['means']['JH']:.5f}, versus {primary['means']['ID']:.5f} for ID. "
         f"The JH-minus-ID effect is {primary['estimate']:+.5f}, with 95% interval [{lo:+.5f}, {hi:+.5f}]. {conclusion}")
    text(f"The interval uses {comparison['bootstrap_repetitions']:,} paired crossed bootstrap draws over five training seeds and "
         f"{comparison['topology_clusters']} topology clusters. Cluster draws are shared across methods, seeds, and all four fixed conditions. "
         "Main-figure error bars instead summarize training-seed variation conditional on this test pool. Secondary intervals are descriptive and not multiplicity-adjusted.")
    eligible=comparison['eligible_per_condition'];empty=comparison['no_feasible_per_condition']
    text(f"Each core condition retains {eligible:,} eligible snapshots and {empty:,} without a feasible candidate. "
         "Only conditional recall excludes the latter; regret and probability scores retain them. The bootstrap does not cover uncertainty in the generator design or the shared finite training/validation sample.","CaptionV2")
    head("S7  Limits of the second approach")
    text("The structural guarantee covers the supplied one-level prerequisite forest. It does not capture arbitrary posterior correlations, prove reports truthful, or implement sequential cyber operations. "
         "Larger tests change both fact count and root density. Coherent overwrites can leave already-positive reports unchanged, so intervention rates are not equal report-error rates. Every method shares a sparse symbolic report codec; an HDC communication-compression benefit remains untested. "
         "Both favorable and unfavorable controls remain visible in the main figures and full results report.","SmallV2")
    head("Primary-source lineage")
    text("[1] Manhaeve et al. DeepProbLog: Neural Probabilistic Logic Programming. NeurIPS, 2018. "
         "[2] Ahmed, Chang, and Van den Broeck. Semantic Strengthening of Neuro-Symbolic Learning. AISTATS, 2023. "
         "[3] Sagawa et al. Distributionally Robust Neural Networks for Group Shifts. ICLR, 2020.","SmallV2")
    text("Submission note. This separate three-page supplement is prepared at the requested length. Public MVML author guidelines inspected for this project do not explicitly confirm a separate supplemental upload allowance: "
         '<link href="https://mvml.org/author-guidelines/" color="#007f72">mvml.org/author-guidelines</link>.',"SmallV2")
    path=SUPP/"Supplement_v2.pdf"
    def footer(c,doc):
        c.saveState();c.setStrokeColor(colors.HexColor("#cadbd6"));c.line(48,40,547,40)
        c.setFont(normal,7.3);c.setFillColor(colors.HexColor("#536779"))
        c.drawString(48,27,"Dependency-factored regeneration | Second-study supplemental record")
        c.drawRightString(547,27,str(doc.page));c.restoreState()
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=52,
                      title="Second-study supplementary record",author="Research project").build(story,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(path).pages)
    if pages!=3: raise RuntimeError(f"Expected three pages, produced {pages}")
    inputs=[DATA/"summary.csv",DATA/"curves.csv",manifest_path,comparison_path,
            SUPP/"v2_figS1_training.png",SUPP/"v2_figS2_seed_effects.png"]
    if pipeline_path.is_file(): inputs.append(pipeline_path)
    provenance={"inputs":{str(p.relative_to(ROOT)):sha(p) for p in inputs},"script_sha256":sha(Path(__file__)),
                "pdf_sha256":sha(path),"pages":pages,"figures":["v2_figS1_training","v2_figS2_seed_effects"],
                "supplement_policy":"Separate supplemental allowance not explicitly confirmed in public instructions"}
    (SUPP/"Supplement_v2.provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(f"Created {path.relative_to(ROOT)}: {pages} pages, {path.stat().st_size} bytes")


if __name__=="__main__":
    main()
