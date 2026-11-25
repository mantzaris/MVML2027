#!/usr/bin/env python3
"""Build the separate three-page third-study supplement from final artifacts."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_v3_results import ROOT, DATA, SUPP, NEW, SEEDS, load, core, subset, estimate, sha, validate


def main():
    manifest_path = DATA / "manifest.json"
    comparison_path = DATA / "comparisons.json"
    manifest = json.loads(manifest_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    rows = load("summary.csv")
    validate(rows, manifest)
    cc = core(rows)
    settings = manifest["settings"]
    primary = comparison["primary"]
    if (primary["method"], primary["comparator"], primary["metric"]) != ("SCR", "ID", "macro_regret"):
        raise ValueError("Unexpected prespecified primary contrast")
    for stem in ["v3_figS1_training", "v3_figS2_seed_effects"]:
        provenance = json.loads((SUPP / f"{stem}.provenance.json").read_text())
        if sha(ROOT / provenance["script"]) != provenance["script_sha256"]:
            raise ValueError("Supplementary plot script changed; regenerate figures")
        for relative, digest in provenance["inputs"].items():
            if sha(ROOT / relative) != digest:
                raise ValueError("Supplementary figure inputs changed")
        if sha(SUPP / f"{stem}.png") != provenance["output_sha256"]["png"]:
            raise ValueError("Supplementary image hash mismatch")
    SUPP.mkdir(parents=True, exist_ok=True)
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    if (fonts / "DejaVuSans.ttf").exists():
        pdfmetrics.registerFont(TTFont("StudyV3", str(fonts / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("StudyV3Bold", str(fonts / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("StudyV3", normal="StudyV3", bold="StudyV3Bold", italic="StudyV3", boldItalic="StudyV3Bold")
        normal, bold = "StudyV3", "StudyV3Bold"
    else:
        normal, bold = "Helvetica", "Helvetica-Bold"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleV3", fontName=bold, fontSize=17.5, leading=22, textColor=colors.HexColor("#17283f"), spaceAfter=8))
    styles.add(ParagraphStyle("SubtitleV3", fontName=normal, fontSize=9.8, leading=13.3, textColor=colors.HexColor("#536779"), spaceAfter=10))
    styles.add(ParagraphStyle("HeadV3", fontName=bold, fontSize=10.8, leading=14, textColor=colors.HexColor("#a32b55"), spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle("BodyV3", fontName=normal, fontSize=8.7, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle("CaptionV3", fontName=normal, fontSize=8, leading=11, textColor=colors.HexColor("#42576c"), spaceAfter=7))
    styles.add(ParagraphStyle("SmallV3", fontName=normal, fontSize=7.5, leading=10.2, spaceAfter=5))
    story = []

    def text(value, style="BodyV3"):
        story.append(Paragraph(value, styles[style]))

    def head(value):
        text(value, "HeadV3")

    def image(stem):
        from PIL import Image as PILImage
        path = SUPP / f"{stem}.png"
        with PILImage.open(path) as im:
            width, height = im.size
        story.append(Image(str(path), width=499, height=499 * height / width))

    def equation():
        import matplotlib.pyplot as plt
        path = ROOT / "work/v3-equation.png"
        fig = plt.figure(figsize=(7.3, 1.1), facecolor="#f9eef3")
        fig.text(.02, .71, r"$u_j=[\mu_{\mathcal{C}_j}h,\ \sigma_{\mathcal{C}_j}h,\ \mu_{\mathcal{A}_j}h,\ \log(1+|\mathcal{C}_j|),\ \log(1+|\mathcal{A}_j|)]$", fontsize=11)
        fig.text(.02, .21, r"$\widehat{s}_j(x)=\sum_{k=1}^{4}\mathrm{softmax}_k(a(g))\,\sigma(\rho_k([u_j,g])),\qquad\widehat{U}_j=v_j\widehat{s}_j-c_j$", fontsize=11.5)
        fig.savefig(path, dpi=400, facecolor=fig.get_facecolor())
        plt.close(fig)
        story.append(Image(str(path), width=499, height=499 * 1.1 / 7.3))
        story.append(Spacer(1, 7))

    def table(data, widths):
        grid = Table(data, colWidths=widths, repeatRows=1)
        grid.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), normal), ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 11),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4e5ed")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#faf5f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#dbb6c7")),
        ]))
        story.append(grid)

    text("Third-study supplementary record", "TitleV3")
    text("Direct candidate inference and lightweight denoising models under deception", "SubtitleV3")
    text("Earlier studies reconstructed fact beliefs before scoring candidate courses of action. This follow-up tests a different prediction target: directly estimating candidate feasibility from canonical prerequisite sets and pooled evidence. "
         "A denoising autoencoder and logistic model test whether inexpensive belief reconstruction is competitive. All 22 methods are reevaluated on fresh test worlds (generator seed 62027); the first two studies remain unchanged.")
    head("S1  Shared-context candidate ranking equation")
    text("Let h<sub>f</sub> encode a fact's observed features and its parent's features. C<sub>j</sub> contains required nonroot facts; A<sub>j</sub> contains their distinct touched roots plus explicitly required roots. "
         "Means and standard deviations pool these sets; g pools all fact embeddings. A common gate mixes four candidate heads:")
    equation()
    text("SCR composes permutation-invariant set learning [1] with a shared mixture of experts [2]. Canonicalization removes redundant required roots and the empty conjunction has probability one. "
         "The latent regimes are not identified attack labels. This is a task-specific methodological hypothesis, not a claim to invent set learning or mixtures; candidate predictions need not form a globally coherent fact posterior.", "CaptionV3")
    head("S2  Four-condition macro measurements")
    data = [["Method", "Regret", "Recall@3", "Plan Brier", "Fact Brier"]]
    for method in ["B2", "ID", "JH", "LOGIT", "DAE", "CAND_LOCAL", "CAND_CTX", "SCR"]:
        selected = subset(cc, method=method)
        values = [method] + [f"{estimate(selected, metric)[0]:.4f}" for metric in ["regret", "recall3", "plan_brier"]]
        values.append("N/A" if method in ["CAND_LOCAL", "CAND_CTX", "SCR"] else f"{estimate(selected, 'brier')[0]:.4f}")
        data.append(values)
    table(data, [105, 98.5, 98.5, 98.5, 98.5])
    text("Table S1. Means over five training seeds and equally weighted benign, targeted, random, and coherent core conditions. The three interventions request rate 0.2; all use 64 facts, 20% missing observations and a common sparse report codec. "
         "Recall is conditional on a feasible candidate. N/A means no fact posterior exists; it does not mean zero error.", "CaptionV3")
    story.append(PageBreak())

    text("Optimization and lightweight controls", "TitleV3")
    image("v3_figS1_training")
    text("Figure S1. Additional recorded optimization traces for the five new models. Curves average all five training seeds at every epoch; all 25 new models reached the 40-epoch cap. "
         "Candidate and belief objectives differ, so their numeric levels are not directly comparable. DAE pretraining traces remain in the recorded curves file.", "CaptionV3")
    head("S3  Frozen training and architectural controls")
    text(f"Each new method receives {settings['train']:,} training and {settings['val']:,} validation worlds, five seeds, balanced benign/targeted/random/coherent observation groups, and common batch size {settings['batch']}. "
         "The maximum supervised budget is 40 epochs with patience 8; validation selects checkpoints without consulting the fresh test pool. Every new model reached that cap, making this a fixed-budget comparison rather than a convergence claim. Familiar intervention families occur in training at rates 0.1, 0.2 and 0.4; this is not an unseen-family evaluation.")
    text("SCR uses 32-dimensional fact embeddings, a 98-dimensional canonical candidate summary, and 64-dimensional global context. CAND_CTX uses one head with global context; CAND_LOCAL has the identical head dimensions and parameter count but zeros the global vector. "
         "The context comparison is matched in parameter count; the four-expert versus one-head comparison also changes capacity. Candidate supervision combines feasibility BCE, candidate Brier, and soft decision regret (weight 0.25). Shared input features include catalogue-value-weighted criticality; only candidate-specific value and cost tensors are absent from feasibility heads.")
    head("S4  Autoencoder and logistic alternatives")
    text("DAE builds a 56-dimensional local/parent/global observed-feature vector and uses a 56-32-8 encoder and 8-32-56 decoder. Five denoising epochs reconstruct the original observed vector after independent 25% zero masking and Gaussian noise (standard deviation 0.05), following the denoising principle [3]. "
         "This target is observed evidence, not hidden truth. A supervised 8-to-1 head then predicts fact readiness; a product scores candidate feasibility. LOGIT applies a single 56-to-1 linear readiness head to the same features. "
         "Both retain the belief-model supervision used for the earlier controls; no DAE-without-pretraining ablation is included, so any DAE effect does not isolate pretraining.")
    head("S5  Compute and retained-state accounting")
    duration = manifest.get("wall_seconds")
    duration_text = f"The complete run took {duration / 60:.1f} minutes. " if duration is not None else ""
    text(f"Training and inference run on {escape(str(manifest['hardware']))}. {duration_text}"
         "New checkpoint parameter counts are SCR 14,888; CAND_CTX/CAND_LOCAL 12,481 each; DAE 4,233 including the retained pretraining decoder; and LOGIT 57. "
         "Reported CUDA latency covers warmed batch-one inference and ranking; incremental peak allocation excludes resident models/data. Separate host-wall timing includes the prepared-evidence pipeline, not sensing or network transit.")
    text("Raw arrays, model checkpoints, installation files, and large logs remain ignored. Concise measurements, source/data digests, reproducible scripts, selected figures, and this supplement are retained.", "SmallV3")
    story.append(PageBreak())

    text("Paired effects and inference limits", "TitleV3")
    image("v3_figS2_seed_effects")
    text("Figure S2. Additional SCR-minus-ID seed effects in the four core conditions, plus incremental warmed inference allocation. Negative differences favor SCR. "
         "The same worlds are shared across methods, training seeds and conditions; GPU allocation is not total deployment memory.", "CaptionV3")
    head("S6  Prespecified primary contrast")
    lo, hi = primary["percentile_95_ci"]
    conclusion = ("The interval favors SCR in this synthetic setting." if hi < 0 else
                  "The interval favors ID in this synthetic setting." if lo > 0 else
                  "The interval includes zero and establishes neither a gain nor equivalence.")
    text(f"SCR macro regret is {primary['means']['SCR']:.5f}, versus {primary['means']['ID']:.5f} for ID. "
         f"The paired SCR-minus-ID effect is {primary['estimate']:+.5f}, with 95% interval [{lo:+.5f}, {hi:+.5f}]. {conclusion} "
         f"The interval uses {comparison['bootstrap_repetitions']:,} paired crossed bootstrap draws over five training seeds and {comparison['topology_clusters']} topology clusters, with shared cluster draws across all four conditions.")
    mixture = next(c for c in comparison["contrasts"] if c["comparator"] == "CAND_CTX")["metrics"]["macro_regret"]
    mlo, mhi = mixture["percentile_95_ci"]
    text(f"Within the direct family, SCR minus CAND_CTX is {mixture['estimate']:+.5f} "
         f"with interval [{mlo:+.5f}, {mhi:+.5f}]. This descriptive mixture comparison also changes model capacity.", "CaptionV3")
    text(f"There are {comparison['eligible_per_condition']:,} eligible snapshots and {comparison['no_feasible_per_condition']:,} without a feasible candidate per core condition. "
         "Only conditional recall excludes the latter. Figure intervals summarize seed variation conditional on this pool; secondary contrasts are descriptive and not multiplicity-adjusted. The analysis does not capture generator-design or shared finite-training-data uncertainty.", "CaptionV3")
    head("S7  Boundaries of the alternative approach")
    union_rate = estimate(subset(cc, method="SCR"), "shadow_union_violation")[0]
    for method in ["ID", "JH", "LOGIT", "DAE"]:
        if estimate(subset(cc, method=method), "shadow_union_violation")[0] != 0:
            raise ValueError("Update the supplement's zero-violation control statement")
    text(f"SCR violates shadow-union monotonicity in {100 * union_rate:.2f}% of core-condition queries; product/factored controls have zero violations. "
         "SCR replaces fact reconstruction with candidate feasibility learning; it cannot provide fact-level repairs or the joint model's marginal consistency guarantee. The adjacent-pair shadow-union check measures a limited conjunction-consistency condition and can contain up to 10 facts, beyond training sets of 2-5. "
         "Global context and mixture weights do not verify source truth or identify an adversary. Larger tests change fact count and root density together. All methods share a sparse symbolic codec, so HDC communication savings remain untested. "
         "Results concern synthetic static planning snapshots with fixed candidate sets, not sequential cyber operations or operational mission effectiveness. Comparing SCR with old P/B2 also changes training exposure; comparing SCR with ID changes architecture, prediction target and loss.", "SmallV3")
    head("Primary-source lineage")
    text('[1] Zaheer et al. <link href="https://papers.neurips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf" color="#a32b55">Deep Sets</link>. NeurIPS, 2017. '
         '[2] Jacobs et al. <link href="https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf" color="#a32b55">Adaptive Mixtures of Local Experts</link>. Neural Computation, 1991. '
         '[3] Vincent et al. <link href="https://icml.cc/2008/papers/592.pdf" color="#a32b55">Extracting and Composing Robust Features with Denoising Autoencoders</link>. ICML, 2008.', "SmallV3")
    text('Submission note. This separate three-page record is prepared at the requested length. Public <link href="https://mvml.org/author-guidelines/" color="#a32b55">MVML author guidelines</link> inspected for the project do not explicitly confirm a separate supplemental upload allowance.', "SmallV3")
    path = SUPP / "Supplement_v3.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dac4ce"))
        canvas.line(48, 40, 547, 40)
        canvas.setFont(normal, 7.3)
        canvas.setFillColor(colors.HexColor("#536779"))
        canvas.drawString(48, 27, "Direct candidate learning and denoising | Third-study supplemental record")
        canvas.drawRightString(547, 27, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=48, leftMargin=48, topMargin=43, bottomMargin=52,
                      title="Third-study supplementary record", author="Research project").build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(path).pages)
    if pages != 3:
        raise RuntimeError(f"Expected exactly three pages; produced {pages}")
    inputs = [DATA / "summary.csv", DATA / "curves.csv", manifest_path, comparison_path,
              SUPP / "v3_figS1_training.png", SUPP / "v3_figS2_seed_effects.png"]
    pipeline_path = DATA / "pipeline_timing.json"
    if pipeline_path.is_file():
        inputs.append(pipeline_path)
    provenance = {"inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                  "script": "scripts/make_v3_supplement.py", "script_sha256": sha(Path(__file__)),
                  "pdf_sha256": sha(path), "pages": pages,
                  "figures": ["v3_figS1_training", "v3_figS2_seed_effects"],
                  "supplement_policy": "Separate supplemental allowance not explicitly confirmed in public instructions"}
    (SUPP / "Supplement_v3.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Created {path.relative_to(ROOT)}: {pages} pages, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
