#!/usr/bin/env python3
"""Render original research-design diagrams as editable SVG, without result data."""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures" / "design"
INK, MUTED, BLUE, TEAL, GOLD, RED = "#17283f", "#586b82", "#355bd6", "#087f8c", "#b87b13", "#ba4b43"
PALE, LINE = "#f4f7fb", "#d7e0ec"

class Figure:
    def __init__(self, title, subtitle):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1100" viewBox="0 0 1600 1100" role="img">
<title>{escape(title)}</title><desc>{escape(subtitle)}. Research design schematic, not measured results.</desc>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="{MUTED}"/></marker></defs>
<rect width="1600" height="1100" fill="{PALE}"/>
<g font-family="DejaVu Sans, Arial, sans-serif">
''']
        self.text(48, 62, title, 34, INK, "bold")
        self.text(48, 103, subtitle, 20, MUTED)
        self.text(48, 1060, "MVML 2027  |  HYPER-CAP research design  |  Schematic only — no measured results", 18, MUTED)

    def text(self, x, y, value, size=20, color=INK, weight="normal", anchor="start"):
        for i, line in enumerate(str(value).split("\n")):
            self.parts.append(f'<text x="{x}" y="{y+i*(size+8)}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(line)}</text>')

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=14, dashed=False):
        dash = ' stroke-dasharray="9 6"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')

    def panel(self, x, y, label, title):
        self.rect(x, y, 728, 410)
        self.rect(x+22, y+20, 35, 35, fill=BLUE, stroke=BLUE, radius=8)
        self.text(x+39, y+45, label, 22, "white", "bold", "middle")
        self.text(x+74, y+46, title, 24, INK, "bold")

    def arrow(self, x1, y1, x2, y2, color=MUTED, dashed=False):
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5" marker-end="url(#arrow)"{dash}/>')

    def box(self, x, y, w, h, title, body="", color=BLUE, title_size=21):
        self.rect(x, y, w, h, fill=PALE, stroke=color)
        self.text(x+16, y+32, title, title_size, color, "bold")
        if body:
            self.text(x+16, y+64, body, 18, INK)

    def dot(self, x, y, label, color=BLUE, r=28):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="3"/>')
        self.text(x, y+7, label, 18, color, "bold", "middle")

    def write(self, name):
        (OUT/name).write_text("".join(self.parts)+"</g></svg>", encoding="utf-8")


def architecture():
    f=Figure("Selective regeneration for cyber planning", "Partial evidence becomes a budgeted belief repair and an inspectable course of action.")
    f.panel(48,150,"A","Distributed evidence")
    for x,t,b in [(76,"Observer 1","Host status\nLocal history"),(308,"Observer 2","Service checks\nDelayed reports"),(540,"Observer 3","Mission limits\nResource evidence")]:
        f.box(x,238,204,123,t,b)
        f.arrow(x+102,363,x+102,405)
    f.box(91,414,641,103,"Provenance-preserving exchange","Origin IDs · time windows · bounded serialized bytes",TEAL)
    f.text(88,542,"No observer receives the complete hidden state.",19,MUTED)

    f.panel(824,150,"B","GPU representation and learning")
    for x,t,b in [(852,"HDC memory","Fixed codebook\nBatched retrieval"),(1086,"Graph network","Dependencies\nBelief decoder"),(1320,"Repair gate","Learned impact\nSelect up to K")]:
        f.box(x,242,204,136,t,b,TEAL,title_size=20)
    f.arrow(1058,308,1081,308)
    f.arrow(1292,308,1315,308)
    f.box(852,416,672,99,"CUDA execution","Mixed-precision layers · FP32 beliefs and losses",BLUE)
    f.text(852,542,"Training and learned inference require a usable GPU.",19,MUTED)

    f.panel(48,600,"C","Selective regeneration")
    colors=[TEAL]*32
    for i in [2,7,13,18,25]: colors[i]=GOLD
    for i in [6,16,29]: colors[i]=MUTED
    for i,col in enumerate(colors):
        x=90+(i%8)*71
        y=697+(i//8)*45
        f.rect(x,y,54,31,fill=col,stroke=col,radius=6)
    f.box(673,697,69,166,"K","cap",GOLD,title_size=22)
    for x,c,label in [(90,TEAL,"Retain"),(280,GOLD,"Regenerate"),(523,MUTED,"Unresolved")]:
        f.rect(x,900,18,18,fill=c,stroke=c,radius=4)
        f.text(x+27,916,label,19,c)
    f.text(90,968,"Recompute affected dependencies; keep competing beliefs.",19,MUTED)

    f.panel(824,600,"D","Shared planner and audit trace")
    f.box(851,703,173,100,"Evidence","Source and age",BLUE,title_size=20)
    f.box(1072,703,194,100,"Prerequisite","Belief + rule",TEAL,title_size=20)
    f.box(1313,677,211,88,"COA A","Conditional plan",GOLD,title_size=20)
    f.box(1313,830,211,88,"COA B","Alternative plan",BLUE,title_size=20)
    f.arrow(1026,750,1067,750)
    f.arrow(1268,745,1309,724)
    f.arrow(1242,805,1309,860)
    f.text(852,877,"Same generator and planner\nfor every comparison method.",20,MUTED)
    f.text(852,970,"Report top-three feasibility, utility, diversity, and assumptions.",18,MUTED)
    f.write("design01_architecture.svg")


def equation():
    f=Figure("A decision-impact teacher for regeneration", "The proposed extension teaches where a correction matters to downstream planning.")
    f.panel(48,150,"A","Build on established methods")
    for y,t,b,c in [(234,"GraphMAE","Masked graph-feature reconstruction",BLUE),(329,"Decision-focused learning","Prediction trained for decision quality",TEAL),(424,"Semantic loss","Probabilistic outputs constrained by logic",GOLD)]:
        f.box(79,y,665,79,t,"",c)
        f.text(96,y+61,b,18,MUTED)
    f.panel(824,150,"B","Counterfactual teacher at training")
    f.box(854,235,184,108,"Current belief","Provisional b⁻",BLUE)
    f.box(1087,235,184,108,"Repair fact f","Training truth y",GOLD)
    f.box(1320,235,204,108,"Score plans","Same candidates",TEAL)
    f.arrow(1040,287,1083,287)
    f.arrow(1273,287,1316,287)
    f.box(854,386,670,109,"Equation 3","Δf = [ regret(b⁻) − regret(b⁻ with f corrected) ]+",BLUE)
    f.text(854,535,"Choose up to K positive gains; truth never enters inference.",18,MUTED)

    f.panel(48,600,"C","Learn the repair operator")
    f.box(79,688,290,104,"Observable evidence","History + received messages",BLUE,title_size=20)
    f.box(431,688,312,104,"Learned gate g","Priority scores → up to K facts",TEAL,title_size=20)
    f.arrow(373,738,426,738)
    f.rect(79,832,664,94,fill=PALE,stroke=GOLD)
    f.text(411,889,"bᵍ = (1 − g) ⊙ b⁻ + g ⊙ r",29,INK,"bold","middle")
    f.text(79,969,"r is the regenerated belief; g limits which facts change.",19,MUTED)

    f.panel(824,600,"D","Checks and failure boundaries")
    entries=[("K = 0","Retain the provisional state",BLUE),("All facts selected","Recover full reconstruction",TEAL),("Entropy-only gate","Comparator at the same budget",GOLD)]
    for i,(a,b,c) in enumerate(entries):
        yy=691+i*71
        f.rect(854,yy,670,55,fill=PALE,stroke=LINE,radius=9)
        f.text(870,yy+35,a,20,c,"bold")
        f.text(1140,yy+35,b,18,INK)
    f.text(854,942,"One-fact gains can miss interacting repairs.",20,RED,"bold")
    f.text(854,975,"Evaluate paired-repair cases and irreducible ambiguity.",18,MUTED)
    f.write("design02_equation.svg")


def benchmark():
    f=Figure("A benchmark that separates deception from change", "Hidden-state evaluation and topology-disjoint splits protect the scientific comparison.")
    f.panel(48,150,"A","Synthetic service-dependency world")
    edges=[(166,299,364,260),(166,299,364,401),(364,260,596,326),(364,401,596,326)]
    for a,b,c,d in edges: f.arrow(a,b,c,d)
    for x,y,t,c in [(144,300,"Net",BLUE),(366,249,"Auth",TEAL),(366,420,"Data",GOLD),(622,325,"Svc",BLUE)]:
        f.dot(x,y,t,c,38)
    f.text(83,496,"Known action schema; hidden readiness and resource facts.",19,MUTED)
    f.text(83,532,"Deterministic base actions support exact rollout checks.",19,MUTED)

    f.panel(824,150,"B","Partial and conflicting observations")
    for i,label in enumerate(["Observer 1","Observer 2","Observer 3"]):
        yy=249+i*76
        f.text(856,yy+23,label,20,INK,"bold")
        for j in range(8):
            active=(i+j)%3!=0
            col=TEAL if active else LINE
            if (i,j) in [(1,3),(2,5)]: col=RED
            f.rect(1015+j*62,yy,46,32,fill=col,stroke=col,radius=5)
    f.text(856,505,"Teal: observed   Grey: missing   Red: manipulated",19,MUTED)
    f.text(856,540,"Illustrative pattern only; masks are evaluator-only labels.",18,MUTED)

    f.panel(48,600,"C","Controlled perturbation families")
    tiles=[("Random reports","Matched corruption count",BLUE),("Critical prerequisite","Targeted decision impact",RED),("Coherent falsehood","Local rules still satisfied",RED),("Genuine change","Truth changes; no deception",TEAL),("Link outage","Missing or delayed exchange",GOLD),("Joint repairs","Two corrections needed",GOLD)]
    for i,(t,b,c) in enumerate(tiles):
        xx=79+(i%2)*343
        yy=687+(i//2)*94
        f.box(xx,yy,322,80,t,"",c,title_size=20)
        f.text(xx+16,yy+62,b,17,MUTED)
    f.text(79,990,"Freeze reference conditions before inspecting test outcomes.",18,MUTED)

    f.panel(824,600,"D","Splits and information access")
    for xx,t,b,c in [(854,"Train","Fit and supervise",BLUE),(1087,"Validation","Tune and calibrate",TEAL),(1320,"Test","Score once frozen",GOLD)]:
        f.box(xx,691,204,99,t,b,c,title_size=21)
    f.text(854,836,"Disjoint topology families, episodes, and generator seeds.",19,MUTED)
    f.rect(854,869,670,96,fill=PALE,stroke=RED,dashed=True)
    f.text(872,901,"Hidden truth → training labels and evaluation metrics",20,RED,"bold")
    f.text(872,939,"Inference input: observable history, rules, and received messages",17,INK)
    f.write("design03_benchmark.svg")


def evidence():
    f=Figure("Evidence required for the paper and the proposal", "Controlled baselines, measured GPU costs, and four mandatory empirical figures.")
    f.panel(48,150,"A","Comparison ladder")
    rows=[("B0","Probabilistic + symbolic","Conventional reference"),("B1","GraphMAE adapted","Established ML method"),("B2","GraphMAE + decision loss","Strong matched-loss control"),("B3","HDC cleanup","Representation alone"),("B4","Full regeneration","Value of selectivity"),("B5","Entropy gate","Value of impact supervision"),("P","Proposed gate + HDC","Complete method")]
    for i,(a,b,c) in enumerate(rows):
        yy=230+i*43
        f.rect(78,yy,667,37,fill=PALE if i<6 else "#e7f3f3",stroke=LINE,radius=5)
        f.text(91,yy+25,a,18,BLUE if i<6 else TEAL,"bold")
        f.text(143,yy+25,b,17,INK)
        f.text(440,yy+25,c,16,MUTED)

    f.panel(824,150,"B","Six paper figures")
    labels=[("01  Architecture","Mechanism and trace",BLUE),("02  Recovery","Feasibility and regret",TEAL),("03  Calibration","Truth and uncertainty",TEAL),("04  Efficiency","Bytes, memory, latency",TEAL),("05  Ablations","What causes the gains",TEAL),("06  Generalization","Transfer and failures",BLUE)]
    for i,(t,b,c) in enumerate(labels):
        xx=854+(i%2)*343
        yy=234+(i//2)*97
        f.box(xx,yy,322,80,t,"",c,title_size=20)
        f.text(xx+16,yy+61,b,18,MUTED)
    f.text(854,547,"Figures 02–05 require real experimental measurements.",18,TEAL,"bold")

    f.panel(48,600,"C","GPU measurement contract")
    f.box(79,687,665,95,"CUDA model computation","HDC retrieval · graph layers · counterfactual batches",TEAL)
    for i,(t,b) in enumerate([("VRAM","All buffers"),("p95","Batch one"),("Bytes","Serialized"),("GPU h","Run cost")]):
        xx=79+i*171
        f.box(xx,818,151,94,t,b,BLUE,title_size=22)
    f.text(79,953,"Report complete loop time, including symbolic planning.",19,MUTED)
    f.text(79,985,"No silent CPU fallback for research training or inference.",18,RED)

    f.panel(824,600,"D","Submission evidence")
    f.box(854,687,670,93,"MVML 2027","Learning objective · baselines · ablations · uncertainty",BLUE)
    f.box(854,811,670,93,"DARPA HYPER-CAP","Alternative COAs · replanning · communication · traceability",TEAL)
    f.text(854,954,"Plan → implementation → GPU logs → empirical figures",20,INK,"bold")
    f.text(854,986,"A research plan alone is not prior feasibility evidence.",18,MUTED)
    f.write("design04_evidence.svg")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    architecture()
    equation()
    benchmark()
    evidence()
    titles=[
        ("design01_architecture.svg","Selective regeneration architecture"),
        ("design02_equation.svg","New equation and inherited methods"),
        ("design03_benchmark.svg","Benchmark and information boundaries"),
        ("design04_evidence.svg","Comparisons and required evidence"),
    ]
    cards="".join(f'<section><h2>{escape(title)}</h2><a href="{name}"><img src="{name}" alt="{escape(title)}"></a></section>' for name,title in titles)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cyber regeneration research design figures</title><style>
body{margin:0;background:#f4f7fb;color:#17283f;font-family:Arial,sans-serif}
main{max-width:1480px;margin:auto;padding:32px}h1{font-size:30px}p{line-height:1.6;max-width:960px}
section{margin:32px 0}h2{font-size:22px}img{width:100%;height:auto;border:1px solid #d7e0ec;border-radius:14px;background:white}a{color:#355bd6}
</style><main><h1>Cyber regeneration research design figures</h1>
<p>Four original design schematics for the GPU research plan. They show the proposed architecture, decision-impact equation, experimental controls, and evidence requirements. They contain no measured performance results.</p>
<p><a href="../../RESEARCH_PLAN.md">Research plan</a> · <a href="../../docs/FIGURE_PLAN.md">Empirical figure specification</a></p>
'''+cards+"</main></html>"
    (OUT/"index.html").write_text(page,encoding="utf-8")
    print("Rendered four SVG design figures and their HTML gallery.")

if __name__ == "__main__":
    main()
