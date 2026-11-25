#!/usr/bin/env python3
"""Plot recorded GPU experiment summaries; no synthetic or placeholder values."""
import csv
import hashlib
import html
import json
from pathlib import Path
import os
import subprocess
ROOT=Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"]=str(ROOT/"work/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True,exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures/paper"
SUPP=ROOT/"supplemental"
CORE=["B0","B1","B2","B3","B4","B5","P"]
MAIN=["B0","B1","B2","B3","B5","P"]
ALL=CORE+["random_gate","no_provenance","no_semantic","dense_gate"]
LABEL={"B0":"Constrained prior", "B1":"GraphMAE adapted","B2":"Decision-aware graph",
       "B3":"HDC cleanup","B4":"Full regeneration","B5":"Entropy gate","P":"Proposed",
       "random_gate":"Random gate","no_provenance":"No direct source cues",
       "no_semantic":"No semantic loss","dense_gate":"Dense + learned gate"}
COLORS=dict(zip(ALL,["#687487","#0072B2","#D55E00","#9970AB","#CC79A7","#E69F00",
                     "#00866C","#729EA1","#8C6D31","#B04B50","#53599A"]))
MARKERS=dict(zip(ALL,["o","s","^","v","P","D","o","x","+","*","h"]))
NUM={"seed","corruption","facts","budget","recall3","regret","brier","false_repair",
     "false_certainty","preservation","latency_ms","peak_vram_mb","train_seconds",
     "bytes","eligible","coverage","changed_report_fraction","intervention_fraction","bin","confidence","accuracy","count","epoch","loss","repair_fraction"}

def load(name):
    path=ROOT/"results"/name
    if not path.is_file(): raise FileNotFoundError(f"Real experimental input required: {path}")
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError(f"Empty measurement file: {path}")
    for row in rows:
        for key in NUM & row.keys():
            if row[key]!="": row[key]=float(row[key])
    return rows

def subset(rows, **filters):
    return [r for r in rows if all(r.get(k)==v for k,v in filters.items())]

def mean_seed(rows, key):
    grouped={}
    for row in rows:
        val=row.get(key)
        if isinstance(val,(float,int)) and np.isfinite(val):
            grouped.setdefault(int(row["seed"]),[]).append(val)
    return {s:float(np.mean(v)) for s,v in grouped.items()}

def estimate(rows,key):
    values=np.array(list(mean_seed(rows,key).values()))
    if not len(values): return np.nan,np.nan
    if len(values)==1: return float(values[0]),0.
    critical={2:12.706,3:4.303,4:3.182,5:2.776}.get(len(values),1.96)
    return float(values.mean()),float(critical*values.std(ddof=1)/np.sqrt(len(values)))

def reference(rows):
    selected=subset(rows,condition="severity",corruption=.2)
    if not selected: raise ValueError("Reference severity=.2 measurements required")
    return selected

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,
                         "axes.titleweight":"bold","axes.labelsize":9,"legend.fontsize":8,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":.17,"axes.axisbelow":True,
                         "pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"})

def panel(ax,title,y=None,x=None):
    ax.set_title(title,loc="left",pad=10)
    if y: ax.set_ylabel(y)
    if x: ax.set_xlabel(x)

def lines(ax, rows, xkey, ykey, methods=MAIN):
    for m in methods:
        mr=subset(rows,method=m)
        xs=sorted({r[xkey] for r in mr if isinstance(r.get(xkey),(float,int))})
        if not xs: continue
        vals=[estimate(subset(mr,**{xkey:x}),ykey) for x in xs]
        ys=np.array([p[0] for p in vals]); es=np.array([p[1] for p in vals])
        ax.plot(xs,ys,label=LABEL[m],color=COLORS[m],marker=MARKERS[m],
                lw=2.1 if m=="P" else 1.4,ms=4)
        ax.fill_between(xs,ys-es,ys+es,color=COLORS[m],alpha=.09)

def bars(ax,rows,key,methods=CORE,horizontal=False):
    means=[]; errors=[]
    methods=[m for m in methods if subset(rows,method=m)]
    for m in methods:
        v,e=estimate(subset(rows,method=m),key); means.append(v);errors.append(e)
    pos=np.arange(len(methods))
    if horizontal:
        ax.barh(pos,means,xerr=errors,color=[COLORS[m] for m in methods],alpha=.9,capsize=2)
        ax.set_yticks(pos,[LABEL[m] for m in methods]);ax.invert_yaxis()
    else:
        ax.bar(pos,means,yerr=errors,color=[COLORS[m] for m in methods],alpha=.9,capsize=2)
        ax.set_xticks(pos,methods)

def paired(rows,a="P",b="B2",key="recall3"):
    aa=mean_seed(subset(rows,method=a),key);bb=mean_seed(subset(rows,method=b),key)
    return {seed:aa[seed]-bb[seed] for seed in aa.keys() & bb.keys()}

def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(fig, stem, inputs, description, supplementary=False):
    directory=SUPP if supplementary else OUT
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/(stem+".pdf"),bbox_inches="tight")
    fig.savefig(directory/(stem+".png"),dpi=300,bbox_inches="tight",pil_kwargs={"optimize":True})
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError: commit=None
    metadata={"title":description,"script":"scripts/make_results.py",
              "script_sha256":hashfile(Path(__file__)),"base_commit":commit,
              "inputs":{name:hashfile(ROOT/"results"/name) for name in inputs},
              "aggregation":"Mean within training seed; intervals use Student t across independent trained seeds.",
              "uncertainty_limit":"Intervals do not include additional resampling of test topology clusters.",
              "output_sha256":{ext:hashfile(directory/(stem+"."+ext)) for ext in ["pdf","png"]}}
    (directory/(stem+".provenance.json")).write_text(json.dumps(metadata,indent=2)+"\n")
    plt.close(fig)

def gallery(manifest):
    cards=[
        ("fig02_deception_recovery","Planning quality under deception",
         "Requested intervention severity, deception mechanisms, and paired proposed-minus-B2 recall contrasts. Error bars and bands are 95% Student-t intervals over training seeds; small dots in panel D are individual seed contrasts."),
        ("fig03_repair_calibration","Calibration and repair diagnostics",
         "Plan-success calibration pools candidate predictions across seeds. Fact-level Brier scores and mean snapshot-level conditional repair/confidence errors use the reference condition. Empty conditional denominators contribute zero; marker area in panel D encodes confident-fact coverage, which is also tabulated in the results report."),
        ("fig04_gpu_communication","GPU resources and available evidence",
         "All methods share one sparse symbolic wirecodec: payload limits change evidence availability, so this is not an HDC compression comparison. CUDA timing covers warmed batch-one inference and candidate ranking. Memory is incremental peak allocation above resident models/data. B4 and B5 reuse P training."),
        ("fig05_ablation_selection","Controls, repair allowances, and extrapolation",
         "The full-repair, entropy, and random selectors reuse P weights. Dense + learned gate is independently trained. The source-cue ablation retains the shared age-weighted prior. Larger 128/256-fact tests also change the topology family; facts are not host counts."),
    ]
    sections=[]
    for stem,title,caption in cards:
        sections.append(f'<section><h2>{html.escape(title)}</h2><img src="{stem}.png" alt="{html.escape(title)}"><p>{html.escape(caption)}</p><p><a href="{stem}.pdf">Vector PDF</a> · <a href="{stem}.png">PNG</a> · <a href="{stem}.provenance.json">Provenance</a></p></section>')
    for stem,title in [("figS1_training","S1: Recorded optimization curves"),("figS2_seed_variation","S2: Seed contrasts and intervention audit")]:
        sections.append(f'<section><h2>{title}</h2><img src="../../supplemental/{stem}.png" alt="{title}"><p><a href="../../supplemental/{stem}.pdf">Vector PDF</a> · <a href="../../supplemental/{stem}.provenance.json">Provenance</a></p></section>')
    hw=html.escape(str(manifest.get("hardware","See study manifest")))
    seed_count=len(manifest.get("settings",{}).get("seeds",[]))
    page="""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cyber regeneration: empirical figures</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#243247;background:#eef3f7;margin:0}main{max-width:1080px;margin:auto;padding:32px 24px}h1{line-height:1.15;font-size:36px;color:#17283f}h2{font-size:23px;color:#076e76}section{margin:28px 0;padding:24px;background:white;border:1px solid #d8e2e9;border-radius:10px}img{display:block;width:100%;height:auto}a{color:#075fa0}p{max-width:90ch}.meta{color:#536779}footer{padding:12px 0 30px}</style>
<main><h1>Selective regeneration under deceptive evidence</h1>
"""
    page+=f'<p class="meta">{seed_count} training seeds · {hw} · 4 main empirical figures · 2 additional supplemental figures</p>'
    page+='<p>A controlled GPU study of belief repair and candidate ranking. Reference condition: 64 facts, 20% requested targeted report interventions, a 2,048-byte budget per remote observer, and a 10% repair allowance. Measurements support this synthetic setting; they do not establish a complete operational planner.</p>'
    page+='<p><a href="../../docs/RESULTS.md">Measured findings and limitations</a> · <a href="../../results/study_manifest.json">Study manifest</a> · <a href="../../supplemental/Supplement.pdf">Three-page supplement</a></p>'
    page+='<p class="meta">Method key: '+html.escape('; '.join(f"{m}: {LABEL[m]}" for m in CORE))+'.</p>'
    page+=''.join(sections)
    page+='<footer>Figures are generated from recorded measurements. Each provenance file records the input, script, and output hashes. Primary crossed-bootstrap intervals are reported separately in the <a href="../../results/paired_comparison.json">paired comparison</a>.</footer></main></html>'
    (OUT/"index.html").write_text(page)

def main():
    style()
    rows=load("summary.csv"); cal=load("calibration.csv"); curves=load("curves.csv")
    manifest=json.loads((ROOT/"results/study_manifest.json").read_text())
    expected=manifest.get("settings",{}).get("seeds",[])
    observed=sorted({int(r["seed"]) for r in rows})
    if expected and observed!=sorted(expected): raise ValueError("All recorded study seeds are required")
    if {r["method"] for r in rows}!=set(ALL): raise ValueError("All prespecified methods and ablations are required")
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ["method","condition","corruption","attack","facts","budget","repair_fraction"])
        groups.setdefault(key,[]).append(int(row["seed"]))
    if any(sorted(seeds)!=observed for seeds in groups.values()):
        raise ValueError("Every method-condition cell must contain exactly one record per training seed")
    ref=reference(rows)
    # Figure 2: paired planning comparisons.
    fig,axs=plt.subplots(2,2,figsize=(9.2,7.1),layout="constrained")
    sev=subset(rows,condition="severity")
    attack=subset(rows,condition="attack")+ref
    bandwidth=subset(rows,condition="bandwidth")+ref
    repair=subset(rows,condition="repair")+ref
    sizes=subset(rows,condition="size")+subset(rows,condition="attack",attack="coherent")
    lines(axs[0,0],sev,"corruption","recall3")
    panel(axs[0,0],"A  Feasible COA recall","Recall among eligible instances","Requested intervention fraction")
    axs[0,0].set_ylim(0,1.04);axs[0,0].legend(ncol=2,loc="lower left")
    lines(axs[0,1],sev,"corruption","regret")
    panel(axs[0,1],"B  Decision consequences","Candidate-set utility regret","Requested intervention fraction")
    families=sorted({r["attack"] for r in attack})
    for i,m in enumerate(["B0","B1","B2","P"]):
        vals=[estimate(subset(attack,attack=a,method=m),"recall3") for a in families]
        axs[1,0].bar(np.arange(len(families))+(i-1.5)*.18,[v for v,e in vals],
                     yerr=[e for v,e in vals],width=.18,color=COLORS[m],label=LABEL[m],capsize=2)
    axs[1,0].set_xticks(np.arange(len(families)),families);axs[1,0].set_ylim(0,1.04)
    panel(axs[1,0],"C  Deception mechanism","Feasible COA recall","Mechanism at requested intervention = 0.2")
    axs[1,0].legend(ncol=2,loc="lower left")
    for i,a in enumerate(families):
        contrasts=paired(subset(attack,attack=a))
        v=np.array(list(contrasts.values()))
        if len(v):
            e=2.776*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0
            axs[1,1].errorbar(v.mean(),i,xerr=e,color=COLORS["P"],fmt="o",capsize=4)
            axs[1,1].scatter(v,i+np.linspace(-.1,.1,len(v)),s=15,color=COLORS["B2"],alpha=.7)
    axs[1,1].axvline(0,color="#555555",lw=1,ls="--");axs[1,1].set_yticks(range(len(families)),families)
    panel(axs[1,1],"D  Paired proposed minus B2","Deception mechanism","Recall difference; dots = seeds")
    save(fig,"fig02_deception_recovery",["summary.csv"],"Planning quality under deceptive reports; 95% training-seed intervals.")

    # Figure 3: calibration and belief repair diagnostics.
    fig,axs=plt.subplots(2,2,figsize=(9.2,7.1),layout="constrained")
    for m in ["B0","B1","B2","B5","P"]:
        mr=subset(cal,method=m); bs=sorted({r["bin"] for r in mr});xx=[];yy=[];cc=[]
        for b in bs:
            br=[r for r in subset(mr,bin=b) if r["count"]>0];n=sum(r["count"] for r in br)
            if n>0:
                xx.append(sum(r["confidence"]*r["count"] for r in br)/n)
                yy.append(sum(r["accuracy"]*r["count"] for r in br)/n);cc.append(n)
        axs[0,0].plot(xx,yy,marker=MARKERS[m],color=COLORS[m],label=LABEL[m],ms=4)
        if m in ["B2","P"]: axs[0,1].plot(xx,cc,marker=MARKERS[m],color=COLORS[m],label=LABEL[m])
    axs[0,0].plot([0,1],[0,1],"--",color="gray",lw=1)
    panel(axs[0,0],"A  Plan-success calibration","Observed plan feasibility","Predicted plan-success probability")
    axs[0,0].set_xlim(0,1);axs[0,0].set_ylim(0,1);axs[0,0].legend()
    panel(axs[0,1],"B  Calibration sample support","Predictions pooled across seeds","Mean probability in bin")
    axs[0,1].legend()
    bars(axs[1,0],ref,"brier")
    panel(axs[1,0],"C  Belief reconstruction","Fact-level Brier score (lower is better)","Method")
    for m in CORE:
        mr=subset(ref,method=m); x,xe=estimate(mr,"false_repair");y,ye=estimate(mr,"false_certainty")
        coverage,_=estimate(mr,"coverage")
        axs[1,1].errorbar(x,y,xerr=xe,yerr=ye,fmt="none",color=COLORS[m],capsize=2)
        axs[1,1].scatter(x,y,s=30+100*coverage,marker=MARKERS[m],color=COLORS[m],zorder=3)
        offset={"B1":(58,0),"B2":(58,15),"B4":(58,30),"P":(12,27)}.get(m,(5,4))
        axs[1,1].annotate(m,(x,y),xytext=offset,textcoords="offset points",fontsize=8,
                           arrowprops={"arrowstyle":"-","color":COLORS[m],"lw":.6})
    panel(axs[1,1],"D  Repair risk","Error rate among confident facts","Fraction of changes that worsen truth error")
    axs[1,1].legend(handles=[Line2D([],[],color="#555555",marker="o",ls="none",markersize=np.sqrt(30+100*c),label=f"{c:.0%}") for c in [.5,.9]],
                    title="Confident-fact coverage",loc="upper left",fontsize=7,title_fontsize=7)
    save(fig,"fig03_repair_calibration",["summary.csv","calibration.csv"],"Calibration and fact reconstruction at reference conditions; marker area in panel D encodes confident-fact coverage.")

    # Figure 4: measured compute and payload variation.
    fig,axs=plt.subplots(2,2,figsize=(9.2,7.1),layout="constrained")
    lines(axs[0,0],bandwidth,"budget","recall3")
    panel(axs[0,0],"A  Evidence availability under payload limits","Feasible COA recall","Payload budget per observer (bytes)")
    axs[0,0].set_ylim(0,1.04);axs[0,0].legend(ncol=2,loc="lower left")
    for m in CORE:
        rr=subset(ref,method=m); x,xe=estimate(rr,"latency_ms"); y,ye=estimate(rr,"recall3")
        axs[0,1].errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[m],color=COLORS[m],capsize=2)
        offset={"B1":(5,9),"B2":(-25,-20)}.get(m,(5,4))
        axs[0,1].annotate(m,(x,y),xytext=offset,textcoords="offset points",fontsize=8,
                          arrowprops={"arrowstyle":"-","color":COLORS[m],"lw":.6})
    panel(axs[0,1],"B  Inference cost and quality","Feasible COA recall","Batch-one GPU inference + ranking (ms)")
    bars(axs[1,0],ref,"peak_vram_mb")
    panel(axs[1,0],"C  Incremental peak GPU allocation","Increment above resident baseline (MiB)","Method")
    axs[1,0].yaxis.set_major_locator(MultipleLocator(.05))
    axs[1,0].yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    bars(axs[1,1],ref,"train_seconds")
    panel(axs[1,1],"D  Underlying model training cost","Measured training time (seconds)","Method (B4 / B5 reuse P)")
    save(fig,"fig04_gpu_communication",["summary.csv"],"GPU resources and evidence availability under a shared sparse symbolic wirecodec; not a compression comparison. Excludes host serialization and transfers.")

    # Figure 5: complete ablations and size sensitivity.
    fig,axs=plt.subplots(2,2,figsize=(9.2,7.6),layout="constrained")
    bars(axs[0,0],ref,"recall3",ALL,horizontal=True)
    panel(axs[0,0],"A  All controls and ablations",None,"Feasible COA recall")
    axs[0,0].set_xlim(0,1.04)
    lines(axs[0,1],repair,"repair_fraction","regret",["B4","B5","P"])
    panel(axs[0,1],"B  Repair allowance","Candidate-set regret","Fraction of facts eligible for change")
    axs[0,1].legend()
    lines(axs[1,0],sizes,"facts","recall3",["B0","B1","B2","P"])
    panel(axs[1,0],"C  Size and topology extrapolation","Feasible COA recall","Fact slots (not host count)")
    axs[1,0].legend()
    for i,m in enumerate(["random_gate","no_provenance","no_semantic","dense_gate"]):
        aa=paired(ref,a="P",b=m,key="regret"); values=np.array(list(aa.values()))
        if len(values):
            err=2.776*values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0
            axs[1,1].errorbar(values.mean(),i,xerr=err,fmt="o",color=COLORS[m],capsize=4)
    axs[1,1].axvline(0,color="#555",ls="--",lw=1)
    axs[1,1].set_yticks(range(4),[LABEL[x] for x in ["random_gate","no_provenance","no_semantic","dense_gate"]])
    panel(axs[1,1],"D  Paired regret contrasts",None,"Proposed minus ablation (negative favors proposed)")
    zoom=axs[1,1].inset_axes([.12,.52,.49,.36])
    for i,m in enumerate(["no_provenance","no_semantic","dense_gate"]):
        values=np.array(list(paired(ref,a="P",b=m,key="regret").values()))
        err=2.776*values.std(ddof=1)/np.sqrt(len(values))
        zoom.errorbar(values.mean(),i,xerr=err,fmt="o",color=COLORS[m],capsize=3,ms=4)
    zoom.axvline(0,color="#555",ls="--",lw=.7)
    zoom.set_yticks(range(3),["Source cues","Semantic","Dense"])
    zoom.tick_params(labelsize=6.5);zoom.set_title("Near-zero contrasts",fontsize=8,pad=5)
    zoom.spines["top"].set_visible(True);zoom.spines["right"].set_visible(True)
    zoom.set_ylim(-.45,2.45)
    save(fig,"fig05_ablation_selection",["summary.csv"],"Controls, repair budgets, larger fact states, and paired ablation effects.")

    # Supplement figure S1: additional training evidence.
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.1),layout="constrained")
    for j,split in enumerate(["train","validation"]):
        cr=subset(curves,split=split)
        if not cr and split=="validation": cr=subset(curves,split="val")
        for m in ["B1","B2","P","dense_gate"]:
            mr=subset(cr,method=m)
            if not mr: continue
            xs=sorted({r["epoch"] for r in mr});ys=[estimate(subset(mr,epoch=e),"loss")[0] for e in xs]
            axs[j].plot(xs,ys,color=COLORS[m],label=LABEL[m],lw=1.5)
        panel(axs[j],f"{'AB'[j]}  {split.capitalize()} objective","Recorded loss","Epoch")
        axs[j].legend(fontsize=7)
    fig.suptitle("Epoch-wise means use available seeds; the cohort changes after early stopping",fontsize=8,fontweight="normal")
    save(fig,"figS1_training",["curves.csv"],"Recorded training and validation objectives; objective scales differ and the contributing seed cohort may change with early stopping.",True)

    # Supplement figure S2: seed-level heterogeneity and information saturation.
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.1),layout="constrained")
    seedlist=sorted({int(r["seed"]) for r in rows});families=sorted({r["attack"] for r in attack})
    matrix=np.full((len(families),len(seedlist)),np.nan)
    for i,a in enumerate(families):
        pp=paired(subset(attack,attack=a))
        for j,s in enumerate(seedlist): matrix[i,j]=pp.get(s,np.nan)
    lim=max(float(np.nanmax(np.abs(matrix))),.01)
    im=axs[0].imshow(matrix,cmap="RdBu",vmin=-lim,vmax=lim,aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): axs[0].text(j,i,f"{matrix[i,j]:+.3f}",ha="center",va="center",fontsize=7,color="white" if abs(matrix[i,j])>.55*lim else "black")
    axs[0].set_xticks(range(len(seedlist)),seedlist);axs[0].set_yticks(range(len(families)),families)
    panel(axs[0],"A  Paired recall difference","Deception mechanism","Training seed")
    fig.colorbar(im,ax=axs[0],label="Proposed minus B2",shrink=.8)
    for i,(key,label,color) in enumerate([("intervention_fraction","Delivered interventions","#0072B2"),("changed_report_fraction","Delivered value changes","#E69F00")]):
        stats=[estimate(subset(attack,method="B0",attack=a),key) for a in families]
        axs[1].bar(np.arange(len(families))+(i-.5)*.30,[v for v,e in stats],
                   yerr=[e for v,e in stats],width=.30,color=color,label=label,capsize=2)
    axs[1].axhline(.2,color="#666666",ls="--",lw=1,label="Requested fraction")
    axs[1].set_xticks(range(len(families)),families)
    panel(axs[1],"B  Report-intervention audit","Fraction of delivered reports","Deception mechanism")
    axs[1].legend(fontsize=7,loc="lower left")
    save(fig,"figS2_seed_variation",["summary.csv"],"Paired seed-level effects and actual delivered report changes under matched requested intervention rates.",True)
    gallery(manifest)
    print("Generated four main empirical figures, two additional supplemental figures, and the HTML gallery.")

if __name__=="__main__":
    main()
