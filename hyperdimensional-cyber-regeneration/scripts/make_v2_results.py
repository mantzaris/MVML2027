#!/usr/bin/env python3
"""Render the frozen second study without modifying first-study artifacts."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "work/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DATA = ROOT / "results/v2"
OUT = ROOT / "figures/v2"
SUPP = ROOT / "supplemental/v2"
NEW = ["ID", "IH", "JD", "JH_ERM", "JH", "JH_product"]
ALL = ["B0", "B1", "B2", "B3", "B4", "B5", "P", "random_gate", "no_provenance", "no_semantic", "dense_gate"] + NEW
FOCUS = ["B2", "P", "ID", "JD", "JH"]
LABEL = {"B0":"Constrained prior", "B1":"Old GraphMAE", "B2":"Old dense full", "B3":"HDC cleanup",
         "B4":"Old P, full", "B5":"Old entropy gate", "P":"Old learned gate",
         "random_gate":"Old random gate", "no_provenance":"Old source ablation",
         "no_semantic":"Old semantic ablation", "dense_gate":"Old dense gate",
         "ID":"Independent dense", "IH":"Independent HDC", "JD":"Joint dense",
         "JH_ERM":"Joint HDC, mean risk", "JH":"Joint HDC, robust", "JH_product":"JH marginal product"}
COLORS = {"B0":"#718096", "B1":"#4c78a8", "B2":"#d55e00", "B3":"#9e77b1", "B4":"#cc79a7", "B5":"#e69f00",
          "P":"#6b7280", "random_gate":"#97a6b5", "no_provenance":"#a18a69", "no_semantic":"#bd7372", "dense_gate":"#8887a2",
          "ID":"#0072b2", "IH":"#56b4e9", "JD":"#b05a9d", "JH_ERM":"#947133", "JH":"#007f72", "JH_product":"#e69f00"}
MARKERS = {m:mark for m,mark in zip(ALL, ["o","s","^","v","P","D","o","x","+","*","h","s","d","^","v","o","X"])}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    path = DATA / name
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty recorded input: {path}")
    for row in rows:
        for key, val in row.items():
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                pass
    return rows


def subset(rows, **filters):
    return [r for r in rows if all(r.get(k) == v for k,v in filters.items())]


def ordinary(row):
    return row.get("missing", .2) == .2 and row.get("facts", 64) == 64 and row.get("budget", 2048) == 2048


def core(rows):
    return [r for r in rows if ordinary(r) and (r["corruption"] == 0 or
            (r["corruption"] == .2 and r["attack"] in ["targeted","random","coherent"]))]


def reference(rows):
    return [r for r in rows if ordinary(r) and r["corruption"] == .2 and r["attack"] == "targeted"]


def seed_values(rows, metric):
    values = {}
    for r in rows:
        v = r.get(metric)
        if isinstance(v, (int,float)) and np.isfinite(v):
            values.setdefault(int(r["seed"]), []).append(float(v))
    return {s:float(np.mean(v)) for s,v in values.items()}


def estimate(rows, metric):
    values = np.array(list(seed_values(rows, metric).values()))
    if not len(values):
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.
    critical = {2:12.706,3:4.303,4:3.182,5:2.776}.get(len(values),1.96)
    return values.mean(), critical * values.std(ddof=1) / np.sqrt(len(values))


def paired(rows, comparator, metric="regret"):
    a = seed_values(subset(rows, method="JH"), metric)
    b = seed_values(subset(rows, method=comparator), metric)
    return {s:a[s]-b[s] for s in sorted(a.keys() & b.keys())}


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":11,
        "axes.titleweight":"bold", "axes.labelsize":9, "legend.fontsize":7.7,
        "axes.spines.top":False, "axes.spines.right":False, "axes.grid":True,
        "grid.alpha":.16, "axes.axisbelow":True, "pdf.fonttype":42, "ps.fonttype":42,
        "savefig.facecolor":"white"})


def panel(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, loc="left", pad=9)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)


def line(ax, rows, xkey, metric, methods=FOCUS):
    for method in methods:
        rr = subset(rows, method=method)
        xs = sorted({r[xkey] for r in rr})
        if not xs: continue
        stats = [estimate(subset(rr, **{xkey:x}), metric) for x in xs]
        vals, errs = np.array(stats).T
        ax.plot(xs, vals, marker=MARKERS[method], color=COLORS[method], label=LABEL[method],
                lw=2.2 if method=="JH" else 1.4, ms=4)
        ax.fill_between(xs, vals-errs, vals+errs, color=COLORS[method], alpha=.09)


def bar(ax, rows, metric, methods, horizontal=False, labels=True):
    methods = [m for m in methods if subset(rows, method=m)]
    stats = [estimate(subset(rows, method=m), metric) for m in methods]
    vals, errs = np.array(stats).T
    x = np.arange(len(methods))
    if horizontal:
        ax.barh(x, vals, xerr=errs, color=[COLORS[m] for m in methods], capsize=2)
        ax.set_yticks(x, [LABEL[m] if labels else m for m in methods]); ax.invert_yaxis()
    else:
        ax.bar(x, vals, yerr=errs, color=[COLORS[m] for m in methods], capsize=2)
        ax.set_xticks(x, [m for m in methods])


def save(fig, stem, inputs, title, supplementary=False):
    directory = SUPP if supplementary else OUT
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", pil_kwargs={"optimize":True})
    metadata = {"title":title, "script":"scripts/make_v2_results.py", "script_sha256":sha(Path(__file__)),
                "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "inputs":{str((DATA/p).relative_to(ROOT)):sha(DATA/p) for p in inputs},
                "aggregation":"Where drawn, error bars/bands use equal condition means within seed and Student-t 95% intervals across five training seeds. Reliability curves pool candidate predictions by bin count; training curves average available seeds at each epoch.",
                "uncertainty_limit":"Figure intervals condition on the shared fresh test pool; primary crossed seed/topology bootstrap is recorded separately.",
                "output_sha256":{ext:sha(directory / f"{stem}.{ext}") for ext in ["pdf","png"]}}
    (directory / f"{stem}.provenance.json").write_text(json.dumps(metadata,indent=2)+"\n")
    plt.close(fig)


def mechanism_name(row):
    return "benign" if row["corruption"] == 0 else row["attack"]


def gallery(manifest):
    cards = [
        ("v2_fig01_generalization","Fresh-test decision quality", "Four-condition macro averages, mechanism-specific regret, severity sensitivity, and paired seed effects. Fresh test worlds are shared by every model; old checkpoints are reevaluated without retraining."),
        ("v2_fig02_joint_calibration","Dependency-aware probability scoring", "Actual candidate-success probabilities, shared-root diagnostics, and plan Brier scores. Root-duplication bins show candidate counts in the fixed test pool; the final bin has only two candidates, not ten independent cases across five seeds. Reliability curves pool candidate predictions across seeds; this does not make reused worlds independent. The violation metric is the fraction of fact slots whose marginal exceeds the parent marginal by more than 1e-6. JH_product uses the same checkpoint with marginal-product scoring, isolating the inference equation."),
        ("v2_fig03_matched_controls","Matched controls and ablations", "All methods remain visible. ID controls mixed training data and full regeneration; JD controls representation; JH_ERM controls the group-risk objective. Primary inference is JH versus ID, not old P versus JH."),
        ("v2_fig04_evidence_compute","Evidence availability and compute", "Common sparse report transport, additional missing observations, size plus topology shift, and measured CUDA inference cost. Full regeneration has no fixed six-fact repair constraint; CUDA timing excludes host I/O."),
    ]
    sections=[]
    for stem,title,caption in cards:
        sections.append(f'<section><h2>{html.escape(title)}</h2><img src="{stem}.png" alt="{html.escape(title)}"><p>{html.escape(caption)}</p><a href="{stem}.pdf">Vector PDF</a> · <a href="{stem}.provenance.json">Provenance</a></section>')
    for stem,title in [("v2_figS1_training","S1: Optimization traces"),("v2_figS2_seed_effects","S2: Paired seed effects and GPU allocation")]:
        sections.append(f'<section><h2>{title}</h2><img src="../../supplemental/v2/{stem}.png" alt="{title}"><a href="../../supplemental/v2/{stem}.pdf">Vector PDF</a> · <a href="../../supplemental/v2/{stem}.provenance.json">Provenance</a></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Second regeneration study</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:30px;color:#17283f;background:#f5f8fb}section{background:white;padding:24px;margin:24px 0;border:1px solid #dae3ed;border-radius:12px}img{width:100%;height:auto}a{color:#006b8b}h1{line-height:1.2}</style><h1>Joint posterior regeneration: measured second study</h1><p>Four empirical figures plus two additional supplemental figures. Bands and error bars are 95% training-seed t intervals, conditional on the shared fresh test scenarios. Main comparisons use five seeds and equal condition weights.</p><p><a href="../../supplemental/v2/Supplement_v2.pdf">Three-page supplement</a> · <a href="../../docs/RESULTS_V2.md">Results and inference limits</a> · <a href="../paper/index.html">Preserved first-study figures</a></p>'''+''.join(sections)+'</html>'
    (OUT/"index.html").write_text(page)


def main():
    style()
    rows, cal, curves, deps = [load(n) for n in ["summary.csv","calibration.csv","curves.csv","dependency_bins.csv"]]
    manifest = json.loads((DATA/"manifest.json").read_text())
    seeds = sorted({int(r["seed"]) for r in rows})
    if seeds != [11,23,37,51,71]: raise ValueError("Completed five-seed v2 measurements required")
    if len(rows) != 5*17*18: raise ValueError(f"Expected 1,530 seed/method/condition rows; found {len(rows)}")
    keys=[(r["seed"],r["method"],r["condition"],r["attack"],r["corruption"],r["facts"],r["budget"],r.get("missing",.2)) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError("Duplicate measurements")
    cc, rr = core(rows), reference(rows)
    for method in ALL:
        if len(subset(cc,method=method)) != 20: raise ValueError(f"Expected four core conditions per seed for {method}")
    mechanisms = ["benign","targeted","random","coherent"]
    # 1: Primary robust generalization and actual post-development test evidence.
    fig,axs=plt.subplots(2,2,figsize=(10.2,7.4),layout="constrained")
    bar(axs[0,0],cc,"regret",["B2","P","ID","IH","JD","JH_ERM","JH"])
    panel(axs[0,0],"A  Four-condition macro regret","Candidate-set regret (lower is better)","Method")
    for i,m in enumerate(FOCUS):
        vals=[estimate([r for r in subset(cc,method=m) if mechanism_name(r)==a],"regret") for a in mechanisms]
        axs[0,1].bar(np.arange(4)+(i-2)*.15,[v for v,e in vals],yerr=[e for v,e in vals],width=.15,color=COLORS[m],label=LABEL[m],capsize=2)
    axs[0,1].set_xticks(range(4),mechanisms)
    panel(axs[0,1],"B  Mechanism-specific decisions","Candidate-set regret","Core test condition")
    axs[0,1].set_ylim(0,axs[0,1].get_ylim()[1]*1.28)
    axs[0,1].legend(ncol=2,fontsize=7,loc="upper left")
    targeted=[r for r in rows if ordinary(r) and r["attack"]=="targeted"]
    line(axs[1,0],targeted,"corruption","regret",["B2","P","ID","JH"])
    panel(axs[1,0],"C  Targeted-intervention severity","Candidate-set regret","Requested report-intervention fraction")
    axs[1,0].legend(fontsize=7)
    comps=["ID","IH","JD","JH_ERM","JH_product"]
    for i,m in enumerate(comps):
        values=np.array(list(paired(cc,m).values())); err=2.776*values.std(ddof=1)/np.sqrt(len(values))
        axs[1,1].errorbar(values.mean(),i,xerr=err,fmt="o",color=COLORS[m],capsize=3)
        axs[1,1].scatter(values,i+np.linspace(-.1,.1,len(values)),s=11,color=COLORS[m],alpha=.6)
    axs[1,1].axvline(0,color="#555",ls="--",lw=1);axs[1,1].set_yticks(range(len(comps)),[LABEL[m] for m in comps])
    panel(axs[1,1],"D  Matched paired contrasts",None,"JH minus comparator regret (negative favors JH)")
    save(fig,"v2_fig01_generalization",["summary.csv","manifest.json"],"Fresh test decision quality and paired seed effects.")
    # 2: Joint probabilities, direct scoring ablation and symbolic consistency.
    fig,axs=plt.subplots(2,2,figsize=(10.2,7.4),layout="constrained")
    cr=reference(cal)
    for m in ["B2","ID","IH","JH_product","JH"]:
        mr=subset(cr,method=m);xx=[];yy=[]
        for b in sorted({r["bin"] for r in mr}):
            br=[r for r in subset(mr,bin=b) if r["count"]>0];n=sum(r["count"] for r in br)
            if n:
                xx.append(sum(r["confidence"]*r["count"] for r in br)/n);yy.append(sum(r["accuracy"]*r["count"] for r in br)/n)
        axs[0,0].plot(xx,yy,marker=MARKERS[m],color=COLORS[m],label=LABEL[m],ms=4)
    axs[0,0].plot([0,1],[0,1],ls="--",color="#777",lw=1);axs[0,0].set_xlim(0,1);axs[0,0].set_ylim(0,1)
    panel(axs[0,0],"A  Candidate-success calibration","Observed feasibility","Predicted success probability")
    axs[0,0].legend(fontsize=7)
    dr=reference(deps) if "facts" in deps[0] else deps
    line(axs[0,1],dr,"duplicate_roots","plan_brier",["ID","IH","JH_product","JH"])
    counts={int(r["duplicate_roots"]):int(r["count"]) for r in subset(dr,method="JH",seed=11)}
    axs[0,1].set_xticks(sorted(counts),[f"{d}\nn={counts[d]:,}" for d in sorted(counts)],fontsize=8)
    axs[0,1].legend(fontsize=7)
    panel(axs[0,1],"B  Shared-root dependency diagnostic","Candidate probability Brier score","Repeated roots; n = candidates in the fixed test pool")
    bar(axs[1,0],cc,"logical_violation",["B0","B2","P","ID","IH","JD","JH"])
    panel(axs[1,0],"C  Logical probability consistency","Fraction with child probability > parent","Method")
    for i,m in enumerate(["ID","JD","JH_product","JH"]):
        vals=[estimate([r for r in subset(cc,method=m) if mechanism_name(r)==a],"plan_brier") for a in mechanisms]
        axs[1,1].bar(np.arange(4)+(i-1.5)*.18,[v for v,e in vals],yerr=[e for v,e in vals],width=.18,color=COLORS[m],label=LABEL[m],capsize=2)
    axs[1,1].set_xticks(range(4),mechanisms)
    axs[1,1].set_ylim(0,axs[1,1].get_ylim()[1]*1.3)
    axs[1,1].legend(fontsize=7,ncol=2,loc="upper left")
    panel(axs[1,1],"D  Candidate probability quality","Plan Brier score (lower is better)","Core test condition")
    save(fig,"v2_fig02_joint_calibration",["summary.csv","calibration.csv","dependency_bins.csv"],"Joint candidate probabilities, dependency duplication and semantic consistency.")
    # 3: Retain every control, and separate data/full-update effects from new mechanism.
    fig,axs=plt.subplots(2,2,figsize=(10.5,9.1),layout="constrained")
    bar(axs[0,0],cc,"regret",ALL,horizontal=True)
    panel(axs[0,0],"A  All retained comparisons",None,"Four-condition macro regret")
    for i,m in enumerate(["ID","IH","JD","JH_ERM","JH_product"]):
        values=np.array(list(paired(cc,m,"plan_brier").values()));err=2.776*values.std(ddof=1)/np.sqrt(len(values))
        axs[0,1].errorbar(values.mean(),i,xerr=err,fmt="o",color=COLORS[m],capsize=3)
        axs[0,1].scatter(values,i+np.linspace(-.1,.1,len(values)),s=13,color=COLORS[m],alpha=.65)
    axs[0,1].axvline(0,color="#555",ls="--",lw=1);axs[0,1].set_yticks(range(5),[LABEL[m] for m in ["ID","IH","JD","JH_ERM","JH_product"]])
    panel(axs[0,1],"B  Paired probability-score controls",None,"JH minus comparator plan Brier")
    for m in ["B2","P"]+NEW:
        mr=subset(cc,method=m);means=seed_values(mr,"regret")
        worst={s:max(r["regret"] for r in mr if int(r["seed"])==s) for s in seeds}
        x,y=np.mean(list(means.values())),np.mean(list(worst.values()))
        axs[1,0].scatter(x,y,color=COLORS[m],marker=MARKERS[m],s=40,label=m)
    panel(axs[1,0],"C  Average and worst core condition","Worst-condition regret, then seed mean","Macro regret")
    axs[1,0].legend(ncol=2,fontsize=7,loc="lower right")
    for m in ["B2","P"]+NEW:
        x,xe=estimate(subset(cc,method=m),"brier"); y,ye=estimate(subset(cc,method=m),"plan_brier")
        axs[1,1].errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[m],color=COLORS[m],capsize=2,label=LABEL[m])
    panel(axs[1,1],"D  Fact versus plan reconstruction","Plan Brier score","Fact Brier score")
    axs[1,1].legend(ncol=2,fontsize=6.8)
    save(fig,"v2_fig03_matched_controls",["summary.csv"],"All retained methods and matched comparisons for representation, objective and joint scoring.")
    # 4: Evidence regime and measured inference cost, not a deployment claim.
    fig,axs=plt.subplots(2,2,figsize=(10.2,7.4),layout="constrained")
    bw=[r for r in rows if r["facts"]==64 and r["attack"]=="targeted" and r["corruption"]==.2 and r.get("missing",.2)==.2]
    line(axs[0,0],bw,"budget","regret")
    panel(axs[0,0],"A  Shared-codec evidence budgets","Candidate-set regret","Allowed bytes per remote source")
    axs[0,0].legend(fontsize=7,ncol=2)
    miss=[r for r in rows if r["facts"]==64 and r["budget"]==2048 and r["corruption"]==.2]
    for i,m in enumerate(FOCUS):
        vals=[estimate(subset(miss,method=m,missing=x),"regret") for x in [.2,.5]]
        axs[0,1].bar(np.arange(2)+(i-2)*.15,[v for v,e in vals],yerr=[e for v,e in vals],width=.15,color=COLORS[m],label=LABEL[m],capsize=2)
    axs[0,1].set_xticks(range(2),["20% missing","50% missing"])
    panel(axs[0,1],"B  Missing-observation shift","Regret averaged across three attacks","Observation missingness")
    axs[0,1].set_ylim(0,axs[0,1].get_ylim()[1]*1.3)
    axs[0,1].legend(fontsize=7,ncol=2,loc="upper left")
    sizes=[r for r in rows if r["attack"]=="coherent" and r["corruption"]==.2 and r["budget"]==2048 and r.get("missing",.2)==.2]
    line(axs[1,0],sizes,"facts","regret")
    panel(axs[1,0],"C  Size plus topology extrapolation","Candidate-set regret","Fact slots (not host count)")
    axs[1,0].legend(fontsize=7,ncol=2)
    for m in ["B2","P","ID","IH","JD","JH_ERM","JH"]:
        x,xe=estimate(subset(rr,method=m),"latency_ms");y,ye=estimate(subset(cc,method=m),"regret")
        axs[1,1].errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[m],color=COLORS[m],capsize=2,label=m)
    panel(axs[1,1],"D  GPU decision cost","Four-condition macro regret","Batch-one CUDA inference + ranking (ms)")
    axs[1,1].legend(ncol=2,fontsize=7,loc="center right")
    save(fig,"v2_fig04_evidence_compute",["summary.csv"],"Common-codec evidence, missingness and topology shifts, with CUDA inference cost.")
    # Additional figures included only in the three-page supplementary record.
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.0),layout="constrained")
    for j,split in enumerate(["train","validation"]):
        for m in ["ID","JD","JH_ERM","JH"]:
            mr=subset(curves,method=m,split=split)
            if not mr: continue
            xs=sorted({r["epoch"] for r in mr});ys=[estimate(subset(mr,epoch=x),"loss")[0] for x in xs]
            axs[j].plot(xs,ys,color=COLORS[m],label=LABEL[m],lw=1.4)
        panel(axs[j],f"{'AB'[j]}  {split.capitalize()} objective","Recorded objective","Epoch")
        axs[j].legend(fontsize=8)
        axs[j].tick_params(labelsize=9)
        axs[j].xaxis.label.set_size(10)
        axs[j].yaxis.label.set_size(10)
    fig.suptitle("Epoch-wise means use available seeds; early stopping changes the contributing cohort",fontsize=8)
    save(fig,"v2_figS1_training",["curves.csv","summary.csv"],"New model training and validation optimization traces.",True)
    fig,axs=plt.subplots(1,2,figsize=(8.7,3.15),layout="constrained")
    matrix=np.zeros((4,5))
    for i,a in enumerate(mechanisms):
        effects=paired([r for r in cc if mechanism_name(r)==a],"ID")
        matrix[i]=[effects[s] for s in seeds]
    lim=max(float(np.abs(matrix).max()),.001)
    im=axs[0].imshow(matrix,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
    for i in range(4):
        for j in range(5): axs[0].text(j,i,f"{matrix[i,j]:+.3f}",ha="center",va="center",fontsize=7,color="white" if abs(matrix[i,j])>.5*lim else "black")
    axs[0].set_xticks(range(5),seeds);axs[0].set_yticks(range(4),mechanisms)
    panel(axs[0],"A  JH minus ID regret","Core condition","Training seed")
    fig.colorbar(im,ax=axs[0],label="Negative favors JH",shrink=.8)
    bar(axs[1],rr,"peak_vram_mb",["B2","P","ID","IH","JD","JH"])
    panel(axs[1],"B  Incremental GPU allocation","Peak above resident baseline (MiB)","Method")
    save(fig,"v2_figS2_seed_effects",["summary.csv"],"Per-seed effects and incremental GPU allocation; does not measure total deployment memory.",True)
    gallery(manifest)
    print("Created four v2 main figures, two extra figures and gallery.")


if __name__=="__main__":
    main()
