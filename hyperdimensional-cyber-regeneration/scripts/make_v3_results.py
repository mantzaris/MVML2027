#!/usr/bin/env python3
"""Render the third study from complete recorded GPU measurements only.

The first and second study's inputs and artifacts are never rewritten. Fact
metrics are undefined for direct candidate models and are not imputed.
"""
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

DATA = ROOT / "results/v3"
OUT = ROOT / "figures/v3"
SUPP = ROOT / "supplemental/v3"
SEEDS = [11, 23, 37, 51, 71]
NEW = ["LOGIT", "DAE", "CAND_LOCAL", "CAND_CTX", "SCR"]
DIRECT = ["CAND_LOCAL", "CAND_CTX", "SCR"]
FOCUS = ["ID", "JH", "LOGIT", "DAE", "CAND_CTX", "SCR"]
LABEL = {
    "B0": "Constrained prior", "B1": "Old GraphMAE", "B2": "Old dense full",
    "B3": "HDC cleanup", "B4": "Old P, full", "B5": "Old entropy gate",
    "P": "Old learned gate", "random_gate": "Old random gate",
    "no_provenance": "Old source ablation", "no_semantic": "Old semantic ablation",
    "dense_gate": "Old dense gate", "ID": "Independent dense",
    "IH": "Independent HDC", "JD": "Joint dense", "JH": "Joint HDC",
    "JH_ERM": "Joint HDC, mean risk", "JH_product": "JH marginal product",
    "LOGIT": "Fact logistic", "DAE": "Denoising autoencoder",
    "CAND_LOCAL": "Local candidate", "CAND_CTX": "Context candidate",
    "SCR": "Shared-context ranking",
}
COLORS = {
    "B0": "#718096", "B1": "#64829c", "B2": "#b98249", "B3": "#9e77b1",
    "B4": "#cc79a7", "B5": "#e69f00", "P": "#737373", "random_gate": "#97a6b5",
    "no_provenance": "#a18a69", "no_semantic": "#bd7372", "dense_gate": "#8887a2",
    "ID": "#0072b2", "IH": "#56b4e9", "JD": "#b05a9d", "JH": "#009e73",
    "JH_ERM": "#947133", "JH_product": "#e69f00", "LOGIT": "#8d6b45",
    "DAE": "#9b59b6", "CAND_LOCAL": "#df9500", "CAND_CTX": "#d55e00",
    "SCR": "#cc3366",
}
MARKERS = {m: marker for m, marker in zip(LABEL, ("o", "s", "^", "v", "P", "D", "h", "x", "+", "*", "X", "s", "d", "^", "o", "v", "X", "P", "D", "v", "^", "o"))}
MECHANISMS = ["benign", "targeted", "random", "coherent"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    path = DATA / name
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty measurement file: {path}")
    for row in rows:
        for key, value in row.items():
            if value in ("", "NA", "N/A", "null", "None"):
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return rows


def subset(rows, **filters):
    return [row for row in rows if all(row.get(key) == val for key, val in filters.items())]


def ordinary(row):
    return row.get("missing", .2) == .2 and row["facts"] == 64 and row["budget"] == 2048


def core(rows):
    return [row for row in rows if ordinary(row) and
            (row["corruption"] == 0 or (row["corruption"] == .2 and row["attack"] in ["targeted", "random", "coherent"]))]


def reference(rows):
    return [row for row in rows if ordinary(row) and row["corruption"] == .2 and row["attack"] == "targeted"]


def mechanism_name(row):
    return "benign" if row["corruption"] == 0 else row["attack"]


def seed_values(rows, metric):
    values = {}
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.setdefault(int(row["seed"]), []).append(float(value))
    return {seed: float(np.mean(value)) for seed, value in values.items()}


def estimate(rows, metric):
    values = np.array(list(seed_values(rows, metric).values()))
    if not len(values):
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return values.mean(), critical * values.std(ddof=1) / np.sqrt(len(values))


def paired(rows, method, comparator, metric="regret"):
    left = seed_values(subset(rows, method=method), metric)
    right = seed_values(subset(rows, method=comparator), metric)
    return {seed: left[seed] - right[seed] for seed in sorted(left.keys() & right.keys())}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .15, "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def panel(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, loc="left", pad=9)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def line(ax, rows, xkey, metric, methods=FOCUS):
    for method in methods:
        selected = subset(rows, method=method)
        xs = sorted({row[xkey] for row in selected})
        if not xs:
            continue
        vals, errs = np.array([estimate(subset(selected, **{xkey: x}), metric) for x in xs]).T
        ax.plot(xs, vals, marker=MARKERS[method], color=COLORS[method], label=method,
                lw=2.2 if method == "SCR" else 1.4, ms=4)
        ax.fill_between(xs, vals - errs, vals + errs, color=COLORS[method], alpha=.08)


def bar(ax, rows, metric, methods, horizontal=False, long_labels=False):
    methods = [method for method in methods if subset(rows, method=method)]
    vals, errs = np.array([estimate(subset(rows, method=method), metric) for method in methods]).T
    x = np.arange(len(methods))
    labels = [LABEL[method] if long_labels else method for method in methods]
    if horizontal:
        ax.barh(x, vals, xerr=errs, color=[COLORS[m] for m in methods], capsize=2)
        ax.set_yticks(x, labels)
        ax.invert_yaxis()
    else:
        ax.bar(x, vals, yerr=errs, color=[COLORS[m] for m in methods], capsize=2)
        ax.set_xticks(x, labels)
    return methods


def save(fig, stem, inputs, title, supplementary=False):
    directory = SUPP if supplementary else OUT
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", pil_kwargs={"optimize": True})
    manifest = json.loads((DATA / "manifest.json").read_text())
    metadata = {
        "title": title, "script": "scripts/make_v3_results.py", "script_sha256": sha(Path(__file__)),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": {str((DATA / name).relative_to(ROOT)): sha(DATA / name) for name in inputs},
        "seeds": SEEDS, "test_generator_seed": manifest["settings"]["test_seed"],
        "filtering": "Core: 64 facts, 2048-byte budget, 20% missing; benign plus targeted/random/coherent at 20% requested intervention. Reference: targeted 20% only. Shift panels label changed variable.",
        "aggregation": "Equal condition mean within seed, then mean and Student-t 95% interval across five training seeds. Calibration pools probabilities by bin counts. Optimization traces average all five training seeds at every epoch; all 25 new models reached the 40-epoch cap.",
        "uncertainty_limit": "Intervals condition on one shared fresh test pool; crossed seed/topology bootstrap is recorded separately. Pooled candidate counts are not independent sample counts.",
        "not_applicable": "Direct candidate models have no fact posterior: fact Brier and logical consistency are not applicable, not zero.",
        "output_sha256": {ext: sha(directory / f"{stem}.{ext}") for ext in ["pdf", "png"]},
    }
    (directory / f"{stem}.provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)


def validate(rows, manifest):
    if manifest["settings"]["test_seed"] != 62027 or manifest["settings"].get("pilot"):
        raise ValueError("Final figures require the frozen fresh third-study test pool")
    methods = manifest["methods"]
    if len(methods) != 22 or len(rows) != 5 * 22 * 18:
        raise ValueError(f"Complete 1,980-row third study required; found {len(rows)} rows")
    if sorted({int(row["seed"]) for row in rows}) != SEEDS:
        raise ValueError("The complete five-seed study is required")
    keys = [(r["seed"], r["method"], r["condition"], r["attack"], r["corruption"], r["facts"], r["budget"], r.get("missing", .2)) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate condition measurements")
    for method in methods:
        if len(subset(core(rows), method=method)) != 20:
            raise ValueError(f"Expected four core conditions for each seed of {method}")
        if method not in LABEL or method not in MARKERS:
            raise ValueError(f"Missing plot identity for {method}")
    for row in rows:
        if row["method"] in DIRECT and isinstance(row.get("brier"), (int, float)) and np.isfinite(row["brier"]):
            raise ValueError("Candidate-only fact Brier must be not applicable")
    return methods


def effect_panel(ax, rows, comparisons, metric="regret"):
    for i, (method, comparator) in enumerate(comparisons):
        vals = np.array(list(paired(rows, method, comparator, metric).values()))
        if len(vals) != len(SEEDS):
            raise ValueError("Incomplete paired training-seed contrast")
        err = 2.776 * vals.std(ddof=1) / np.sqrt(len(vals))
        ax.errorbar(vals.mean(), i, xerr=err, fmt="o", color=COLORS[comparator], capsize=3)
        ax.scatter(vals, i + np.linspace(-.1, .1, len(vals)), s=12, color=COLORS[comparator], alpha=.65)
    ax.axvline(0, color="#555", ls="--", lw=1)
    ax.set_yticks(range(len(comparisons)), [f"{a} - {b}" for a, b in comparisons])
    ax.invert_yaxis()


def gallery(manifest):
    cards = [
        ("v3_fig01_decision_quality", "Direct candidate learning and decision quality",
         "Fresh test seed 62027, five training seeds, four equally weighted core conditions. The primary comparison is SCR versus ID; context and mixture contrasts are descriptive. Old models are reevaluated from frozen checkpoints."),
        ("v3_fig02_probability_semantics", "Probability quality and symbolic limits",
         "Panels A/B use targeted 20% interventions; C/D are four-condition macro averages. Candidate calibration and Brier scores are comparable across all methods. The final repeated-root bin has only seven candidates and is a fragile descriptive diagnostic. Dependency bins report candidates in the one fixed test pool; reused candidates across training seeds are not independent samples. Direct candidate methods have no fact posterior. The shadow-union audit tests whether a union of adjacent prerequisite sets is assigned more success probability than either constituent; this is a limited diagnostic, not a proof of full logical coherence."),
        ("v3_fig03_model_cost", "All comparisons and measured model cost",
         "All 22 methods remain visible. CUDA timing includes warmed inference and candidate ranking but excludes host I/O. Training cost and parameter counts compare only newly trained methods; DAE checkpoint parameters include its pretraining decoder. Comparing candidate learners with fact learners also changes the prediction target and training loss."),
        ("v3_fig04_transfer", "Evidence shifts and latent-mixture behavior",
         "Every method uses the same sparse symbolic report codec. Larger fact sets also change root density. SCR gate entropy is averaged over snapshots and training seeds; it is invariant to expert label permutations. Latent experts are not identified deception classes. No wire-codec compression gain is tested."),
    ]
    sections = []
    for stem, title, caption in cards:
        sections.append(f'<section><h2>{html.escape(title)}</h2><img src="{stem}.png" alt="{html.escape(title)}"><p>{html.escape(caption)}</p><a href="{stem}.pdf">Vector PDF</a> · <a href="{stem}.provenance.json">Provenance</a></section>')
    for stem, title in [("v3_figS1_training", "S1: Optimization traces"), ("v3_figS2_seed_effects", "S2: Paired effects and GPU allocation")]:
        sections.append(f'<section><h2>{title}</h2><img src="../../supplemental/v3/{stem}.png" alt="{title}"><a href="../../supplemental/v3/{stem}.pdf">Vector PDF</a> · <a href="../../supplemental/v3/{stem}.provenance.json">Provenance</a></section>')
    legend = "".join(f"<tr><td><code>{html.escape(m)}</code></td><td>{html.escape(LABEL[m])}</td></tr>" for m in manifest["methods"])
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Third GPU study</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:30px;color:#17283f;background:#f5f8fb}section{background:white;padding:24px;margin:24px 0;border:1px solid #dae3ed;border-radius:12px}img{width:100%;height:auto}a{color:#8c2450}h1{line-height:1.2}td{padding:3px 20px 3px 0}</style><h1>Candidate learning and cheap ML: third measured study</h1><p>Four empirical figures and two additional supplemental figures from the complete CUDA study. Error bars are 95% training-seed t intervals conditional on the shared test pool; the primary crossed seed/topology bootstrap is reported separately.</p><p><a href="../../supplemental/v3/Supplement_v3.pdf">Three-page supplement</a> · <a href="../../docs/RESULTS_V3.md">Results and limits</a> · <a href="../v2/index.html">Preserved second study</a></p>''' + ''.join(sections) + '<section><h2>Method key</h2><table>' + legend + '</table></section></html>'
    (OUT / "index.html").write_text(page)


def main():
    style()
    rows, cal, curves, deps = [load(name) for name in ["summary.csv", "calibration.csv", "curves.csv", "dependency_bins.csv"]]
    manifest = json.loads((DATA / "manifest.json").read_text())
    methods = validate(rows, manifest)
    cc, rr = core(rows), reference(rows)

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.7), layout="constrained")
    bar(axs[0, 0], cc, "regret", ["B2", "ID", "JH"] + NEW)
    axs[0, 0].tick_params(axis="x", labelrotation=28, labelsize=8)
    panel(axs[0, 0], "A  Four-condition macro regret", "Candidate-set regret", "Method")
    width = .135
    for i, method in enumerate(FOCUS):
        stats = [estimate([r for r in subset(cc, method=method) if mechanism_name(r) == attack], "regret") for attack in MECHANISMS]
        axs[0, 1].bar(np.arange(4) + (i - 2.5) * width, [v for v, _ in stats],
                      yerr=[e for _, e in stats], width=width, color=COLORS[method], label=method, capsize=2)
    axs[0, 1].set_xticks(range(4), MECHANISMS)
    axs[0, 1].set_ylim(0, axs[0, 1].get_ylim()[1] * 1.30)
    axs[0, 1].legend(ncol=3, loc="upper left")
    panel(axs[0, 1], "B  Mechanism-specific decisions", "Candidate-set regret", "Core test condition")
    targeted = [r for r in rows if ordinary(r) and r["attack"] == "targeted"]
    line(axs[1, 0], targeted, "corruption", "regret", ["B2", "ID", "JH", "DAE", "CAND_CTX", "SCR"])
    axs[1, 0].legend(ncol=3)
    panel(axs[1, 0], "C  Targeted-intervention severity", "Candidate-set regret", "Requested report-intervention fraction")
    effect_panel(axs[1, 1], cc, [("SCR", "ID"), ("SCR", "CAND_CTX"), ("CAND_CTX", "CAND_LOCAL"), ("SCR", "DAE"), ("SCR", "LOGIT")])
    panel(axs[1, 1], "D  Paired architectural contrasts", None, "Regret difference (negative favors first method)")
    save(fig, "v3_fig01_decision_quality", ["summary.csv", "manifest.json"], "Fresh-test decisions and paired contrasts among candidate and belief models.")

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.7), layout="constrained")
    cr = [r for r in cal if r["condition"] == "severity" and r["attack"] == "targeted" and r["corruption"] == .2]
    for method in FOCUS:
        mr = subset(cr, method=method)
        xx, yy = [], []
        for bin_id in sorted({r["bin"] for r in mr}):
            br = [r for r in subset(mr, bin=bin_id) if r["count"] > 0]
            count = sum(r["count"] for r in br)
            if count:
                xx.append(sum(r["confidence"] * r["count"] for r in br) / count)
                yy.append(sum(r["accuracy"] * r["count"] for r in br) / count)
        axs[0, 0].plot(xx, yy, marker=MARKERS[method], color=COLORS[method], label=method, ms=4)
    axs[0, 0].plot([0, 1], [0, 1], ls="--", color="#777", lw=1)
    axs[0, 0].set(xlim=(0, 1), ylim=(0, 1))
    axs[0, 0].legend(ncol=2)
    panel(axs[0, 0], "A  Candidate-success calibration", "Observed feasibility", "Predicted success probability")
    dr = reference(deps) if "facts" in deps[0] else [r for r in deps if r.get("attack", "targeted") == "targeted" and r.get("corruption", .2) == .2]
    supported_deps = [r for r in dr if r["count"] > 0]
    line(axs[0, 1], supported_deps, "duplicate_roots", "plan_brier", ["ID", "JH", "DAE", "CAND_CTX", "SCR"])
    counts = {int(r["duplicate_roots"]): int(r["count"]) for r in subset(dr, method="SCR", seed=11)}
    axs[0, 1].set_xticks(sorted(counts), [f"{d}\nn={counts[d]:,}" + (" (omitted)" if counts[d] == 0 else "") for d in sorted(counts)], fontsize=8)
    axs[0, 1].legend(ncol=2)
    panel(axs[0, 1], "B  Shared-prerequisite diagnostic", "Candidate Brier score", "Repeated roots; n = candidates in fixed test pool")
    bar(axs[1, 0], cc, "brier", ["B2", "ID", "JH", "LOGIT", "DAE"])
    axs[1, 0].text(.03, .96, "SCR / CAND_CTX / CAND_LOCAL: N/A\nDirect models do not output fact beliefs", transform=axs[1, 0].transAxes, va="top", fontsize=8,
                   bbox={"facecolor": "white", "edgecolor": "#d0d5dc", "boxstyle": "round,pad=.4"})
    axs[1, 0].set_ylim(0, axs[1, 0].get_ylim()[1] * 1.35)
    panel(axs[1, 0], "C  Fact reconstruction, when defined", "Macro fact Brier score", "Belief-producing methods only")
    bar(axs[1, 1], cc, "shadow_union_violation", ["ID", "JH", "LOGIT", "DAE", "CAND_LOCAL", "CAND_CTX", "SCR"])
    axs[1, 1].tick_params(axis="x", labelrotation=28, labelsize=8)
    panel(axs[1, 1], "D  Candidate conjunction coherence", "Fraction violating union monotonicity", "Adjacent candidate-pair shadow audit")
    save(fig, "v3_fig02_probability_semantics", ["summary.csv", "calibration.csv", "dependency_bins.csv", "manifest.json"], "Candidate calibration, dependency support and the limits of direct probability semantics.")

    fig, axs = plt.subplots(2, 2, figsize=(10.9, 10.1), layout="constrained")
    order = [m for m in LABEL if m in methods]
    bar(axs[0, 0], cc, "regret", order, horizontal=True)
    axs[0, 0].tick_params(axis="y", labelsize=7.2)
    panel(axs[0, 0], "A  All 22 retained comparisons", None, "Four-condition macro regret")
    for method in ["B2", "P", "ID", "JH"] + NEW:
        x, xe = estimate(subset(rr, method=method), "latency_ms")
        y, ye = estimate(subset(cc, method=method), "regret")
        axs[0, 1].errorbar(x, y, xerr=xe, yerr=ye, fmt=MARKERS[method], color=COLORS[method], capsize=2, label=method)
    axs[0, 1].legend(ncol=2, loc="best")
    panel(axs[0, 1], "B  Warm GPU decision cost", "Four-condition macro regret", "Batch-one CUDA inference + ranking (ms)")
    checkpoint_rows = manifest["checkpoints"]
    for method in NEW:
        mr = [r for r in checkpoint_rows if r["method"] == method]
        if len(mr) != 5:
            raise ValueError(f"Expected five checkpoint records for {method}")
        counts = [r.get("parameters", r.get("parameter")) for r in mr]
        if any(v is None for v in counts):
            raise ValueError("Checkpoint parameter counts must be measured")
        times = np.array([r["train_seconds"] for r in mr])
        err = 2.776 * times.std(ddof=1) / np.sqrt(len(times))
        axs[1, 0].errorbar(np.mean(counts), times.mean(), yerr=err, fmt=MARKERS[method], color=COLORS[method], capsize=2, label=method, ms=6)
    axs[1, 0].set_xscale("log")
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].legend(ncol=1)
    panel(axs[1, 0], "C  New-model training cost", "Pretraining + supervised wall time (s)", "Checkpoint parameters (DAE includes decoder)")
    for method in ["B2", "P", "ID", "JH"] + NEW:
        x, xe = estimate(subset(cc, method=method), "plan_brier")
        y, ye = estimate(subset(cc, method=method), "regret")
        axs[1, 1].errorbar(x, y, xerr=xe, yerr=ye, fmt=MARKERS[method], color=COLORS[method], capsize=2, label=method)
    axs[1, 1].legend(ncol=2, loc="best")
    panel(axs[1, 1], "D  Probability and decision quality", "Four-condition macro regret", "Four-condition candidate Brier score")
    save(fig, "v3_fig03_model_cost", ["summary.csv", "manifest.json"], "All retained controls, measured CUDA costs and new-model training-resource comparisons.")

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.7), layout="constrained")
    bw = [r for r in rows if r["facts"] == 64 and r["attack"] == "targeted" and r["corruption"] == .2 and r.get("missing", .2) == .2]
    line(axs[0, 0], bw, "budget", "regret")
    axs[0, 0].legend(ncol=3)
    panel(axs[0, 0], "A  Shared-codec evidence budgets", "Candidate-set regret", "Allowed bytes per remote source")
    miss = [r for r in rows if r["facts"] == 64 and r["budget"] == 2048 and r["corruption"] == .2]
    for i, method in enumerate(FOCUS):
        stats = [estimate(subset(miss, method=method, missing=value), "regret") for value in [.2, .5]]
        axs[0, 1].bar(np.arange(2) + (i - 2.5) * width, [v for v, _ in stats], yerr=[e for _, e in stats], width=width, color=COLORS[method], label=method, capsize=2)
    axs[0, 1].set_xticks(range(2), ["20% missing", "50% missing"])
    axs[0, 1].set_ylim(0, axs[0, 1].get_ylim()[1] * 1.3)
    axs[0, 1].legend(ncol=3, loc="upper left")
    panel(axs[0, 1], "B  Missing-observation shift", "Regret averaged across three attacks", "Observation missingness")
    sizes = [r for r in rows if r["attack"] == "coherent" and r["corruption"] == .2 and r["budget"] == 2048 and r.get("missing", .2) == .2]
    line(axs[1, 0], sizes, "facts", "regret")
    axs[1, 0].legend(ncol=3)
    panel(axs[1, 0], "C  Size plus topology extrapolation", "Candidate-set regret", "Fact slots (not host count)")
    stats = [estimate([r for r in subset(cc, method="SCR") if mechanism_name(r) == attack], "gate_entropy") for attack in MECHANISMS]
    if any(not np.isfinite(value) for value, _ in stats):
        raise ValueError("SCR gate entropy diagnostics are required")
    axs[1, 1].bar(np.arange(4), [v for v, _ in stats], yerr=[e for _, e in stats], color=COLORS["SCR"], capsize=3, alpha=.85)
    axs[1, 1].axhline(np.log(4), color="#555", ls="--", label="Uniform four-expert gate")
    axs[1, 1].set_xticks(range(4), MECHANISMS)
    axs[1, 1].set_ylim(0, np.log(4) * 1.17)
    axs[1, 1].legend(loc="upper left")
    panel(axs[1, 1], "D  SCR latent-mixture behavior", "Mean gate entropy (nats)", "Core condition; experts are not attack labels")
    save(fig, "v3_fig04_transfer", ["summary.csv", "manifest.json"], "Evidence and topology shifts with permutation-invariant latent-mixture diagnostics.")

    fig, axs = plt.subplots(1, 2, figsize=(8.7, 3.1), layout="constrained")
    for j, split in enumerate(["train", "validation"]):
        for method in NEW:
            mr = subset(curves, method=method, split=split)
            xs = sorted({r["epoch"] for r in mr})
            ys = [estimate(subset(mr, epoch=x), "loss")[0] for x in xs]
            axs[j].plot(xs, ys, color=COLORS[method], label=method, lw=1.4)
        panel(axs[j], f"{'AB'[j]}  {split.capitalize()} objective", "Recorded objective", "Supervised epoch")
        axs[j].legend(ncol=2, fontsize=7)
    fig.suptitle("Objectives differ by prediction target; five-seed means, all models reached the 40-epoch cap", fontsize=8)
    save(fig, "v3_figS1_training", ["curves.csv", "manifest.json"], "Recorded optimization traces for newly trained methods, with differing objective scales.", True)

    fig, axs = plt.subplots(1, 2, figsize=(8.7, 3.15), layout="constrained")
    matrix = np.array([[paired([r for r in cc if mechanism_name(r) == attack], "SCR", "ID")[seed] for seed in SEEDS] for attack in MECHANISMS])
    lim = max(float(np.abs(matrix).max()), .001)
    im = axs[0].imshow(matrix, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    axs[0].grid(False)
    for i in range(4):
        for j in range(5):
            axs[0].text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(matrix[i, j]) > .5 * lim else "black")
    axs[0].set_xticks(range(5), SEEDS)
    axs[0].set_yticks(range(4), MECHANISMS)
    panel(axs[0], "A  SCR minus ID regret", "Core condition", "Training seed")
    fig.colorbar(im, ax=axs[0], label="Negative favors SCR", shrink=.8)
    bar(axs[1], rr, "peak_vram_mb", ["ID", "JH"] + NEW)
    axs[1].tick_params(axis="x", labelrotation=35, labelsize=7)
    panel(axs[1], "B  Incremental GPU allocation", "Peak above resident baseline (MiB)", "Warmed batch-one inference")
    save(fig, "v3_figS2_seed_effects", ["summary.csv", "manifest.json"], "Paired seed-level effects and incremental GPU memory, excluding resident state.", True)
    gallery(manifest)
    print("Created four v3 main figures, two additional figures and an HTML gallery.")


if __name__ == "__main__":
    main()
