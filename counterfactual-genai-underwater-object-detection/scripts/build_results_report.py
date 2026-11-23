#!/usr/bin/env python3
"""Build honest, result-driven MVML figures and anonymous manuscript drafts.

Missing empirical inputs are fatal. --foundations-only builds only the method
diagram and already-measured numerical validation figure; no research results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uac.runtime import atomic_json, completed_stage, fingerprint, sha256_file, project_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    KeepTogether,
)
from pypdf import PdfReader

MANUSCRIPT = ROOT / "manuscript"
FIGURES = MANUSCRIPT / "figures"
RENDERS = MANUSCRIPT / "renders"
BLUE, ORANGE, TEAL, GRAY, RED = "#1f4e79", "#d97b2b", "#14877b", "#677482", "#b23b42"
COLORS = [BLUE, ORANGE, TEAL, GRAY, RED, "#7866a8", "#6c8d35", "#be7c99"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def read_json(path):
    path = project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required empirical artifact is missing: {path}")
    return json.loads(path.read_text())


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} requires an observed finite numeric value")
    return float(value)


def pct(value, digits=1):
    return "NA" if value is None else f"{100 * finite(value, 'proportion'):.{digits}f}%"


def num(value, digits=3):
    return "NA" if value is None else f"{finite(value, 'value'):.{digits}f}"


def short_method(name):
    names = {
        "full_agent": "Generative agent", "proposed": "Generative agent",
        "P": "Generative agent", "cnn": "Spectral CNN", "stft_cnn": "Spectral CNN",
        "foundation": "Frozen wav2vec 2.0", "fourier": "Fourier",
        "stft": "STFT", "wavelet": "Wavelet", "fusion": "Numeric fusion",
        "fixed_verification": "Fixed verification", "heuristic": "Heuristic",
        "heuristic_verification": "Heuristic verification", "main": "Numeric fusion",
        "proposed_agent": "Generative agent", "agent_no_generation": "Agent without generation",
        "fixed_validity_only": "Fixed, V = 0", "agent_validity_only": "Agent, V = 0",
        "full_record_null": "Full-record null",
        "no_generation": "Agent without generation", "no_null": "No null adjustment",
        "all_views": "All views", "all": "All four views", "stft_wavelet": "STFT + wavelet", "stft_hht": "STFT + HHT", "synthetic_supervised_temporal_cnn": "Controlled temporal CNN",
    }
    return names.get(str(name), str(name).replace("_", " "))


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def method_figure():
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    def box(x, y, w, h, text, color=BLUE, dashed=False):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                               linewidth=1.2, edgecolor=color, facecolor=color + "10",
                               linestyle="--" if dashed else "-")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, color="#172b3a", linespacing=1.2)
    def arrow(a, b, color=GRAY):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=color, lw=1))
    box(.1, 3.65, 2.05, 1, "Real acoustic clip\nChannel 0 / 16 kHz")
    box(2.65, 3.65, 3.25, 1, "Fourier  |  STFT  |  Wavelet\nMeasured numeric evidence")
    box(6.55, 3.65, 3.15, 1, "Frozen Qwen controller\nChoose tool, verify or stop")
    arrow((2.2, 4.15), (2.6, 4.15))
    arrow((5.95, 4.15), (6.5, 4.15))
    box(.1, 1.85, 2.05, 1.05, "HHT approximation\nExploratory only", GRAY, True)
    box(2.65, 1.85, 3.25, 1.05, "Central 1.5 s probe\nGenerated background +\nsame-support nuisance edit", ORANGE)
    box(6.55, 1.85, 3.15, 1.05, "Paired score change\nDispersion and validity checks", TEAL)
    arrow((8.1, 3.6), (8.1, 3.15))
    arrow((8.1, 3.15), (4.3, 3.15))
    arrow((4.3, 3.15), (4.3, 2.95))
    arrow((5.95, 2.4), (6.5, 2.4))
    box(2.65, .15, 3.25, 1, "Calibrated numeric detector\nNo LLM probability override", TEAL)
    box(6.55, .15, 3.15, 1, "Saved predictions and traces\nAtomic restart checkpoints", GRAY)
    arrow((8.1, 1.8), (8.1, 1.45))
    arrow((8.1, 1.45), (4.3, 1.45))
    arrow((4.3, 1.45), (4.3, 1.2))
    arrow((5.95, .65), (6.5, .65))
    ax.text(.1, .8, "One waveform:\ncorrelated views,\nnot independent evidence", fontsize=8, color=GRAY)
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    save_figure(fig, "figure1_method")


def hht_figure(dsp):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), constrained_layout=True)
    tone = dsp["known_tone"]
    errors = [tone["if_relative_error"], tone["wavelet_relative_error"]]
    axes[0].bar(["HHT tone\nfrequency", "Wavelet\npeak grid"], errors, color=[GRAY, BLUE])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Relative frequency error")
    axes[0].set_title("(a) 440 Hz fixture")
    for i, value in enumerate(errors):
        axes[0].annotate(f"{value:.2g}", (i, value), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8)
    rows = [r for r in dsp["benchmarks"] if r["representation"] == "hht"]
    failures = [100 * r["unconverged_or_extrema_failed_fraction"] for r in rows]
    axes[1].bar([f'{r["samples"] / 16000:g} s' for r in rows], failures, color=RED)
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Flagged modes (%)")
    axes[1].set_title("(b) Noisy scene limitation")
    for i, value in enumerate(failures):
        axes[1].text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=9)
    reconstruction = [tone["reconstruction_relative_rms"]] + [
        r["maximum_reconstruction_relative_rms"] for r in rows]
    axes[2].bar(["Tone", "3 s", "10 s"], reconstruction, color=TEAL)
    axes[2].axhline(dsp["thresholds"]["reconstruction_relative_rms_max"],
                    color=GRAY, ls="--", label="Fixture threshold")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Relative reconstruction RMS")
    axes[2].set_title("(c) Reconstruction")
    axes[2].legend(loc="upper left", fontsize=7)
    save_figure(fig, "figureS1_hht_numerics")


def group_records(rows, key="method"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def headline_records(real):
    """A result-independent display set; the development-locked comparator is compulsory.

    Every evaluated method remains in machine-readable tables. Membership in the display set
    never depends on test outcomes; plotting may order the included methods by AP.
    """
    methods = {"main", "cnn", "foundation", "fixed_verification", "fixed_validity_only",
               "heuristic_verification", "proposed_agent", "agent_validity_only", "agent_no_generation",
               real["primary_comparison"]["method_b"]}
    return [r for r in real["records"] if r["split"] == "test" and r["method"] in methods]


def development_policy_summary():
    paths = Counter(); sources = {}; count = 0
    for seed in (17, 29, 43):
        path = ROOT / "results/raw/policies/development" / f"agent_seed{seed}" / "all.json"
        index = read_json(path.parent / "index.json")
        if index.get("status") != "complete" or index.get("sha256") != sha256_file(path):
            raise ValueError("Final manuscript requires complete checksummed constrained development policies")
        records = read_json(path)["records"]
        for row in records:
            if not all(t.get("constrained_decoding") == "finite_legal_json_trie_v1" for t in row["trace"]):
                raise ValueError("Development policy description must use final constrained traces")
            paths[tuple(t["action"] for t in row["trace"])] += 1
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
        count += len(records)
    if len(paths) == 1:
        sequence = " then ".join(next(iter(paths)))
        description = (f"All {count:,} development clip-seed decisions followed {sequence}. "
                       "The observed controller therefore behaved as a fixed acquisition sequence; "
                       "these traces do not establish evidence-dependent adaptation or a reasoning benefit.")
    else:
        description = (f"The development controller used {len(paths)} distinct action sequences across "
                       f"{count:,} clip-seed decisions. Sequence variation alone does not establish useful adaptation.")
    return {"description": description, "source_sha256": sources,
            "path_counts": [{"actions": list(path), "n": n} for path, n in paths.items()], "n": count}


def test_policy_summary(real):
    """Describe usage from final metrics; validate action order against saved audit."""
    rows = [row for row in real["policy_usage"] if row["method"] == "proposed_agent"]
    if len(rows) != 3 or {row["seed"] for row in rows} != {17, 29, 43}:
        raise ValueError("Final test policy usage must cover each of the three detector seeds")
    count = sum(int(row["n"]) for row in rows)
    all_views_and_verify = all(math.isclose(row["mean_acquired_views"], 3., abs_tol=1e-12)
                               and math.isclose(row["verification_fraction"], 1., abs_tol=1e-12) for row in rows)
    audit_path = "reports/test_agent_behavior_audit.json"
    audit = read_json(audit_path)
    if len(audit["records"]) != 3 or {row["seed"] for row in audit["records"]} != {17, 29, 43}:
        raise ValueError("Test action-order audit seed coverage differs")
    source_sha256 = {audit_path: sha256_file(audit_path)}
    path_counts = Counter()
    for record in audit["records"]:
        expected_n = next(row["n"] for row in rows if row["seed"] == record["seed"])
        if record["n"] != expected_n or sum(path["count"] for path in record["paths"]) != expected_n:
            raise ValueError("Test action-order audit and final usage counts differ")
        if sha256_file(record["source"]) != record["source_sha256"]:
            raise ValueError("Test action-order source changed after audit")
        source_sha256[record["source"]] = record["source_sha256"]
        for path in record["paths"]:
            path_counts[tuple(path["actions"])] += path["count"]
    verification_last = (all_views_and_verify and len(path_counts) == 1
                         and next(iter(path_counts)) == ("wavelet", "fourier", "verify"))
    if verification_last:
        description = (f"All {count:,} held-out test clip-seed decisions acquired all three views and requested "
                       "verification. Every audited path was wavelet, Fourier, then verify. Verification was the "
                       "final action, so generated evidence informed no subsequent tool choice. The observed "
                       "test policy did not perform adaptive tool selection.")
        short = (f"All {count:,} test clip-seed paths were wavelet, Fourier, then verify; generation informed "
                 "no later action and did not demonstrate adaptive selection.")
    else:
        fraction = sum(row["n"] * row["verification_fraction"] for row in rows) / count
        views = sum(row["n"] * row["mean_acquired_views"] for row in rows) / count
        description = (f"Across {count:,} test clip-seed decisions, verification was requested in {pct(fraction)} "
                       f"and the mean acquired-view count was {views:.2f}. Usage variation alone does not establish useful adaptation.")
        short = description
    return {"description": description, "short_description": short, "n": count,
            "all_three_views_and_verification": all_views_and_verify,
            "verification_last_for_all": verification_last, "source_sha256": source_sha256,
            "path_counts": [{"actions": list(path), "n": n} for path, n in path_counts.items()]}


def result_figure(real):
    rows = headline_records(real)
    grouped = group_records(rows)
    order = sorted(grouped, key=lambda m: np.mean([r["ap"] for r in grouped[m]]))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, max(3.0, .24 * len(order) + 1.0)),
                             gridspec_kw={"width_ratios": [1.3, 1]}, constrained_layout=True)
    for index, method in enumerate(order):
        values = [r["ap"] for r in grouped[method]]
        axes[0].hlines(index, min(values), max(values), color=BLUE, lw=2)
        axes[0].scatter(values, [index] * len(values), s=22, color=BLUE, alpha=.7)
        recalls = [r["recall"] for r in grouped[method] if r["recall"] is not None]
        fprs = [r["fpr"] for r in grouped[method] if r["fpr"] is not None]
        axes[1].scatter(100 * np.mean(recalls), index + .12, s=28, color=BLUE,
                        label="Recall" if index == 0 else None)
        axes[1].scatter(100 * np.mean(fprs), index - .12, s=25, marker="s", color=ORANGE,
                        label="FPR" if index == 0 else None)
    axes[0].set_yticks(range(len(order)), [f"{i+1}. {short_method(m)}" for i, m in enumerate(order)])
    axes[0].set_xlim(0, 1.02)
    axes[0].set_xlabel("Average precision (each detector seed)")
    axes[0].set_title("(a) Natural real / AIS labels")
    axes[0].grid(axis="x", alpha=.2)
    axes[0].set_ylim(-.5, len(order) - .5)
    axes[1].set_xlabel("Mean test rate (%)")
    axes[1].set_yticks(range(len(order)), [""] * len(order))
    axes[1].set_ylim(-.5, len(order) - .5)
    largest_rate = 100 * max(np.mean([r[key] for r in grouped[m]])
                             for m in order for key in ("recall", "fpr"))
    rate_max = max(2.5, largest_rate * 1.5)
    axes[1].set_xlim(-.08, rate_max)
    axes[1].set_xticks(np.linspace(0, rate_max, 6))
    axes[1].set_title("(b) Calibration-frozen rates")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(axis="x", alpha=.2)
    save_figure(fig, "figure2_real_detection")


def robustness_figure(real):
    rows = real["robustness"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    snr_rows = [r for r in rows if r.get("snr_db") is not None]
    others = [r for r in rows if r.get("snr_db") is None]
    for index, (method, entries) in enumerate(group_records(snr_rows).items()):
        by_snr = defaultdict(list)
        for entry in entries:
            by_snr[float(entry["snr_db"])].append(entry["ap"])
        snrs = sorted(by_snr)
        styles = {"main": ("o", "-", 4), "stft": ("s", "--", 4),
                  "fixed_verification": ("D", "-", 5), "proposed_agent": ("o", ":", 8)}
        marker, line, size = styles.get(method, ("o", "-", 4))
        axes[0].plot(snrs, [np.mean(by_snr[s]) for s in snrs], marker=marker,
                     markerfacecolor="none" if method == "proposed_agent" else COLORS[index % len(COLORS)],
                     linestyle=line, ms=size, lw=1.4, label=short_method(method), color=COLORS[index % len(COLORS)])
    axes[0].set_xlabel("Added-noise SNR (dB)")
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("(a) Artificial corruption of real audio")
    axes[0].legend(fontsize=6.5, loc="best")
    if others:
        conditions = list(dict.fromkeys(r["condition"] for r in others))
        methods = list(group_records(others))
        width = .8 / max(len(methods), 1)
        for j, method in enumerate(methods):
            values = [np.mean([r["ap"] for r in others if r["method"] == method and r["condition"] == c])
                      if any(r["method"] == method and r["condition"] == c for r in others) else np.nan
                      for c in conditions]
            axes[1].bar(np.arange(len(conditions)) + j * width, values, width,
                        color=COLORS[j % len(COLORS)], label=short_method(method))
        axes[1].set_xticks(np.arange(len(conditions)) + (len(methods) - 1) * width / 2,
                          [c.replace("_", "\n") for c in conditions], fontsize=7)
    else:
        # This is an explicit data-origin warning, not a fabricated empty curve.
        axes[1].axis("off")
        axes[1].text(.05, .7, "No separate natural-domain shift\nresults supplied.\n\nAdded noise does not establish\nindependent-deployment generalization.",
                     transform=axes[1].transAxes, va="top", fontsize=10)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("(b) Clean shared subset" if others and all(r["condition"] == "clean" for r in others) else "(b) Other recorded conditions")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "figure3_robustness")


def mechanism_figure(real, synthetic):
    rows = real["ablations"]
    grouped = group_records(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, max(3.0, .3 * len(grouped) + 1)),
                             gridspec_kw={"width_ratios": [1.2, 1]}, constrained_layout=True)
    methods = list(grouped)
    for index, method in enumerate(methods):
        values = [r["ap"] for r in grouped[method]]
        axes[0].scatter(values, [index] * len(values), color=BLUE, s=22)
        axes[0].hlines(index, min(values), max(values), color=BLUE)
    axes[0].set_yticks(range(len(methods)), [short_method(m) for m in methods])
    axes[0].set_xlim(0, 1.02)
    axes[0].set_xlabel("Average precision")
    axes[0].set_title("(a) Recorded component ablations")
    contrasts = real["continuous_score_contrasts"]
    for index, row in enumerate(contrasts):
        values = list(row["per_seed_differences"].values())
        axes[1].scatter(values, [index] * len(values), color=ORANGE, s=20, label="Seed differences" if index == 0 else None)
        axes[1].hlines(index, min(values), max(values), color=ORANGE, lw=1)
        axes[1].scatter(row["difference_a_minus_b"], index, color=BLUE, marker="D", s=38,
                        label="AP of mean-seed probability" if index == 0 else None, zorder=4)
    axes[1].set_yticks(range(len(contrasts)), ["Fixed: V vs 0" if r["method_a"] == "fixed_verification" else "Agent: V vs 0" for r in contrasts])
    limit = max(.01, max(abs(value) for row in contrasts for value in
                        [row["difference_a_minus_b"]] + list(row["per_seed_differences"].values()))) * 1.3
    axes[1].set_xlim(-limit, limit)
    axes[1].set_ylim(-.8, len(contrasts) - .2)
    axes[1].axvline(0, color=GRAY, lw=.8)
    axes[1].set_xlabel("Descriptive paired AP difference")
    axes[1].set_title("(b) Continuous score contribution")
    axes[1].legend(fontsize=6.5, loc="lower center")
    axes[1].grid(axis="x", alpha=.2)
    axes[0].grid(axis="x", alpha=.2)
    save_figure(fig, "figure4_mechanism_ablation")


def compute_figure(real):
    rows = real["cost"]
    ap = {m: float(np.mean([r["ap"] for r in group]))
          for m, group in group_records([r for r in real["records"] if r["split"] == "test"]).items()}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3), constrained_layout=True)
    for index, row in enumerate(rows):
        method = row["method"]
        if method not in ap:
            raise ValueError(f"Cost method has no matching empirical AP: {method}")
        marker = "o" if row["mode"] == "live" else "x"
        if row.get("milliseconds_per_clip_min") is not None and row.get("milliseconds_per_clip_max") is not None:
            axes[0].errorbar(row["milliseconds_per_clip"], ap[method],
                            xerr=[[row["milliseconds_per_clip"] - row["milliseconds_per_clip_min"]],
                                  [row["milliseconds_per_clip_max"] - row["milliseconds_per_clip"]]],
                            fmt="none", ecolor=COLORS[index % len(COLORS)], capsize=2, lw=1)
        axes[0].scatter(row["milliseconds_per_clip"], ap[method], marker=marker, s=45,
                        color=COLORS[index % len(COLORS)])
        axes[0].annotate(str(index + 1), (row["milliseconds_per_clip"], ap[method]),
                         xytext=(4, 3), textcoords="offset points", fontsize=9)
        axes[1].bar(index, row["peak_memory_mib"] / 1024, color=COLORS[index % len(COLORS)],
                    hatch="" if row["mode"] == "live" else "//")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("Warm batch 32 milliseconds per clip (log scale)")
    axes[0].set_ylabel("Mean detector-seed AP")
    axes[0].set_title("(a) Accuracy and recorded cost")
    axes[0].plot([], [], "o", color=GRAY, label="Live measurement")
    if any(r["mode"] == "replay_estimate" for r in rows):
        axes[0].plot([], [], "x", color=GRAY, label="Replay cost estimate")
    axes[0].legend(loc="lower left", fontsize=7)
    axes[1].set_xticks(range(len(rows)), [f"{i+1}. {short_method(r['method'])}" for i, r in enumerate(rows)],
                      rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Peak allocated memory (GiB)")
    axes[1].set_title("(b) Recorded GPU memory")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "figure5_accuracy_compute")


def synthetic_figure(synthetic):
    rows = synthetic["records"]
    families = list(dict.fromkeys(r.get("family", "unspecified") for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    for i, family in enumerate(families):
        selected = [r for r in rows if r.get("family", "unspecified") == family]
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["iou_threshold"]].append(row["temporal_ap"])
        thresholds = sorted(grouped)
        axes[0].plot(thresholds, [np.mean(grouped[t]) for t in thresholds],
                     marker="o", label=family, color=COLORS[i % len(COLORS)])
        snr = defaultdict(list)
        for row in selected:
            value = row.get("per_source_snr_db", row.get("snr_db"))
            if value is not None and abs(row["iou_threshold"] - .5) < 1e-8:
                snr[value].append(row["temporal_ap"])
        if snr:
            levels = sorted(snr)
            axes[1].plot(levels, [np.mean(snr[s]) for s in levels],
                         marker="o", label=family, color=COLORS[i % len(COLORS)])
    axes[0].set_xlabel("Temporal IoU threshold")
    axes[0].set_ylabel("Union-activity average precision")
    axes[0].set_title("(a) Controlled temporal evaluation")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Per-source received-support SNR (dB)")
    axes[1].set_ylabel("Temporal AP at IoU 0.5")
    axes[1].set_title("(b) Controlled noise sensitivity")
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.2)
    save_figure(fig, "figureS2_controlled_temporal")



def supplemental_family_label(family):
    """Clarify held-out evaluation scope without changing frozen family keys."""
    return {"train": "in-family test", "heldout": "held-out test"}.get(family, str(family))


def supplemental_regime_label(regime):
    """The historical key counts two sources; temporal overlap is not required."""
    return "two-source scenes" if regime == "overlapping_sources" else regime


def supplemental_view_figure(views):
    rows = views["view_ablation"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    families = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    offsets = np.linspace(-.18, .18, max(len(families), 1))
    for j, family in enumerate(families):
        for i, method in enumerate(methods):
            values = [r["ap"] for r in rows if r["family"] == family and r["method"] == method]
            if values:
                axes[0].scatter([i + offsets[j]] * len(values), values,
                                color=COLORS[j % len(COLORS)], s=23,
                                label=supplemental_family_label(family) if i == 0 else None)
                axes[0].hlines(np.mean(values), i + offsets[j] - .08, i + offsets[j] + .08,
                               color=COLORS[j % len(COLORS)], lw=2)
    axes[0].set_xticks(range(len(methods)), [short_method(m).replace(" ", "\n") for m in methods], fontsize=7)
    axes[0].set_ylabel("Controlled presence AP")
    axes[0].set_title("(a) Frozen representation heads")
    axes[0].legend(fontsize=7)
    missing = views["missing_view"]
    conditions = list(dict.fromkeys(r["corruption"] + ":" + str(r.get("removed_view", "none")) for r in missing))
    missing_methods = list(dict.fromkeys(r["method"] for r in missing))
    for family_index, family in enumerate(families):
        for j, method in enumerate(missing_methods):
            values = []
            for condition in conditions:
                selected = [r["ap"] for r in missing if r["family"] == family and r["method"] == method
                            and r["corruption"] + ":" + str(r.get("removed_view", "none")) == condition]
                values.append(np.mean(selected) if selected else np.nan)
            axes[1].plot(range(len(conditions)), values, marker="o" if family_index == 0 else "x", ms=3,
                         linestyle="-" if family_index == 0 else "--", label=short_method(method) + " / " + supplemental_family_label(family),
                         color=COLORS[j % len(COLORS)])
    axes[1].set_xticks(range(len(conditions)), [c.split(":")[-1].replace("_", "\n") for c in conditions], fontsize=7)
    axes[1].set_ylabel("Controlled presence AP")
    axes[1].set_title("(b) Recorded missing-view tests")
    axes[1].legend(fontsize=6.5)
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "figureS3_representation_study")


def view_summary_rows(views):
    grouped = defaultdict(list)
    for row in views["view_ablation"]:
        grouped[(row["family"], row["method"])].append(row)
    return [[supplemental_family_label(family), short_method(method), num(np.mean([r["ap"] for r in rows])),
             len({r["seed"] for r in rows}), rows[0]["n_scenes"]]
            for (family, method), rows in sorted(grouped.items())]


def validate_supplement(views):
    for name in ("view_ablation", "regime_ablation", "missing_view"):
        if not isinstance(views.get(name), list) or not views[name]:
            raise ValueError(f"Required focused supplemental result missing: {name}")
        for row in views[name]:
            for key in ("method", "seed", "ap", "n_scenes"):
                if key not in row:
                    raise ValueError(f"Supplemental row missing {key}")
            finite(row["ap"], "supplement AP")
            finite(row["n_scenes"], "supplement n_scenes")


def validate_inputs(real, synthetic):
    required = ("records", "primary_comparison", "robustness", "ablations", "cost", "continuous_score_contrasts", "policy_usage")
    for name in required:
        if name not in real or not real[name]:
            raise ValueError(f"Required real-result field is missing/empty: {name}")
    for row in real["records"]:
        for name in ("method", "seed", "split", "ap", "auroc", "brier", "fpr", "recall",
                     "threshold", "n", "group_count"):
            if name not in row:
                raise ValueError(f"Missing real record field {name}")
        for name in ("ap", "auroc", "brier", "threshold", "n", "group_count"):
            finite(row[name], name)
    if not any(row["split"] == "test" for row in real["records"]):
        raise ValueError("No real test records supplied")
    comparison = real["primary_comparison"]
    for name in ("method_a", "method_b", "difference_a_minus_b", "ci", "independent_groups"):
        if name not in comparison:
            raise ValueError(f"Primary comparison missing {name}")
    finite(comparison["difference_a_minus_b"], "primary difference")
    if comparison["ci"] is None:
        if comparison["independent_groups"] != 1:
            raise ValueError("Missing primary interval must state the one-component dependence limitation")
    else:
        if not isinstance(comparison["ci"], list) or len(comparison["ci"]) != 2:
            raise ValueError("Primary interval must be null or two observed endpoints")
        if comparison["independent_groups"] < 2:
            raise ValueError("One dependence component cannot support an independent-component interval")
        for value in comparison["ci"]:
            finite(value, "primary interval")
    if not isinstance(synthetic.get("records"), list) or not synthetic["records"]:
        raise ValueError("Synthetic empirical records are missing")
    for row in synthetic["records"]:
        for name in ("method", "seed", "origin", "split", "iou_threshold", "temporal_ap",
                     "event_f1", "n_events", "n_scenes"):
            if name not in row:
                raise ValueError(f"Synthetic record missing {name}")
        for name in ("iou_threshold", "temporal_ap", "event_f1", "n_events", "n_scenes"):
            finite(row[name], name)
    for row in real["robustness"]:
        for key in ("method", "condition", "ap"):
            if key not in row:
                raise ValueError(f"Robustness missing {key}")
        finite(row["ap"], "robustness AP")
    for row in real["ablations"]:
        if "method" not in row:
            raise ValueError("Ablation requires method")
        finite(row.get("ap"), "ablation AP")
    expected_pairs = {("fixed_verification", "fixed_validity_only"), ("proposed_agent", "agent_validity_only")}
    if {(r.get("method_a"), r.get("method_b")) for r in real["continuous_score_contrasts"]} != expected_pairs:
        raise ValueError("Matched fixed/agent continuous-score contrasts are incomplete")
    for row in real["continuous_score_contrasts"]:
        finite(row.get("difference_a_minus_b"), "continuous-score AP contrast")
        if row.get("ci") is not None or row.get("independent_components") != 1:
            raise ValueError("Continuous-score contrasts must retain the one-component uncertainty limitation")
        if set(row.get("per_seed_differences", {})) != {"17", "29", "43"}:
            raise ValueError("Continuous-score contrast needs all three paired seed differences")
        for value in row["per_seed_differences"].values():
            finite(value, "continuous-score seed difference")
    expected_cost_methods = {"main", "fixed_verification", "heuristic_verification", "proposed_agent", "agent_no_generation"}
    if {r.get("method") for r in real["cost"]} != expected_cost_methods:
        raise ValueError("Five-method live cost coverage is incomplete")
    for row in real["cost"]:
        if row.get("mode") != "live":
            raise ValueError("Final cost figure requires actual live measurements")
        if finite(row.get("milliseconds_per_clip"), "latency") <= 0:
            raise ValueError("Cost latency must be positive")
        finite(row.get("peak_memory_mib"), "peak memory")


def configure_fonts():
    directory = Path("/usr/share/fonts/truetype/croscore")
    for name, filename in [
        ("Tinos", "Tinos-Regular.ttf"), ("Tinos-Bold", "Tinos-Bold.ttf"),
        ("Tinos-Italic", "Tinos-Italic.ttf"), ("Tinos-BoldItalic", "Tinos-BoldItalic.ttf"),
        ("Arimo", "Arimo-Regular.ttf"), ("Arimo-Bold", "Arimo-Bold.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
    pdfmetrics.registerFontFamily("Tinos", normal="Tinos", bold="Tinos-Bold",
                                 italic="Tinos-Italic", boldItalic="Tinos-BoldItalic")
    pdfmetrics.registerFontFamily("Arimo", normal="Arimo", bold="Arimo-Bold")
    styles = {
        "body": ParagraphStyle("body", fontName="Tinos", fontSize=11, leading=13,
                               alignment=TA_JUSTIFY, spaceAfter=6, firstLineIndent=7 * mm),
        "small": ParagraphStyle("small", fontName="Tinos", fontSize=9, leading=11, spaceAfter=4),
        "abstract": ParagraphStyle("abstract", fontName="Tinos", fontSize=10, leading=11.5, spaceAfter=5),
        "title": ParagraphStyle("title", fontName="Arimo-Bold", fontSize=16, leading=19,
                                alignment=TA_CENTER, spaceAfter=10),
        "h1": ParagraphStyle("h1", fontName="Arimo-Bold", fontSize=12, leading=14,
                             spaceBefore=8, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName="Arimo-Bold", fontSize=11, leading=13,
                             spaceBefore=6, spaceAfter=4),
        "caption": ParagraphStyle("caption", fontName="Tinos", fontSize=9.5, leading=11,
                                  alignment=TA_CENTER, spaceBefore=4, spaceAfter=8),
        "table": ParagraphStyle("table", fontName="Tinos", fontSize=8.5, leading=10),
        "table_head": ParagraphStyle("table_head", fontName="Arimo-Bold", fontSize=8.5, leading=10),
    }
    return styles


STYLES = None
WIDTH = letter[0] - 50 * mm - 12


def paragraph(text, kind="body"):
    return Paragraph(text, STYLES[kind])


def figure(name, caption, width=WIDTH, max_height=3.55 * inch):
    image = Image(str(FIGURES / f"{name}.png"))
    factor = min(width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth, image.drawHeight = image.imageWidth * factor, image.imageHeight * factor
    image.hAlign = "CENTER"
    return [image, paragraph(escape(caption), "caption")]


def table(headers, rows, widths=None):
    content = [[paragraph(escape(str(c)), "table_head") for c in headers]]
    content += [[paragraph(escape(str(c)), "table") for c in row] for row in rows]
    if widths and sum(widths) > WIDTH:
        widths = [value * WIDTH / sum(widths) for value in widths]
    output = Table(content, colWidths=widths, repeatRows=1, hAlign="CENTER")
    output.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f7")),
        ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor(BLUE)),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return output


def equation(text, number, name):
    fig = plt.figure(figsize=(6.3, .48))
    fig.text(.01, .45, text, fontsize=12, va="center")
    fig.text(.98, .45, f"({number})", fontsize=10, ha="right", va="center")
    fig.savefig(FIGURES / f"{name}.png", dpi=400, transparent=False, bbox_inches="tight")
    plt.close(fig)
    img = Image(str(FIGURES / f"{name}.png"))
    factor = WIDTH / img.imageWidth
    img.drawWidth, img.drawHeight = WIDTH, img.imageHeight * factor
    img.hAlign = "CENTER"
    return img


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arimo", 8)
    canvas.setFillColor(colors.HexColor("#66717b"))
    canvas.drawString(25 * mm, 16 * mm, "MVML 2027 | Anonymous research draft")
    canvas.drawRightString(letter[0] - 25 * mm, 16 * mm, str(doc.page))
    canvas.restoreState()


REFERENCES = [
    '[1] Fernández García et al., "PORTIA: A Large-Scale Port Acoustic Dataset with AIS-Grounded Annotations for AI-Based Vessel Monitoring," Scientific Data, 2026. doi:10.1038/s41597-026-08384-9. Dataset v2: doi:10.5281/zenodo.21409381.',
    '[2] Y. Zhang et al., "From Signals to Evidence: Interpretable Underwater Acoustic Target Sensing with Large Language Models - Part I," arXiv:2610.22064, 2026.',
    '[3] Z. Li et al., "From Signals to Evidence - Part II: Passive Detection," arXiv:2610.31346, 2026.',
    '[4] Z. Sun et al., "UA-Bench: Benchmarking Large Audio-Language Models on Underwater Acoustic Recognition," arXiv:2610.16942, 2026.',
    '[5] X. Feng, F. Xiao, L. Zhang, P. Feng, H. Pan, and J. Guan, "Physics and metadata-guided generative learning for underwater acoustic target recognition," JASA, vol. 160, no. 4, pp. 3505-3516, 2026. doi:10.1121/10.0046886.',
    '[6] N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis," Proc. R. Soc. A, vol. 454, pp. 903-995, 1998. doi:10.1098/rspa.1998.0193.',
    '[7] Y. Lipman et al., "Flow Matching for Generative Modeling," ICLR, 2023. arXiv:2210.02747.',
    '[8] Qwen Team, "Qwen3-4B-Instruct-2507," model card, 2025. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507.',
    '[9] Xie, Ren and Xu, "Adaptive ship-radiated noise recognition with learnable fine-grained wavelet transform," Ocean Engineering, vol. 265, 112626, 2022. doi:10.1016/j.oceaneng.2022.112626.',
    '[10] Hashemi et al., "UniqueShip: Mitigating Data Leakage in Acoustic Ship Classification Benchmark Datasets," arXiv:2609.13659, 2026.',
    '[11] Huang et al., "From Human Speech to Ocean Signals: Transferring Speech Large Models for Underwater Acoustic Target Recognition," arXiv:2601.18086, 2026.',
    '[12] A. Baevski, H. Zhou, A. Mohamed and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," 2020. arXiv:2006.11477. Implemented frozen checkpoint: https://huggingface.co/facebook/wav2vec2-base-960h.',
]


def aggregate_table(real):
    groups = group_records(headline_records(real))
    return [[short_method(method), str(len({r["seed"] for r in rows})),
             num(np.mean([r["ap"] for r in rows])), num(np.mean([r["auroc"] for r in rows])),
             num(np.mean([r["brier"] for r in rows])), pct(np.mean([r["recall"] for r in rows])),
             pct(np.mean([r["fpr"] for r in rows]))]
            for method, rows in groups.items()]


def comparison_text(real, compact=False):
    p = real["primary_comparison"]
    difference = p["difference_a_minus_b"]
    statement = (
        f"For {escape(short_method(p['method_a']))} minus the development-selected {escape(short_method(p['method_b']))}, "
        f"AP of mean-seed per-record probabilities differs by {difference:+.3f}. "
    )
    if p["ci"] is None:
        statement += ("The five test collection days form one connected day-vessel component; "
                      "independent-component uncertainty is not estimable. No superiority significance "
                      "or equivalence claim follows. ")
    else:
        lo, hi = p["ci"]
        statement += f"The recorded independent-component 95% interval is [{lo:+.3f}, {hi:+.3f}]. "
    if compact:
        return statement
    return statement + (
        "Tables instead average seed-specific AP; the estimands differ."
    )


def independent_test_summary(real):
    """Read audited summaries without recomputing predictions or fitting a model."""
    path = "manifests/real_test_independent_audit.json"
    audit = read_json(path)
    if audit.get("status") != "passed":
        raise ValueError("Independent real-test metrics audit has not passed")
    for source, checksum in audit["source_artifacts"].items():
        if sha256_file(source) != checksum:
            raise ValueError("Independent-audit source changed: " + source)
    methods = {row["method"]: row for row in audit["method_summary"]}
    for method, rows in group_records([r for r in real["records"] if r["split"] == "test"]).items():
        if not math.isclose(methods[method]["mean_seed_ap"], float(np.mean([r["ap"] for r in rows])), abs_tol=1e-12):
            raise ValueError("Audited and final-table mean seed AP differ")
    primary = real["primary_comparison"]
    difference = methods[primary["method_a"]]["ensemble_ap"] - methods[primary["method_b"]]["ensemble_ap"]
    if not math.isclose(difference, primary["difference_a_minus_b"], abs_tol=1e-12):
        raise ValueError("Audited primary AP contrast differs from final table")
    return {"methods": methods, "proposed_vs_fixed": audit["proposed_vs_fixed"],
            "source_sha256": {path: sha256_file(path), **audit["source_artifacts"]}}


def measured_result_text(real, audit):
    comparator = real["primary_comparison"]["method_b"]
    agent, baseline = audit["methods"]["proposed_agent"], audit["methods"][comparator]
    fixed = audit["methods"]["fixed_verification"]
    difference = real["primary_comparison"]["difference_a_minus_b"]
    text = (f"The agent reached ensemble AP {agent['ensemble_ap']:.4f}, versus {baseline['ensemble_ap']:.4f} "
            f"for {escape(short_method(comparator))}, the comparator selected on development data (difference {difference:+.4f}). "
            f"Fixed verification reached {fixed['ensemble_ap']:.4f}; its ensemble-AP difference from the agent was "
            f"{audit['proposed_vs_fixed']['ensemble_ap_difference']:+.6f}, with maximum probability difference "
            f"{audit['proposed_vs_fixed']['maximum_ensemble_probability_abs_difference']:.2g}. "
            "Ensemble AP evaluates averaged per-record probabilities; Table2 instead averages per-seed AP.")
    return text.replace("Table2", "Table 2")


def measured_robustness_text(real):
    rows = real["robustness"]
    agent = {r["condition"]: r for r in rows if r["method"] == "proposed_agent"}
    fixed = {r["condition"]: r for r in rows if r["method"] == "fixed_verification"}
    coincident = set(agent) == set(fixed) and all(math.isclose(agent[c]["ap"], fixed[c]["ap"], rel_tol=0, abs_tol=1e-12) for c in agent)
    clean = next(r for r in agent.values() if r.get("snr_db") is None)
    lowest = min((r for r in agent.values() if r.get("snr_db") is not None), key=lambda r: r["snr_db"])
    prefix = (f"Fixed and agent AP coincided in all {len(agent)} conditions. " if coincident else "The paired method curves are reported without an independent-component interval. ")
    return prefix + (f"Agent seed-17 AP was {clean['ap']:.4f} on this clean subset and {lowest['ap']:.4f} at "
                     f"{lowest['snr_db']:g} dB added-noise SNR. These results do not demonstrate an adaptive robustness benefit.")


def measured_continuous_text(real):
    rows = {row["method_a"]: row for row in real["continuous_score_contrasts"]}
    fixed, agent = rows["fixed_verification"], rows["proposed_agent"]
    values = [value for row in rows.values() for value in row["per_seed_differences"].values()]
    direction = ("All three detector-seed differences were negative in each control family. "
                 if all(value < 0 for value in values) else "Paired detector-seed differences are shown separately. ")
    return ("Figure4 reports the continuous-score controls. Keeping V instead of setting it to zero changed "
            f"ensemble AP by {fixed['difference_a_minus_b']:+.4f} for fixed verification and "
            f"{agent['difference_a_minus_b']:+.4f} for the agent. " + direction +
            "These descriptive contrasts preserve queries, validity and calibration routing.").replace("Figure4", "Figure 4")


def measured_cost_text(real):
    rows = {row["method"]: row for row in real["cost"]}
    agent, fixed = rows["proposed_agent"], rows["fixed_verification"]
    ratio = agent["milliseconds_per_clip"] / fixed["milliseconds_per_clip"]
    return (f"The agent cost {agent['milliseconds_per_clip']:.1f} ms/clip, versus {fixed['milliseconds_per_clip']:.1f} "
            f"for fixed verification, {rows['heuristic_verification']['milliseconds_per_clip']:.1f} for the heuristic, "
            f"{rows['main']['milliseconds_per_clip']:.1f} for numeric fusion and "
            f"{rows['agent_no_generation']['milliseconds_per_clip']:.1f} without generation. "
            f"The agent/fixed cost ratio was {ratio:.2f}; allocated agent peak was {agent['peak_memory_mib']/1024:.2f} GiB. "
            "These are warm batch-throughput measurements, not single-request latency.")


def measured_conclusion(real, audit):
    difference = real["primary_comparison"]["difference_a_minus_b"]
    comparator = escape(short_method(real["primary_comparison"]["method_b"]))
    direction = "below" if difference < 0 else "above" if difference > 0 else "equal to"
    score = next(row for row in real["continuous_score_contrasts"] if row["method_a"] == "proposed_agent")
    ratio = next(row for row in real["cost"] if row["method"] == "proposed_agent")["milliseconds_per_clip"] / next(row for row in real["cost"] if row["method"] == "fixed_verification")["milliseconds_per_clip"]
    return (f"On this balanced AIS-window cohort, the agent's ensemble AP was {direction} the development-selected "
            f"{comparator} comparator by a signed difference of {difference:+.4f}. Its ensemble AP matched fixed "
            f"verification while measured cost was {ratio:.2f} times higher. The continuous-score contrast against "
            f"the matched V=0 agent control was {score['difference_a_minus_b']:+.4f}. All test paths acquired every view "
            "and verified last; generated evidence informed no subsequent choice. This execution did not demonstrate "
            "improved detection or adaptive tool selection. One connected test component prevents population-level "
            "superiority or equivalence inference.")


def build_main(real, synthetic, semisynthetic, config, dsp):
    story = []
    failure = read_json("manifests/agent_unconstrained_development_failure.json")
    audited = independent_test_summary(real)
    title = "Multi-Representation Underwater Acoustic Object Detection with Generative AI Reasoning Agents"
    story += [paragraph(title, "title"), paragraph("Anonymous authors", "caption")]
    story += [paragraph("<b><i>Abstract - </i></b>Underwater acoustic object detection depends on how the same waveform "
                        "is represented. We compare Fourier spectra, multi-resolution STFT features and Morlet wavelets, "
                        "individually and in fusion, alongside a two-dimensional log-mel spectrogram CNN. A bounded "
                        "generative AI controller receives numerical detector evidence from acquired views and may request "
                        "class-free background replacement. A same-support nuisance edit controls waveform disturbance. "
                        "A deliberately balanced PORTIA cohort measures three-second "
                        "AIS-linked window presence, not natural acoustic-event boundaries. Separate temporal-CNN "
                        "studies evaluate constructed activity unions rather than agent benefit. " +
                        comparison_text(real, compact=True) +
                        " The study measures component ablations, corruptions and actual end-to-end "
                        "compute. A bounded HHT approximation remains exploratory after noisy-scene convergence "
                        "failures. Generated interval replacement is treated as a sensitivity intervention, not "
                        "physical source removal or causal identification.", "abstract"),
              paragraph("<b>Keywords:</b> underwater acoustics; object detection; generative AI; reasoning agents; "
                        "signal representations; spectrograms; wavelets; Hilbert-Huang transform", "small"),
              paragraph("1 Introduction", "h1"),
              paragraph("Fourier spectra, spectrograms and wavelets expose different frequency and time structure in "
                        "the same underwater waveform. We compare their individual and fused detection performance and "
                        "test bounded tool selection across these representations, with generative verification as an "
                        "additional probe (Fig. 1). The controller receives numerical head outputs, not images; a separate "
                        "CNN processes two-dimensional log-mel spectrograms. Detection here means AIS-linked window presence.")]
    story += figure("figure1_method", "Fig. 1. Implemented evidence and verification workflow. Fourier, STFT and wavelet "
                    "measurements are primary. HHT is exploratory. The language model selects tools; the numeric head "
                    "retains responsibility for calibrated predictions.", max_height=3.4 * inch)
    story += [PageBreak()]

    story += [paragraph("2 Related Work and Method", "h1"),
              paragraph("Physics-aware audio-language frameworks already combine complementary acoustic representations "
                        "with generated reports [2,3]. Underwater audio-language benchmarks test waveform dependence [4], "
                        "and generative augmentation addresses data scarcity [5]. Wavelet front ends [9] and the HHT "
                        "decomposition [6] are established methods. Speech-to-ocean transfer [11] motivates a pretrained "
                        "baseline. The candidate contribution is the paired intervention and tool-selection protocol."),
              paragraph("2.1 Acoustic representations and detection", "h2"),
              paragraph("Native audio is preserved; channel 0 is CUDA-resampled to 16 kHz. Fourier, multi-resolution STFT "
                        "and analytic Morlet views provide 32, 64 and 56 features. Scaling uses training data only. "
                        "The 32-feature bounded linear-envelope HHT approximation is exploratory after its numerical failure."),
              equation(r"$X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N}$", 1, "equation_fourier"),
              equation(r"$S(m,k)=\left|\sum_n x[n]w[n-mH]e^{-j2\pi kn/N}\right|^2$", 2, "equation_stft"),
              equation(r"$x(t)=\sum_q c_q(t)+r(t),\quad f_q(t)=\frac{1}{2\pi}\frac{d\phi_q(t)}{dt}$", 3, "equation_hht"),
              paragraph("The primary verification replaces the fixed central 1.5 seconds [0.75, 2.25] of each "
                        "three-second clip while preserving exterior context. Scoring remains window presence, not source deletion "
                        "or localization. Baselines include feature heads, a 2D log-mel spectrogram CNN, and frozen speech-ASR "
                        "pretrained/fine-tuned facebook/wav2vec2-base-960h [12], mean/SD pooled into 1,536 features."),
              paragraph("2.2 Generative replacement and paired control", "h2"),
              paragraph("A conditional waveform flow [7] learns background completion. For editable mask M, background b, "
                        "observation u and Gaussian z, training uses y(t)=M[(1-t)z+tb]+(1-M)u. "
                        "A same-support sham approximately matches perturbation energy and coarse spectrum: after masking, "
                        "RMS mismatch must be below 0.05 and eight-band normalized-power L1 error below 0.2. Other engineering "
                        "gates check finite features, exterior invariance and amplitude. Rejection returns the base score. "
                        "The full-record null is context-unconditional except for observed-RMS scaling, not a physical counterfactual."),
              equation(r"$\delta_{vk}=[L_v(x)-L_v(x^{-k})]-[L_v(x)-L_v(x^{0k})]$", 4, "equation_delta"),
              equation(r"$V=\mathrm{median}_{v,k}(\delta_{vk})-\lambda\,\mathrm{MAD}_{v,k}(\delta_{vk})-\gamma R$", 5, "equation_score"),
              paragraph("L is a view-head logit clipped to [-8, 8]. We use K=4 draws, 8 Euler steps, lambda=0.5 and gamma=0.1; "
                        "medians use the lower central order statistic. R measures exterior change and boundary perturbation. "
                        "Nonnegative logistic calibration combines base "
                        "evidence with V. Views are correlated; this acoustic formulation combines familiar robust "
                        "aggregation and calibration, whose utility must be established empirically."),
              PageBreak()]

    cohort = read_json("data/manifests/split_audit.json")
    if not cohort.get("passed"):
        raise ValueError("Real cohort audit has not passed")
    cohort_rows = [[label, cohort["counts"][split]["labels"]["1"],
                    cohort["counts"][split]["labels"]["0"],
                    cohort["counts"][split]["vessel_ids"], cohort["counts"][split]["days"]]
                   for split, label in [("train", "Training"), ("development", "Development"),
                                         ("calibration", "Calibration"), ("test", "Test")]]
    story += [paragraph("3 Data and Experimental Protocol", "h1"),
              paragraph("The real study uses a custom PORTIA v2 cohort [1], not the official benchmark split. "
                        "Its labels reflect AIS-linked vessel co-occurrence, not human judgments of audibility. "
                        "Single-vessel-task positives and binary-task negatives are combined after excluding "
                        "2,388 window IDs with conflicting cross-task labels. Days are allocated before learning; vessel IDs "
                        "that cross allocated partitions are reserved by test/calibration/development/train priority. "
                        "A 60-second temporal guard separates partitions. The test cohort is deliberately 600 positive and 600 "
                        "background clips, not deployment prevalence: AP/precision and probability calibration are cohort-specific."),
              paragraph("Table 1. Custom cohort: AIS single-vessel positives and binary-task backgrounds.", "caption"),
              table(["Partition", "Vessel clips", "Background clips", "Vessel IDs", "Days"],
                    cohort_rows,
                    [95, 85, 100, 85, 70]),
              paragraph("Repeated vessel identities connect all five test days into one dependence component. "
                        "Neither 1,200 clips nor five days are independent deployment replicates. AIS co-occurrence supplies "
                        "no human audibility judgment or natural acoustic-event boundaries."),
              paragraph("3.1 Models, controller and controls", "h2"),
              paragraph("Three detector seeds are evaluated with one shared generator trained at seed 17. "
                        "The frozen non-thinking Qwen3-4B-Instruct-2507 [8] tool controller runs wholly on the RTX 5000 Ada. It "
                        "starts from STFT evidence and can acquire Fourier/wavelet evidence, verify, or stop within "
                        "three additional actions. A finite token trie masks generation to legal JSON actions and enumerated "
                        "reasons for the current candidate and tool budget; the model greedily chooses legal prefixes. "
                        "The strict parser remains an independent check. Schema validity is enforced by construction, "
                        "not evidence of acoustic correctness. The LLM cannot overwrite detector probabilities."),
              paragraph(f"The archived unconstrained development run achieved only {pct(failure['valid_action_fraction'])} "
                        "action validity. This engineering failure motivated the decoding amendment before test "
                        "evaluation; failed traces and source snapshots remain preserved.", "small"),
              paragraph("Controls include acoustic/foundation baselines, fixed and heuristic verification, the agent "
                        "without generation, matched V = 0 score controls and component ablations."),
              paragraph("3.2 Outcomes and uncertainty", "h2"),
              paragraph("The real primary endpoint is AP against AIS-presence labels. Thresholds are selected on "
                        "calibration data for a target empirical negative-clip FPR; test recall and FPR are both reported. "
                        "AP, AUROC and Brier score are computed for every retained record before any referral. "
                        "Controlled temporal AP uses one-to-one matching at stated IoU thresholds. No primary "
                        "confidence interval is reported for the single day-vessel component. An optional 2,000-draw "
                        "paired day-bootstrap interval is nonconfirmatory sensitivity analysis because vessels recur "
                        "across days. Detector-seed variability is descriptive."),
              PageBreak()]

    story += [paragraph("4 Real Detection Results", "h1"), paragraph(measured_result_text(real, audited))]
    sensitivity = real["primary_comparison"].get("day_cluster_sensitivity", {}).get("ci")
    if sensitivity is not None:
        lo, hi = sensitivity
        story += [paragraph(f"For sensitivity only, the paired day-bootstrap interval is [{lo:+.3f}, {hi:+.3f}]. "
                            "Recurring vessel IDs violate day independence; this interval is nonconfirmatory.", "small")]
    story += figure("figure2_real_detection", "Fig. 2. Natural-real detection against AIS labels. Dots/ranges show "
                    "detector seeds, not independent-record confidence intervals. The operating-point panel uses "
                    "thresholds fixed on calibration data and shows mean seed-specific rates in percent. Core methods plus the locked comparator are shown; all methods "
                    "are retained in the exported tables.", max_height=2.85 * inch)
    story += [paragraph("Table 2. Core comparisons and locked comparator: mean seed-specific metrics. This differs from AP of averaged seed probabilities. All methods are exported.", "caption"),
              table(["Method", "Seeds", "AP", "AUROC", "Brier", "Recall", "FPR"],
                    aggregate_table(real), [145, 40, 50, 55, 50, 55, 55]),
              paragraph(f"Mean-seed agent AP was {audited['methods']['proposed_agent']['mean_seed_ap']:.4f}; "
                        f"the spectral CNN reached {audited['methods']['cnn']['mean_seed_ap']:.4f}. These descriptive "
                        "test comparisons do not replace the development-locked comparator. At the frozen operating point, "
                        f"agent recall was {100 * np.mean([r['recall'] for r in real['records'] if r['method'] == 'proposed_agent']):.2f}% "
                        f"and FPR {100 * np.mean([r['fpr'] for r in real['records'] if r['method'] == 'proposed_agent']):.2f}%; "
                        "the low recall limits detection utility.", "small"),
              PageBreak()]

    story += [paragraph("5 Robustness and Controlled Temporal Detection", "h1"),
              paragraph("Figure 3 tests a separate 120-record metadata-selected subset at seed 17/batch 8, retaining AIS labels under added noise. "
                        "It covers only numeric fusion, STFT, fixed verification and the agent, not the locked wavelet "
                        "or all baselines. This is corruption sensitivity, not natural domain shift. "
                        "The fully synthetic and semi-synthetic temporal studies evaluate a separately trained spectral CNN, "
                        "not the Qwen/flow method. Their construction labels describe unions of activity; overlapping sources "
                        "are not resolved instances, and these outcomes cannot establish an agent benefit.")]
    story += figure("figure3_robustness", "Fig. 3. Seed-17 robustness on a separate 120-record metadata-selected subset, not the full 1,200-record three-seed test cohort. "
                    "Added-noise curves share parents and noise draws; the clean subset is shown separately. "
                    "The locked wavelet comparator is not part of this four-method study.", max_height=2.6 * inch)
    story += [paragraph(measured_robustness_text(real), "small")]
    syn_rows = synthetic_summary_rows(synthetic) + synthetic_summary_rows(semisynthetic)
    story += [paragraph("Table 3. Temporal AP means over seeds and declared conditions within each origin/family and IoU: mixed plus fixed SNRs for fully synthetic, three insertion SNRs for semi-synthetic. Origins are not pooled.", "caption"),
              table(["Origin / family", "IoU", "Mean temporal AP", "Seeds"],
                    syn_rows, [185, 55, 120, 65]),
              paragraph("Controlled source stems make received interval support observable, but the signals are "
                        "bounded harmonic, modulated and chirped constructions. A separate semi-synthetic study inserts "
                        "real AIS-positive excerpts into same-day looped real backgrounds. Its truth is the known "
                        "insertion union, not audible event truth; source excerpts include their original noise. "
                        "Origins remain separate; neither supplies human-verified natural event boundaries."),
              PageBreak()]

    story += [paragraph("6 Mechanism and Component Ablations", "h1"),
              paragraph(measured_continuous_text(real)),
              paragraph(development_policy_summary()["description"], "small"),
              paragraph(test_policy_summary(real)["description"], "small"),
              paragraph("The fixed and agent V = 0 controls preserve base logits, acquired views, queries, validity "
                        "masks, fitting rows and the same two-input positive calibration procedure. Only the entire "
                        "continuous score V is set to zero. The agent control preserves the original verification-enabled "
                        "actions, rather than rerunning the no-generation controller. V = 0 removes contrast, dispersion "
                        "and boundary penalty together, rather than isolating delta alone.")]
    story += figure("figure4_mechanism_ablation", "Fig. 4. Component AP and matched continuous-score contrasts. Diamonds compare AP of mean-seed "
                    "probabilities; dots/ranges are paired detector-seed differences, not confidence intervals. "
                    "Both V = 0 controls retain query, validity and calibration routing. One dependence component "
                    "precludes an independent-component interval; these are descriptive contributions.", max_height=3.0 * inch)
    story += [paragraph("6.1 Counterfactual interpretation", "h2"),
              paragraph("The intervention is computational. Its sham is constrained to the same support and matched "
                        "in perturbation energy, but nuisance edits can still change target detectability. The measured "
                        "paired controls test component utility, not strict physical counterfactual identification. "
                        "Invalid or unavailable verification returns the base detector output "
                        "and remains in the denominator."),
              paragraph("6.2 HHT numerical failure", "h2"),
              paragraph("HHT reconstruction and a simple-tone check passed, but noisy modes frequently failed extrema "
                        "or convergence checks. "
                        f"The recorded fractions are {pct(dsp['benchmarks'][3]['unconverged_or_extrema_failed_fraction'])} "
                        f"for the three-second fixture and {pct(dsp['benchmarks'][7]['unconverged_or_extrema_failed_fraction'])} "
                        f"for the ten-second fixture. The exploratory HHT head nevertheless reached mean-seed AP "
                        f"{audited['methods']['hht_exploratory']['mean_seed_ap']:.4f}. Its descriptive accuracy does not "
                        "validate the decomposition or promote it into the primary analysis."),
              PageBreak()]

    story += [paragraph("7 Compute, Reproducibility and Limitations", "h1")]
    story += figure("figure5_accuracy_compute", "Fig. 5. Mean detector-seed test AP and actual seed 17 live cost on 32 calibration clips at batch 32, "
                    "with three warmed passes; horizontal bars show their range, not a confidence interval. Five timed "
                    "methods include heuristic verification. Model loading "
                    "is excluded and separately recorded; memory is allocated GPU peak. This measured-method "
                    "comparison is not an overall cost frontier over unmeasured stronger baselines.", max_height=3.1 * inch)
    story += [paragraph(measured_cost_text(real)),
              paragraph("All substantial transforms, fitting and inference use the 32 GB RTX 5000 Ada, "
                        "selected by UUID. CPU orchestration, I/O, statistical reporting and a small independent "
                        "spline-reference fixture are incidental. The Thunderbolt connection makes transfer-inclusive "
                        "measurement essential. Live timings include decoding, transfer, GPU transforms, requested generation, "
                        "controller calls, classification and frozen calibration. Full primary transform recomputation "
                        "and unqueried canonical generation companions are charged; no transform-savings claim is made."),
              paragraph("Checkpoints preserve model, optimizer, scheduler, random states, configuration fingerprints "
                        "and the next-batch cursor. Atomic file replacement, fsync, checksum validation and a prior "
                        "generation support interrupted-run recovery. A separate-process CUDA fixture reproduced "
                        "the next optimizer step exactly and recovered from a corrupted latest checkpoint copy. "
                        "This does not protect against storage destruction or establish bitwise reproducibility of "
                        "every final research kernel."),
              paragraph("The strongest limitations are AIS proxies, artificial class balance, one test dependence component, "
                        "cohort construction, a shared generator seed, a single external GPU/system, and controlled "
                        "rather than independently annotated real temporal intervals. An agent's rationale is not "
                        "ground truth. Any gains must be attributed only to contrasts supported by the saved outputs."),
              PageBreak()]

    story += [paragraph("8 Conclusion", "h1"), paragraph(measured_conclusion(real, audited)),
              paragraph("The contribution is the measured failure analysis and matched controls, not a claimed "
                        "performance gain. Further work requires real event annotations, additional deployments and "
                        "a demonstrably selective intervention before stronger acoustic or causal claims."),
              paragraph("References", "h1")]
    story += [paragraph(escape(reference), "small") for reference in REFERENCES]
    destination = MANUSCRIPT / "main_paper.pdf"
    doc = SimpleDocTemplate(str(destination), pagesize=letter, leftMargin=25 * mm, rightMargin=25 * mm,
                            topMargin=25 * mm, bottomMargin=25 * mm, title=title, author="Anonymous",
                            allowSplitting=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return destination


def origin_family_label(origin, family):
    if origin == "semi_synthetic":
        return "Real-excerpt insertion"
    return {"train": "Fully synthetic / within-family", "heldout": "Fully synthetic / held-out"}.get(family, str(origin) + " / " + str(family))


def synthetic_summary_rows(synthetic):
    grouped = defaultdict(list)
    for row in synthetic["records"]:
        if row["split"] == "test":
            grouped[(row["origin"], row.get("family", "unspecified"), row["iou_threshold"])].append(row)
    return [[origin_family_label(origin, family), num(iou, 1), num(np.mean([r["temporal_ap"] for r in rows])),
             len({r["seed"] for r in rows})]
            for (origin, family, iou), rows in sorted(grouped.items())]


def build_supplement(real, synthetic, config, dsp, views):
    failure = read_json("manifests/agent_unconstrained_development_failure.json")
    story = [paragraph("Supplementary Study", "title"),
             paragraph("Hilbert-Huang and Wavelet Evidence for Underwater Acoustic Detection", "h1"),
             paragraph("S1 Controlled Representation Study", "h1"),
             paragraph("This companion study separates exact construction labels from unknown real-world audibility. "
                       "Fully synthetic scenes comprise colored ambient noise and zero, one or two harmonic, "
                       "amplitude-modulated or chirped sources. Received supports include bounded multipath delay. "
                       "Overlapping source supports are merged into the union of activity for temporal scoring. "
                       "A union-activity detector does not resolve overlapping physical objects. The separately trained "
                       "temporal spectral CNN supplies the fully/semi-synthetic results; these do not test Qwen/flow benefit."),
             paragraph("The focused representation study uses three-second controlled clips and fixed detection "
                       "heads for STFT, STFT plus wavelets, STFT plus bounded HHT, and all views. It separates "
                       "training/development/test generation identities and a held-out signal family. Presence AP "
                       "in this study is not temporal AP and does not directly establish a reasoning-agent benefit. "
                       "HHT remains an explicitly exploratory approximation."),
             paragraph("Table S1. Recorded controlled representation results, grouped by family and method.", "caption"),
             table(["Family", "Method", "Mean AP", "Seeds", "Scenes"],
                   view_summary_rows(views), [95, 160, 65, 45, 65]),
             paragraph("Temporal AP uses greedy confidence-ordered one-to-one matching with the monotone precision "
                       "envelope. Duplicate predictions count as false positives; classes with no ground-truth "
                       "events are excluded from mAP and their false positives remain reported. Thresholded event "
                       "F1 and presence discrimination are additional outcomes, not replacements for a failed "
                       "temporal endpoint."),
             paragraph("No human-verified real temporal intervals were available in this execution. Therefore this "
                       "supplement makes no claim of real event-boundary accuracy, range/bearing localization, "
                       "source identity or silent-object detection."),
             PageBreak(),
             paragraph("S2 HHT and Wavelet Numerical Validity", "h1"),
             paragraph("The bounded GPU EMD approximation uses up to six modes and ten sifts, piecewise-linear "
                       "envelopes and masked instantaneous-frequency summaries. It is not relabeled as classical "
                       "cubic-spline EMD. HHT remains exploratory because the noisy-scene convergence gate failed.")]
    story += figure("figureS1_hht_numerics", "Fig. S1. Measured numerical fixtures only. The 440 Hz frequency check and "
                    "small reconstruction errors do not repair the noisy-scene mode failures. The combined flagged-mode "
                    "fraction includes nonconvergence and extrema failures.", max_height=3.2 * inch)
    tone = dsp["known_tone"]
    story += [table(["Diagnostic", "Observed value", "Interpretation"],
                    [["Tone first-mode frequency", f"{tone['imf0_mean_if_hz']:.6f} Hz", "Simple fixture only"],
                     ["Morlet peak", f"{tone['wavelet_peak_hz']:.4f} Hz", "Discrete frequency-grid error"],
                     ["Spline-reference discrepancy", pct(dsp["classical_spline_reference"]["imf0_relative_rms_difference_interior"]),
                      "One 4,096-sample CPU reference"],
                     ["HHT status", dsp["hht_readiness"].replace("_", " "), "Primary analysis excludes HHT"]],
                    [165, 135, 140]),
              paragraph("The FFT/Hilbert and wavelet calculations ran on the selected Ada GPU. The small independent "
                        "spline reference ran on CPU solely as a correctness fixture. Edge effects, mode mixing and "
                        "low-amplitude instantaneous-frequency instability remain relevant. Exact reconstruction "
                        "tests summation consistency and cannot establish physically meaningful decomposition."),
              PageBreak(),
              paragraph("S3 Controlled Noise Sensitivity and Representation Scope", "h1")]
    story += figure("figureS3_representation_study", "Fig. S2. Presence-detection AP for the focused representation "
                    "study and missing-view tests. Points show available seeds; short horizontal markers are seed "
                    "means. Missing-view curves retain each family separately. HHT remains exploratory after the "
                    "numerical convergence failure.", max_height=3.3 * inch)
    regime_rows = views["regime_ablation"]
    regime_methods = list(dict.fromkeys(r["method"] for r in regime_rows))
    keys = list(dict.fromkeys((r["family"], r["regime"]) for r in regime_rows))
    rows = []
    for family, regime in keys:
        values = []
        for method in regime_methods:
            scores = [r["ap"] for r in regime_rows
                      if r["family"] == family and r["regime"] == regime and r["method"] == method]
            values.append(num(np.mean(scores)) if scores else "NA")
        rows.append([supplemental_family_label(family) + " / " + supplemental_regime_label(regime)] + values)
    story += [paragraph("Table S2. Mean AP by controlled source regime. Regimes are synthetic forms, not vessel classes. Two-source scenes include overlapping and nonoverlapping supports; interval overlap is not required.", "caption"),
              table(["Family / regime"] + [short_method(m) for m in regime_methods], rows,
                    [150] + [75] * len(regime_methods))]
    story += [paragraph("Generative replacement can erase all activity in the editable interval. A matched sham "
                        "may preserve the original waveform additively while still masking it perceptually. "
                        "Accordingly, detector changes are sensitivity measurements. Generator uncertainty, "
                        "boundary distortion and invalid replacements must remain visible in the evaluation."),
              PageBreak(),
              paragraph("S4 Reproducibility and Resource Accounting", "h1"),
              paragraph("Table S3. Numerical pilot timing. These are single synchronized tensor-resident calls with "
                        "unequal first-use overhead, not a warmed throughput comparison.", "caption"),
              table(["Clip", "View", "ms / clip", "Peak MiB"],
                    [[f"{r['samples']/16000:g} s", r["representation"],
                      num(r["milliseconds_per_clip"], 3), num(r["peak_allocated_mib"], 2)]
                     for r in dsp["benchmarks"]], [65, 140, 100, 110]),
              paragraph(f"The unconstrained development replay had {pct(failure['valid_action_fraction'])} valid actions across "
                        f"{failure['action_count']:,} actions from {failure['completed_clip_seed_records']:,} clip-seed records "
                        "(including partial runs). It failed by proposing already observed STFT after wavelet acquisition. "
                        "This engineering failure and its traces are preserved. Before test evaluation, the controller was amended to finite "
                        "legal-JSON token-trie decoding, enforcing schema validity by construction and retaining the independent parser. "
                        "EOS is required, observed "
                        "tools are excluded, and no retries choose a favorable action. The 160-state GPU fixture checks "
                        "integration, not reasoning competence. " + test_policy_summary(real)["short_description"]),
              paragraph("Experiment recovery uses project-local atomic checkpoints, checksummed generation metadata, "
                        "a hash-chained ledger and verified prediction shards. Restarting recomputes unfinished "
                        "work and preserves completed outputs. Source/configuration/data fingerprints guard "
                        "against accidental resume under a different experiment."),
              paragraph("All code, data, caches, models, figures, reports and restart artifacts remain inside the "
                        "new MVML2027 submission directory. The earlier submission is unchanged. Large raw data "
                        "and checkpoints remain outside version control; no publication or submission is performed "
                        "by generating these drafts."),
              paragraph("The main limitations remain five collection days in one dependence component, AIS-derived labels, "
                        "one port/deployment, 600/600 artificial test balance, a fixed generator seed and construction-only temporal "
                        "annotation. AP/precision and calibration are not deployment-prevalence estimates. The supplement reports "
                        "failed numerical assumptions alongside successful fixtures."),
              paragraph("Supplement references", "h1"),
              paragraph(escape(REFERENCES[0]), "small"),
              paragraph(escape(REFERENCES[5]), "small"),
              paragraph(escape(REFERENCES[6]), "small")]
    destination = MANUSCRIPT / "supplement.pdf"
    doc = SimpleDocTemplate(str(destination), pagesize=letter, leftMargin=25 * mm, rightMargin=25 * mm,
                            topMargin=25 * mm, bottomMargin=25 * mm,
                            title="Supplementary controlled study", author="Anonymous")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return destination


def export_figure_data(real, synthetic, semisynthetic):
    output = MANUSCRIPT / "figure_data"
    output.mkdir(exist_ok=True)
    for name, rows in (("real_records", real["records"]), ("robustness", real["robustness"]),
                       ("ablations", real["ablations"]), ("continuous_score_contrasts", real["continuous_score_contrasts"]), ("cost", real["cost"]),
                       ("synthetic_records", synthetic["records"]), ("semisynthetic_records", semisynthetic["records"])):
        keys = sorted({key for row in rows for key in row})
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def render_and_check(paths):
    RENDERS.mkdir(exist_ok=True)
    counts = {}
    for path in paths:
        reader = PdfReader(path)
        count = len(reader.pages)
        counts[path.name] = count
        if path.name == "main_paper.pdf" and count != 8:
            raise RuntimeError(f"Main layout is {count} pages, expected exactly 8; revise before delivery")
        if path.name == "supplement.pdf" and not 3 <= count <= 4:
            raise RuntimeError(f"Supplement layout is {count} pages, expected 3-4; revise before delivery")
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < 80:
                raise RuntimeError(f"Unexpected nearly blank page {index+1}: {path.name}")
            if any(marker in text for marker in ("RESULT PENDING", "LAYOUT ONLY", "EMPIRICAL RESULT PENDING")):
                raise RuntimeError(f"Layout-only content leaked into empirical manuscript: {path.name}, page{index+1}")
            if path.name == "main_paper.pdf" and index == 0:
                if "Abstract -" not in text or "Keywords:" not in text:
                    raise RuntimeError("Main abstract/keywords could not be verified")
                abstract = text.split("Abstract -", 1)[1].split("Keywords:", 1)[0]
                if len(abstract.split()) > 300:
                    raise RuntimeError("Main abstract exceeds the conference300-word limit")
        subprocess.run(["pdftoppm", "-r", "110", "-png", str(path),
                        str(RENDERS / path.stem)], cwd=ROOT, check=True, capture_output=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default="results/tables/final_real_results.json")
    parser.add_argument("--synthetic", default="reports/synthetic_results.json")
    parser.add_argument("--views", default="results/tables/supplemental_views.json")
    parser.add_argument("--semisynthetic", default="results/tables/semisynthetic_results.json")
    parser.add_argument("--foundations-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    MANUSCRIPT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    dsp = read_json("manifests/dsp_validation.json")
    if args.foundations_only:
        method_figure()
        hht_figure(dsp)
        atomic_json("manuscript/foundations_manifest.json", {
            "status": "method_and_numeric_fixture_figures_only",
            "source_sha256": sha256_file("manifests/dsp_validation.json"),
            "figures": ["figure1_method", "figureS1_hht_numerics"],
            "research_result_claims": False,
        })
        print("Built method diagram and measured numerical-fixture figure only.")
        return 0
    real, synthetic = read_json(args.real), read_json(args.synthetic)
    validate_inputs(real, synthetic)
    semisynthetic = read_json(args.semisynthetic)
    validate_inputs(real, semisynthetic)
    if semisynthetic.get("status") != "complete" or any(r["origin"] != "semi_synthetic" for r in semisynthetic["records"]):
        raise ValueError("Complete, separately labelled semi-synthetic results are required")
    views = read_json(args.views)
    validate_supplement(views)
    if args.validate_only:
        print("Empirical result schemas validated; no manuscript generated.")
        return 0
    config = read_json("configs/research.json")
    global STYLES
    STYLES = configure_fonts()
    method_figure()
    hht_figure(dsp)
    result_figure(real)
    robustness_figure(real)
    mechanism_figure(real, synthetic)
    compute_figure(real)
    synthetic_figure(synthetic)
    supplemental_view_figure(views)
    export_figure_data(real, synthetic, semisynthetic)
    for label in ("view_ablation", "regime_ablation", "missing_view"):
        rows = views[label]
        keys = sorted({key for row in rows for key in row})
        with (MANUSCRIPT / "figure_data" / f"supplement_{label}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    paths = [build_main(real, synthetic, semisynthetic, config, dsp),
             build_supplement(real, synthetic, config, dsp, views)]
    counts = render_and_check(paths)
    receipt = {
        "status": "rendered_pending_visual_inspection",
        "page_counts": counts,
        "input_sha256": {args.real: sha256_file(args.real), args.synthetic: sha256_file(args.synthetic),
                         args.views: sha256_file(args.views), args.semisynthetic: sha256_file(args.semisynthetic),
                         "manifests/dsp_validation.json": sha256_file("manifests/dsp_validation.json"),
                         "manifests/agent_unconstrained_development_failure.json": sha256_file("manifests/agent_unconstrained_development_failure.json"),
                         "manifests/agent_constrained_validation.json": sha256_file("manifests/agent_constrained_validation.json"),
                         "data/manifests/split_audit.json": sha256_file("data/manifests/split_audit.json"),
                         "configs/research.json": sha256_file("configs/research.json")},
        "outputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
        "visual_inspection_complete": False,
        "report_builder_sha256": sha256_file("scripts/build_results_report.py"),
        "development_policy_description": development_policy_summary(),
        "test_policy_description": test_policy_summary(real),
        "independent_test_audit": independent_test_summary(real),
        "scope": "Anonymous drafts; figures reflect supplied measured artifacts, not promised gains.",
    }
    atomic_json("manuscript/build_manifest.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
