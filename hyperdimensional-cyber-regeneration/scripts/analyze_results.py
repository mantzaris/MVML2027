#!/usr/bin/env python3
"""Paired crossed seed/topology-cluster bootstrap; compact audited study report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_TAG = "severity_0.2_targeted_64_2048_0.1"
METHODS = {
    "B0": "Symbolic constrained prior",
    "B1": "GraphMAE-adapted reconstruction",
    "B2": "Dense decision-aware baseline",
    "B3": "HDC associative cleanup",
    "B4": "P decoder, full repair (inference ablation)",
    "B5": "P decoder, entropy selection (inference ablation)",
    "P": "Decision-gated HDC regeneration",
    "random_gate": "P decoder, random selection (inference ablation)",
    "no_provenance": "Direct source/age-channel ablation",
    "no_semantic": "Semantic-loss ablation",
    "dense_gate": "Dense representation with learned gate",
}

def raw_path(seed, method):
    return ROOT / "results/runs" / str(seed) / f"{method}_{PRIMARY_TAG}.npz"

def paired_bootstrap(seeds, repetitions=2000):
    numerators = {"recall3": [], "regret": []}
    denominators = {"recall3": [], "regret": []}
    hashes = {}
    expected_group = None
    expected_eligible = None
    point_estimates = {"P": {"recall3": [], "regret": []},
                       "B2": {"recall3": [], "regret": []}}
    for seed in seeds:
        paths = {method: raw_path(seed, method) for method in ("P", "B2")}
        raw = {method: dict(np.load(path, allow_pickle=False))
               for method, path in paths.items()}
        for method, path in paths.items():
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        p, b = raw["P"], raw["B2"]
        if "group" not in p or "group" not in b:
            raise ValueError("Raw results require topology cluster IDs; do not replace with snapshot bootstrap.")
        if not np.array_equal(p["group"], b["group"]) or not np.array_equal(p["eligible"], b["eligible"]):
            raise ValueError("P and B2 are not paired on test episodes.")
        groups = p["group"]
        if expected_group is None:
            expected_group = groups.copy()
            expected_eligible = p["eligible"].copy()
        elif not np.array_equal(groups, expected_group):
            raise ValueError("Test cluster ordering differs across seeds; crossed bootstrap assumption fails.")
        if not np.array_equal(p["eligible"], expected_eligible):
            raise ValueError("Test eligibility differs across seeds; shared test set assumption fails.")
        unique, inverse = np.unique(groups, return_inverse=True)
        eligible = p["eligible"].astype(bool)
        for metric in numerators:
            weight = eligible.astype(float) if metric == "recall3" else np.ones(len(groups))
            key = "recall" if metric == "recall3" else metric
            delta = np.asarray(p[key], dtype=float) - np.asarray(b[key], dtype=float)
            numerators[metric].append(np.bincount(inverse, weights=delta * weight, minlength=len(unique)))
            denominators[metric].append(np.bincount(inverse, weights=weight, minlength=len(unique)))
            for method, values in raw.items():
                point_estimates[method][metric].append(float(np.average(values[key], weights=weight)))
    ns, ng = len(seeds), len(unique)
    num = {m: np.stack(v) for m, v in numerators.items()}
    den = {m: np.stack(v) for m, v in denominators.items()}
    rng = np.random.default_rng(20271125)
    draws = {m: np.empty(repetitions) for m in num}
    for i in range(repetitions):
        seed_idx = rng.integers(0, ns, size=ns)
        cluster_idx = rng.integers(0, ng, size=ng)
        # Shared cluster draw preserves dependence from reusing one test set across models.
        for metric in num:
            numerator = num[metric][seed_idx][:, cluster_idx].sum(axis=1)
            denominator = den[metric][seed_idx][:, cluster_idx].sum(axis=1)
            if np.any(denominator == 0):
                raise ValueError("No eligible snapshots in a bootstrap draw.")
            draws[metric][i] = np.mean(numerator / denominator)
    contrasts = {}
    for metric in num:
        effects = num[metric].sum(axis=1) / den[metric].sum(axis=1)
        contrasts[metric] = {
            "direction": "P minus B2",
            "higher_is_better": metric == "recall3",
            "estimate": float(effects.mean()),
            "percentile_95_ci": np.quantile(draws[metric], [.025, .975]).tolist(),
            "seed_effects": dict(zip(map(str, seeds), effects.tolist())),
            "P_mean": float(np.mean(point_estimates["P"][metric])),
            "B2_mean": float(np.mean(point_estimates["B2"][metric])),
        }
    return {
        "condition": {"corruption": .2, "attack": "targeted", "facts": 64, "budget": 2048,
                      "repair_fraction": .1, "raw_tag": PRIMARY_TAG},
        "seeds": seeds, "snapshots_per_seed": len(expected_group), "topology_clusters": ng,
        "eligible_snapshots": int(expected_eligible.sum()),
        "no_feasible_snapshots": int((~expected_eligible.astype(bool)).sum()),
        "no_feasible_rate": float((~expected_eligible.astype(bool)).mean()),
        "bootstrap_repetitions": repetitions, "bootstrap_rng_seed": 20271125,
        "bootstrap_design": "Paired crossed two-factor bootstrap: resample training seeds and topology clusters with replacement; the same resampled clusters are used for every resampled seed. Within each cluster retain all snapshots. Percentile 95% intervals.",
        "estimand": "Mean across training seeds of conditional Recall@3 and unconditional candidate-oracle regret. Conditions and candidate sets are fixed; uncertainty covers seeds and this synthetic topology generator.",
        "limitations": "Only five training seeds; shared finite train/validation data; no uncertainty over generator design or external real-world environments. Two primary intervals are descriptive and not family-wise multiplicity adjusted. An interval including zero does not establish equivalence.",
        "contrasts": contrasts, "raw_input_sha256": hashes,
    }

def interpretation(metric, ci):
    lo, hi = ci
    if metric == "recall3":
        return "The interval supports higher recall in this controlled setting." if lo > 0 else (
            "The interval supports lower recall in this controlled setting." if hi < 0 else
            "The interval includes zero; these data do not establish a recall improvement or equivalence.")
    return "The interval supports lower regret in this controlled setting." if hi < 0 else (
        "The interval supports higher regret in this controlled setting." if lo > 0 else
        "The interval includes zero; these data do not establish a regret improvement or equivalence.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repetitions", type=int, default=2000)
    args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / "results/summary.csv").open()))
    reference = [r for r in rows if r["condition"] == "severity" and float(r["corruption"]) == .2
                 and r["attack"] == "targeted" and int(r["facts"]) == 64
                 and int(r["budget"]) == 2048 and float(r["repair_fraction"]) == .1]
    seeds = sorted({int(r["seed"]) for r in reference})
    if len(seeds) != 5:
        raise ValueError(f"Expected completed five-seed experiment, found {seeds}")
    result = paired_bootstrap(seeds, args.repetitions)
    secondary = {}
    for comparator in ("B5", "random_gate", "dense_gate", "B4"):
        pair = {}
        for metric in ("recall3", "regret"):
            effects = []
            for seed in seeds:
                selected = {r["method"]: r for r in reference if int(r["seed"]) == seed}
                effects.append(float(selected["P"][metric]) - float(selected[comparator][metric]))
            pair[metric] = {"direction": f"P minus {comparator}",
                            "mean_difference": float(np.mean(effects)),
                            "seed_differences": dict(zip(map(str, seeds), effects)),
                            "inference": "Descriptive secondary comparison; no significance or multiplicity claim."}
        secondary[comparator] = pair
    result["secondary_descriptive_comparisons"] = secondary
    output = ROOT / "results/paired_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    manifest = json.loads((ROOT / "results/study_manifest.json").read_text())
    settings = manifest.get("settings", {})
    recall, regret = result["contrasts"]["recall3"], result["contrasts"]["regret"]
    rci = np.asarray(recall["percentile_95_ci"]) * 100
    gci = regret["percentile_95_ci"]
    lines = [
        "# Experimental results and scope",
        "",
        "This report is generated from completed experiment outputs. It reports observed results, including null or negative comparisons; no missing result is imputed.",
        "",
        f"Hardware: {manifest.get('hardware', 'see manifest')}; PyTorch {manifest.get('torch', 'see manifest')}, CUDA {manifest.get('cuda', 'see manifest')}. "
        f"Five training seeds ({', '.join(map(str,seeds))}), {settings.get('train')} training snapshots, {settings.get('val')} validation snapshots, "
        f"{settings.get('test')} shared test snapshots per condition, {settings.get('pretrain')} masked-feature pretraining epochs and at most {settings.get('epochs')} supervised epochs (patience {settings.get('patience')}). "
        f"Recorded study wall time: {manifest.get('gpu_wall_seconds', 0)/3600:.2f} hours.",
        "",
        "## Primary paired comparison",
        "",
        f"At the reference condition, {result['eligible_snapshots']:,}/{result['snapshots_per_seed']:,} snapshots contain at least one truly feasible candidate. "
        f"The remaining {result['no_feasible_snapshots']:,} snapshots ({100*result['no_feasible_rate']:.3f}%) have no feasible candidate; "
        "they are excluded only from the conditional Recall@3 denominator and retained for regret and other snapshot metrics.",
        "",
        "Prespecified reference condition: 64 facts, targeted corruption intervention fraction 0.2, 2,048-byte budget per remote observer and 10% repair budget. P is the proposed decision-gated HDC model; B2 is the dense decision-aware reconstruction baseline.",
        "",
        f"- Recall@3: P {100*recall['P_mean']:.3f}%, B2 {100*recall['B2_mean']:.3f}%; paired P−B2 difference **{100*recall['estimate']:+.3f} percentage points**, 95% interval [{rci[0]:+.3f}, {rci[1]:+.3f}]. {interpretation('recall3', recall['percentile_95_ci'])}",
        f"- Candidate-oracle regret (lower is better): P {regret['P_mean']:.5f}, B2 {regret['B2_mean']:.5f}; paired P−B2 difference **{regret['estimate']:+.5f}**, 95% interval [{gci[0]:+.5f}, {gci[1]:+.5f}]. {interpretation('regret', gci)}",
        "",
        f"Intervals use {args.repetitions:,} paired bootstrap draws over the crossed factors of five training seeds and {result['topology_clusters']} topology clusters. "
        "A cluster draw is shared across the resampled seeds because all models use the same test set. All snapshots inside a sampled cluster are retained. "
        "Recall is conditional on at least one truly feasible candidate and is recomputed from eligible numerators and denominators. Regret includes all snapshots. "
        "These are descriptive 95% percentile intervals, without multiplicity adjustment. The fixed synthetic generator and finite shared training/validation set are outside their uncertainty scope.",
        "",
        "## Reference-condition results",
        "",
        "Recall entries are the mean ± sample standard deviation over training seeds, not confidence intervals. Other columns are seed means. "
        "Confidence error is the mean per-snapshot error rate among beliefs below 0.1 or above 0.9; coverage is the fraction of beliefs meeting that threshold. Zero-coverage snapshots contribute zero to the error statistic.",
        "",
        "| Method | Recall@3 (%) | Regret ↓ | Brier ↓ | Confidence error (%) ↓ | Confidence coverage (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in METHODS.items():
        values = [r for r in reference if r["method"] == method]
        if len(values) != len(seeds):
            raise ValueError(f"Incomplete reference rows for {method}")
        get = lambda field: np.asarray([float(r[field]) for r in values])
        lines.append(f"| {method}: {label} | {100*get('recall3').mean():.3f} ± {100*get('recall3').std(ddof=1):.3f} | "
                     f"{get('regret').mean():.5f} | {get('brier').mean():.5f} | "
                     f"{100*get('false_certainty').mean():.2f} | {100*get('coverage').mean():.2f} |")
    lines += [
        "",
        "## Secondary comparisons and original hypotheses",
        "",
        "The following reference-condition effects are descriptive seed means. All contrasts are P minus the named comparator. "
        "A positive recall difference favors P; a positive regret difference favors the comparator. No secondary significance claim is made.",
        "",
        "| Comparator | Recall difference (percentage points) | Regret difference | Interpretation |",
        "|---|---:|---:|---|",
    ]
    descriptions = {
        "B5": "Same P decoder, entropy selection; same maximum repair budget.",
        "random_gate": "Same P decoder, random selection; same maximum repair budget.",
        "dense_gate": "Separately trained dense model with the gated objective; representation ablation.",
        "B4": "Same P decoder with full reconstruction; difference quantifies the cost of sparse selection.",
    }
    for comparator, comparison in secondary.items():
        lines.append(f"| {comparator} | {100*comparison['recall3']['mean_difference']:+.3f} | "
                     f"{comparison['regret']['mean_difference']:+.5f} | {descriptions[comparator]} |")
    h1_positive = all(secondary[c]["recall3"]["mean_difference"] > 0 for c in ("B5", "random_gate"))
    h1_all_seeds = all(v > 0 for c in ("B5", "random_gate")
                      for v in secondary[c]["recall3"]["seed_differences"].values())
    h1 = ("Narrowly supported descriptively: learned selection has higher mean Recall@3 than entropy and random selection at the same maximum repair budget"
          + (" in every training seed." if h1_all_seeds else ", with seed-level variability."))
    if not h1_positive:
        h1 = "Not supported by both mean comparisons: learned selection does not beat both entropy and random selection on reference Recall@3."
    def scenario_mean(condition, corruption, attack, method, metric):
        values = [float(r[metric]) for r in rows if r["condition"] == condition
                  and float(r["corruption"]) == corruption and r["attack"] == attack
                  and r["method"] == method and int(r["facts"]) == 64]
        if len(values) != len(seeds):
            raise ValueError("Incomplete distribution-shift diagnostic")
        return float(np.mean(values))
    clean_p = scenario_mean("severity", 0., "targeted", "P", "regret")
    clean_entropy = scenario_mean("severity", 0., "targeted", "B5", "regret")
    random_p = scenario_mean("attack", .2, "random", "P", "regret")
    random_entropy = scenario_mean("attack", .2, "random", "B5", "regret")
    coherent_p = scenario_mean("attack", .2, "coherent", "P", "regret")
    coherent_entropy = scenario_mean("attack", .2, "coherent", "B5", "regret")
    lines += [
        "",
        "The learned-selection advantage does not extend to all tested observation conditions. "
        "Training and validation use 20% targeted interventions. "
        f"At zero injected attack, mean regret is {clean_p:.5f} for P versus {clean_entropy:.5f} for entropy selection; "
        f"under random attacks it is {random_p:.5f} versus {random_entropy:.5f}. "
        f"Under coherent attacks the means are {coherent_p:.5f} versus {coherent_entropy:.5f}. "
        "These descriptive diagnostics show a substantial dependence on the training-matched corruption mechanism; "
        "the H1 finding below is confined to the reference condition.",
    ]
    lines += [
        "",
        "| Hypothesis | Status supported by this experiment |",
        "|---|---|",
        f"| H1: selective regeneration improves viable-candidate recovery at fixed repair budget | {h1} This is a static candidate-ranking finding; validation thresholds can use fewer than the maximum repairs. |",
        "| H2: decision-impact teacher improves planning beyond generic reconstruction | Not isolated. There is no otherwise identical retrained model without the teacher. P versus B2 changes representation and repair policy, while B4/B5/random share teacher-trained weights. The primary comparison cannot attribute a benefit to teacher supervision. |",
        "| H3: HDC improves quality–bandwidth tradeoffs | Untested as specified: all methods share the symbolic report codec. dense_gate addresses the internal HDC feature contribution, not learned-message or wire-codec superiority. |",
        "| H4: provenance/calibration reduces false certainty at matched coverage | Untested as specified: confidence-error and coverage diagnostics are reported, but coverage is not matched and the source/age ablation retains the common fused prior. |",
        "",
        "Full reconstruction and sparse selection answer different budget questions. A useful gate relative to entropy or random selection can coexist with worse regret than full reconstruction. "
        "Any small P-versus-dense_gate difference is a descriptive result in this generator, not evidence of universal HDC superiority. "
        "The primary comparison and all ablations remain visible regardless of their direction.",
    ]
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "- B1 adapts GraphMAE-style masked feature reconstruction to this synthetic task; it is not an official benchmark reproduction. B2 adds decision and symbolic training terms. The separately trained dense_gate model is the appropriate representation ablation.",
        "- B4 (full repair), B5 (entropy selection), and random_gate reuse P's weights and change inference selection. They are inference ablations, not independently optimized algorithms.",
        "- The direct source/age-channel ablation removes explicit source distinctions and ages but retains the shared fused provisional belief. It does not remove every indirect provenance influence.",
        "- All methods receive the same sparse symbolic report serialization. HDC is an internal associative memory; bandwidth curves do not demonstrate an HDC wire-compression advantage.",
        "- The benchmark contains independent decision snapshots with a noisy prior-time observation and genuine state changes. Candidates are abstract conjunctions with terminal utilities. It does not establish sequential planning, temporal recovery delay, real cyber effectiveness, or distributed consensus.",
        "- Coherent attack severity counts interventions; some overwritten report values remain correct. The compact summary also records actual changed-report fractions.",
        "- The complete decoder runs before selecting repairs. Fewer changed beliefs do not establish inference compute savings. Preservation in the raw summaries is an unnecessary-change rate and is better when lower.",
        "- CUDA latency is warmed batch-one model inference plus candidate scoring and excludes serialization, transfer, and end-to-end system overhead. Peak VRAM is incremental above already resident models and data, not total deployment memory.",
        "- The experiment tests a particular composition of prior methods and the one-fact regret-benefit gate target. It does not establish a first-ever novelty claim or satisfy the complete proposed DARPA capability.",
        "",
        "## Reproducibility and artifacts",
        "",
        "The compact summary is in `results/summary.csv`; bin-level calibration and training curves are in `results/calibration.csv` and `results/curves.csv`. "
        "`results/study_manifest.json` records actual settings, hardware, source digests and data hashes. "
        "`results/paired_comparison.json` records the paired contrasts, bootstrap design, per-seed effects and hashes of raw inputs. "
        "The raw per-snapshot arrays and checkpoints remain local and ignored by Git. Regenerate this report with `python scripts/analyze_results.py` after the complete GPU study.",
        "",
        "The separate 2–4 page supplemental PDF is prepared as requested. Public conference instructions checked for this project do not verify acceptance of a separate supplement; confirm with the conference before upload.",
        "",
    ]
    timing_path = ROOT / "results/pipeline_timing.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        selected_timing = {row["method"]: row for row in timing["rows"]}
        lines += [
            "## Prepared-evidence pipeline timing",
            "",
            f"A separate audit on training seed {timing['seed']} and one fixed 64-fact reference snapshot "
            f"measured {selected_timing['P']['repeats']} repeats after {timing['warmups']} warmups. "
            "It includes actual report packing/unpacking, host-to-device evidence and metadata transfer, "
            "GPU feature fusion, inference/ranking, and top-three IDs returned to the host. "
            "All eleven reconstructed prediction paths were checked against their original CUDA inputs before timing.",
            "",
            f"P median: {selected_timing['P']['median_ms']:.3f} ms "
            f"(10th–95th percentiles {selected_timing['P']['p10_ms']:.3f}–{selected_timing['P']['p95_ms']:.3f} ms); "
            f"B2 median: {selected_timing['B2']['median_ms']:.3f} ms "
            f"(10th–95th percentiles {selected_timing['B2']['p10_ms']:.3f}–{selected_timing['B2']['p95_ms']:.3f} ms). "
            f"The common remote payload was {timing['payload_bytes']} bytes.",
            "",
            "This bounded host-wall measurement excludes sensing, bandwidth admission, graph/catalogue construction, "
            "transport delay and cold model loading. Repetition percentiles describe timing variability for that one snapshot, "
            "not seed or environment uncertainty. All methods and the exact scope are in `results/pipeline_timing.json`.",
            "",
        ]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"paired_output": str(output), "recall3": recall, "regret": regret}, indent=2))

if __name__ == "__main__":
    main()
