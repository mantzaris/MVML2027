#!/usr/bin/env python3
"""Frozen second-study comparisons with paired crossed seed/topology bootstrap.

The old test set motivated the second approach. This analysis only uses the new
held-out pool and preserves the original experiment's reports unchanged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v2"
SEEDS = [11, 23, 37, 51, 71]
CONDITIONS = [
    ("benign", "severity", 0.0, "targeted"),
    ("targeted", "severity", 0.2, "targeted"),
    ("random", "attack", 0.2, "random"),
    ("coherent", "attack", 0.2, "coherent"),
]
COMPARATORS = ["ID", "IH", "JD", "JH_ERM", "JH_product", "P", "B2"]
METRICS = {"macro_regret": "regret", "macro_recall3": "recall", "macro_plan_brier": "plan_brier"}
METHODS = {
    "JH": "HDC dependency-factorized regeneration",
    "JD": "Dense dependency-factorized regeneration",
    "IH": "HDC independent reconstruction",
    "ID": "Dense independent reconstruction",
    "JH_ERM": "JH with mean group loss",
    "JH_product": "JH checkpoint, product-of-marginals scoring",
    "B0": "Original constrained prior",
    "B1": "Original GraphMAE-adapted reconstruction",
    "B2": "Original dense decision-aware reconstruction",
    "B3": "Original HDC cleanup",
    "B4": "Original P decoder, full repair",
    "B5": "Original P decoder, entropy gate",
    "P": "Original learned selective regeneration",
    "random_gate": "Original random-gate ablation",
    "no_provenance": "Original direct-source-cue ablation",
    "no_semantic": "Original semantic-loss ablation",
    "dense_gate": "Original dense-gate ablation",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition_rows(rows, method, seed=None, label=None):
    out = []
    for row in rows:
        if row["method"] != method or (seed is not None and int(row["seed"]) != seed):
            continue
        if int(row["facts"]) != 64 or int(row["budget"]) != 2048 or float(row["repair_fraction"]) != .1:
            continue
        for name, family, corruption, attack in CONDITIONS:
            if (label is None or name == label) and row["condition"] == family and row["attack"] == attack and float(row["corruption"]) == corruption:
                out.append(row)
                break
    return out


def raw_path(seed, method, condition):
    _, name, corruption, attack = condition
    return ROOT / "results/runs/v2" / str(seed) / f"{method}_{name}_{corruption}_{attack}_64_2048_0.1.npz"


def bootstrap_weights(repetitions, n_seeds, n_clusters, rng_seed=20271126):
    rng = np.random.default_rng(rng_seed)
    seed_counts = np.empty((repetitions, n_seeds), dtype=float)
    cluster_counts = np.empty((repetitions, n_clusters), dtype=float)
    for i in range(repetitions):
        seed_counts[i] = np.bincount(rng.integers(n_seeds, size=n_seeds), minlength=n_seeds)
        cluster_counts[i] = np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters)
    return seed_counts, cluster_counts


def bootstrap_draws(numerator, denominator, seed_counts, cluster_counts):
    """Arrays are seed × condition × cluster; conditions remain fixed/equal-weight."""
    num = np.einsum("scg,rg->rsc", numerator, cluster_counts, optimize=True)
    den = np.einsum("scg,rg->rsc", denominator, cluster_counts, optimize=True)
    if np.any(den <= 0):
        raise ValueError("A resampled condition has an empty metric denominator")
    per_seed = (num / den).mean(axis=2)
    return (per_seed * seed_counts).sum(axis=1) / seed_counts.sum(axis=1)


def summarize_contrasts(repetitions=2000):
    raw, hashes = {}, {}
    expected_group = expected_eligible = None
    for method in ["JH"] + COMPARATORS:
        raw[method] = []
        for seed in SEEDS:
            per_condition = []
            for condition in CONDITIONS:
                path = raw_path(seed, method, condition)
                hashes[str(path.relative_to(ROOT))] = sha256(path)
                with np.load(path, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in ["group", "eligible", "recall", "regret", "plan_brier"]}
                if len(data["group"]) != 4000:
                    raise ValueError(f"Expected 4,000 test snapshots: {path}")
                if expected_group is None:
                    expected_group = data["group"].copy()
                    expected_eligible = data["eligible"].astype(bool).copy()
                if not np.array_equal(data["group"], expected_group):
                    raise ValueError(f"Topology cluster ordering is not paired: {path}")
                if not np.array_equal(data["eligible"].astype(bool), expected_eligible):
                    raise ValueError(f"Underlying candidate feasibility changed across paired cases: {path}")
                for metric in METRICS.values():
                    if data[metric].shape != (4000,) or not np.isfinite(data[metric]).all():
                        raise ValueError(f"Invalid per-snapshot {metric}: {path}")
                per_condition.append(data)
            raw[method].append(per_condition)
    unique, inverse = np.unique(expected_group, return_inverse=True)
    if len(unique) != 250:
        raise ValueError(f"Expected 250 topology clusters, found {len(unique)}")
    seed_counts, cluster_counts = bootstrap_weights(repetitions, len(SEEDS), len(unique))
    contrasts = []
    for comparator in COMPARATORS:
        contrast = {"method": "JH", "comparator": comparator,
                    "direction": f"JH minus {comparator}", "metrics": {}, "per_condition_effects": {}}
        for metric, key in METRICS.items():
            numerators, denominators, a_means, b_means = [], [], [], []
            for si, seed in enumerate(SEEDS):
                ns, ds, aa, bb = [], [], [], []
                for ci, condition in enumerate(CONDITIONS):
                    a, b = raw["JH"][si][ci], raw[comparator][si][ci]
                    weight = expected_eligible.astype(float) if key == "recall" else np.ones(4000)
                    delta = a[key].astype(float) - b[key].astype(float)
                    ns.append(np.bincount(inverse, weights=delta * weight, minlength=len(unique)))
                    ds.append(np.bincount(inverse, weights=weight, minlength=len(unique)))
                    aa.append(float(np.average(a[key], weights=weight)))
                    bb.append(float(np.average(b[key], weights=weight)))
                numerators.append(ns); denominators.append(ds); a_means.append(aa); b_means.append(bb)
            numerator, denominator = np.asarray(numerators), np.asarray(denominators)
            condition_effect = numerator.sum(axis=2) / denominator.sum(axis=2)
            seed_effect = condition_effect.mean(axis=1)
            draws = bootstrap_draws(numerator, denominator, seed_counts, cluster_counts)
            contrast["metrics"][metric] = {
                "estimate": float(seed_effect.mean()),
                "percentile_95_ci": np.quantile(draws, [.025, .975]).tolist(),
                "higher_is_better": key == "recall",
                "means": {"JH": float(np.mean(a_means)), comparator: float(np.mean(b_means))},
                "seed_effects": dict(zip(map(str, SEEDS), seed_effect.tolist())),
            }
            contrast["per_condition_effects"][metric] = {
                condition[0]: {"mean_difference": float(condition_effect[:, ci].mean()),
                               "seed_effects": dict(zip(map(str, SEEDS), condition_effect[:, ci].tolist()))}
                for ci, condition in enumerate(CONDITIONS)
            }
        contrasts.append(contrast)
    primary = dict(contrasts[0]["metrics"]["macro_regret"])
    primary.update(method="JH", comparator="ID", metric="macro_regret", direction="JH minus ID")
    return {
        "primary": primary, "contrasts": contrasts, "seeds": SEEDS,
        "test_generator_seed": 42027, "snapshots_per_condition": 4000,
        "topology_clusters": len(unique),
        "conditions": [{"label": label, "condition": name, "corruption": c, "attack": a}
                       for label, name, c, a in CONDITIONS],
        "eligible_per_condition": int(expected_eligible.sum()),
        "no_feasible_per_condition": int((~expected_eligible).sum()),
        "bootstrap_repetitions": repetitions, "bootstrap_rng_seed": 20271126,
        "bootstrap_design": "Paired crossed training-seed/topology bootstrap. Resample five training seeds and 250 topology clusters with replacement; use the same cluster draw across every selected seed, method and all four paired conditions. Keep all snapshots within a cluster. Conditions are fixed and equally weighted, not resampled.",
        "estimand": "Equal-weight mean over benign0, targeted0.2, random0.2 and coherent0.2 conditions, then mean over model-training seeds. Regret and plan Brier include every snapshot. Recall is conditional on a feasible candidate within each condition before macro averaging.",
        "inference_scope": "Primary: JH-minus-ID macro regret. Secondary intervals are descriptive, unadjusted for multiple comparisons, and do not establish equivalence when they contain zero. Only five seeds; finite common train/validation data and generator-design uncertainty are outside the bootstrap.",
        "development_disclosure": "The first-study test pool (seed32027) was inspected and motivated this approach. It is development evidence. This study evaluates the frozen second protocol on fresh seed42027; familiar attack families are covered in mixed training and are not unseen-mechanism tests.",
        "raw_input_sha256": hashes,
    }


def write_report(rows, result, manifest):
    primary = result["primary"]
    lo, hi = primary["percentile_95_ci"]
    if hi < 0:
        conclusion = "The primary interval favors JH within this synthetic generator."
    elif lo > 0:
        conclusion = "The primary interval favors the matched dense independent baseline."
    else:
        conclusion = "The primary interval contains zero; the study does not establish an improvement or equivalence."
    lines = [
        "# Second approach: dependency-factorized regeneration",
        "",
        "This report records the second, prospectively frozen experiment. The first study's results motivated its design; they were not a fresh confirmation set. Original reports and measurements are preserved.",
        "",
        f"Hardware: {manifest.get('hardware', 'see manifest')}; PyTorch {manifest.get('torch', 'see manifest')}, CUDA {manifest.get('cuda', 'see manifest')}. The run trains 25 new model instances (five variants × five seeds), using {manifest.get('settings', {}).get('train')} training and {manifest.get('settings', {}).get('val')} validation snapshots. Recorded study wall time: {manifest.get('wall_seconds', 0)/60:.2f} minutes. It evaluates 17 methods across 18 conditions, giving 1,530 seed/method/condition records.",
        "",
        "## Changes and comparison logic",
        "",
        "The new approach (JH) regenerates the full belief state, predicts root readiness and child readiness conditional on its root, and computes candidate success with each shared root counted once. It replaces the first method's fixed six-fact gate and perfect-truth single-fact teacher. HDC retrieval remains an optional input augmentation, tested against a dense representation.",
        "",
        "All five new trained variants receive the same mixed training data and optimization budget. JH, JD, IH and ID use the same smooth maximum over group losses (beta 2); JH_ERM replaces that aggregation with a group mean. JH_product reuses the JH checkpoint but scores candidates from the product of fact marginals. The original eleven methods are reevaluated on the new test pool without retraining. Their comparison combines multiple changes, whereas the new controls separate them.",
        "",
        f"Fresh evaluation seed: 42027. Five model seeds ({', '.join(map(str, SEEDS))}); 4,000 snapshots for each evaluated condition. The primary macro average covers benign observations and targeted, random and coherent interventions at 20%. The same hidden worlds and candidates are paired across these four conditions. {result['eligible_per_condition']:,} snapshots per condition have a feasible candidate; {result['no_feasible_per_condition']:,} do not. No-feasible cases are retained in regret and plan Brier.",
        "",
        "## Primary result",
        "",
        f"Mean candidate-oracle regret: JH **{primary['means']['JH']:.5f}**, ID **{primary['means']['ID']:.5f}**. Paired difference JH−ID: **{primary['estimate']:+.5f}**, 95% crossed-bootstrap interval **[{lo:+.5f}, {hi:+.5f}]**. Lower is better. {conclusion}",
        "",
        f"The interval uses {result['bootstrap_repetitions']:,} draws across five training seeds and 250 topology clusters. Cluster draws are shared across all selected seeds and all four paired conditions; each cluster retains its snapshots. Conditions receive fixed equal weights. This estimates seed/topology variation for the fixed generator, not uncertainty about real cyber environments or the generator design.",
        "",
        "## Macro comparison across four conditions",
        "",
        "Entries are averages over conditions and training seeds. Recall is conditional on a truly feasible candidate and is reported as a percentage. Fact and plan Brier are distinct scoring rules; plan Brier directly checks the candidate-success probabilities used for decisions.",
        "",
        "| Method | Recall@3 (%) | Regret ↓ | Fact Brier ↓ | Plan Brier ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in METHODS.items():
        chosen = condition_rows(rows, method)
        if len(chosen) != 20:
            raise ValueError(f"Incomplete four-condition comparison: {method}")
        avg = lambda key: float(np.mean([float(r[key]) for r in chosen]))
        lines.append(f"| {method}: {label} | {100*avg('recall3'):.3f} | {avg('regret'):.5f} | {avg('brier'):.5f} | {avg('plan_brier'):.5f} |")
    diagnostic = lambda method, key: float(np.mean([float(r[key]) for r in condition_rows(rows, method)]))
    lines += ["", "Coherence and prediction quality are separate properties. The mean fraction of facts violating the supplied child≤parent marginal inequality (tolerance 1e-6) is "
              f"{100*diagnostic('JH', 'logical_violation'):.4f}% for JH, {100*diagnostic('JD', 'logical_violation'):.4f}% for JD, "
              f"{100*diagnostic('IH', 'logical_violation'):.4f}% for IH and {100*diagnostic('ID', 'logical_violation'):.4f}% for ID. "
              "A zero violation rate does not prove that the inferred facts are true or that candidate ranking is better.", "",
              f"JH changes {100*diagnostic('JH', 'changed_fraction'):.2f}% of beliefs numerically from their provisional values, versus {100*diagnostic('P', 'changed_fraction'):.2f}% for original P (absolute-change threshold 1e-4). This measured intervention difference is part of the full-versus-sparse policy change, not a compute-saving result."]
    lines += ["", "## Secondary mechanism comparisons", "",
              "All effects below are JH minus comparator macro regret. Secondary intervals are descriptive and not corrected for multiple comparisons. A negative effect favors JH.", "",
              "| Comparator | Effect | 95% interval | What changes |", "|---|---:|---|---|"]
    meanings = {"ID": "Representation and factorized inference; primary comparator.",
                "IH": "Factorized versus independent regeneration with HDC retained.",
                "JD": "HDC input augmentation within factorized regeneration.",
                "JH_ERM": "Smooth worst-group emphasis versus mean group loss.",
                "JH_product": "Joint scoring versus product of marginals using identical weights.",
                "P": "Full redesign versus original sparse learned gate; data and repair policy also differ.",
                "B2": "Mixed-data structured design versus original targeted-only dense baseline."}
    for contrast in result["contrasts"]:
        comp = contrast["comparator"]; metric = contrast["metrics"]["macro_regret"]
        ci = metric["percentile_95_ci"]
        lines.append(f"| {comp} | {metric['estimate']:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] | {meanings[comp]} |")
    lines += ["", "## Condition-specific diagnosis", "",
              "These descriptive means expose regressions that an overall macro average could conceal. Training now covers the listed mechanisms; these are fresh worlds, not unseen attack-family tests.", "",
              "| Condition | JH regret | ID regret | Original P regret | Original B2 regret | JH − ID | JH − B2 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, _, _, _ in CONDITIONS:
        mean = lambda method: float(np.mean([float(r["regret"]) for r in condition_rows(rows, method, label=label)]))
        jh, independent, old, specialist = mean("JH"), mean("ID"), mean("P"), mean("B2")
        lines.append(f"| {label} | {jh:.5f} | {independent:.5f} | {old:.5f} | {specialist:.5f} | {jh-independent:+.5f} | {jh-specialist:+.5f} |")
    lower, higher = [], []
    for label, _, _, _ in CONDITIONS:
        avg = lambda method: float(np.mean([float(r["regret"]) for r in condition_rows(rows, method, label=label)]))
        difference = avg("JH") - avg("B2")
        if difference < -1e-10:
            lower.append(label)
        elif difference > 1e-10:
            higher.append(label)
    lines += ["", "Relative to original B2, JH has lower mean regret in " + (", ".join(lower) if lower else "none of the four core conditions")
              + " and higher mean regret in " + (", ".join(higher) if higher else "none of the four core conditions")
              + ". These are descriptive condition means. The old specialist was trained only on targeted 20% interventions, whereas new models train on a broader mixture; a macro improvement cannot be interpreted as universal dominance or a benefit of factorization alone."]
    lines += ["", "## Limits and evidence boundaries", "",
              "- Full regeneration changes all fact beliefs as needed. It does not satisfy the old six-fact intervention cap, and improvements against old P must not be presented as a win at an equal sparse-repair budget.",
              "- The factorization is coherent under the model's assumed root/child conditional structure. It is not an exact posterior for arbitrary correlated deception, arbitrary graphs or multiple causal parents. Corrupted observations can still make the inferred probabilities wrong.",
              "- The smooth group-loss maximum emphasizes predefined training groups. It does not guarantee robustness to every attack distribution or certify worst-case deployment safety.",
              "- HDC benefit is isolated by JH versus JD. A successful joint model alone does not establish a useful HDC contribution or a wire-compression advantage. Every method still receives the common sparse report codec.",
              "- These are independent synthetic snapshots and shared candidate conjunctions. No operational multistep cyber planning, temporal recovery delay or external cyber-range performance is established.",
              "- The old test pool was used to motivate the redesign. The fresh pool is protected from model selection by the frozen second-study protocol; five seeds and a single generator still limit generalization claims.",
              "- New method composition and equations are proposed research contributions. Probabilistic factorization, semantic constraints, masked feature learning and robust group losses have established antecedents; this experiment alone does not establish a first-ever novelty claim.",
              "", "## Reproducibility", "",
              "`results/v2/manifest.json` records the executed settings, source identity and environment. `results/v2/comparisons.json` records paired effects, per-seed/per-condition effects, bootstrap design and hashes of every raw input used in the contrasts. Aggregate metrics and curves are retained under `results/v2/`; raw arrays and checkpoints remain ignored. Regenerate this report with `.venv/bin/python scripts/analyze_v2.py` only after the full second run completes.", ""]
    pipeline_path = OUT / "pipeline_timing.json"
    if pipeline_path.is_file():
        pipeline = json.loads(pipeline_path.read_text())
        timing = {r["method"]: r for r in pipeline["rows"]}
        lines += ["## Prepared-evidence host-wall timing", "",
                  "A separate audit measures 50 repetitions after five warmups on seed 37 and the first 64-fact fresh-test snapshot. It includes actual packet packing/unpacking, transfers, GPU feature construction, model inference, ranking and candidate IDs returned to the host. Every measured marginal and candidate-probability path is checked against the original inputs before timing.", "",
                  "| Method | Median (ms) | 10th–95th percentiles (ms) |", "|---|---:|---|" ]
        for method in ["JH", "JD", "IH", "ID", "JH_ERM", "JH_product", "P", "B2"]:
            row = timing[method]
            lines.append(f"| {method} | {row['median_ms']:.3f} | {row['p10_ms']:.3f}–{row['p95_ms']:.3f} |")
        lines += ["", f"The shared packet payload contains {pipeline['payload_bytes']} bytes. This bounded loop excludes sensing, bandwidth admission, topology/catalogue construction, network delays and cold model loading. These percentiles describe repeated timings on one fixed snapshot, not uncertainty across deployments or training seeds. Main figure CUDA timings exclude this host work and must be interpreted separately.", ""]
    (ROOT / "docs/RESULTS_V2.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("A completed second-study manifest is required")
    manifest = json.loads(manifest_path.read_text())
    expected = {"seeds": SEEDS, "test_seed": 42027, "test": 4000, "pilot": False}
    settings = manifest.get("settings", {})
    mismatches = {key: settings.get(key) for key, value in expected.items() if settings.get(key) != value}
    if mismatches:
        raise ValueError(f"Manifest does not describe the frozen fresh-test run: {mismatches}")
    if len(manifest.get("conditions", [])) != 18 or set(manifest.get("methods", [])) != set(METHODS):
        raise ValueError("Completed manifest must list all 18 conditions and 17 methods")
    for relative, digest in manifest.get("source_hashes", {}).items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Executed source changed since the study: {relative}")
    with (OUT / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if sorted({int(r["seed"]) for r in rows}) != SEEDS:
        raise ValueError("All five frozen training seeds must be complete")
    if len(rows) != 1530:
        raise ValueError(f"Expected 1,530 rows (5 seeds x 17 methods x 18 conditions), found {len(rows)}")
    if set(r["method"] for r in rows) != set(METHODS):
        raise ValueError("Expected all seventeen methods, including matched controls and original comparators")
    keys = ["seed", "method", "condition", "corruption", "attack", "facts", "budget", "repair_fraction"]
    if len({tuple(row[k] for k in keys) for row in rows}) != len(rows):
        raise ValueError("Duplicate method/seed/condition summary rows")
    expected_conditions = {(s[0], float(s[1]), s[2], int(s[3]), int(s[4]), float(s[5]), float(s[7]))
                           for s in manifest["conditions"]}
    for method in METHODS:
        for seed in SEEDS:
            observed_conditions = {(r["condition"], float(r["corruption"]), r["attack"], int(r["facts"]),
                                    int(r["budget"]), float(r["repair_fraction"]), float(r["missing"]))
                                   for r in rows if r["method"] == method and int(r["seed"]) == seed}
            if observed_conditions != expected_conditions:
                raise ValueError(f"Condition catalogue mismatch: {method}, seed {seed}")
            if sum(r["method"] == method and int(r["seed"]) == seed for r in rows) != 18:
                raise ValueError(f"Expected 18 conditions: {method}, seed {seed}")
            if len(condition_rows(rows, method, seed)) != 4:
                raise ValueError(f"Incomplete prespecified condition set: {method}, seed {seed}")
    result = summarize_contrasts(args.repetitions)
    result["manifest_sha256"] = sha256(manifest_path)
    result["summary_sha256"] = sha256(OUT / "summary.csv")
    result["analysis_source_sha256"] = sha256(Path(__file__))
    (OUT / "comparisons.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(rows, result, manifest)
    print(json.dumps({"primary": result["primary"], "contrasts": len(result["contrasts"]),
                      "no_feasible_per_condition": result["no_feasible_per_condition"]}, indent=2))


if __name__ == "__main__":
    main()
