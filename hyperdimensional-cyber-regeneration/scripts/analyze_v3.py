#!/usr/bin/env python3
"""Frozen third-study analysis: direct candidates versus fact reconstruction.

The first two test pools and third-study pilot are development evidence. This
script accepts only the completed seed-62027 run, checks its recorded inputs,
and uses paired crossed training-seed/topology bootstrap intervals. No model
training or inference is performed here; statistical analysis runs on the CPU.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3"
SEEDS = [11, 23, 37, 51, 71]
NEW_METHODS = ["SCR", "CAND_CTX", "CAND_LOCAL", "LOGIT", "DAE"]
DIRECT = {"SCR", "CAND_CTX", "CAND_LOCAL"}
FACT_METRICS = ["brier", "false_repair", "false_certainty", "coverage",
                "preservation", "logical_violation", "changed_fraction"]
CONDITIONS = [
    ("benign", "severity", 0.0, "targeted"),
    ("targeted", "severity", 0.2, "targeted"),
    ("random", "attack", 0.2, "random"),
    ("coherent", "attack", 0.2, "coherent"),
]
COMPARATORS = ["ID", "CAND_CTX", "CAND_LOCAL", "DAE", "LOGIT", "JH", "B2", "P"]
METRICS = {"macro_regret": "regret", "macro_recall3": "recall", "macro_plan_brier": "plan_brier"}
METHODS = {
    "SCR": "Context-conditioned four-expert candidate scorer",
    "CAND_CTX": "Single-expert candidate scorer with global context",
    "CAND_LOCAL": "Single-expert candidate scorer without global context",
    "LOGIT": "Cheap logistic fact reconstruction",
    "DAE": "Cheap denoising-autoencoder fact reconstruction",
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
CONTRAST_MEANINGS = {
    "ID": "Direct candidate prediction versus frozen mixed-data dense fact reconstruction; architecture, objective, supervision and pretraining differ.",
    "CAND_CTX": "Four latent experts versus one expert; both use global context. Parameter count also changes.",
    "CAND_LOCAL": "Latent mixture plus global context versus local single-expert scoring; two components change.",
    "DAE": "Direct candidate prediction versus a smaller pretrained denoising autoencoder with supervised fact fine-tuning.",
    "LOGIT": "Direct nonlinear candidate prediction versus a smaller logistic fact model.",
    "JH": "Direct candidate prediction versus HDC dependency-factorized full regeneration.",
    "B2": "New mixed-data design versus original targeted-only dense decision-aware reconstruction; training distributions also differ.",
    "P": "New direct scoring versus original targeted-only sparse regeneration; training distribution and output scope also differ.",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return float(value) if value not in (None, "") else float("nan")


def condition_rows(rows, method, seed=None, label=None):
    result = []
    for row in rows:
        if row["method"] != method or (seed is not None and int(row["seed"]) != seed):
            continue
        if (int(row["facts"]) != 64 or int(row["budget"]) != 2048
                or float(row["repair_fraction"]) != .1 or float(row["missing"]) != .2):
            continue
        for name, family, corruption, attack in CONDITIONS:
            if ((label is None or name == label) and row["condition"] == family
                    and row["attack"] == attack and float(row["corruption"]) == corruption):
                result.append(row)
                break
    return result


def raw_path(seed, method, condition):
    _, name, corruption, attack = condition
    return ROOT / "results/runs/v3" / str(seed) / f"{method}_{name}_{corruption}_{attack}_64_2048_0.1.npz"


def bootstrap_weights(repetitions, n_seeds, n_clusters, rng_seed=20271127):
    rng = np.random.default_rng(rng_seed)
    seed_counts = np.empty((repetitions, n_seeds), dtype=float)
    cluster_counts = np.empty((repetitions, n_clusters), dtype=float)
    for i in range(repetitions):
        seed_counts[i] = np.bincount(rng.integers(n_seeds, size=n_seeds), minlength=n_seeds)
        cluster_counts[i] = np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters)
    return seed_counts, cluster_counts


def bootstrap_draws(numerator, denominator, seed_counts, cluster_counts):
    """Arrays are seed x condition x cluster; condition weights stay fixed."""
    num = np.einsum("scg,rg->rsc", numerator, cluster_counts, optimize=True)
    den = np.einsum("scg,rg->rsc", denominator, cluster_counts, optimize=True)
    if np.any(den <= 0):
        raise ValueError("A resampled condition has an empty metric denominator")
    per_seed = (num / den).mean(axis=2)
    return (per_seed * seed_counts).sum(axis=1) / seed_counts.sum(axis=1)


def summarize_contrasts(repetitions=2000, rows=None):
    raw, hashes = {}, {}
    expected_group = expected_eligible = expected_truth = None
    parity_checks = 0
    for method in ["SCR"] + COMPARATORS:
        raw[method] = []
        for seed in SEEDS:
            per_condition = []
            for condition in CONDITIONS:
                path = raw_path(seed, method, condition)
                hashes[str(path.relative_to(ROOT))] = sha256(path)
                with np.load(path, allow_pickle=False) as archive:
                    data = {key: archive[key] for key in ["group", "eligible", "recall", "regret", "plan_brier"]}
                    prob, truth = archive["planprob"], archive["plantruth"]
                    if method in DIRECT:
                        for key in FACT_METRICS:
                            if key not in archive or not np.isnan(archive[key]).all():
                                raise ValueError(f"Candidate-only {key} must be unavailable: {path}")
                    if prob.shape != (4000, 32) or truth.shape != prob.shape:
                        raise ValueError(f"Invalid candidate probability/label shape: {path}")
                    if not np.isfinite(prob).all() or np.any(prob < 0) or np.any(prob > 1):
                        raise ValueError(f"Invalid candidate probabilities: {path}")
                    if not np.isin(truth, [0., 1.]).all():
                        raise ValueError(f"Non-Boolean evaluator candidate labels: {path}")
                    if not np.allclose(((prob - truth)**2).mean(axis=1), data["plan_brier"], atol=2e-7, rtol=2e-6):
                        raise ValueError(f"Stored candidate Brier disagrees with probabilities: {path}")
                    if expected_truth is None:
                        expected_truth = truth.copy()
                    elif not np.array_equal(truth, expected_truth):
                        raise ValueError(f"Evaluator candidate truth is not paired: {path}")
                if data["group"].shape != (4000,) or data["eligible"].shape != (4000,):
                    raise ValueError(f"Expected 4,000 test snapshots: {path}")
                if expected_group is None:
                    expected_group = data["group"].copy()
                    expected_eligible = data["eligible"].astype(bool).copy()
                if not np.array_equal(data["group"], expected_group):
                    raise ValueError(f"Topology cluster ordering is not paired: {path}")
                if not np.array_equal(data["eligible"].astype(bool), expected_eligible):
                    raise ValueError(f"Feasibility changed across paired cases: {path}")
                if not np.array_equal(data["eligible"].astype(bool), expected_truth.any(axis=1)):
                    raise ValueError(f"Eligibility does not match evaluator labels: {path}")
                for metric in METRICS.values():
                    if data[metric].shape != (4000,) or not np.isfinite(data[metric]).all():
                        raise ValueError(f"Invalid per-snapshot {metric}: {path}")
                if rows is not None:
                    chosen = condition_rows(rows, method, seed, condition[0])
                    if len(chosen) != 1:
                        raise ValueError(f"Expected one summary row: {method}/{seed}/{condition[0]}")
                    for key, column in [("regret", "regret"), ("plan_brier", "plan_brier"), ("recall", "recall3")]:
                        weight = expected_eligible if key == "recall" else np.ones(4000, dtype=bool)
                        value = float(data[key][weight].mean())
                        if not np.isclose(value, float(chosen[0][column]), atol=2e-7, rtol=2e-6):
                            raise ValueError(f"Raw/summary mismatch for {column}: {path}")
                        parity_checks += 1
                per_condition.append(data)
            raw[method].append(per_condition)
    unique, inverse, cluster_sizes = np.unique(expected_group, return_inverse=True, return_counts=True)
    if len(unique) != 250 or not np.all(cluster_sizes == 16):
        raise ValueError("Expected 250 paired topology clusters of sixteen snapshots")
    seed_counts, cluster_counts = bootstrap_weights(repetitions, len(SEEDS), len(unique))
    contrasts = []
    for comparator in COMPARATORS:
        contrast = {"method": "SCR", "comparator": comparator,
                    "direction": f"SCR minus {comparator}", "metrics": {}, "per_condition_effects": {}}
        for metric, key in METRICS.items():
            numerators, denominators, a_means, b_means = [], [], [], []
            for si, seed in enumerate(SEEDS):
                ns, ds, aa, bb = [], [], [], []
                for ci, condition in enumerate(CONDITIONS):
                    a, b = raw["SCR"][si][ci], raw[comparator][si][ci]
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
                "means": {"SCR": float(np.mean(a_means)), comparator: float(np.mean(b_means))},
                "seed_effects": dict(zip(map(str, SEEDS), seed_effect.tolist())),
            }
            contrast["per_condition_effects"][metric] = {
                condition[0]: {"mean_difference": float(condition_effect[:, ci].mean()),
                               "seed_effects": dict(zip(map(str, SEEDS), condition_effect[:, ci].tolist()))}
                for ci, condition in enumerate(CONDITIONS)
            }
        contrasts.append(contrast)
    primary = dict(contrasts[0]["metrics"]["macro_regret"])
    primary.update(method="SCR", comparator="ID", metric="macro_regret", direction="SCR minus ID")
    return {
        "primary": primary, "contrasts": contrasts, "seeds": SEEDS,
        "test_generator_seed": 62027, "snapshots_per_condition": 4000,
        "topology_clusters": len(unique), "snapshots_per_cluster": 16,
        "conditions": [{"label": label, "condition": name, "corruption": c, "attack": a}
                       for label, name, c, a in CONDITIONS],
        "eligible_per_condition": int(expected_eligible.sum()),
        "no_feasible_per_condition": int((~expected_eligible).sum()),
        "bootstrap_repetitions": repetitions, "bootstrap_rng_seed": 20271127,
        "bootstrap_design": "Paired crossed training-seed/topology bootstrap. Resample five training seeds and 250 topology clusters with replacement; use the same cluster draw across every selected seed, method and all four paired conditions. Keep all sixteen snapshots within a cluster. Conditions stay fixed and equally weighted.",
        "estimand": "Equal-weight mean over benign0, targeted0.2, random0.2 and coherent0.2 conditions, then mean over training seeds. Regret and plan Brier retain all snapshots. Recall is conditional on a feasible candidate within each condition before macro averaging.",
        "inference_scope": "Primary: SCR-minus-ID macro regret. Secondary intervals are descriptive and unadjusted for multiple comparisons. A zero-containing interval does not establish equivalence. Five seeds, fixed training/validation data and one generator limit inference; generator-design uncertainty is not bootstrapped.",
        "development_disclosure": "Prior test pools 32027/42027 and pilot pools 52027/72027 are development evidence. The frozen third study evaluates fresh pool62027; familiar attack mechanisms occur in the training mixture and are not unseen-mechanism tests.",
        "raw_input_sha256": hashes, "raw_summary_parity_checks": parity_checks,
    }


def validate_inputs(manifest, rows):
    expected = {"seeds": SEEDS, "test_seed": 62027, "test": 4000, "pilot": False,
                "train": 20000, "val": 2000, "train_seed": 12027, "validation_seed": 22027,
                "batch": 256, "epochs": 40, "patience": 8, "lr": .001, "risk_beta": 2.}
    settings = manifest.get("settings", {})
    mismatches = {key: settings.get(key) for key, value in expected.items() if settings.get(key) != value}
    if mismatches:
        raise ValueError(f"Manifest does not describe the frozen fresh-test run: {mismatches}")
    if len(manifest.get("conditions", [])) != 18 or set(manifest.get("methods", [])) != set(METHODS):
        raise ValueError("Completed manifest must list all eighteen conditions and twenty-two methods")
    source_hashes = manifest.get("source_hashes", {})
    required_sources = {"src/regen_v3.py", "scripts/run_v3.py", "configs/study_v3.json"}
    if not required_sources.issubset(source_hashes):
        raise ValueError("Executed new model, runner and configuration source hashes are required")
    for relative, digest in source_hashes.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Executed source changed since the study: {relative}")
    if not manifest.get("source_commit") or len(manifest.get("test_hashes", {})) != 18 or manifest.get("wall_seconds", 0) <= 0:
        raise ValueError("Completed run needs source commit, eighteen test hashes and positive wall time")
    prior_path = ROOT / "results/v2/manifest.json"
    prior = json.loads(prior_path.read_text())
    if sha256(prior_path) != manifest.get("reused_v2_manifest_sha256"):
        raise ValueError("Reused second-study manifest changed")
    if manifest.get("data_hashes") != prior["data_hashes"]:
        raise ValueError("New models did not use the exact second-study mixed train/validation observations")
    if len(rows) != 1980 or sorted({int(r["seed"]) for r in rows}) != SEEDS:
        raise ValueError(f"Expected 1,980 records across all five frozen seeds, found {len(rows)}")
    if set(r["method"] for r in rows) != set(METHODS):
        raise ValueError("Expected all twenty-two methods")
    keys = ["seed", "method", "condition", "corruption", "attack", "facts", "budget", "repair_fraction", "missing"]
    if len({tuple(row[k] for k in keys) for row in rows}) != len(rows):
        raise ValueError("Duplicate seed/method/condition summary rows")
    expected_conditions = {(s[0], float(s[1]), s[2], int(s[3]), int(s[4]), float(s[5]), float(s[7]))
                           for s in manifest["conditions"]}
    for method in METHODS:
        for seed in SEEDS:
            selected = [r for r in rows if r["method"] == method and int(r["seed"]) == seed]
            observed = {(r["condition"], float(r["corruption"]), r["attack"], int(r["facts"]),
                         int(r["budget"]), float(r["repair_fraction"]), float(r["missing"])) for r in selected}
            if observed != expected_conditions or len(selected) != 18 or len(condition_rows(rows, method, seed)) != 4:
                raise ValueError(f"Incomplete condition catalogue: {method}, seed {seed}")
            for row in selected:
                for metric in ["regret", "recall3", "plan_brier", "latency_ms", "bytes"]:
                    if not np.isfinite(number(row.get(metric))):
                        raise ValueError(f"Missing shared metric {metric}: {method}, seed {seed}")
                if method in DIRECT and any(not np.isnan(number(row.get(k))) for k in FACT_METRICS):
                    raise ValueError(f"Candidate-only fact metrics must be unavailable: {method}, seed {seed}")
                if number(row["bytes"]) < 0 or number(row["bytes"]) > 2 * int(row["budget"]):
                    raise ValueError(f"Packet budget exceeded: {method}, seed {seed}")
    checkpoints = manifest.get("checkpoints", [])
    if {(int(r["seed"]), r["method"]) for r in checkpoints} != {(s, m) for s in SEEDS for m in NEW_METHODS} or len(checkpoints) != 25:
        raise ValueError("Expected one new checkpoint record per method and seed (twenty-five total)")
    for record in checkpoints:
        path = ROOT / "results/checkpoints/v3" / f"{record['seed']}_{record['method']}.pt"
        if sha256(path) != record["checkpoint_sha256"]:
            raise ValueError(f"New checkpoint hash mismatch: {path}")
        if record.get("pretrain_epochs") != (5 if record["method"] == "DAE" else 0):
            raise ValueError(f"Unexpected pretraining budget: {path}")
        if not (0 <= record["best_epoch"] < record["epochs_run"] <= 40):
            raise ValueError(f"Invalid checkpoint epoch accounting: {path}")
        times = [record[k] for k in ["pretrain_seconds", "supervised_seconds", "train_seconds"]]
        if not np.isfinite(times).all() or min(times) < 0 or times[2] + .001 < times[0] + times[1]:
            raise ValueError(f"Invalid training time accounting: {path}")
    old_checkpoint_hashes = manifest.get("legacy_checkpoint_hashes", {})
    if len(old_checkpoint_hashes) != 55:
        raise ValueError("Expected hashes for all twenty-five v2 and thirty v1 reused checkpoints")
    for relative, digest in old_checkpoint_hashes.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Reused checkpoint changed: {relative}")
    if "legacy_thresholds_sha256" in manifest and sha256(ROOT / "results/selected_thresholds.json") != manifest["legacy_thresholds_sha256"]:
        raise ValueError("Frozen original repair thresholds changed")
    return {"summary_records": len(rows), "source_hashes_verified": len(source_hashes),
            "new_models_at_epoch_cap": sum(r["epochs_run"] == 40 for r in checkpoints),
            "selected_epoch_range_zero_based": [min(r["best_epoch"] for r in checkpoints), max(r["best_epoch"] for r in checkpoints)],
            "new_checkpoints_verified": len(checkpoints), "reused_checkpoints_verified": len(old_checkpoint_hashes),
            "direct_candidate_rows_with_unavailable_fact_metrics": sum(r["method"] in DIRECT for r in rows)}



def validate_external_artifacts(manifest):
    """Audit recorded replay/timing outputs; do not rerun CUDA work here."""
    audited = {}
    replay_path = OUT / "replay.json"
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text())
        expected = {"passed": True, "training_seed": 37, "fresh_test_seed": 62027,
                    "model_condition_replays": 12, "model_snapshot_evaluations": 48000}
        if any(replay.get(k) != v for k, v in expected.items()) or len(replay.get("rows", [])) != 12:
            raise ValueError("Incomplete third-study replay record")
        if replay["source_manifest_sha256"] != sha256(OUT / "manifest.json"):
            raise ValueError("Replay used a different study manifest")
        if replay["replay_script_sha256"] != sha256(ROOT / "scripts/replay_v3.py"):
            raise ValueError("Replay script changed after verification")
        expected_cases = {(method, family, corruption, attack)
                          for method in ["SCR", "DAE", "LOGIT"]
                          for _, family, corruption, attack in CONDITIONS}
        observed_cases = set()
        errors = []
        for row in replay["rows"]:
            observed_cases.add((row["method"], row["condition"], row["corruption"], row["attack"]))
            if row["snapshots"] != 4000:
                raise ValueError("Replay snapshot count changed")
            for path_key, hash_key in [("checkpoint_path", "checkpoint_sha256"), ("raw_path", "raw_sha256")]:
                if sha256(ROOT / row[path_key]) != row[hash_key]:
                    raise ValueError(f"Replay input changed: {row[path_key]}")
            tag = f"{row['condition']}_{row['corruption']}_{row['attack']}_64_2048_0.1"
            if row["input_data_sha256"] != manifest["test_hashes"][tag]:
                raise ValueError("Replay input data does not match frozen test hashes")
            errors.extend(row["max_absolute_errors"].values())
        if observed_cases != expected_cases or not np.isfinite(errors).all() or max(errors) > replay["absolute_tolerance"]:
            raise ValueError("Replay cases or errors fail verification")
        if any(value != max(row["max_absolute_errors"][key] for row in replay["rows"])
               for key, value in replay["maximum_absolute_errors"].items()):
            raise ValueError("Replay maximum-error summary is inconsistent")
        audited["replay"] = {"sha256": sha256(replay_path), "model_snapshot_evaluations": 48000,
                             "model_condition_replays": 12, "maximum_absolute_error": max(errors),
                             "methods": ["SCR", "DAE", "LOGIT"], "training_seed": 37}
    pipeline_path = OUT / "pipeline_timing.json"
    if pipeline_path.is_file():
        timing = json.loads(pipeline_path.read_text())
        if (timing.get("seed") != 37 or timing.get("test_seed") != 62027 or timing.get("snapshot") != 0
                or timing.get("facts") != 64 or timing.get("warmups") != 5
                or timing.get("manifest_sha256") != sha256(OUT / "manifest.json")):
            raise ValueError("Timing scope or provenance differs from the declared audit")
        expected_methods = {"SCR", "CAND_CTX", "CAND_LOCAL", "DAE", "LOGIT", "ID", "JH"}
        if {r["method"] for r in timing["rows"]} != expected_methods or len(timing["rows"]) != 7:
            raise ValueError("Timing record needs all seven methods")
        for relative, digest in timing["source_hashes"].items():
            if sha256(ROOT / relative) != digest:
                raise ValueError(f"Timing source changed: {relative}")
        for method, digest in timing["checkpoint_hashes"].items():
            version = "v3" if method in NEW_METHODS else "v2"
            if sha256(ROOT / f"results/checkpoints/{version}/37_{method}.pt") != digest:
                raise ValueError(f"Timing checkpoint changed: {method}")
        for row in timing["rows"]:
            values = [row[k] for k in ["p10_ms", "median_ms", "p95_ms"]]
            if row["repeats"] != 50 or not np.isfinite(values).all() or not 0 <= values[0] <= values[1] <= values[2]:
                raise ValueError("Timing repetition count or quantiles are invalid")
        if not timing.get("prediction_roundtrip") or not 0 <= timing["payload_bytes"] <= 4096:
            raise ValueError("Missing prediction roundtrip record or invalid payload budget")
        audited["pipeline_timing"] = {"sha256": sha256(pipeline_path), "methods": 7,
                                      "repeats": 50, "warmups": 5,
                                      "payload_bytes": timing["payload_bytes"],
                                      "prediction_roundtrip": timing["prediction_roundtrip"]}
    return audited


def avg(rows, method, key, label=None):
    values = np.asarray([number(r.get(key)) for r in condition_rows(rows, method, label=label)])
    return float(values.mean())


def fmt(value, digits=5):
    return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"


def write_report(rows, result, manifest):
    primary = result["primary"]; lo, hi = primary["percentile_95_ci"]
    conclusion = ("The primary interval favors SCR within this synthetic generator." if hi < 0 else
                  "The primary interval favors the frozen mixed-data dense independent baseline." if lo > 0 else
                  "The primary interval includes zero; improvement or equivalence is not established.")
    lines = [
        "# Third approach: direct candidate prediction and cheap reconstruction",
        "",
        "This prospectively frozen study tests a different output representation. SCR predicts course-of-action candidate success directly from observable evidence and symbolic requirements, bypassing fact reconstruction and products of fact marginals. It uses four latent experts mixed by a shared observational context. These experts have no attack-type labels and must not be identified as recovered attack mechanisms without separate evidence.",
        "",
        f"Hardware: {manifest.get('hardware', 'see manifest')}; PyTorch {manifest.get('torch', 'see manifest')}, CUDA {manifest.get('cuda', 'see manifest')}. Twenty-five new models (five variants × five seeds) train on the common 20,000-snapshot mixture and 2,000 validation snapshots. All seventeen previous methods are evaluated from frozen checkpoints or fixed rules. The completed run has 22 methods × 18 conditions × 5 seeds = 1,980 records, with recorded study wall time {manifest['wall_seconds']/60:.2f} minutes.",
        "",
        "## Comparison design",
        "",
        "SCR, CAND_CTX and CAND_LOCAL use the same candidate BCE + candidate Brier + 0.25 soft decision-regret objective and smooth group aggregation (beta 2). SCR has four experts; CAND_CTX has one with global context; CAND_LOCAL removes the global-context features from the single-expert design. These comparisons test mixture capacity and context, not a generic first-ever set-network contribution. SCR versus CAND_CTX isolates the expert construction while also changing parameter count. CAND_CTX versus CAND_LOCAL is a descriptive context ablation; its interval is not part of the prespecified SCR contrast family.",
        "",
        "LOGIT and DAE predict fact readiness and score candidates by products of those probabilities. DAE first reconstructs pre-corruption observed features from additional feature masking/noise, then fine-tunes with the same supervised fact-and-decision loss as v2. Its autoencoder target is the already noisy observed feature vector, not clean hidden readiness. LOGIT receives no such pretraining. Maximum supervised epochs, patience, batch size and learning rate are shared, but actual training time and model capacity differ. Direct candidate models and fact models have different losses and supervision targets. The frozen v2 models also received five graph-masked pretraining epochs, whereas the new direct scorers receive none. Their comparison is a system-level comparison, not an isolated architecture ablation or an equal-total-optimization comparison.",
        "",
        "The new method retains symbolic root/child requirement structure in candidate construction, but produces no reconstructed fact beliefs. Candidate success probabilities are marginal probabilities for each candidate, not a joint distribution over all candidate outcomes. Candidate-specific value/cost tensors enter utility scoring and the decision loss, not the new feasibility heads. However, inherited observable feature x[12] is a known value-weighted candidate-criticality summary used by all methods. The feasibility models therefore retain indirect dependence on the supplied candidate values; changing value/cost tensors while holding x fixed tests only direct tensor blinding, not full value independence.",
        "",
        f"The fresh evaluation pool uses seed 62027, five training seeds ({', '.join(map(str, SEEDS))}), and 4,000 snapshots per condition. Prior pools 32027/42027 and pilots 52027/72027 are development evidence. The four primary conditions pair the same hidden worlds, topologies and candidates. Of 4,000 snapshots, {result['eligible_per_condition']:,} have a feasible candidate and {result['no_feasible_per_condition']:,} do not. All snapshots enter regret and candidate Brier; only feasible snapshots enter conditional Recall@3.",
        "",
        "## Primary result",
        "",
        f"Macro candidate-oracle regret: SCR **{primary['means']['SCR']:.5f}**, ID **{primary['means']['ID']:.5f}**. SCR−ID: **{primary['estimate']:+.5f}**, 95% crossed-bootstrap interval **[{lo:+.5f}, {hi:+.5f}]**. Lower is better. {conclusion}",
        "",
        "The interval uses 2,000 draws across five training seeds and 250 topology clusters, retaining all sixteen snapshots in each cluster. The same cluster draw is shared across seeds, methods and four fixed, equally weighted conditions. It quantifies training-seed and test-topology variation within this generator; it does not capture uncertainty about the generator, real deployments, or the common training and validation datasets.",
        "",
        "## Four-condition macro results",
        "",
        "N/A means the method has no fact-belief output; it is not a zero error or perfect semantic score. Recall is conditional on feasibility and is shown as a percentage. Each entry averages the four fixed conditions and five training seeds.",
        "",
        "| Method | Recall@3 (%) | Regret ↓ | Candidate Brier ↓ | Fact Brier ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in METHODS.items():
        if len(condition_rows(rows, method)) != 20:
            raise ValueError(f"Incomplete macro results: {method}")
        lines.append(f"| {method}: {label} | {100*avg(rows, method, 'recall3'):.3f} | {avg(rows, method, 'regret'):.5f} | {avg(rows, method, 'plan_brier'):.5f} | {fmt(avg(rows, method, 'brier'))} |")
    lines += ["", "## Prespecified secondary contrasts", "",
              "All effects are SCR minus comparator macro regret. Secondary intervals are descriptive and unadjusted for multiple comparisons. Negative values favor SCR; intervals containing zero do not establish equivalence.", "",
              "| Comparator | Effect | 95% interval | Interpretation |", "|---|---:|---|---|"]
    for contrast in result["contrasts"]:
        comp = contrast["comparator"]; metric = contrast["metrics"]["macro_regret"]; ci = metric["percentile_95_ci"]
        lines.append(f"| {comp} | {metric['estimate']:+.5f} | [{ci[0]:+.5f}, {ci[1]:+.5f}] | {CONTRAST_MEANINGS[comp]} |")
    lines += ["", "## Condition-specific comparisons", "",
              "These descriptive means expose condition-specific regressions that a macro average can hide. All four mechanisms occur in mixed training; they are not previously unseen attack families.", "",
              "| Condition | SCR | CAND_CTX | CAND_LOCAL | DAE | LOGIT | ID | JH | Original B2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, _, _, _ in CONDITIONS:
        cells = " | ".join(f"{avg(rows, m, 'regret', label):.5f}" for m in ["SCR", "CAND_CTX", "CAND_LOCAL", "DAE", "LOGIT", "ID", "JH", "B2"])
        lines.append(f"| {label} | {cells} |")
    lines += ["", "## Compute and output scope", "",
              "The following descriptive means are from the common reference condition (targeted 20%, 64 facts, 2,048 bytes per remote observer). CUDA model-plus-ranking latency excludes host packet processing. Training times are recorded actual work, not a promise that fewer parameters always imply lower latency.", "",
              "| Method | Trainable parameters | Mean training seconds | Mean seed-median CUDA latency (ms) | Mean additional peak VRAM (MiB) |",
              "|---|---:|---:|---:|---:|"]
    checkpoint_records = manifest.get("checkpoints", []) + json.loads((ROOT / "results/v2/manifest.json").read_text())["checkpoints"]
    for method in ["SCR", "CAND_CTX", "CAND_LOCAL", "DAE", "LOGIT", "ID", "JH"]:
        selected = condition_rows(rows, method, label="targeted")
        records = [r for r in checkpoint_records if r["method"] == method]
        parameters = int(np.mean([r["parameters"] for r in records])) if records else None
        # Historical checkpoints may be recorded separately; a missing count stays unavailable.
        parameter_text = f"{parameters:,}" if parameters is not None else "See frozen prior manifest"
        mean = lambda key: float(np.mean([number(r.get(key)) for r in selected]))
        lines.append(f"| {method} | {parameter_text} | {fmt(mean('train_seconds'), 2)} | {fmt(mean('latency_ms'), 3)} | {fmt(mean('peak_vram_mb'), 3)} |")
    lines += ["", "Parameter counts include retained pretraining decoders. DAE has 4,233 retained parameters, of which 2,097 participate in readiness inference. Prior-model training durations were recorded in their original runs; they were not retrained for this study. Latency above is the mean of the five seed-specific warm CUDA medians, while memory is additional peak allocation above resident tensors.", "",
              f"DAE and LOGIT trade predictive quality for lower measured cost: their macro regrets are {avg(rows, 'DAE', 'regret'):.5f} and {avg(rows, 'LOGIT', 'regret'):.5f}, compared with ID's {avg(rows, 'ID', 'regret'):.5f}. SCR improves over the smaller new models and its single-expert control, but its fewer parameters do not translate into lower measured inference latency than ID. Architecture size alone is therefore an incomplete efficiency argument."]
    if "shadow_union_violation" in rows[0]:
        lines += ["", "## Candidate monotonicity diagnostic", "",
                  "A diagnostic adds requirements by taking adjacent candidate unions with observable x held fixed. A valid conjunction probability should not increase when requirements are added. The unions contain up to ten requirements, whereas training candidates have two to five, so this is a query-length stress test. The violation rate is a descriptive logical-consistency check, not an accuracy measure or a training objective. No shadow candidate is inserted into the ranking catalogue, and the inherited catalogue-criticality feature is not recomputed for these shadow queries.", "",
                  "| Method | Macro shadow-union violation fraction ↓ |", "|---|---:|"]
        for method in ["SCR", "CAND_CTX", "CAND_LOCAL", "DAE", "LOGIT", "ID", "JH"]:
            lines.append(f"| {method} | {fmt(avg(rows, method, 'shadow_union_violation'))} |")
        lines += ["", f"SCR increases its success probability after adding requirements in **{100*avg(rows, 'SCR', 'shadow_union_violation'):.2f}%** of these shadow-union comparisons. This is a measured shortcoming of the direct scorer: its queried probabilities can violate conjunction monotonicity and do not constitute a guaranteed coherent joint model of candidate outcomes. The reconstruction methods shown above have zero violations in this check. Because the unions extend beyond the training query lengths, this diagnostic establishes a stress-test failure, not the cause of SCR's regret gap on the original catalogue."]
    lines += ["", "## Evidence boundaries", "",
              "- SCR can compare supplied candidates without regenerating a reusable fact state. It cannot claim fact-level repair, marginal logical coherence or fact-level explanations. Fact and repair metrics are intentionally unavailable.",
              "- Every one of the twenty-five new models reached the fixed forty-epoch supervised cap; none stopped early. Selected checkpoints were at zero-based epochs 36–39 (mostly 39). These results compare methods under that optimization budget and do not establish converged architecture rankings or rule out improvements with longer training.",
        "- The four-expert gate is a latent predictive mechanism. Expert identities are not supervised and have no established one-to-one relation to attack types. A low-entropy gate is not proof of deception detection.",
              "- Global context may identify correlations specific to this simulator. Fresh topology snapshots do not establish robustness to arbitrary adversaries or unseen deception mechanisms.",
              "- Relative gains over original P/B2 combine broader training, different output scope and changed architecture/objectives. The old sparse repair budget is not a constraint on candidate-only scoring.",
              "- Every method uses the same sparse report codec. These comparisons do not establish a hyperdimensional wire-compression advantage. The direct candidate design has no HDC encoder; prior HDC methods remain explicit comparators.",
              "- The benchmark is synthetic, with independent snapshots, a one-level prerequisite forest and fixed candidate conjunctions. It does not establish multistep cyber planning, temporal recovery performance, real cyber-range effectiveness or operational mission utility.",
              "- The equation and methodological composition are proposed contributions. Set aggregation, mixtures of experts, denoising autoencoders and decision-focused learning have established antecedents; novelty beyond this composition requires a broader literature assessment.",
              "", "## Reproducibility", "",
              "`results/v3/manifest.json` records executed settings, source identity, data hashes, hardware, checkpoint provenance and training durations. `results/v3/comparisons.json` records the prespecified paired analysis, per-condition/per-seed effects and all raw input hashes. The analysis validates source/checkpoint integrity and raw/summary parity before generating this report. Raw arrays, weights and logs remain ignored by Git. Reproduce the analysis with `.venv/bin/python scripts/analyze_v3.py` only after the complete GPU run.", ""]
    pipeline_path = OUT / "pipeline_timing.json"
    if pipeline_path.is_file():
        pipeline = json.loads(pipeline_path.read_text())
        lines += ["## Prepared-evidence host-wall timing", "",
                  "This separate bounded measurement includes actual packet packing/unpacking, transfers, feature construction, GPU inference, ranking and candidate IDs returned to the host. It excludes sensing, bandwidth admission, topology/catalogue construction, network delay and cold loading. Percentiles concern repeated executions of the recorded snapshot, not deployment uncertainty.", "",
                  "| Method | Median (ms) | 10th–95th percentiles (ms) |", "|---|---:|---|"]
        for record in pipeline.get("rows", []):
            lines.append(f"| {record['method']} | {record['median_ms']:.3f} | {record['p10_ms']:.3f}–{record['p95_ms']:.3f} |")
        if "payload_bytes" in pipeline:
            lines += ["", f"The common measured packet payload is {pipeline['payload_bytes']} bytes."]
        lines.append("")
    (ROOT / "docs/RESULTS_V3.md").write_text("\n".join(lines))


def write_audit(result, manifest):
    checks = result["audit_checks"]; primary = result["primary"]
    lines = ["# Third-study evidence audit", "",
             "This document is generated from the completed third-study manifest and validated raw inputs. It records checks actually performed by `scripts/analyze_v3.py`; model-behavior tests, timing and replay are separate evidence when available.", "",
             "## Completed statistical and provenance checks", "",
             f"- Complete catalogue: {checks['summary_records']:,} unique records, 22 methods, five seeds and eighteen conditions per method/seed.",
             "- Confirmatory pool: generator seed 62027, 4,000 snapshots per condition, with the prescribed 20,000 training/2,000 validation counts and optimization settings. Prior test and pilot pools are disclosed as development evidence.",
             "- Matched data: the prior v2 manifest hash matches, and complete train/validation observation-plus-truth/candidate hashes equal the v2 mixture. New-model pretraining epochs, best/stopping epochs and recorded training durations satisfy the protocol.",
             f"- Executed source identity: {checks['source_hashes_verified']} recorded source hashes match the current files; source commit `{manifest['source_commit']}` is recorded.",
             f"- Weights: all {checks['new_checkpoints_verified']} new checkpoint hashes and {checks['reused_checkpoints_verified']} recorded reused-checkpoint hashes match disk. Recorded original thresholds are checked when present.",
             f"- Candidate-only output scope: all {checks['direct_candidate_rows_with_unavailable_fact_metrics']} SCR/CAND_CTX/CAND_LOCAL rows keep fact Brier, false-repair, false-certainty, coverage, preservation, marginal logical violations and changed-fact rates unavailable.",
             f"- Optimization-budget audit: {checks['new_models_at_epoch_cap']}/25 new models reached all forty supervised epochs; selected checkpoints span zero-based epochs {checks['selected_epoch_range_zero_based'][0]}–{checks['selected_epoch_range_zero_based'][1]}. None stopped early. The ranking is a fixed-budget result, not evidence that every architecture converged.",
             "- Communication: reported mean bytes are nonnegative and do not exceed the two remote-observer budgets.",
             f"- Paired raw inputs: {len(result['raw_input_sha256'])} archives have SHA-256 records; topology groups, evaluator candidate truth and eligibility match across methods, seeds and four conditions.",
             "- Raw candidate probabilities are finite and lie in [0,1]; evaluator labels are Boolean; per-snapshot candidate Brier is recomputed and checked.",
             f"- Raw/summary agreement: {result['raw_summary_parity_checks']} comparisons of regret, candidate Brier and eligible Recall@3 agree within floating-point tolerance.",
             "- Uncertainty: exactly 2,000 paired crossed training-seed/topology bootstrap draws; 250 clusters of sixteen snapshots; fixed equal condition weights; seed and cluster draws shared across paired methods/conditions.",
             "", "## Primary evidence", "",
             f"SCR−ID macro regret = {primary['estimate']:+.8f}; 95% percentile interval [{primary['percentile_95_ci'][0]:+.8f}, {primary['percentile_95_ci'][1]:+.8f}]. The primary outcome and comparator were fixed before the fresh test. Secondary intervals are descriptive and not multiplicity-adjusted.",
             "", "## Separately recorded artifacts", ""]
    external = result.get("external_artifacts", {})
    if "replay" in external:
        record = external["replay"]
        lines += [f"- GPU replay verified {record['model_snapshot_evaluations']:,} model-snapshot evaluations: SCR, DAE and LOGIT at seed 37 across all four core conditions. Candidate probabilities, regret and candidate Brier have maximum absolute error {record['maximum_absolute_error']:.8g}. All twelve replay input/checkpoint hashes, manifest identity and replay-script hash match the recorded files."]
    if "pipeline_timing" in external:
        record = external["pipeline_timing"]
        lines += [f"- Bounded prepared-evidence timing verified seven methods, fifty repetitions after five warmups, seed 37/snapshot 0 and a {record['payload_bytes']}-byte payload. Source/checkpoint/manifest hashes and timing quantiles match; the timing script records successful probability and available-marginal roundtrip assertions. This is a fixed-snapshot wall-time audit, not full operational latency."]
    for name in ["replay.json", "pipeline_timing.json"]:
        path = OUT / name
        if path.is_file():
            lines.append(f"- `{path.relative_to(ROOT)}` exists with SHA-256 `{sha256(path)}`. The validation scope is stated above and in that artifact.")
        else:
            lines.append(f"- `{path.relative_to(ROOT)}` was not present when this analysis ran; no completed check is claimed for it here.")
    lines += ["", "## Interpretation limits", "",
              "The audit establishes consistency of the recorded synthetic experiment, not operational cyber effectiveness or first-ever methodological novelty. Candidate models supply no fact beliefs. A latent expert is not an identified attack class. New feasibility heads exclude candidate-specific value/cost tensors, but inherited x[12] retains a known value-weighted criticality summary; tensor-blinding tests with x fixed do not imply complete value independence. Full-versus-sparse output scope and objective/supervision differences remain material in cross-family comparisons. Fresh-test isolation is the prospective study procedure recorded by the protocol and source commit; file hashes alone cannot prove that no human viewed a test pool early.", ""]
    (ROOT / "docs/V3_AUDIT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.repetitions != 2000:
        raise ValueError("The frozen third-study analysis requires exactly 2,000 bootstrap repetitions")
    manifest_path = OUT / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("A completed third-study manifest is required")
    manifest = json.loads(manifest_path.read_text())
    with (OUT / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    checks = validate_inputs(manifest, rows)
    result = summarize_contrasts(args.repetitions, rows)
    result.update(manifest_sha256=sha256(manifest_path), summary_sha256=sha256(OUT / "summary.csv"),
                  analysis_source_sha256=sha256(Path(__file__)), audit_checks=checks,
                  external_artifacts=validate_external_artifacts(manifest))
    (OUT / "comparisons.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(rows, result, manifest)
    write_audit(result, manifest)
    print(json.dumps({"primary": result["primary"], "contrasts": len(result["contrasts"]),
                      "no_feasible_per_condition": result["no_feasible_per_condition"],
                      "audit_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
