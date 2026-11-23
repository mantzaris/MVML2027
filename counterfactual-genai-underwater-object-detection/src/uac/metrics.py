"""Detection metrics and group-level uncertainty; learned calibration stays CUDA."""
from __future__ import annotations
from typing import Any
import numpy as np


def _binary_inputs(labels, scores, weights=None):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Labels and scores must be nonempty equal-length vectors")
    if not np.all(np.isin(labels, [0, 1])) or not np.all(np.isfinite(scores)):
        raise ValueError("Labels must be binary and scores finite")
    weights = np.ones(len(labels), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != labels.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite, nonnegative and aligned")
    selected = weights > 0
    if not np.any(selected):
        raise ValueError("At least one sample must have positive weight")
    return labels[selected].astype(np.int64), scores[selected], weights[selected]


def _threshold_counts(labels, scores, weights):
    order = np.argsort(-scores, kind="stable")
    labels, scores, weights = labels[order], scores[order], weights[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(weights * labels)[ends]
    fp = np.cumsum(weights * (1 - labels))[ends]
    return tp, fp, scores[ends]


def binary_average_precision(labels, scores, weights=None) -> float | None:
    """Noninterpolated threshold AP, grouping tied scores before accumulating."""
    labels, scores, weights = _binary_inputs(labels, scores, weights)
    positives = float(np.dot(labels, weights))
    if positives == 0:
        return None
    tp, fp, _ = _threshold_counts(labels, scores, weights)
    precision = tp / (tp + fp)
    recall_increments = np.diff(np.r_[0.0, tp]) / positives
    return float(np.dot(recall_increments, precision))


def binary_auroc(labels, scores, weights=None) -> float | None:
    labels, scores, weights = _binary_inputs(labels, scores, weights)
    positives = float(np.dot(labels, weights))
    negatives = float(np.dot(1 - labels, weights))
    if positives == 0 or negatives == 0:
        return None
    tp, fp, _ = _threshold_counts(labels, scores, weights)
    tpr, fpr = np.r_[0.0, tp / positives], np.r_[0.0, fp / negatives]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def binary_brier(labels, probabilities, weights=None) -> float:
    labels, probabilities, weights = _binary_inputs(labels, probabilities, weights)
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Brier score requires probabilities in [0,1]")
    return float(np.average((probabilities - labels) ** 2, weights=weights))


def binary_metrics(labels, probabilities, threshold: float = 0.5) -> dict[str, Any]:
    labels, probabilities, _ = _binary_inputs(labels, probabilities)
    if np.any((probabilities < 0) | (probabilities > 1)) or not np.isfinite(threshold):
        raise ValueError("Probabilities must be in [0,1] and threshold finite")
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    return {
        "n": len(labels), "positives": tp + fn, "negatives": tn + fp,
        "ap": binary_average_precision(labels, probabilities),
        "auroc": binary_auroc(labels, probabilities),
        "brier": binary_brier(labels, probabilities), "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def threshold_for_fpr(labels, probabilities, max_fpr: float = 0.01) -> dict:
    """Choose lowest calibration threshold satisfying the empirical FPR target.

    Prediction uses score >= threshold. A threshold just above the maximum
    negative score yields zero calibration false positives; it may exceed 1.
    This function does not assert a population guarantee.
    """
    labels, probabilities, _ = _binary_inputs(labels, probabilities)
    if not 0 <= max_fpr <= 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid probability/FPR range")
    negatives = probabilities[labels == 0]
    if not len(negatives):
        raise ValueError("Calibration requires negative examples")
    candidates = np.unique(np.r_[0.0, np.nextafter(negatives, np.inf)])
    valid = [float(t) for t in candidates if np.mean(negatives >= t) <= max_fpr]
    threshold = min(valid)
    return {"threshold": threshold, "target_fpr": float(max_fpr),
            "calibration_fpr": float(np.mean(negatives >= threshold)),
            "calibration_negative_count": len(negatives),
            "population_guarantee": False}


def paired_group_bootstrap(labels, prediction_a, prediction_b, groups, *,
                           metric: str = "ap", n_bootstrap: int = 2000,
                           seed: int = 17, confidence: float = 0.95,
                           return_draws: bool = False) -> dict:
    """Paired cluster bootstrap of A minus B. Entire groups are resampled."""
    labels, prediction_a, _ = _binary_inputs(labels, prediction_a)
    labels_b, prediction_b, _ = _binary_inputs(labels, prediction_b)
    if not np.array_equal(labels, labels_b):
        raise ValueError("Predictions must use identical labels")
    raw_groups = np.asarray(groups, dtype=object)
    if raw_groups.ndim != 1 or len(raw_groups) != len(labels):
        raise ValueError("One independent-group identifier is required per sample")
    if any(value is None or str(value) == "" for value in raw_groups):
        raise ValueError("Group identifiers cannot be missing")
    groups = raw_groups.astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap size or confidence")
    functions = {"ap": binary_average_precision, "auroc": binary_auroc, "brier": binary_brier}
    if metric not in functions:
        raise ValueError(f"Unknown metric: {metric}")
    function = functions[metric]
    observed_a, observed_b = function(labels, prediction_a), function(labels, prediction_b)
    if observed_a is None or observed_b is None:
        return {"metric": metric, "difference_a_minus_b": None, "ci": None,
                "independent_groups": len(unique), "reason": "Metric undefined on observed labels"}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        group_counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        sample_weights = group_counts[inverse]
        a = function(labels, prediction_a, sample_weights)
        b = function(labels, prediction_b, sample_weights)
        if a is not None and b is not None:
            draws.append(a - b)
    alpha = (1 - confidence) / 2
    interval = np.quantile(draws, [alpha, 1 - alpha]).tolist() if draws and len(unique) > 1 else None
    result = {
        "metric": metric, "difference_a_minus_b": float(observed_a - observed_b),
        "score_a": observed_a, "score_b": observed_b, "ci": interval,
        "confidence": confidence, "requested_draws": n_bootstrap,
        "valid_draws": len(draws), "undefined_draws": n_bootstrap - len(draws),
        "independent_groups": len(unique), "seed": seed,
        "resampling_unit": "complete supplied group; same draw for both methods",
        "limitation": "Percentile interval, not a p-value; supplied groups must capture dependence.",
    }
    if len(unique) < 20:
        result["small_group_warning"] = "Fewer than 20 independent groups; interval reliability is limited."
    if return_draws:
        result["draws"] = draws
    return result


def temporal_iou(a, b) -> float:
    """Two (onset, offset) pairs in the same physical units, normally seconds."""
    a0, a1 = map(float, a)
    b0, b1 = map(float, b)
    if not np.all(np.isfinite([a0, a1, b0, b1])) or a1 <= a0 or b1 <= b0:
        raise ValueError("Intervals must have finite, strictly increasing endpoints")
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - intersection
    return intersection / union


def temporal_average_precision(predictions: list[dict], ground_truth: list[dict],
                               iou_threshold: float = 0.5) -> dict:
    """Per-class event AP; greedy score-ordered one-to-one temporal matching.

    Required fields: recording_id, class_id, onset, offset; predictions add score.
    Deterministic tied-score order: recording_id, onset, offset, original index.
    AP integrates the monotone precision envelope at observed recall changes.
    Classes with no annotated events are excluded from mAP and reported separately.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    for event in predictions + ground_truth:
        if "recording_id" not in event or "class_id" not in event:
            raise ValueError("Events require recording_id and class_id")
        temporal_iou((event["onset"], event["offset"]), (event["onset"], event["offset"]))
    for pred in predictions:
        if not np.isfinite(pred["score"]):
            raise ValueError("Prediction scores must be finite")
    classes = sorted({str(e["class_id"]) for e in ground_truth} |
                     {str(e["class_id"]) for e in predictions})
    class_results = {}
    for class_id in classes:
        references = [g for g in ground_truth if str(g["class_id"]) == class_id]
        selected = [(index, p) for index, p in enumerate(predictions) if str(p["class_id"]) == class_id]
        selected.sort(key=lambda pair: (-float(pair[1]["score"]), str(pair[1]["recording_id"]),
                                       float(pair[1]["onset"]), float(pair[1]["offset"]), pair[0]))
        used = set()
        matches = []
        for index, pred in selected:
            candidates = []
            for ref_index, ref in enumerate(references):
                if ref_index in used or str(ref["recording_id"]) != str(pred["recording_id"]):
                    continue
                iou = temporal_iou((pred["onset"], pred["offset"]), (ref["onset"], ref["offset"]))
                if iou >= iou_threshold:
                    candidates.append((iou, -ref_index, ref_index))
            if candidates:
                _, _, matched = max(candidates)
                used.add(matched)
                matches.append(1)
            else:
                matches.append(0)
        if not references:
            ap = None
        elif not matches:
            ap = 0.0
        else:
            tp = np.cumsum(matches)
            fp = np.cumsum(1 - np.asarray(matches))
            recall = tp / len(references)
            precision = tp / (tp + fp)
            mrec = np.r_[0.0, recall, 1.0]
            mpre = np.r_[0.0, precision, 0.0]
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            changes = np.flatnonzero(mrec[1:] != mrec[:-1])
            ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
        class_results[class_id] = {
            "ap": ap, "ground_truth_events": len(references), "predictions": len(selected),
            "true_positive_events": len(used), "false_positive_events": len(selected) - len(used),
            "missed_events": len(references) - len(used),
        }
    eligible = [r["ap"] for r in class_results.values() if r["ap"] is not None]
    return {"iou_threshold": iou_threshold,
            "map": float(np.mean(eligible)) if eligible else None, "classes": class_results,
            "ap_convention": "all-points interpolated precision-envelope AP",
            "matching": "greedy score order, one prediction per ground-truth event"}


def fit_monotonic_calibrator(features, labels, *, steps: int = 600,
                             learning_rate: float = 0.03, l2: float = 1e-4) -> dict:
    """Fit positive logistic weights on calibration-only CUDA feature rows."""
    import torch
    if not features.is_cuda or not labels.is_cuda:
        raise ValueError("Learned calibration must run on CUDA; no CPU fallback")
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels) or not len(labels):
        raise ValueError("Expected features [N,D] and labels [N]")
    if not torch.isfinite(features).all().item() or not torch.isfinite(labels).all().item():
        raise ValueError("Calibration data must be finite")
    if not torch.all((labels == 0) | (labels == 1)).item() or torch.unique(labels).numel() != 2:
        raise ValueError("Calibration needs both binary classes")
    if steps < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid optimization settings")
    x = features.detach().float()
    y = labels.detach().float()
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    with torch.enable_grad():
        raw_weights = torch.nn.Parameter(torch.zeros(x.shape[1], device=x.device))
        bias = torch.nn.Parameter(torch.zeros((), device=x.device))
        optimizer = torch.optim.Adam([raw_weights, bias], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            weights = torch.nn.functional.softplus(raw_weights)
            logits = x @ weights + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + l2 * weights.square().mean()
            loss.backward()
            optimizer.step()
        weights = torch.nn.functional.softplus(raw_weights)
        final_loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weights + bias, y)
    return {"type": "positive_logistic_v1", "weights": weights.detach().cpu().tolist(),
            "intercept": float(bias.detach().item()), "mean": mean.cpu().tolist(),
            "scale": scale.cpu().tolist(), "steps": steps, "learning_rate": learning_rate,
            "l2": l2, "calibration_rows": len(y), "bce": float(final_loss.detach().item()),
            "fit_device": "cuda", "feature_count": x.shape[1]}


def apply_calibrator(features, state: dict):
    import torch
    if not features.is_cuda:
        raise ValueError("Calibrator inference must run on CUDA")
    if state.get("type") != "positive_logistic_v1" or features.ndim != 2:
        raise ValueError("Invalid calibration schema/input")
    if features.shape[1] != state["feature_count"]:
        raise ValueError("Feature count differs from calibration")
    def tensor(name):
        return torch.tensor(state[name], dtype=torch.float32, device=features.device)
    return torch.sigmoid(((features.float() - tensor("mean")) / tensor("scale")) @ tensor("weights")
                         + state["intercept"])
