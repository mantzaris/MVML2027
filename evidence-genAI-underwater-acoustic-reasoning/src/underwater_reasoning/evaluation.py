"""Independent-score aggregation; compatibility checks never establish truth."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SLOTS = ("dominant_band", "narrowband", "modulation", "energy_trend")
FAMILIES = {"dominant_band": {"A"}, "narrowband": {"A"}, "modulation": {"B"}, "energy_trend": {"C"}}


def structural_citation_check(slot, evidence_ids, evidence, waveform_hash, numeric=False):
    """Return structural problems only; a clean result is NOT verified support."""
    issues = []
    if slot not in FAMILIES:
        raise ValueError("Unknown physical slot")
    if not evidence_ids:
        issues.append("missing_citation")
    for identifier in evidence_ids:
        item = evidence.get(identifier)
        if item is None:
            issues.append("unknown_evidence")
        elif item["record_hash"] != waveform_hash:
            issues.append("wrong_recording")
        elif item["tool"] == "W" and not numeric:
            issues.append("raw_audio_requires_localized_independent_review")
        elif item["tool"] not in FAMILIES[slot]:
            issues.append("incompatible_tool")
    return issues


@dataclass(frozen=True)
class ScoredClaim:
    """Filled by an independent adjudicated-reference evaluator, not by tools."""
    status: str  # asserted, uncertain, omitted
    identifiable: bool
    correct: bool
    supported: bool
    cited: bool = False

    def __post_init__(self):
        if self.status not in {"asserted", "uncertain", "omitted"}:
            raise ValueError("Unknown claim status")
        if self.status != "asserted" and (self.correct or self.supported or self.cited):
            raise ValueError("Uncertainty/omission cannot count as a supported positive assertion")
        if not self.identifiable and self.correct:
            raise ValueError("Ambiguous reference cannot establish a correct definite assertion")

    def loss(self, rho=.25):
        if self.status == "omitted":
            return rho
        if self.status == "uncertain":
            return rho if self.identifiable else 0.
        return 0. if self.identifiable and self.correct and self.supported else 1.


def terminal_loss(class_correct, claims, rho=.25):
    if len(claims) != 4:
        raise ValueError("Every report must retain all four requested physical slots")
    class_loss = rho if class_correct is None else 0. if class_correct else 1.
    return .5 * class_loss + .5 * sum(c.loss(rho) for c in claims) / 4


def physical_metrics(reports, whole_test_n=None):
    n = len(reports)
    if any(len(row) != 4 for row in reports):
        raise ValueError("Omitted slots must remain in denominators")
    whole = n if whole_test_n is None else whole_test_n
    if whole < n:
        raise ValueError("Whole-test denominator cannot be smaller than retained reports")
    flat = [claim for row in reports for claim in row]
    asserted = [c for c in flat if c.status == "asserted"]
    good = sum(c.correct and c.supported for c in asserted)
    unsupported = sum(not c.supported for c in asserted)
    incorrect = sum(not c.correct for c in asserted)
    cited = [c for c in asserted if c.cited]
    supported_cited = sum(c.supported for c in cited)
    correct = sum(c.correct for c in asserted)
    correct_supported_cited = sum(c.correct and c.supported for c in cited)
    ambiguous = [c for c in flat if not c.identifiable]
    ratio = lambda a, b: a / b if b else None
    return {"reports": n, "requested_slots": 4 * n, "assertions": len(asserted),
            "supported_correct_assertions": good, "unsupported_assertions": unsupported,
            "supported_coverage": ratio(good, 4 * n), "whole_test_useful_coverage": ratio(good, 4 * whole),
            "assertion_coverage": ratio(len(asserted), 4 * n),
            "unsupported_claim_rate": ratio(unsupported, len(asserted)),
            "fact_error_rate": ratio(incorrect, len(asserted)),
            "citation_support_precision": ratio(supported_cited, len(cited)),
            "citation_completeness": ratio(correct_supported_cited, correct),
            "reference_identifiability": ratio(sum(c.identifiable for c in flat), len(flat)),
            "appropriate_uncertainty": ratio(sum(c.status == "uncertain" for c in ambiguous), len(ambiguous))}


def paired_cluster_bootstrap(left, right, groups, statistic, resamples=2000, seed=17):
    """Both methods use the same sampled clusters, including repeated selections."""
    if not (len(left) == len(right) == len(groups)) or not len(groups):
        raise ValueError("Paired arrays must have equal nonzero length")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("At least two independent source groups are required")
    indices = {g: [i for i, value in enumerate(groups) if value == g] for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = [i for g in sampled for i in indices[g]]
        values.append(statistic([left[i] for i in idx]) - statistic([right[i] for i in idx]))
    return {"difference": float(statistic(left) - statistic(right)),
            "ci95": np.quantile(values, [.025, .975]).tolist(), "groups": len(unique),
            "resamples": resamples, "seed": seed}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()
