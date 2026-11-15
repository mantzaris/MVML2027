"""Known-signal verification only. No real-data or human-grounding claims."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import signal
from underwater_reasoning.dsp import spectrum, modulation, temporal
from underwater_reasoning.evaluation import structural_citation_check
from underwater_reasoning.io import atomic_json

fs = 32000
t = np.arange(5 * fs) / fs
rows = []
for i in range(500):
    rng = np.random.default_rng(np.random.SeedSequence([20261115, i]))
    kind = ("tone", "am", "chirp", "noise", "silence")[i % 5]
    f = float(rng.uniform(40, 1800))
    x = .2 * np.sin(2 * np.pi * f * t)
    record = {"fixture": i, "kind": kind, "seed": [20261115, i]}
    if kind == "tone":
        x += rng.normal(0, .01, len(t))
        card = spectrum(x, fs)
        candidates = card["measurements"]["narrowband_candidates"]
        error = min((abs(c["frequency_hz"] - f) for c in candidates), default=None)
        record.update(frequency_error_hz=error, passed=error is not None and error <= max(3, .03 * f))
    elif kind == "am":
        rate = float(rng.uniform(3, 70))
        x = .2 * (1 + .6 * np.sin(2 * np.pi * rate * t)) * np.sin(2 * np.pi * 500 * t)
        candidates = modulation(x, fs)["measurements"]["modulation_candidates"]
        error = min((abs(c["rate_hz"] - rate) for c in candidates if c["carrier_band_hz"] == [100, 1000]), default=None)
        record.update(rate_error_hz=error, passed=error is not None and error <= max(1, .1 * rate))
    elif kind == "chirp":
        x = .2 * signal.chirp(t, f0=50, f1=1800, t1=5)
        count = len(spectrum(x, fs)["measurements"]["narrowband_candidates"])
        record.update(persistent_tone_candidates=count, passed=count == 0)
    elif kind == "noise":
        x = rng.normal(0, .1, len(t))
        count = len(spectrum(x, fs)["measurements"]["narrowband_candidates"])
        record.update(persistent_tone_candidates=count, passed=count == 0)
    else:
        x = np.zeros(len(t))
        a, b, c = spectrum(x, fs), modulation(x, fs), temporal(x, fs)
        record["passed"] = (not a["measurements"]["narrowband_candidates"]
                            and not b["measurements"]["modulation_candidates"]
                            and c["measurements"]["energy_change_db"] is None)
    rows.append(record)
    if (i + 1) % 100 == 0:
        print("Validated", i + 1, "known-signal fixtures", flush=True)

# Structural evidence controls, explicitly separate from model behavior.
interventions = []
for i in range(100):
    item = {"record_hash": "known-query", "tool": "A", "measurements": {"frequency_hz": 100}}
    kind = ("removed", "wrong_recording", "incompatible", "valid_id_false_value")[i % 4]
    cards = {"A0": item}
    if kind == "removed": cards = {}
    if kind == "wrong_recording": item["record_hash"] = "different-query"
    if kind == "incompatible": item["tool"] = "D"
    if kind == "valid_id_false_value": item["measurements"]["frequency_hz"] = 999
    issues = structural_citation_check("narrowband", ["A0"], cards, "known-query", numeric=True)
    expected = [] if kind == "valid_id_false_value" else [dict(removed="unknown_evidence", wrong_recording="wrong_recording", incompatible="incompatible_tool")[kind]]
    interventions.append({"case": i, "kind": kind, "issues": issues, "passed": issues == expected})
report = {"status": "synthetic_tool_validation_only", "real_experiments": 0, "human_annotations": 0,
          "fixtures": rows, "structural_interventions": interventions,
          "summary": {kind: {"n": sum(r["kind"] == kind for r in rows),
                              "passed": sum(r["kind"] == kind and r["passed"] for r in rows)}
                      for kind in ("tone", "am", "chirp", "noise", "silence")},
          "limitations": ["No generator/controller behavior was tested.",
                          "A structurally valid but false numeric card deliberately passes; independent references are necessary.",
                          "Synthetic fixtures are not ship data or validated ocean propagation."]}
atomic_json(Path(".cache/synthetic_validation_full.json"), report)
from underwater_reasoning.io import file_hash
compact = {k: v for k, v in report.items() if k not in {"fixtures", "structural_interventions"}}
compact["structural_interventions"] = {"cases": len(interventions), "passed": sum(r["passed"] for r in interventions)}
compact["detailed_results_sha256"] = file_hash(Path(".cache/synthetic_validation_full.json"))
atomic_json(Path("results/manifests/synthetic_validation.json"), compact)
print(json.dumps(report["summary"], indent=2))
raise SystemExit(0 if all(r["passed"] for r in rows + interventions) else 1)
