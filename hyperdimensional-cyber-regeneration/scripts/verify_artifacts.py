#!/usr/bin/env python3
"""Verify compact publication outputs against their recorded provenance."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MAIN = ["fig02_deception_recovery", "fig03_repair_calibration",
        "fig04_gpu_communication", "fig05_ablation_selection"]
EXTRA = ["figS1_training", "figS2_seed_variation"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "results/study_manifest.json").read_text())
    assert manifest["settings"]["seeds"] == [11, 23, 37, 51, 71]
    assert not manifest["settings"]["pilot"]
    for relative, sha in manifest["source_hashes"].items():
        assert digest(ROOT / relative) == sha, f"Executed source changed: {relative}"
    with (ROOT / "results/summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5 * 11 * 14
    columns = ["seed", "method", "condition", "corruption", "attack", "facts", "budget", "repair_fraction"]
    counts = Counter(tuple(r[c] for c in columns) for r in rows)
    assert max(counts.values()) == 1, "Duplicate method-condition records"
    for row in rows:
        for field in ["recall3", "regret", "brier", "latency_ms", "bytes", "coverage"]:
            assert math.isfinite(float(row[field])), (field, row)
        assert 0 <= float(row["recall3"]) <= 1
        assert 0 <= int(row["eligible"]) <= manifest["settings"]["test"]
        assert float(row["bytes"]) <= 2 * int(row["budget"])
    for directory, stems in [(ROOT / "figures/paper", MAIN), (ROOT / "supplemental", EXTRA)]:
        for stem in stems:
            provenance = json.loads((directory / f"{stem}.provenance.json").read_text())
            assert digest(ROOT / provenance["script"]) == provenance["script_sha256"]
            for relative, sha in provenance["inputs"].items():
                assert digest(ROOT / "results" / relative) == sha
            for ext, sha in provenance["output_sha256"].items():
                assert digest(directory / f"{stem}.{ext}") == sha
            assert len(PdfReader(directory / f"{stem}.pdf").pages) == 1
    pdf = ROOT / "supplemental/Supplement.pdf"
    pages = PdfReader(pdf).pages
    assert len(pages) == 3
    assert all(len(p.extract_text()) > 400 for p in pages), "Missing supplement text"
    provenance = json.loads((ROOT / "supplemental/Supplement.provenance.json").read_text())
    assert digest(pdf) == provenance["pdf_sha256"]
    assert digest(ROOT / "scripts/make_supplement.py") == provenance["script_sha256"]
    for relative, sha in provenance["inputs"].items():
        assert digest(ROOT / relative) == sha
    thresholds = json.loads((ROOT / "results/selected_thresholds.json").read_text())
    assert len(thresholds["rows"]) == 25
    timing = json.loads((ROOT / "results/pipeline_timing.json").read_text())
    assert len(timing["rows"]) == 11
    assert all(r["repeats"] == 50 and r["median_ms"] > 0 for r in timing["rows"])
    print(json.dumps({"verified": True, "method_condition_seed_rows": len(rows),
                      "main_figures": len(MAIN), "extra_figures": len(EXTRA),
                      "supplement_pages": len(pages), "executed_source_hashes_unchanged": True}))


if __name__ == "__main__":
    main()
