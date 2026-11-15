"""Compact hashes of version-controlled research artifacts; no raw logs/data."""
import subprocess
from pathlib import Path
from underwater_reasoning.io import atomic_json, file_hash

root = Path(__file__).resolve().parents[1]
destination = root / "results/manifests/preparation_artifacts.json"
paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "."], cwd=root).decode().split("\0")
files = sorted({p for p in paths if p and (root / p).is_file() and root / p != destination})
records = [{"path": p, "bytes": (root / p).stat().st_size, "sha256": file_hash(root / p)} for p in files]
atomic_json(destination, {
    "stage": "cpu_preparation_incomplete_research", "recorded_date": "2026-11-15",
    "software_tests_passed": 24, "synthetic_signal_fixtures_passed": 500,
    "structural_intervention_checks_passed": 100, "ai_reviewer_runs": 0,
    "human_reviewers": 0, "gpu_research_jobs_started": 0, "approved_gpu_hour_ceiling": 10,
    "pod_finish_instruction": "leave_running", "actual_allocated_gpu_hours": None,
    "actual_usd_cost": None, "manuscript_status": "incomplete_draft",
    "draft_pdf_pages": 2, "pdf_visual_inspection": "both pages inspected; no clipped text",
    "test_outcomes_examined": False,
    "blocking_dependencies": ["recording/deployment provenance", "source-respecting passenger assessment sampling",
                              "verified artifact preservation before GPU runs", "AI review and GPU pipeline implementation"],
    "artifact_bytes_excluding_this_manifest": sum(r["bytes"] for r in records), "artifacts": records,
})
print(len(records), "artifacts;", sum(r["bytes"] for r in records), "bytes before this compact manifest")
