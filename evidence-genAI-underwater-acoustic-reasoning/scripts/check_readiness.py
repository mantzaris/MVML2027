"""Report research gates. This command never starts or stops compute."""
import json
from pathlib import Path
config = json.loads(Path("configs/execution.json").read_text())
unlock = json.loads(Path("protocol/test_unlock_log.json").read_text())
gates = {
    "gpu_hour_cap": isinstance(config["approved_allocated_gpu_hours"], (float, int)) and config["approved_allocated_gpu_hours"] > 0,
    "persistent_storage": config["persistent_storage_verified"],
    "source_integrity": config["primary_data_verified"],
    "ai_reference_assessments": config["ai_reviewer_assessments_complete"],
    "test_remains_locked": not unlock["test_outcomes_examined"],
}
for name, passed in gates.items(): print(("PASS " if passed else "PENDING ") + name)
if config["actual_pod_hourly_usd"] is None:
    print("COST UNKNOWN: actual Pod hourly rate and prior allocated idle time are not available")
raise SystemExit(0 if all(gates.values()) else 2)
