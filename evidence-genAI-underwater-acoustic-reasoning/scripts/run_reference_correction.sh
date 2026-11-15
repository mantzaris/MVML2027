#!/usr/bin/env bash
set -euo pipefail
cd /workspace/evidence-genAI-underwater-acoustic-reasoning
run_stage() {
    local stage_name="$1"
    shift
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_guarded.py --name "$stage_name" "$@" > ".cache/${stage_name}.log" 2>&1
}
run_stage references_v12 .venv/bin/python scripts/prepare_evidence.py --stage references
run_stage reviewer_controls_v12 .venv/bin/python scripts/run_reviewer_controls.py
run_stage ai_reviews_v12 .venv/bin/python scripts/run_generation.py --mode review --partition all
run_stage reviewer_summary_v12 .venv/bin/python scripts/summarize_reviewers.py
run_stage pilot_gate_v12 .venv/bin/python scripts/audit_pilot.py
run_stage policy_terminal .venv/bin/python scripts/run_generation.py --partition policy_fit
run_stage calibration_terminal .venv/bin/python scripts/run_generation.py --partition calibration
run_stage planner_development .venv/bin/python scripts/run_planner.py --partition development
run_stage planner_calibration .venv/bin/python scripts/run_planner.py --partition calibration
run_stage controller .venv/bin/python scripts/train_controller.py
run_stage pretest_design .venv/bin/python scripts/audit_pretest_design.py
run_stage grammar_repeat .venv/bin/python scripts/smoke_omni.py --name grammar_repeat --repeat 3 --report-schema
