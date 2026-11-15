#!/usr/bin/env bash
set -euo pipefail
cd /workspace/evidence-genAI-underwater-acoustic-reasoning
run_stage() {
    local stage_name="$1"
    shift
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_guarded.py --name "$stage_name" "$@" > ".cache/${stage_name}.log" 2>&1
}
# Every invoked evaluator checks the explicit prospective unlock record.
run_stage planner_test .venv/bin/python scripts/run_planner.py --partition test
run_stage test_terminal .venv/bin/python scripts/run_generation.py --partition test
run_stage clean_analysis .venv/bin/python scripts/analyze_experiment.py
run_stage robustness_preparation .venv/bin/python scripts/prepare_robustness.py
run_stage robustness_generation .venv/bin/python scripts/run_robustness.py
run_stage report_audits .venv/bin/python scripts/run_report_audits.py
run_stage reviewer_summary .venv/bin/python scripts/summarize_reviewers.py
