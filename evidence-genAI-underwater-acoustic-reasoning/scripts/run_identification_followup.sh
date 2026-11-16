#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
.venv/bin/python scripts/run_guarded.py --name identification_features .venv/bin/python scripts/extract_identification_features.py test
.venv/bin/python scripts/run_guarded.py --name identification_inference .venv/bin/python scripts/infer_identification.py
.venv/bin/python scripts/run_guarded.py --name identification_predictions .venv/bin/python scripts/analyze_identification.py --predict-only
.venv/bin/python scripts/run_guarded.py --name identification_omni .venv/bin/python scripts/run_identification_omni.py
.venv/bin/python scripts/run_guarded.py --name identification_analysis .venv/bin/python scripts/analyze_identification.py
