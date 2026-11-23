#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/project_env.sh
exec .venv/bin/python scripts/run_research.py "$@"
