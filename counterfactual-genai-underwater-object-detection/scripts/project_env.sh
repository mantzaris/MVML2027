#!/usr/bin/env bash
# Source this from the canonical project; all generated artifacts remain here.
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export TMPDIR="$PROJECT_ROOT/tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/.cache/python"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_CACHE_PATH="$PROJECT_ROOT/.cache/cuda"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export NUMBA_CACHE_DIR="$PROJECT_ROOT/.cache/numba"
