#!/usr/bin/env bash
set -euo pipefail
cd /workspace/evidence-genAI-underwater-acoustic-reasoning
mkdir -p .cache logs data/private checkpoints results/raw
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir 'transformers==4.57.1' 'accelerate==1.11.0' 'numpy==1.26.4' 'scipy==1.15.3' 'soundfile==0.13.1' 'librosa==0.11.0' 'scikit-learn==1.7.2' 'requests==2.32.4' 'huggingface-hub==0.36.0' 'jsonschema==4.23.0'
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip install --no-cache-dir 'xgrammar==0.2.9' 'apache-tvm-ffi==0.1.16' 'pytest==8.3.5'
.venv/bin/python -c 'import torch,torchvision,torchaudio,transformers,scipy; print(torch.__version__,torchvision.__version__,torchaudio.__version__,transformers.__version__,scipy.__version__)'
.venv/bin/python -m pip freeze > .cache/remote_environment.lock
