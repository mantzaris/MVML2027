# NVIDIA Ada GPU setup

Verified on November 25, 2026 using the GPU attached to this laptop.

| Component | Verified value |
| --- | --- |
| GPU | NVIDIA RTX 5000 Ada Generation |
| Physical index at verification | 1 |
| CUDA device inside the project process | `cuda:0` |
| Memory | 32 GB nominal; CUDA reports 31.48 GiB |
| Compute capability | 8.9 |
| NVIDIA driver | 590.48.01 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu128 |
| PyTorch CUDA runtime | 12.8 |
| BF16 | Supported |

The GPU is already attached and its host driver works. The restricted shell hides NVIDIA device access, which caused the earlier driver error. GPU commands must execute in a host shell with access to `/dev/nvidia*`. No driver reinstall or system configuration change was needed.

## Launch on the Ada GPU

From the project directory:

```bash
python3 scripts/run_gpu.py
```

The launcher finds the exact GPU model, selects its UUID, sets `CUDA_VISIBLE_DEVICES` before importing PyTorch, and uses the project's isolated `.venv`. The selected physical GPU becomes `cuda:0` inside that process. This avoids relying on physical index 1 remaining stable. It refuses a missing or ambiguous GPU and does not fall back to the Quadro T2000 or CPU.

Run a project Python script with arguments:

```bash
python3 scripts/run_gpu.py -- scripts/check_gpu.py --output work/gpu_readiness.json
```

For multiple matching Ada GPUs, use the explicit `--gpu-uuid` option. The requested UUID must also match the expected model. The launcher confines temporary files and framework caches to ignored directories under `work/`.

NVIDIA documents UUID-based selection and visibility ordering in its [CUDA environment-variable reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

## Recreate the isolated environment

The existing `.venv` is ready. To recreate it in a fresh checkout using Python 3.12:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-cuda.lock.txt
python3 scripts/run_gpu.py
```

`requirements-cuda.in` records direct requirements. `requirements-cuda.lock.txt` records the installed dependency versions; it is a small version snapshot, not a container image or wheel archive. The selected PyTorch/CUDA combination is available in the [official installation matrix](https://pytorch.org/get-started/previous-versions/).

The setup installs runtime libraries inside this project. Other research projects and their environments remain independent.

## Checks performed

The hardware smoke check verified FP32 CUDA matrix multiplication, bipolar HDC binding and unbinding, a BF16 forward pass with FP32 loss, finite CUDA gradients, and a real AdamW parameter update. The device-selection tests also verified that reordering devices cannot select the Quadro, absent Ada hardware causes failure, an incorrect UUID is rejected, and duplicate matching GPUs require explicit selection.

`pip check` found no broken dependency requirements. The hardware check is not a research benchmark or evidence for the proposed method's accuracy or speed.

Run the device-selection tests without touching the GPU:

```bash
python3 -B -m unittest discover -s tests -v
```

## Keep the repository small

The virtual environment, installation log, raw GPU report, caches, generated datasets, raw traces, checkpoints, and per-step logs remain local and ignored. The setup log is `work/gpu_setup.log` and the small raw verification report is `work/gpu_readiness.json`.

Commit source, dependency pins, concise result summaries, manifests, and selected publication figures. Check the actual staged paths and blob sizes before committing. Do not use force-add to bypass the artifact ignore rules.

The GPU smoke check is complete. The research simulator, training pipeline, baselines, and empirical paper figures remain future implementation work.
