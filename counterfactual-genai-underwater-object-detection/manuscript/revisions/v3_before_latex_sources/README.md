# Multi-Representation Underwater Acoustic Object Detection with Generative AI Reasoning Agents

The full research run and final document review are complete. The study compares Fourier, STFT/spectrogram and wavelet representations, individual views and their combinations, a 2D log-mel CNN, and an exploratory EMD/Hilbert–Huang approximation. The language-model controller receives numerical detector evidence rather than images. Read [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md) for the measured findings and deliverables. The proposed agent did not improve the locked real-data endpoint: ensemble AP was 0.5738 versus 0.5906 for the development-selected wavelet comparator. No improvement is claimed.

The eight-page [main draft](manuscript/main_paper.pdf) contains five figures and the implemented equations; the four-page [companion study](manuscript/supplement.pdf) reports additional controlled experiments. All 38 execution stages and independent numerical/visual checks completed. Scripts, environments, data, checkpoints, raw results, figures and logs remain inside this submission directory. The original [research plan](RESEARCH_PLAN.md) is prospective; [execution amendments](protocol/execution_amendment.md) and the measured results describe what actually ran.

Completion is recorded in `runs/research_complete.json`; the one-command verifier/recovery entry point is `bash scripts/resume_research.sh`. The reviewed drafts are not submitted and still need the user's author metadata and final submission-stage checks.

## Verified local GPU

On 23 November 2026 the local **NVIDIA RTX 5000 Ada Generation** was visible through an authorized Sonnet Thunderbolt connection. NVIDIA reported 32,760 MiB total memory. A brief PyTorch 2.9.0+cu128 check passed CUDA matrix multiplication, a known-frequency FFT and STFT with **only this GPU visible**. See the [verification record](manifests/gpu_verification.json).

The project selects UUID GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf. It excludes the internal 4 GB Quadro T2000. No driver installation, reboot or remote worker was needed. All current research computation uses the local Ada.

The initial smoke check used an existing CUDA-enabled interpreter without modifying its environment. All future project environments, caches, data, scripts, temporary files and outputs must be created inside this submission directory. The dedicated .venv is installed and pinned in requirements-research.lock. Source scripts/project_env.sh before commands to keep caches and temporary files local.

## Continue or recover the full study

```bash
cd /home/meow/Documents/repos/MVML2027/counterfactual-genai-underwater-object-detection
bash scripts/resume_research.sh
```

The supervisor verifies completed stages and resumes unfinished training/inference. Its current state is `runs/horizon_status.json`; durable events are in `runs/events.jsonl`, stage receipts in `runs/stages/`, and model checkpoints in `checkpoints/`. See `protocol/restart.md` for recovery guarantees and limits. Do not launch a second copy while the current supervisor is running. No automatic operating-system startup service has been installed.

The authoritative execution decisions are in `protocol/execution_amendment.md`. They record the PORTIA data substitution, exploratory HHT status, fixed conditional verification probe, and the fact that the five test days form one connected day–vessel component. The final comparison must remain descriptive unless independent evidence supports a stronger inference.

## GPU launcher

From this project directory, verify the identity:

    python3 scripts/run_on_ada.py --dry-run

After creating the project-local CUDA-enabled .venv, run the tiny CUDA check:

    PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_on_ada.py -- .venv/bin/python scripts/verify_cuda.py

For future research commands, use this project's CUDA-enabled environment and project-local program:

    python3 scripts/run_on_ada.py -- .venv/bin/python scripts/your_research_program.py

The launcher checks UUID, model name and at least 30,000 MiB total VRAM, then sets CUDA_VISIBLE_DEVICES before starting the child. It fails if the intended card is unavailable. Each future research program must also assert CUDA residency for models and tensors; the launcher cannot prevent arbitrary child code from choosing CPU operations. The included smoke check performs that residency check.

Codex's filesystem/device sandbox did not expose NVIDIA correctly during the initial query. The host-level check and CUDA execution succeeded outside that sandbox. A sandbox visibility error is not evidence that drivers need reinstalling.

Run the research utility checks:

    source scripts/project_env.sh
    .venv/bin/python -m unittest discover -s tests -v

## Project boundary

All files in this directory belong to the new submission. The previous evidence-genAI-underwater-acoustic-reasoning directory is preserved. Literature and existing manifests were inspected read-only for overlap and hardware history; no previous results were modified, no Git commit or push was made, and no submission was published.

Large future data, checkpoints and raw outputs must stay inside this submission directory under its ignored data/, checkpoints/ and results/raw/ directories. Do not stage or mirror project work into Codex work/ or outputs/ directories or another repository. The existing submission's remote worker and historical compute deadline are not this project's execution configuration.
