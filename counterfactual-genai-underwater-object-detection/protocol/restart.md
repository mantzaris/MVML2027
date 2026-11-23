# Restarting research after an interruption

All runtime state belongs to this submission directory. The orchestrator uses one project-wide kernel lock, an fsynced event ledger, versioned atomic checkpoints, and stage-completion receipts with artifact checksums. No data or checkpoints should be staged outside this directory.

## Checkpoint contract

Use src/uac/runtime.py:

- atomic_json(path, value) writes a same-directory temporary file, flushes and fsyncs it, renames it atomically, then fsyncs the parent directory.
- EventLedger("runs/events.jsonl").append("event_name", stage=..., ...) appends and fsyncs a hash-chained JSON record. A partial final append is quarantined before continuation; corruption in a complete record stops the run for inspection.
- RunLock("runs/active.lock") prevents two orchestrators writing concurrently. The operating system releases this lock after process death or reboot; an old lock file alone does not indicate an active process.
- CheckpointStore(directory, config=..., data_fingerprint=..., code_fingerprint=...) saves versioned checkpoint generations and keeps the last two intact generations.
- save_training(model, optimizer, scheduler=..., scaler=..., epoch=..., batch_cursor=..., global_step=..., extra=...) includes model, optimizer, scheduler, mixed-precision scaler, Python/NumPy/PyTorch CPU and CUDA RNG state, cursor, configuration hash, data fingerprint and code fingerprint.
- load_training(...) returns epoch, batch_cursor, global_step, extra and a recovery report. It verifies checksums and compatibility before loading. A corrupt latest generation falls back to the preceding valid compatible generation. Missing or different configuration/data/code never silently resumes.
- completed_stage(...) records a stage only after every named output is durable. stage_is_complete(...) rechecks all outputs before skipping work.
- StopRequested() installs SIGINT/SIGTERM handlers that only set a flag. The training loop saves at the next safe optimizer boundary and exits; checkpoints are never written from inside a signal handler.

Checkpoint payloads are trusted local pickle-based files. Do not load checkpoints from an untrusted party with this utility.

## Training loop requirements

The checkpoint cursor means the **next batch to process**. Save after a complete optimizer step, at least once each epoch and at a wall-time interval of approximately two minutes during long epochs. Save both on a graceful stop request and at the end of a stage.

For exact continuation, derive each epoch's shuffle order from a dedicated deterministic seed and preserve the order or generator state in extra. Generate augmentation randomness from sample identity, epoch and seed, or save the relevant generator state. Do not consume the restored global RNG while reconstructing an already-partly-completed data-loader epoch. Save gradient-accumulation state or checkpoint only after optimizer steps when gradients have been cleared. Workers that prefetch randomly must not silently advance the checkpointed RNG; use deterministic sample-indexed generation or a separately recorded sampler.

Only mark an inference shard complete after its predictions and metadata are atomically written and checksummed. On restart, skip verified shards and recompute the unfinished shard. Never append duplicate prediction rows merely because a job was restarted.

The CheckpointStore recovers from durable generation metadata even if interruption occurred before latest.json was updated. Uncommitted .tmp files and payloads without committed metadata are ignored. Two generations survive ordinary saving; an interrupted cleanup can leave harmless older files. Do not delete previous generations manually during a run.

## Restart procedure

1. Reconnect/power the local RTX 5000 Ada eGPU using the established workflow.
2. From this project directory, inspect status with the project-local interpreter:

       .venv/bin/python scripts/status.py

3. If the only ledger damage is an interrupted final line, the orchestrator repairs it automatically on its next append. A read-side repair can be requested explicitly:

       .venv/bin/python scripts/status.py --repair-tail

4. Start the same research orchestrator configuration through scripts/run_on_ada.py. It must acquire RunLock, verify CUDA/device identity, verify config/data/code fingerprints, and call checkpoint/stage recovery before doing work.
5. Record the recovered generation, prior progress and any rejected corrupt generation in the event ledger. Resume the saved next-batch cursor and recompute only unfinished stages or shards.

The complete restart command is:

```bash
cd /home/meow/Documents/repos/MVML2027/counterfactual-genai-underwater-object-detection
bash scripts/resume_research.sh
```

This command resumes acquisition, verifies completed stages, restores unfinished training and inference, enforces the test freeze, executes the remaining evaluations and rebuilds result-driven manuscript drafts. It writes the active stage to `runs/horizon_status.json`, stage receipts to `runs/stages/`, and detailed stage logs to `logs/horizon_*.log`. A second orchestrator cannot start while the kernel lock is held. Each GPU subprocess independently verifies the RTX 5000 Ada UUID; it will fail rather than use the smaller GPU.

Do not use `--skip-download-if-active` after a power outage; that switch is only for the current supervised handoff while an existing downloader remains alive. The normal command above actively resumes missing downloads. These utilities do not automatically start a process after a machine reboot. A returning user or assistant must launch the documented resume command. No service, scheduled automation, or system-startup entry is created by these utilities.

## Limits

Power loss between checkpoints can lose work since the last durable checkpoint; it cannot be eliminated by logging alone. Atomic writes and fsync protect against ordinary interrupted writes on a functioning local filesystem, not disk destruction, filesystem failure or defective hardware. This protocol also cannot guarantee bitwise reproducibility for nondeterministic CUDA kernels; record deterministic settings and tolerances in the experiment manifest.

Tests cover interrupted index writes, a torn latest checkpoint, a partial event append, mismatched configuration, changed stage artifacts, lock contention, RNG restoration and signal handling. They exercise file recovery with a small serializer adapter; the first CUDA pilot must additionally test an actual model/optimizer save-and-resume round trip.

## CUDA resume validation performed

On 23 November 2026, scripts/validate_cuda_resume.py ran a small deterministic
CUDA model in separate processes on the pinned RTX 5000 Ada. Restarted training
reproduced the next optimizer step exactly, including model tensors, AdamW state,
scheduler state, Python/NumPy/PyTorch CPU/CUDA RNG effects and the next data cursor.
A deliberately corrupted latest checkpoint in a copied fixture recovered the
previous generation and reproduced its next step exactly; the original clean
checkpoint stayed intact. The result and checksummed artifact receipt are in
manifests/cuda_resume_validation.json and
manifests/cuda_resume_validation_receipt.json.

This validates the runtime mechanism for the small fixture. Each final research
training loop must still use the cursor/RNG contract and save its own checkpoints.


## Supervisor signal propagation

Each newly launched stage owns a separate process group. A supervisor stop request is sent to that entire owned group, including the GPU-launcher descendant, so the research program can save at its next safe boundary. The supervisor waits up to60seconds for that group to exit, then force-stops only that owned group if it has not exited. Native checkpoint/shard recovery still handles abrupt power loss or forced termination. This avoids treating termination of a launcher wrapper alone as proof that its worker has stopped.
