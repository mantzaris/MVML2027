# Final-batch checkpoint boundary correction

Recorded before real detector fitting on 23 November 2026.

A code review found a diagnostic-only resume defect in the original supervised
training helper: if a checkpoint records `batch_cursor == len(order)`, resuming
skips the training loop and enters validation, where the local `loss` variable
has not been assigned. The process would raise an error while constructing the
history row. It would not silently change the trained parameters, but would
prevent automatic continuation at that boundary.

The completed supplemental models retain the exact original `training.py` and
its source hashes. Their training finished without this interruption and their
healthy completed model index permits skipping all finished fitting. Completed
synthetic temporal and flow models use their separate training loops. No prior
source hash or checkpoint is rewritten to conceal the defect.

Future real feature/CNN/foundation and augmentation fitting uses
`src/uac/training_v2.py`. It preserves the optimization and selection rules while
saving the last training-loss diagnostic in checkpoint metadata. Validation reads
that saved value; an absent historical diagnostic is explicitly null. The real
training scripts fingerprint this versioned helper. The original helper remains
unchanged for prior provenance.

The CUDA regression fixture writes an actual optimizer checkpoint at the final
batch boundary, resumes validation without another optimizer update, verifies
model and AdamW state exactly, and compares the next optimizer step with the
continuous reference. It tests both present and absent saved loss diagnostics.
The result is stored in `manifests/training_v2_boundary_validation.json`.
