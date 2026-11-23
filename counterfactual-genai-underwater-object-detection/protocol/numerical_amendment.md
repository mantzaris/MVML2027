# Numerical pilot and HHT scope amendment

Recorded 23 November 2026, before any detector outcomes from this new project.
The machine-readable measurements are in `manifests/dsp_validation.json`; the
implementation is `src/uac/dsp.py` and `src/uac/synthetic.py`.

## Decision

The primary representation set is **Fourier, multi-resolution STFT and analytic
Morlet wavelets** (32 + 64 + 56 = **152 features**). HHT contributes another 32
features for a separately labeled exploratory representation study. Extraction
can retain all **184 features** so a later supplement can inspect the HHT branch,
but the primary experiment must explicitly select the first three groups.

The implemented HHT branch uses at most six modes and ten sifts per mode, with
piecewise-linear GPU envelopes and amplitude/edge masks. It is a **bounded EMD
approximation**, not a validated equivalent of classical cubic-spline EMD. It
passed narrow numerical fixtures but failed the intended general convergence
gate on noisy controlled scenes: 95.8–100% of attempted modes either reached the
sift bound without satisfying the strict stop criteria or encountered an extrema
failure. The combined flag is reported rather than relabeled as successful IMF
convergence. Such modes must not be presented as reliable physical components.

This applies the fallback already specified in the prospective research plan:
retain validated Fourier/STFT/wavelet computation as primary and report limited
HHT behavior openly. No detector outcome was used to select this scope. A later
HHT development experiment may examine more sifts or improved envelopes on
training/development fixtures, but needs its own recorded settings and validity
assessment before any confirmatory inclusion. It must not retroactively alter
this pilot's ten-sift measurements.

## Numerical observations

All computations below used the exact pinned local RTX 5000 Ada UUID. The
small 4,096-sample classical-spline reference alone used CPU SciPy as an
independent correctness fixture; it is not the research preprocessing path.

- For a 440 Hz sine, the first approximate IMF's amplitude-weighted frequency
  was **440.000214 Hz**, a relative error of **4.86 × 10⁻⁷**. This fixture passed
  the prespecified 5% frequency-error threshold.
- The largest mean-power Morlet scale was **412.5047 Hz**, a **6.249%** discrepancy
  on the discrete 48-scale grid, within the 15% fixture threshold.
- Relative waveform reconstruction RMS errors were approximately **4.15 × 10⁻⁸**
  for the tone and at most **6.65 × 10⁻⁸** in the benchmark batches, below 10⁻⁵.
  Reconstruction alone is not evidence that the extracted modes are useful IMFs.
- On the independent 128 Hz + 0.3 × 1,024 Hz fixture, the interior first-mode
  relative RMS difference from a natural-cubic-spline EMD reference was **4.03%**.
  This is a diagnostic from one fixture, not an equivalence result.
- Controlled zero/one/two-source scenes reconstructed exactly as background plus
  source stems; no source samples lay outside the received-support masks. Measured
  per-source received-support SNR matched the assigned SNR to the displayed
  floating-point precision (maximum recorded discrepancy **0.0 dB**).
- Every extracted feature was finite and CUDA-resident. Nine API/numerical tests
  passed, including seeded generation repeatability and explicit CPU rejection.

## Bounded timing and memory pilot

Times are one synchronized call per configuration, measured with tensors already
on the GPU. They include differing first-use overhead and are **not a warmed,
transfer-inclusive latency benchmark**. Do not use these rows as the paper's
accuracy–compute evidence. Peak memory includes tensors already live during the
validation process; it is not incremental per-tool memory.

| Samples per clip | Batch | View | Feature width | Milliseconds per clip | Peak allocated MiB |
|---:|---:|---|---:|---:|---:|
| 48,000 (3 s) | 4 | Fourier | 32 | 20.534 | 40.68 |
| 48,000 (3 s) | 4 | STFT | 64 | 24.302 | 51.70 |
| 48,000 (3 s) | 4 | Wavelet | 56 | 2.324 | 103.38 |
| 48,000 (3 s) | 4 | Bounded HHT approximation | 32 | 29.853 | 86.83 |
| 160,000 (10 s) | 4 | Fourier | 32 | 0.500 | 55.59 |
| 160,000 (10 s) | 4 | STFT | 64 | 1.079 | 93.91 |
| 160,000 (10 s) | 4 | Wavelet | 56 | 2.735 | 263.26 |
| 160,000 (10 s) | 4 | Bounded HHT approximation | 32 | 29.719 | 209.88 |

The noisy-scene mean sift count was 9.83 at 48,000 samples and 10.0 at 160,000.
The bounded branch is computationally feasible on this GPU, but computational
feasibility does not repair its mode-convergence limitation.

## Remaining validity work

Validate real-recording preprocessing and missing/short-input handling. Confirm
feature distributions using only training/development groups. Test wavelet and
HHT edge behavior, chirps, overlapping components, mode mixing, amplitude masks
and the alternative synthetic family. Record exact learned normalization and
any feature exclusions before locked test inference. These limited fixtures do
not validate object-detection performance, field acoustics, or source-selective
counterfactual removal.
