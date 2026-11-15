"""Independent review views: DPSS multitaper and rectified-envelope autocorrelation.

No imports from the agent's DSP tools. Outputs are visual aids, never human labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from .io import atomic_json, canonical_hash, safe_relative


def reference_figure(audio, fs, output: Path):
    x = np.asarray(audio, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Reference views require finite mono audio")
    n = len(x)
    tapers = signal.windows.dpss(n, NW=3, Kmax=5)
    spectra = np.abs(np.fft.rfft(tapers * (x - x.mean()), axis=1)) ** 2 / fs
    psd = spectra.mean(axis=0)
    psd[1:-1] *= 2
    f = np.fft.rfftfreq(n, 1 / fs)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].plot(np.arange(n) / fs, x, linewidth=.4)
    axes[0].set(xlabel="Time (s)", ylabel="Sample amplitude", title="Independent review aids — not reference labels")
    frequencies, times, power = signal.spectrogram(x, fs, window="hann", nperseg=fs,
                                                  noverlap=3 * fs // 4, mode="psd")
    # Peak pooling is display-only: preserve narrow tones when reducing image size.
    edges = np.geomspace(20, min(8000, fs / 2), 220)
    pooled = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        pooled.append(power[mask].max(axis=0) if mask.any() else np.zeros(len(times)))
    display = 10 * np.log10(np.maximum(pooled, 1e-30))
    image = axes[1].pcolormesh(times, np.sqrt(edges[:-1] * edges[1:]), display,
                               shading="nearest", cmap="magma", vmin=display.max() - 90, vmax=display.max())
    axes[1].set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(20, 8000), yscale="log",
                title="Log-frequency spectrogram; peak pooling for display only")
    fig.colorbar(image, ax=axes[1], label="Relative dB/Hz")
    axes[2].plot(f, 10 * np.log10(np.maximum(psd, 1e-30)), linewidth=.7)
    axes[2].set(xlabel="Frequency (Hz)", ylabel="PSD (relative dB/Hz)", xlim=(20, 8000), xscale="log")
    # Squared-amplitude, low-pass envelope; deliberately not Hilbert/spectral peaks.
    for low, high in ((100, 1000), (1000, 8000)):
        sos = signal.butter(3, [low, min(high, fs / 2 - 1)], fs=fs, btype="bandpass", output="sos")
        y = signal.sosfiltfilt(sos, x)
        power = signal.sosfiltfilt(signal.butter(3, 100, fs=fs, output="sos"), y * y)
        env = np.sqrt(np.maximum(power, 0))[fs // 4:-fs // 4:max(1, fs // 250)]
        env -= env.mean()
        ac = signal.correlate(env, env, mode="full", method="fft")[len(env) - 1:]
        ac /= max(ac[0], 1e-30)
        lag = np.arange(len(ac)) * max(1, fs // 250) / fs
        axes[3].plot(lag, ac, label=f"Carrier {low}–{high} Hz")
    axes[3].set(xlabel="Envelope lag (s), candidate period = 1/rate", ylabel="Autocorrelation", xlim=(0, .5))
    axes[3].legend()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def create_pack(rows, audio_root: Path, output: Path, private_key: Path, seed=17):
    if private_key.resolve().is_relative_to(output.resolve()):
        raise ValueError("Private identity key must remain outside the served review directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty pack directory to preserve earlier reviews")
    output.mkdir(parents=True, exist_ok=True)
    visible, keys = [], []
    for row in rows:
        anonymous = "clip_" + canonical_hash([seed, row["clip_id"], row["waveform_sha256"]])[:20]
        x, fs = sf.read(safe_relative(audio_root, row["audio_path"]), dtype="float32")
        # Re-encode without source names or embedded metadata; preserve samples.
        sf.write(output / (anonymous + ".wav"), x, fs, subtype="FLOAT")
        reference_figure(x, fs, output / (anonymous + ".png"))
        visible.append({"id": anonymous, "duration_seconds": len(x) / fs})
        keys.append({"anonymous_id": anonymous, "clip_id": row["clip_id"],
                     "waveform_sha256": row["waveform_sha256"]})
    visible.sort(key=lambda r: canonical_hash([seed, r["id"]]))
    payload = {"pack_id": canonical_hash(visible), "clips": visible,
               "status": "unannotated", "reference_view_version": "0.1.0"}
    atomic_json(private_key, {"pack_id": payload["pack_id"], "keys": keys})
    template = Path(__file__).with_name("review.html").read_text()
    (output / "index.html").write_text(template.replace("/*PACK_DATA*/", json.dumps(payload)))
    return payload
