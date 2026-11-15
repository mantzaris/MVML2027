"""Versioned analytical candidates; these are not independent reference labels."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy import signal, ndimage

VERSION = "0.1.0"
BANDS = ((20, 100), (100, 500), (500, 2000), (2000, 8000))


def validate_audio(audio, sample_rate):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < sample_rate:
        raise ValueError("Expected finite mono audio of at least one second")
    if sample_rate < 16000:
        raise ValueError("Sample rate cannot represent the declared analysis bands")
    return x


def record_hash(x, fs):
    return hashlib.sha256(str(fs).encode() + np.asarray(x, dtype="<f4").tobytes()).hexdigest()


def card(family, x, fs, values, parameters):
    return {"record_hash": record_hash(x, fs), "evidence_id": family + "0", "tool": family,
            "version": VERSION, "sample_rate": fs, "time_span_seconds": [0, len(x) / fs],
            "parameters": parameters, "measurements": values,
            "quality": {"candidate_measurements_only": True, "independently_verified": False}}


def spectrum(audio, fs):
    x = validate_audio(audio, fs)
    f, psd = signal.welch(x, fs, window="hann", nperseg=fs, noverlap=3 * fs // 4, detrend="constant")
    _, t, stft = signal.stft(x, fs, window="hann", nperseg=fs, noverlap=3 * fs // 4,
                            detrend="constant", boundary=None, padded=False)
    bands = [float(np.sum(psd[(f >= low) & (f < high)]) * (f[1] - f[0])) for low, high in BANDS]
    local = ndimage.median_filter(psd, size=31, mode="reflect")
    contrast = 10 * np.log10(np.maximum(psd, 1e-30) / np.maximum(local, 1e-30))
    peaks, _ = signal.find_peaks(contrast, height=12, distance=4)
    candidates = []
    frame_power = np.abs(stft) ** 2
    frame_floor = ndimage.median_filter(frame_power, size=(31, 1), mode="reflect")
    for i in sorted(peaks, key=lambda i: psd[i], reverse=True):
        if not 20 <= f[i] <= 2000 or psd[i] < 1e-20:
            continue
        active = frame_power[i] >= 10 ** 1.2 * np.maximum(frame_floor[i], 1e-30)
        # Contiguous runs, rather than their bounding box, determine persistence.
        changes = np.diff(np.r_[False, active, False].astype(int))
        spans = [[max(0., float(t[a] - .5)), min(len(x) / fs, float(t[b - 1] + .5))]
                 for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
        longest = max((end - start for start, end in spans), default=0.)
        if longest >= len(x) / fs / 2:
            candidates.append({"frequency_hz": float(f[i]), "frequency_interval_hz": [float(f[i] - 1), float(f[i] + 1)],
                               "support_spans_seconds": spans, "longest_persistence_seconds": longest,
                               "peak_to_local_floor_db": float(contrast[i])})
        if len(candidates) == 3:
            break
    total = sum(bands)
    values = {"band_bounds_hz": BANDS, "integrated_band_power": bands, "power_units": "sample_amplitude_squared",
              "dominant_band_hz": list(BANDS[int(np.argmax(bands))]) if total > 1e-20 else None,
              "narrowband_candidates": candidates, "no_detection_is_not_verified_absence": True}
    return card("A", x, fs, values, {"window_seconds": 1, "hop_seconds": .25, "peak_threshold_db": 12})


def modulation(audio, fs):
    x = validate_audio(audio, fs)
    candidates = []
    for low, high in ((100, 1000), (1000, 8000)):
        high = min(high, fs / 2 - 1)
        filtered = signal.sosfiltfilt(signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos"), x)
        envelope = np.abs(signal.hilbert(filtered))
        # Remove filter/Hilbert edge regions before measuring modulation.
        trim = fs // 4
        env = signal.detrend(envelope[trim:-trim])
        if np.mean(filtered ** 2) <= 1e-20:
            continue
        f, p = signal.welch(env, fs, nperseg=min(len(env), 2 * fs), noverlap=fs)
        mask = (f >= 2) & (f <= 80)
        selected = np.flatnonzero(mask)
        peak = selected[np.argmax(p[mask])]
        floor = max(float(np.median(p[mask])), 1e-30)
        strength = 10 * np.log10(max(p[peak], 1e-30) / floor)
        relative = float(np.std(env) / max(np.mean(envelope[trim:-trim]), 1e-30))
        if strength >= 12 and relative >= .03:
            candidates.append({"rate_hz": float(f[peak]), "rate_interval_hz": [float(f[peak] - .5), float(f[peak] + .5)],
                               "carrier_band_hz": [low, high], "peak_to_floor_db": float(strength),
                               "relative_envelope_std": relative, "time_span_seconds": [.25, len(x) / fs - .25]})
    return card("B", x, fs, {"modulation_candidates": candidates,
                             "no_detection_is_not_verified_absence": True},
                {"rate_range_hz": [2, 80], "peak_threshold_db": 12, "minimum_relative_std": .03})


def temporal(audio, fs):
    x = validate_audio(audio, fs)
    step = fs // 4
    rms = [float(np.sqrt(np.mean(x[i:i + step] ** 2))) for i in range(0, len(x), step)]
    first, last = float(np.mean(x[:fs] ** 2)), float(np.mean(x[-fs:] ** 2))
    change = float(10 * np.log10(last / first)) if min(first, last) > 1e-20 else None
    trend = "uncertain" if change is None else "increase" if change >= 3 else "decrease" if change <= -3 else "steady"
    return card("C", x, fs, {"rms": rms, "rms_time_step_seconds": .25, "energy_change_db": change,
                            "energy_trend_candidate": trend, "fraction_zero": float(np.mean(x == 0)),
                            "fraction_at_digital_full_scale": float(np.mean(np.abs(x) >= 1))},
                {"first_last_window_seconds": 1, "trend_threshold_db": 3})


def retrieve(query, bank, metadata, query_group, k=5):
    """Distinct-group, acoustic-fit-only cosine neighbors with safe output fields."""
    q, x = np.asarray(query, dtype=float), np.asarray(bank, dtype=float)
    if x.ndim != 2 or q.shape != (x.shape[1],) or len(metadata) != len(x):
        raise ValueError("Embedding shape mismatch")
    if not np.isfinite(q).all() or not np.isfinite(x).all() or np.linalg.norm(q) == 0:
        raise ValueError("Invalid embeddings")
    if any(row["partition"] != "acoustic_fit" for row in metadata):
        raise ValueError("Retrieval bank contains non-acoustic-fit examples")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-norm retrieval embedding")
    scores = x @ q / (norms * np.linalg.norm(q))
    result, groups = [], {query_group}
    for i in np.argsort(-scores, kind="stable"):
        row = metadata[i]
        if row["source_group"] in groups:
            continue
        groups.add(row["source_group"])
        result.append({"exemplar_id": row["anonymous_id"], "training_label": row["label"],
                       "cosine_distance": float(np.clip(1 - scores[i], 0, 2))})
        if len(result) == k:
            break
    return result
