"""CUDA acoustic feature views with fixed, named dimensions.

All substantial tensor operations require CUDA. Inputs are unpadded, equal-length
[B,N] float waveforms in native amplitude units; split variable-duration records
before calling. HHT uses *piecewise-linear GPU EMD*, an approximation to classical
spline-envelope EMD, with explicitly bounded sifting and reported diagnostics.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

GROUP_DIMS = {"fourier": 32, "stft": 64, "wavelet": 56, "hht": 32}
ALL_GROUPS = tuple(GROUP_DIMS)
PRIMARY_GROUPS = ("fourier", "stft", "wavelet")
# Low-level extraction retains all views; primary models explicitly select PRIMARY_GROUPS.
DEFAULT_GROUPS = ALL_GROUPS


def require_cuda(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda":
        raise RuntimeError("Research DSP requires the UUID-pinned CUDA GPU; CPU fallback is prohibited")
    if x.ndim != 2 or x.shape[-1] < 2048 or not x.is_floating_point():
        raise ValueError("Expected unpadded floating [batch,samples], samples >= 2048")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Waveform contains nonfinite samples")
    return x.float()


def _rms(x, dim=-1):
    return x.square().mean(dim).clamp_min(1e-20).sqrt()


def _bands(sample_rate, device, count=24):
    high = min(7500.0, sample_rate * 0.47)
    if high <= 20:
        raise ValueError("Sample rate is too low for configured acoustic bands")
    return torch.logspace(math.log10(20), math.log10(high), count + 1, device=device)


def _band_pool(power, frequencies, edges):
    # [...,F] -> [...,bands]. Power is summed rather than averaging empty bins.
    membership = (frequencies[:, None] >= edges[:-1]) & (frequencies[:, None] < edges[1:])
    return power @ membership.to(power.dtype)


def _distribution_stats(power, frequencies, sample_rate):
    mass = power.sum(-1).clamp_min(1e-20)
    weights = power / mass[..., None]
    centroid = (weights * frequencies).sum(-1)
    bandwidth = (weights * (frequencies - centroid[..., None]).square()).sum(-1).clamp_min(0).sqrt()
    entropy = -(weights * weights.clamp_min(1e-20).log()).sum(-1) / math.log(max(power.shape[-1], 2))
    flatness = (power.clamp_min(1e-20).log().mean(-1) - power.mean(-1).clamp_min(1e-20).log()).clamp(-30, 0)
    return centroid / (sample_rate / 2), bandwidth / (sample_rate / 2), entropy, flatness, weights.amax(-1)


def fourier_features(x, sample_rate=16000):
    x = require_cuda(x)
    n = x.shape[-1]
    frequencies = torch.fft.rfftfreq(n, d=1 / sample_rate, device=x.device)
    power = (torch.fft.rfft(x) / n).abs().square()
    weights = torch.full_like(frequencies, 2)
    weights[0] = 1
    if n % 2 == 0:
        weights[-1] = 1
    power = power * weights
    bands = _band_pool(power, frequencies, _bands(sample_rate, x.device)).clamp_min(1e-20).log()
    center, width, entropy, flatness, concentration = _distribution_stats(power, frequencies, sample_rate)
    rms = _rms(x)
    extra = torch.stack((rms.log(), x.abs().amax(-1) / rms,
                         center, width, flatness, entropy, concentration,
                         (x[:, 1:] * x[:, :-1] < 0).float().mean(-1)), -1)
    return torch.cat((bands, extra), -1)


def stft_features(x, sample_rate=16000):
    x = require_cuda(x)
    features = []
    for n_fft in (512, 2048):
        window = torch.hann_window(n_fft, device=x.device)
        spec = torch.stft(x, n_fft=n_fft, hop_length=n_fft // 4,
                          window=window, center=False, return_complex=True)
        # [B,T,F], normalized by window energy and transform length.
        power = spec.abs().square().transpose(1, 2) / (window.square().sum() * n_fft)
        frequencies = torch.fft.rfftfreq(n_fft, d=1 / sample_rate, device=x.device)
        mean_power = power.mean(1)
        bands = _band_pool(mean_power, frequencies, _bands(sample_rate, x.device)).clamp_min(1e-20).log()
        center, _, _, _, _ = _distribution_stats(power, frequencies, sample_rate)
        frame_energy = power.sum(-1)
        normalized = power / power.sum(-1, keepdim=True).clamp_min(1e-20)
        if power.shape[1] > 1:
            flux = (normalized[:, 1:] - normalized[:, :-1]).square().mean(-1).sqrt()
        else:
            flux = torch.zeros_like(frame_energy)
        extra = torch.stack((frame_energy.mean(-1).clamp_min(1e-20).log(),
                             flux.mean(-1), flux.std(-1, correction=0),
                             center.mean(-1), center.std(-1, correction=0),
                             frame_energy.std(-1, correction=0) / frame_energy.mean(-1).clamp_min(1e-20),
                             (normalized < 1 / (10 * normalized.shape[-1])).float().mean((1, 2)),
                             frame_energy.amax(-1) / frame_energy.mean(-1).clamp_min(1e-20)), -1)
        features.append(torch.cat((bands, extra), -1))
    return torch.cat(features, -1)


def morlet_power(x, sample_rate=16000, scales=48, time_bins=16, scale_chunk=8):
    """Analytic Morlet mean and coarse time power, with edge/COI masks.

    Positive-frequency Morlet response, omega0=6, admissibility correction,
    discrete unit-energy filters. Negative frequencies are zero. Frequencies
    are logarithmic from 20 Hz to min(7500 Hz, .47*sample_rate).
    Returns [B,scales], [B,scales,time_bins], frequencies and valid fractions.
    Full [B,scales,N] coefficients are deliberately not retained.
    """
    x = require_cuda(x)
    n = x.shape[-1]
    centers = torch.logspace(math.log10(20), math.log10(min(7500.0, sample_rate * .47)), scales, device=x.device)
    frequencies = torch.fft.fftfreq(n, d=1 / sample_rate, device=x.device)
    spectrum = torch.fft.fft(x)
    sample_index = torch.arange(n, device=x.device)
    powers, temporal, coverage = [], [], []
    for start in range(0, scales, scale_chunk):
        f0 = centers[start:start + scale_chunk]
        scale_seconds = 6 / (2 * torch.pi * f0)
        omega = 2 * torch.pi * scale_seconds[:, None] * frequencies[None, :]
        response = (torch.exp(-0.5 * (omega - 6).square())
                    - math.exp(-18) * torch.exp(-0.5 * omega.square()))
        response = response * (frequencies[None, :] > 0)
        response = response / (response.square().sum(-1, keepdim=True) / n).clamp_min(1e-20).sqrt()
        coefficients = torch.fft.ifft(spectrum[:, None, :] * response[None, :, :], dim=-1)
        power = coefficients.abs().square()
        edge = torch.ceil(3 * scale_seconds * sample_rate).long()
        valid = (sample_index[None, :] >= edge[:, None]) & (sample_index[None, :] < n - edge[:, None])
        denom = valid.sum(-1).clamp_min(1)
        masked = power * valid[None]
        powers.append(masked.sum(-1) / denom[None])
        # Numerator/denominator pooling makes partially covered time bins honest.
        pooled_num = F.adaptive_avg_pool1d(masked, time_bins)
        pooled_den = F.adaptive_avg_pool1d(valid.float()[None], time_bins)
        temporal.append(pooled_num / pooled_den.clamp_min(1e-12))
        coverage.append(valid.float().mean(-1))
    return torch.cat(powers, 1), torch.cat(temporal, 1), centers, torch.cat(coverage)


def wavelet_features(x, sample_rate=16000):
    power, temporal, frequencies, coverage = morlet_power(x, sample_rate)
    center, width, entropy, flatness, concentration = _distribution_stats(power, frequencies, sample_rate)
    energy = temporal.sum(1)
    extra = torch.stack((power.sum(-1).clamp_min(1e-20).log(), center, width, entropy,
                         flatness, concentration,
                         energy.std(-1, correction=0) / energy.mean(-1).clamp_min(1e-20),
                         coverage.mean().expand(x.shape[0])), -1)
    return torch.cat((power.clamp_min(1e-20).log(), extra), -1)


def _extrema(x):
    middle = x[:, 1:-1]
    maxima = (middle > x[:, :-2]) & (middle >= x[:, 2:])
    minima = (middle < x[:, :-2]) & (middle <= x[:, 2:])
    return F.pad(maxima, (1, 1), value=False), F.pad(minima, (1, 1), value=False)


def _linear_envelope(x, extrema):
    """Interpolate bracketing extrema entirely on GPU; endpoints anchor x."""
    b, n = x.shape
    indices = torch.arange(n, device=x.device)[None].expand(b, -1)
    anchors = extrema.clone()
    anchors[:, 0] = True
    anchors[:, -1] = True
    left = torch.where(anchors, indices, -1).cummax(-1).values
    right = torch.where(anchors, indices, n).flip(-1).cummin(-1).values.flip(-1)
    y_left = x.gather(-1, left)
    y_right = x.gather(-1, right)
    fraction = (indices - left).float() / (right - left).clamp_min(1).float()
    return y_left + fraction * (y_right - y_left)


def emd_linear(x, max_imfs=6, max_sifts=10, sd_threshold=0.05, mean_threshold=0.05):
    """Bounded piecewise-linear EMD approximation, never a spline-EMD claim."""
    x = require_cuda(x)
    if not 1 <= max_imfs <= 6 or not 1 <= max_sifts <= 10:
        raise ValueError("The protocol permits 1..6 IMFs and 1..10 sifts")
    residual = x.clone()
    imfs, counts, valid_modes, failures = [], [], [], []
    original_energy = x.square().mean(-1).clamp_min(1e-20)
    for _ in range(max_imfs):
        maxima, minima = _extrema(residual)
        enabled = ((maxima.sum(-1) >= 2) & (minima.sum(-1) >= 2)
                   & (residual.square().mean(-1) > original_energy * 1e-8))
        h = residual.clone()
        active = enabled.clone()
        count = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        failed = torch.zeros_like(enabled)
        for _ in range(max_sifts):
            maxima, minima = _extrema(h)
            enough = (maxima.sum(-1) >= 2) & (minima.sum(-1) >= 2)
            failed = failed | (active & ~enough)
            active = active & enough
            mean_envelope = (_linear_envelope(h, maxima) + _linear_envelope(h, minima)) / 2
            candidate = h - mean_envelope
            denom = h.square().mean(-1).clamp_min(1e-20)
            sd = mean_envelope.square().mean(-1) / denom
            normalized_mean = mean_envelope.abs().mean(-1) / _rms(h)
            cand_max, cand_min = _extrema(candidate)
            extrema_count = cand_max.sum(-1) + cand_min.sum(-1)
            zero_count = (candidate[:, 1:] * candidate[:, :-1] < 0).sum(-1)
            converged = (sd <= sd_threshold) & (normalized_mean <= mean_threshold) & ((extrema_count - zero_count).abs() <= 1)
            h = torch.where(active[:, None], candidate, h)
            count = count + active.long()
            active = active & ~converged
        mode = torch.where(enabled[:, None], h, torch.zeros_like(h))
        residual = residual - mode
        imfs.append(mode)
        counts.append(count)
        valid_modes.append(enabled)
        failures.append(failed | (enabled & active))
    modes = torch.stack(imfs, 1)
    reconstruction_error = _rms(x - modes.sum(1) - residual) / _rms(x)
    diagnostics = {"sift_counts": torch.stack(counts, 1),
                   "valid_modes": torch.stack(valid_modes, 1),
                   "unconverged_or_extrema_failed": torch.stack(failures, 1),
                   "reconstruction_relative_rms": reconstruction_error}
    return modes, residual, diagnostics


def analytic_signal(x):
    if x.device.type != "cuda":
        raise RuntimeError("Hilbert analytic signal requires CUDA")
    n = x.shape[-1]
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    h[0] = 1
    if n % 2 == 0:
        h[1:n // 2] = 2
        h[n // 2] = 1
    else:
        h[1:(n + 1) // 2] = 2
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * h, dim=-1)


def hht_features(x, sample_rate=16000, return_diagnostics=False):
    x = require_cuda(x)
    imfs, residual, diagnostics = emd_linear(x)
    analytic = analytic_signal(imfs)
    amplitude = analytic.abs()
    # Consecutive complex phase difference avoids a CPU unwrap operation.
    phase_product = analytic[..., 1:] * analytic[..., :-1].conj()
    instantaneous_frequency = torch.atan2(phase_product.imag, phase_product.real) * sample_rate / (2 * torch.pi)
    threshold = 0.1 * amplitude.square().mean(-1, keepdim=True).sqrt()
    valid = (amplitude[..., 1:] > threshold) & (amplitude[..., :-1] > threshold)
    edge = max(32, x.shape[-1] // 100)
    indices = torch.arange(x.shape[-1] - 1, device=x.device)
    valid = valid & (indices >= edge) & (indices < x.shape[-1] - edge - 1)
    valid = valid & diagnostics["valid_modes"][..., None]
    before_band = valid
    in_band = (instantaneous_frequency >= 20) & (instantaneous_frequency <= min(7500, sample_rate * .47))
    valid = valid & in_band
    weight = amplitude[..., 1:].square() * valid
    total_weight = weight.sum(-1).clamp_min(1e-20)
    mean_if = (instantaneous_frequency * weight).sum(-1) / total_weight
    variance = ((instantaneous_frequency - mean_if[..., None]).square() * weight).sum(-1) / total_weight
    energy_fraction = imfs.square().mean(-1) / x.square().mean(-1, keepdim=True).clamp_min(1e-20)
    per_mode = torch.stack((energy_fraction, mean_if / (sample_rate / 2),
                            variance.clamp_min(0).sqrt() / (sample_rate / 2), valid.float().mean(-1)), -1).flatten(1)
    n_valid = before_band.float().sum((1, 2)).clamp_min(1)
    extra = torch.stack((_rms(x).log(), residual.square().mean(-1) / x.square().mean(-1).clamp_min(1e-20),
                         diagnostics["reconstruction_relative_rms"],
                         diagnostics["valid_modes"].float().mean(-1),
                         diagnostics["sift_counts"].float().mean(-1) / 10,
                         diagnostics["unconverged_or_extrema_failed"].float().mean(-1),
                         ((instantaneous_frequency < 0) & before_band).sum((1, 2)) / n_valid,
                         ((~in_band) & before_band).sum((1, 2)) / n_valid), -1)
    features = torch.cat((per_mode, extra), -1)
    diagnostics["mean_instantaneous_frequency_hz"] = mean_if
    diagnostics["valid_if_fraction"] = valid.float().mean(-1)
    return (features, diagnostics) if return_diagnostics else features


def feature_names(groups: Sequence[str] = DEFAULT_GROUPS):
    names = []
    for group in groups:
        if group == "fourier":
            names.extend([f"fourier.log_band_power_{i:02}" for i in range(24)])
            names.extend(f"fourier.{name}" for name in ("log_rms", "crest", "centroid", "bandwidth", "log_flatness", "entropy", "peak_mass", "zero_crossing"))
        elif group == "stft":
            for size in (512, 2048):
                names.extend([f"stft{size}.log_band_power_{i:02}" for i in range(24)])
                names.extend(f"stft{size}.{name}" for name in ("log_energy", "flux_mean", "flux_sd", "centroid_mean", "centroid_sd", "energy_cv", "sparsity", "peak_mean_energy"))
        elif group == "wavelet":
            names.extend([f"wavelet.log_scale_power_{i:02}" for i in range(48)])
            names.extend(f"wavelet.{name}" for name in ("log_energy", "centroid", "bandwidth", "entropy", "log_flatness", "peak_mass", "temporal_cv", "edge_valid_fraction"))
        elif group == "hht":
            names.extend(f"hht.imf{q}.{name}" for q in range(6) for name in ("energy_fraction", "mean_if", "sd_if", "valid_if_fraction"))
            names.extend(f"hht.{name}" for name in ("log_rms", "residual_energy_fraction", "reconstruction_error", "extracted_fraction", "normalized_sifts", "failed_fraction", "negative_if_fraction", "outband_if_fraction"))
        else:
            raise ValueError(f"Unknown representation {group}")
    return names


def extract_features(x, sample_rate=16000, groups: Sequence[str] = DEFAULT_GROUPS):
    """Return stable [B,D] features, names, slices and CUDA diagnostics.

    All groups: D=184; select groups to measure genuinely acquired-view cost.
    No fitted standardization occurs here: fit scalers on training groups only.
    """
    x = require_cuda(x)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Choose one or more distinct feature groups")
    outputs, slices, diagnostics = [], {}, {}
    offset = 0
    for group in groups:
        if group == "hht":
            value, diag = hht_features(x, sample_rate, return_diagnostics=True)
            diagnostics["hht"] = diag
        elif group == "fourier":
            value = fourier_features(x, sample_rate)
        elif group == "stft":
            value = stft_features(x, sample_rate)
        elif group == "wavelet":
            value = wavelet_features(x, sample_rate)
        else:
            raise ValueError(f"Unknown representation {group}")
        if value.shape != (x.shape[0], GROUP_DIMS[group]) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Invalid {group} feature output")
        outputs.append(value)
        slices[group] = (offset, offset + GROUP_DIMS[group])
        offset += GROUP_DIMS[group]
    return {"features": torch.cat(outputs, -1), "feature_names": feature_names(groups),
            "group_slices": slices, "diagnostics": diagnostics}


def feature_tensor(x, sample_rate=16000, groups: Sequence[str] = DEFAULT_GROUPS):
    return extract_features(x, sample_rate, groups)["features"]
