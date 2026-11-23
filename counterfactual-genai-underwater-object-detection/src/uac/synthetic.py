"""CUDA-only controlled underwater-acoustic-like signal mixtures.

These bounded simulations do not claim validated vessel or ocean acoustics.
The heldout family changes background color, nonstationarity, modulation and
multipath delay ranges. Numeric outputs remain on CUDA; integer/string metadata
can be recorded directly in manifests. Seeded batches are reproducible.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _device(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Synthetic research generation requires the pinned CUDA GPU")
    return device


def _rand(shape, device, generator, low=0., high=1.):
    return low + (high - low) * torch.rand(shape, device=device, generator=generator)


def _delay(x, samples):
    indices = torch.arange(x.shape[-1], device=x.device)
    shifted = indices - samples[..., None]
    return x.gather(-1, shifted.clamp(0, x.shape[-1] - 1)) * (shifted >= 0)


def _colored_noise(shape, sample_rate, exponent, device, generator):
    white = torch.randn(shape, device=device, generator=generator)
    frequency = torch.fft.rfftfreq(shape[-1], d=1 / sample_rate, device=device)
    response = frequency.clamp_min(20).pow(-exponent[..., None] / 2)
    response[..., 0] = 0
    noise = torch.fft.irfft(torch.fft.rfft(white) * response, n=shape[-1])
    return noise / noise.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()


def generate_batch(batch_size, num_samples=160000, sample_rate=16000, seed=17,
                   family="train", device="cuda", source_counts=None):
    """Generate zero/one/two-source scenes with exact received event supports.

    audio/background:[B,N], stems/event_masks:[B,2,N]; intervals:[B,2,2]
    in seconds, -1 for inactive. `snr_db` is per-source received RMS SNR within
    that source's received support, *not* the combined overlapping-source SNR.
    Sources are harmonic/AM/chirped signals, not real vessel class exemplars.
    No per-clip normalization or clipping is applied to the resulting mixture.
    """
    device = _device(device)
    if family not in ("train", "heldout"):
        raise ValueError("family must be train or heldout")
    if batch_size < 1 or num_samples < 2048 or sample_rate < 8000:
        raise ValueError("Require positive batch, >=2048 samples and sample_rate>=8000")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    duration = num_samples / sample_rate
    time = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate
    exponent = _rand((batch_size,), device, generator, .5, 1.5) if family == "train" else _rand((batch_size,), device, generator, 1.7, 2.2)
    background = _colored_noise((batch_size, num_samples), sample_rate, exponent, device, generator)
    background_scale = torch.exp(_rand((batch_size, 1), device, generator, math.log(.015), math.log(.08)))
    if family == "heldout":
        envelope_rate = _rand((batch_size, 1), device, generator, .15, .8)
        envelope_phase = _rand((batch_size, 1), device, generator, 0, 2 * math.pi)
        background = background * (1 + .4 * torch.sin(2 * torch.pi * envelope_rate * time + envelope_phase))
        # A second stochastic family: sparse exponentially decaying interference.
        impulse_at = _rand((batch_size, 1), device, generator, .1 * duration, .8 * duration)
        lag = time[None] - impulse_at
        impulse_noise = torch.randn((batch_size, num_samples), device=device, generator=generator)
        background = background + .5 * torch.exp(-lag.clamp_min(0) / .04) * (lag >= 0) * impulse_noise
        background = background / background.square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
    background = background * background_scale
    if source_counts is None:
        counts = torch.randint(0, 3, (batch_size,), device=device, generator=generator)
    elif isinstance(source_counts, int):
        counts = torch.full((batch_size,), source_counts, device=device, dtype=torch.long)
    else:
        counts = torch.as_tensor(source_counts, device=device, dtype=torch.long)
    if counts.shape != (batch_size,) or bool(((counts < 0) | (counts > 2)).any()):
        raise ValueError("source_counts must contain one value 0..2 per scene")
    active = torch.arange(2, device=device)[None] < counts[:, None]
    maximum_duration = min(8., duration * .8)
    minimum_duration = min(.5, maximum_duration * .5)
    event_duration = _rand((batch_size, 2), device, generator, minimum_duration, maximum_duration)
    onset = _rand((batch_size, 2), device, generator) * (duration - event_duration)
    offset = onset + event_duration
    local_time = time[None, None] - onset[..., None]
    direct_masks = (local_time >= 0) & (local_time < event_duration[..., None]) & active[..., None]
    taper_seconds = min(.05, duration * .05)
    fade_in = (local_time / taper_seconds).clamp(0, 1)
    fade_out = ((event_duration[..., None] - local_time) / taper_seconds).clamp(0, 1)
    envelope = (.5 - .5 * torch.cos(torch.pi * fade_in)) * (.5 - .5 * torch.cos(torch.pi * fade_out)) * direct_masks
    f0 = _rand((batch_size, 2), device, generator, 35, 240 if family == "train" else 360)
    chirp_rate = _rand((batch_size, 2), device, generator, -8, 8) if family == "train" else _rand((batch_size, 2), device, generator, -18, 18)
    # Avoid unphysical zero/negative instantaneous fundamentals during an event.
    chirp_rate = torch.maximum(chirp_rate, -.8 * f0 / event_duration.clamp_min(1e-3))
    types = torch.randint(0, 3, (batch_size, 2), device=device, generator=generator)
    fundamental_phase = 2 * torch.pi * (f0[..., None] * local_time + .5 * chirp_rate[..., None] * local_time.square())
    if family == "heldout":
        fundamental_phase = fundamental_phase + 1.2 * torch.sin(2 * torch.pi * .4 * local_time)
    source = torch.zeros((batch_size, 2, num_samples), device=device)
    for harmonic in range(1, 9):
        phase = _rand((batch_size, 2, 1), device, generator, 0, 2 * math.pi)
        source = source + torch.sin(harmonic * fundamental_phase + phase) / harmonic
    am_rate = _rand((batch_size, 2, 1), device, generator, 2, 20)
    am_depth = _rand((batch_size, 2, 1), device, generator, .15, .8)
    source = source * (1 + am_depth * torch.sin(2 * torch.pi * am_rate * local_time))
    chirp = torch.sin(2 * torch.pi * (2.7 * f0[..., None] * local_time + .5 * (70 + 2 * chirp_rate[..., None]) * local_time.square()))
    source = torch.where((types == 1)[..., None], .3 * source + chirp, source)
    source = torch.where((types == 2)[..., None], source * torch.exp(-local_time.clamp_min(0) / event_duration[..., None].clamp_min(.01)), source)
    dry = source * envelope
    received = dry.clone()
    received_masks = direct_masks.clone()
    maximum_delay = torch.zeros((batch_size, 2), device=device, dtype=torch.long)
    kernel_width = 9 if family == "train" else 17
    # Finite causal low-pass paths give frequency attenuation without inventing
    # infinitely long support from circular FFT filtering.
    smooth = F.avg_pool1d(F.pad(dry.reshape(-1, 1, num_samples), (kernel_width - 1, 0)), kernel_width, stride=1).reshape_as(dry)
    support = F.max_pool1d(F.pad(direct_masks.float().reshape(-1, 1, num_samples), (kernel_width - 1, 0)), kernel_width, stride=1).reshape_as(dry)
    for path in range(2):
        lower, upper = (.002, .08) if family == "train" else (.08, .25)
        delay = (_rand((batch_size, 2), device, generator, lower, upper) * sample_rate).long()
        gain = _rand((batch_size, 2, 1), device, generator, .1, .45) * (1 if path == 0 else -.7)
        received = received + gain * _delay(smooth, delay)
        received_masks = received_masks | (_delay(support, delay) > 0)
        maximum_delay = torch.maximum(maximum_delay, delay)
    support_count = received_masks.sum(-1).clamp_min(1)
    source_rms = ((received.square() * received_masks).sum(-1) / support_count).clamp_min(1e-20).sqrt()
    background_rms = ((background[:, None].square() * received_masks).sum(-1) / support_count).clamp_min(1e-20).sqrt()
    snr_choices = torch.tensor([-15., -10., -5., 0., 5., 10., 20.], device=device)
    snr = snr_choices[torch.randint(0, len(snr_choices), (batch_size, 2), device=device, generator=generator)]
    received = received * (background_rms / source_rms * torch.pow(10., snr / 20))[..., None] * active[..., None]
    intervals = torch.stack((onset, (offset + (maximum_delay + kernel_width - 1) / sample_rate).clamp_max(duration)), -1)
    intervals = torch.where(active[..., None], intervals, torch.full_like(intervals, -1))
    direct_intervals = torch.stack((onset, offset), -1)
    direct_intervals = torch.where(active[..., None], direct_intervals, torch.full_like(direct_intervals, -1))
    audio = background + received.sum(1)
    return {"audio": audio, "background": background, "stems": received,
            "event_masks": received_masks, "intervals": intervals,
            "direct_intervals": direct_intervals, "source_count": counts,
            "labels": (counts > 0).float(), "source_types": torch.where(active, types, -1),
            "source_active": active, "snr_db": snr, "fundamental_hz": f0,
            "seed": int(seed), "family": family, "sample_rate": int(sample_rate),
            "description": "controlled harmonic/AM/chirp simulations; not validated vessel classes"}
