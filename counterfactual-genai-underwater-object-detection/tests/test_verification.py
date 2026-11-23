"""Numerical verification contracts; GPU fixtures opt in explicitly."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from uac.models import interval_mask, matched_sham
from uac.verification import (band_mismatch, generate_intervention_features,
                              intervention_components, verification_score)


class MeasurementFixtureBank:
    """Deterministic measurement logits only; not a trained detection model."""
    def view_logits(self, features):
        return features[:, [0, 32, 96]].clamp(-8, 8)


class VerificationGuard(unittest.TestCase):
    def test_cpu_interventions_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            generate_intervention_features(None, torch.zeros(1, 2048), seed=1)


@unittest.skipUnless(os.environ.get("UAC_RUN_GPU_TESTS") == "1", "GPU checks explicitly opt in")
class VerificationNumerics(unittest.TestCase):
    def test_same_support_sham_and_wrong_band_control(self):
        t = torch.arange(48000, device="cuda") / 16000
        original = (.05 * torch.sin(2 * torch.pi * 200 * t))[None]
        mask = interval_mask(1, 48000, 12000, 36000, taper_samples=160)
        edit = .02 * torch.sin(2 * torch.pi * 1200 * t)[None] * mask
        replacement = original + edit
        noise = torch.randn(original.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(791))
        sham, diag = matched_sham(original, replacement, mask, noise)
        self.assertLess(float(diag["relative_energy_mismatch"].max()), 1e-5)
        self.assertTrue(torch.equal(sham[mask == 0], original[mask == 0]))
        self.assertLess(float(band_mismatch(edit, sham - original).max()), .2)
        wrong_band = .02 * torch.sin(2 * torch.pi * 3000 * t)[None] * mask
        self.assertGreater(float(band_mismatch(edit, wrong_band).min()), 1.8)

    def test_lower_median_and_nonfinite_base_fallback(self):
        delta = torch.tensor([[[0., 0., 0.], [2., 2., 2.], [8., 8., 8.], [10., 10., 10.]],
                              [[float("nan"), 0., 0.]] * 4], device="cuda")
        components = {"delta": delta, "penalty": torch.tensor([0., float("nan")], device="cuda"),
                      "valid": torch.tensor([True, False], device="cuda")}
        score = verification_score(components, ["fourier"], dispersion=.5, invalid_penalty=.1)
        # lower median=2, lower MAD=2 ->1. Midpoint median would yield3.
        self.assertTrue(torch.equal(score, torch.tensor([1., 0.], device="cuda")))
        self.assertTrue(bool(torch.isfinite(score).all()))
        class NonfiniteFlow:
            def replace(self, audio, mask, **kwargs):
                result = audio.clone()
                result[0, audio.shape[-1] // 2] = float("nan")
                return result
        t = torch.arange(48000, device="cuda") / 16000
        x = .03 * torch.sin(2 * torch.pi * 440 * t)[None].repeat(2, 1)
        cache = generate_intervention_features(NonfiniteFlow(), x, seed=17, k=1, steps=1)
        self.assertFalse(bool(cache["valid"][0]))
        self.assertTrue(bool(torch.isfinite(cache["replacement_features"]).all()))

    def test_cached_batch_and_each_case_scores_are_identical(self):
        values = torch.arange(3 * 4 * 152, device="cuda", dtype=torch.float32).reshape(3, 4, 152) / 1024
        cache = {"replacement_features": values,
                 "sham_features": values + torch.tensor([.125, .25, .5], device="cuda")[:, None, None],
                 "outside_error": torch.zeros(3, 4, device="cuda"),
                 "boundary_penalty": torch.full((3, 4), .25, device="cuda"),
                 "valid": torch.tensor([True, False, True], device="cuda")}
        bank = MeasurementFixtureBank()
        batched = verification_score(intervention_components(bank, cache, {}), ["stft", "wavelet"])
        single = torch.cat([verification_score(intervention_components(bank, {key: value[i:i + 1] for key, value in cache.items()}, {}),
                                               ["stft", "wavelet"]) for i in range(3)])
        self.assertTrue(torch.equal(batched, single))


if __name__ == "__main__":
    unittest.main()
