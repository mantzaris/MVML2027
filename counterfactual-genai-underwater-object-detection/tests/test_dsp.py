"""Small API safeguards plus opt-in GPU numerical checks.

Run GPU tests only through run_on_ada.py with UAC_RUN_GPU_TESTS=1. CPU fixtures
are limited to verifying rejection and metadata; research transforms never run
on CPU.
"""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from uac.dsp import GROUP_DIMS, feature_names, extract_features, emd_linear
from uac.synthetic import generate_batch


class MetadataAndGuards(unittest.TestCase):
    def test_feature_dimensions_and_unique_names(self):
        self.assertEqual(sum(GROUP_DIMS.values()), 184)
        names = feature_names()
        self.assertEqual(len(names), 184)
        self.assertEqual(len(names), len(set(names)))

    def test_each_view_names_match_dimension(self):
        for name, count in GROUP_DIMS.items():
            self.assertEqual(len(feature_names((name,))), count)

    def test_unknown_view_rejected(self):
        with self.assertRaises(ValueError):
            feature_names(("image",))

    def test_cpu_transform_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
            extract_features(torch.zeros(1, 2048))

    def test_cpu_generator_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            generate_batch(1, device="cpu")


@unittest.skipUnless(os.environ.get("UAC_RUN_GPU_TESTS") == "1", "GPU tests explicitly opt in")
class GpuNumericalTests(unittest.TestCase):
    def test_exact_synthetic_decomposition_and_snr(self):
        batch = generate_batch(3, 8192, source_counts=[0, 1, 2], seed=31)
        self.assertTrue(torch.equal(batch["audio"], batch["background"] + batch["stems"].sum(1)))
        self.assertTrue(torch.equal(batch["labels"], torch.tensor([0., 1., 1.], device="cuda")))
        mask = batch["event_masks"]
        observed = 10 * torch.log10((batch["stems"].square() * mask).sum(-1).clamp_min(1e-20)
                                   / (batch["background"][:, None].square() * mask).sum(-1).clamp_min(1e-20))
        active = batch["source_active"]
        self.assertLess(float((observed[active] - batch["snr_db"][active]).abs().max()), 1e-3)
        self.assertEqual(float(batch["stems"][~mask].abs().max()), 0.)

    def test_seed_reproducibility(self):
        first = generate_batch(2, 4096, seed=51)
        second = generate_batch(2, 4096, seed=51)
        self.assertTrue(torch.equal(first["audio"], second["audio"]))

    def test_feature_shape_and_finiteness(self):
        batch = generate_batch(2, 8192, seed=61)
        result = extract_features(batch["audio"])
        self.assertEqual(result["features"].shape, (2, 184))
        self.assertEqual(result["features"].device.type, "cuda")
        self.assertTrue(bool(torch.isfinite(result["features"]).all()))

    def test_emd_reconstructs_silent_and_tonal_signals(self):
        t = torch.arange(8192, device="cuda") / 16000
        x = torch.stack((torch.zeros_like(t), torch.sin(2 * torch.pi * 440 * t)))
        modes, residual, diagnostics = emd_linear(x)
        self.assertTrue(torch.allclose(modes.sum(1) + residual, x, atol=1e-5))
        self.assertEqual(float(modes[0].abs().max()), 0.)
        self.assertFalse(bool(diagnostics["valid_modes"][0].any()))


if __name__ == "__main__":
    unittest.main()
