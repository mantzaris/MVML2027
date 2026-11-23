"""Identity and memory safeguards; fixtures do not require a GPU or torch."""

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_on_ada.py"
SPEC = importlib.util.spec_from_file_location("run_on_ada", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GpuGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = {"gpu_uuid": "GPU-ada", "gpu_name": "NVIDIA RTX 5000 Ada Generation",
                       "minimum_memory_mib": 30000}

    def test_selects_uuid_instead_of_index_or_first_gpu(self):
        inventory = "GPU-small, Quadro T2000, 4096\nGPU-ada, NVIDIA RTX 5000 Ada Generation, 32760\n"
        self.assertEqual(MODULE.validate_gpu_inventory(inventory, self.config)["uuid"], "GPU-ada")

    def test_rejects_only_smaller_gpu(self):
        with self.assertRaises(ValueError):
            MODULE.validate_gpu_inventory("GPU-small, Quadro T2000, 4096\n", self.config)

    def test_rejects_name_mismatch_even_at_correct_uuid(self):
        with self.assertRaises(ValueError):
            MODULE.validate_gpu_inventory("GPU-ada, NVIDIA RTX PRO 4500 Blackwell, 32760\n", self.config)

    def test_rejects_insufficient_memory_even_at_correct_uuid(self):
        with self.assertRaises(ValueError):
            MODULE.validate_gpu_inventory("GPU-ada, NVIDIA RTX 5000 Ada Generation, 4096\n", self.config)

    def test_rejects_other_large_gpu_uuid(self):
        with self.assertRaises(ValueError):
            MODULE.validate_gpu_inventory("GPU-other, NVIDIA RTX 5000 Ada Generation, 32760\n", self.config)


if __name__ == "__main__":
    unittest.main()
