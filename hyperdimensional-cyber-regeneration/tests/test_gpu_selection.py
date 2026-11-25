"""Selection behavior must never silently use the laptop's other GPU."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_gpu import DEFAULT_GPU, GPU, select_gpu

class GPUSelectionTests(unittest.TestCase):
    def setUp(self):
        self.quadro = GPU("0", "GPU-quadro", "Quadro T2000", "test-driver")
        self.ada = GPU("1", "GPU-ada", DEFAULT_GPU, "test-driver")

    def test_ada_is_selected_independently_of_inventory_order(self):
        for devices in [[self.quadro, self.ada], [self.ada, self.quadro]]:
            self.assertEqual(select_gpu(devices).uuid, "GPU-ada")

    def test_absent_ada_never_falls_back_to_quadro(self):
        with self.assertRaises(RuntimeError):
            select_gpu([self.quadro])

    def test_explicit_uuid_must_match_the_expected_model(self):
        with self.assertRaises(RuntimeError):
            select_gpu([self.quadro, self.ada], requested_uuid="GPU-quadro")

    def test_multiple_adas_require_explicit_selection(self):
        second = GPU("2", "GPU-second-ada", DEFAULT_GPU, "test-driver")
        with self.assertRaises(RuntimeError):
            select_gpu([self.ada, second])
        self.assertEqual(select_gpu([self.ada, second], requested_uuid=second.uuid), second)

if __name__ == "__main__":
    unittest.main()
