import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uac.metrics import (
    binary_average_precision, binary_auroc, binary_brier, binary_metrics,
    paired_group_bootstrap, temporal_average_precision, temporal_iou, threshold_for_fpr,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_binary_ranking(self):
        y, p = [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
        result = binary_metrics(y, p)
        self.assertEqual(result["ap"], 1.0)
        self.assertEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["brier"], 0.025)
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (2, 2, 0, 0))

    def test_tied_scores_respect_joint_threshold(self):
        y, score = [1, 0, 1, 0], [0.5] * 4
        self.assertEqual(binary_average_precision(y, score), 0.5)
        self.assertEqual(binary_auroc(y, score), 0.5)

    def test_weighted_scores_equal_explicit_group_repetition(self):
        y, score, weights = np.array([1, 0, 1, 0]), np.array([0.6, 0.7, 0.4, 0.2]), np.array([2, 2, 0, 1])
        for fn in (binary_average_precision, binary_auroc, binary_brier):
            self.assertAlmostEqual(fn(y, score, weights), fn(np.repeat(y, weights), np.repeat(score, weights)))

    def test_undefined_class_endpoints_remain_missing(self):
        self.assertIsNone(binary_average_precision([0, 0], [0.1, 0.2]))
        self.assertIsNone(binary_auroc([1, 1], [0.1, 0.2]))
        self.assertIsNone(binary_metrics([0, 0], [0.1, 0.2])["recall"])

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            binary_brier([0, 1], [-0.1, 1.1])

    def test_group_bootstrap_identical_methods_has_zero_difference(self):
        result = paired_group_bootstrap([0, 1, 0, 1], [0.1, 0.7, 0.3, 0.8],
                                        [0.1, 0.7, 0.3, 0.8], ["a", "a", "b", "b"],
                                        n_bootstrap=2000)
        self.assertEqual(result["ci"], [0.0, 0.0])
        self.assertEqual(result["independent_groups"], 2)
        self.assertEqual(result["valid_draws"], 2000)

    def test_group_bootstrap_repeats_whole_group(self):
        # Each group contains both classes, so no resample should lose a class.
        result = paired_group_bootstrap([0, 1, 0, 1], [0.1, 0.7, 0.3, 0.8],
                                        [0.6, 0.2, 0.7, 0.1], ["a", "a", "b", "b"],
                                        n_bootstrap=100)
        self.assertEqual(result["undefined_draws"], 0)
        self.assertGreater(result["ci"][0], 0)

    def test_calibration_threshold_handles_negative_ties(self):
        result = threshold_for_fpr([0, 0, 0, 1], [0.1, 0.8, 0.8, 0.9], max_fpr=0.3)
        self.assertGreater(result["threshold"], 0.8)
        self.assertEqual(result["calibration_fpr"], 0)

    def test_zero_fpr_threshold_can_exceed_one(self):
        result = threshold_for_fpr([0, 1], [1.0, 1.0], max_fpr=0)
        self.assertGreater(result["threshold"], 1)
        scored = binary_metrics([0, 1], [1.0, 1.0], threshold=result["threshold"])
        self.assertEqual(scored["fpr"], 0)
        self.assertEqual(scored["recall"], 0)

    def test_temporal_iou(self):
        self.assertAlmostEqual(temporal_iou((0, 2), (1, 3)), 1 / 3)
        self.assertEqual(temporal_iou((0, 1), (1, 2)), 0)

    def test_duplicate_events_match_only_once(self):
        gt = [{"recording_id": "r", "class_id": "vessel", "onset": 1, "offset": 2},
              {"recording_id": "r", "class_id": "vessel", "onset": 3, "offset": 4}]
        predictions = [
            {**gt[0], "score": 0.9},
            {**gt[0], "score": 0.8},
            {**gt[1], "score": 0.7},
        ]
        result = temporal_average_precision(predictions, gt)
        self.assertAlmostEqual(result["map"], 5 / 6)
        self.assertEqual(result["classes"]["vessel"]["false_positive_events"], 1)

    def test_wrong_recording_cannot_match(self):
        gt = [{"recording_id": "r1", "class_id": "vessel", "onset": 1, "offset": 2}]
        pred = [{"recording_id": "r2", "class_id": "vessel", "onset": 1, "offset": 2, "score": 0.9}]
        self.assertEqual(temporal_average_precision(pred, gt)["map"], 0)

    def test_missing_class_not_silently_counted_as_zero_ap(self):
        pred = [{"recording_id": "r", "class_id": "unknown", "onset": 1, "offset": 2, "score": 0.9}]
        result = temporal_average_precision(pred, [])
        self.assertIsNone(result["map"])
        self.assertEqual(result["classes"]["unknown"]["false_positive_events"], 1)


if __name__ == "__main__":
    unittest.main()
