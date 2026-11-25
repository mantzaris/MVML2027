"""CUDA checks for real-data preprocessing, corruptions and tied-score metrics.

Metric references are independent NumPy calculations on tiny fixtures. The
optional captured-data smoke test reads training capture 1 only; no held-out
capture or model prediction is used by this test suite.
"""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from real_data import (CONDITIONS, fit_preprocessing, windows, corrupt,
                       condition_batch, load_capture, input_hash)
from real_metrics import metrics, select_threshold, GaussianNB


def reference_metrics(y, q, threshold):
    """AP by explicit threshold sets; AUC by positive/negative pair counting."""
    y = np.asarray(y, dtype=bool)
    q = np.asarray(q, dtype=np.float32)
    positives, negatives = y.sum(), (~y).sum()
    last_tp, ap = 0, 0.
    choices = []
    for cut in sorted(set(q.tolist()), reverse=True):
        selected = q >= cut
        tp = int((selected & y).sum())
        fp = int((selected & ~y).sum())
        ap += (tp - last_tp) * tp / max(tp + fp, 1)
        last_tp = tp
        f1 = 2 * tp / max(int(selected.sum()) + int(positives), 1)
        choices.append((cut, f1))
    ap = ap / positives if positives else np.nan
    if positives and negatives:
        pairs = q[y, None] - q[None, ~y]
        auc = ((pairs > 0).astype(float) + .5 * (pairs == 0)).mean()
    else:
        auc = np.nan
    selected = q >= threshold
    tp = int((selected & y).sum())
    fp = int((selected & ~y).sum())
    fn = int((~selected & y).sum())
    recall = tp / max(int(positives), 1)
    fpr = fp / max(int(negatives), 1)
    best_f1 = max(v for _, v in choices)
    chosen = max(cut for cut, f1 in choices if f1 == best_f1)
    return dict(ap=ap, auc=auc, brier=np.mean((q.astype(float) - y)**2),
                recall=recall, fpr=fpr, f1=2*tp/max(2*tp+fp+fn, 1),
                balanced_accuracy=(recall+1-fpr)/2), chosen


def fixture_capture(count=37, offset=0):
    generator = torch.Generator(device="cuda").manual_seed(7731 + offset)
    features = torch.rand(count, 14, generator=generator, device="cuda")
    features[:, :6] = features[:, :6] * 4 + offset
    features[:, 8:] = (features[:, 8:] > .5).float()
    observed = torch.rand(count, 14, generator=generator, device="cuda") > .15
    observed[0, 0] = False
    features[0, 0] = 1e6  # An unobserved placeholder must not influence fitting.
    return dict(features=features, observed=observed,
                label=(torch.arange(count, device="cuda") % 3 == 0).float(),
                src=torch.arange(1, count+1, device="cuda", dtype=torch.long),
                dst=torch.arange(1001, 1001+count, device="cuda", dtype=torch.long),
                row_id=torch.arange(count, device="cuda", dtype=torch.long),
                timestamp=torch.arange(count, device="cuda", dtype=torch.float64) + offset*1000)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RealPipelineTests(unittest.TestCase):
    def test_tied_metrics_and_threshold_match_independent_numpy(self):
        rng = np.random.default_rng(414)
        cases = [
            (np.array([0, 1, 0, 1]), np.array([.5, .5, .5, .5])),
            (np.array([0, 0, 1, 1]), np.array([0., .1, .9, 1.])),
            (np.array([1, 1, 0, 0]), np.array([0., .1, .9, 1.])),
        ]
        for _ in range(12):
            cases.append((rng.integers(0, 2, 71), rng.integers(0, 9, 71) / 8))
        for index, (y, q) in enumerate(cases):
            with self.subTest(case=index):
                expected, threshold = reference_metrics(y, q, .5)
                yt = torch.as_tensor(y, dtype=torch.float32, device="cuda")
                qt = torch.as_tensor(q, dtype=torch.float32, device="cuda")
                actual = metrics(yt, qt, .5)
                for key in expected:
                    self.assertAlmostEqual(actual[key], expected[key], places=10, msg=key)
                self.assertAlmostEqual(select_threshold(yt, qt), threshold, places=7)

    def test_metrics_ignore_order_inside_ties(self):
        y = torch.tensor([1, 0, 0, 1, 1, 0], device="cuda")
        q = torch.tensor([.9, .9, .9, .4, .4, .1], device="cuda")
        order = torch.tensor([2, 0, 1, 4, 3, 5], device="cuda")
        original, changed = metrics(y, q), metrics(y[order], q[order])
        for key in original:
            self.assertEqual(original[key], changed[key])
        self.assertEqual(select_threshold(y, q), select_threshold(y[order], q[order]))

    def test_preprocessing_is_fit_from_supplied_training_only(self):
        train = [fixture_capture(37), fixture_capture(19, 2)]
        fitted = fit_preprocessing(train)
        values = torch.cat([b["features"] for b in train]).cpu().numpy()
        observed = torch.cat([b["observed"] for b in train]).cpu().numpy()
        labels = torch.cat([b["label"] for b in train]).cpu().numpy()
        expected_mean = np.array([values[observed[:, i], i].mean(dtype=np.float64) for i in range(14)])
        expected_std = np.array([values[observed[:, i], i].std(dtype=np.float64) for i in range(14)])
        expected_mean[6:] = 0
        expected_std[6:] = 1
        np.testing.assert_allclose(fitted["mean"].cpu().numpy(), expected_mean, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(fitted["std"].cpu().numpy(), expected_std, atol=2e-6, rtol=2e-6)
        normalized = np.clip((values - expected_mean) / expected_std, -5, 5)
        centroid = np.array([normalized[observed[:, i] & (labels == 0), i].mean() for i in range(14)])
        np.testing.assert_allclose(fitted["benign_centroid"].cpu().numpy(), centroid, atol=3e-6, rtol=3e-6)
        before = {k:v.clone() for k,v in fitted.items()}
        unrelated = fixture_capture(37, 90)
        windows(unrelated, 1, fitted)  # Transforming another capture cannot refit.
        for key in before:
            self.assertTrue(torch.equal(before[key], fitted[key]))
        relabeled = [dict(b, label=1-b["label"]) for b in train]
        changed = fit_preprocessing(relabeled)
        torch.testing.assert_close(changed["mean"], fitted["mean"], atol=0., rtol=0.)
        torch.testing.assert_close(changed["std"], fitted["std"], atol=0., rtol=0.)

    def test_window_order_padding_and_observed_graph(self):
        capture = fixture_capture()
        preprocessing = fit_preprocessing([capture])
        capture["src"][1] = capture["src"][0]
        b = windows(capture, 1, preprocessing)
        self.assertEqual(b["x"].shape, (2, 32, 28))
        self.assertEqual(int(b["valid"].sum()), 37)
        torch.testing.assert_close(b["row_id"][b["valid"]], capture["row_id"])
        torch.testing.assert_close(b["timestamp"][b["valid"]], capture["timestamp"])
        self.assertEqual(float(b["adj"][0, 0, 1]), 1.)
        self.assertTrue((b["x"][~b["valid"]] == 0).all())
        padding_edges = (~b["valid"])[:, :, None] | (~b["valid"])[:, None, :]
        self.assertTrue((b["adj"][padding_edges] == 0).all())
        self.assertTrue((b["row_id"][~b["valid"]] == -1).all())

    def test_unknown_endpoint_zero_does_not_create_shared_host_edges(self):
        capture = fixture_capture()
        capture["src"][:3] = 0
        b = windows(capture, 1, fit_preprocessing([capture]))
        self.assertEqual(float(b["adj"][0, 0, 1]), 0.)
        self.assertEqual(float(b["adj"][0, 0, 2]), 0.)

    def test_corruptions_are_label_blind_paired_and_keep_padding(self):
        capture = fixture_capture(37)
        preprocessing = fit_preprocessing([capture])
        clean = windows(capture, 1, preprocessing)
        relabeled = dict(clean, y=1-clean["y"])
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                before = torch.cuda.get_rng_state()
                a = condition_batch(clean, condition, preprocessing)
                after = torch.cuda.get_rng_state()
                b = condition_batch(relabeled, condition, preprocessing)
                self.assertTrue(torch.equal(before, after))
                self.assertTrue(torch.equal(a["x"], b["x"]))
                self.assertTrue(torch.equal(a["adj"], clean["adj"]))
                self.assertTrue(torch.equal(a["y"], clean["y"]))
                self.assertTrue(torch.equal(a["row_id"], clean["row_id"]))
                self.assertTrue((a["x"][~clean["valid"]] == 0).all())
                if condition.startswith("missing"):
                    observed = a["x"][..., 14:].bool()
                    self.assertTrue((a["x"][..., :14][~observed] == 0).all())
                    self.assertTrue((a["x"][..., 14:] <= clean["x"][..., 14:]).all())
                elif condition.startswith("spoof"):
                    self.assertTrue(torch.equal(a["x"][..., 8:], clean["x"][..., 8:]))
                if condition != "clean":
                    self.assertNotEqual(input_hash(a), input_hash(clean))

    def test_gaussian_reference_omits_missing_features_and_padding(self):
        capture = fixture_capture(37)
        b = windows(capture, 1, fit_preprocessing([capture]))
        model = GaussianNB().cuda().fit(b, torch.ones(len(b["x"]), device="cuda"))
        empty = dict(b, x=torch.zeros_like(b["x"]))
        q = model.probabilities(empty)
        torch.testing.assert_close(q[b["valid"]], model.prior[1].expand(int(b["valid"].sum())), atol=1e-6, rtol=1e-6)
        self.assertTrue((q[~b["valid"]] == 0).all())
        changed = dict(b, y=1-b["y"])
        torch.testing.assert_close(model.probabilities(b), model.probabilities(changed), atol=0., rtol=0.)

    @unittest.skipUnless((ROOT / "data/external/ctu13/parsed/scenario_01.npz").is_file(),
                         "Optional training-capture file not present")
    def test_real_training_capture_only_smoke(self):
        capture = load_capture(1)
        preprocessing = fit_preprocessing([capture])
        b = windows(capture, 1, preprocessing, limit=2)
        self.assertEqual(b["x"].shape, (2, 32, 28))
        self.assertTrue(torch.isfinite(b["x"]).all())
        self.assertTrue((b["scenario"] == 1).all())
        self.assertTrue(torch.equal(b["row_id"].flatten(), capture["row_id"][:64]))
        self.assertTrue(torch.equal(b["y"].flatten(), capture["label"][:64]))


if __name__ == "__main__":
    unittest.main()
