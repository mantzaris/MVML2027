"""Scientific CUDA checks for generic flow-model adaptations."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real_models import (KEYS, make_model, infer, loss_terms, observable_input,
                             normalized_adjacency)


def batch(flows=32):
    values = torch.randn(4, flows, 14, device="cuda")
    observed = (torch.rand(4, flows, 14, device="cuda") > .2).float()
    valid = torch.ones(4, flows, device="cuda", dtype=torch.bool)
    valid[0, -5:] = False
    adj = (torch.rand(4, flows, flows, device="cuda") > .8).float()
    adj = torch.maximum(adj, adj.transpose(1, 2))
    y = (torch.rand(4, flows, device="cuda") > .7).float()
    return dict(x=torch.cat((values * observed, observed), -1), valid=valid, adj=adj, y=y)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RealFlowModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91017)
        self.b = batch()

    def test_all_models_finite_supervised_gradients_under_autocast(self):
        for key in KEYS:
            with self.subTest(key=key):
                model = make_model(key).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    q = infer(model, self.b)
                self.assertEqual(q.shape, (4, 32))
                self.assertEqual(q.dtype, torch.float32)
                self.assertTrue(torch.isfinite(q).all())
                self.assertTrue(((q >= 0) & (q <= 1)).all())
                self.assertTrue((q[~self.b["valid"]] == 0).all())
                terms = loss_terms(q, self.b)
                self.assertEqual(terms["total"].shape, (4,))
                terms["total"].mean().backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                if key == "SCR_FLOW":
                    self.assertGreater(model.gate[-1].weight.grad.abs().sum().item(), 0)

    def test_hidden_labels_and_metadata_do_not_enter_inference(self):
        observable = {k:self.b[k] for k in ("x", "valid", "adj")}
        changed = dict(self.b, y=1-self.b["y"], attack_type=torch.rand(4, 32, device="cuda"),
                       source_ip="not an input", scenario=999, clean_truth=torch.randn(4, 32, 14, device="cuda"))
        with torch.no_grad():
            for key in KEYS:
                with self.subTest(key=key):
                    model = make_model(key).cuda().eval()
                    original = infer(model, self.b)
                    self.assertTrue(torch.equal(original, infer(model, observable)))
                    self.assertTrue(torch.equal(original, infer(model, changed)))

    def test_padding_values_labels_and_edges_do_not_affect_real_rows(self):
        altered = {k:v.clone() for k,v in self.b.items()}
        padding = ~self.b["valid"]
        altered["x"][padding] = float("nan")
        altered["y"][padding] = float("nan")
        edge_padding = padding[:, :, None] | padding[:, None, :]
        altered["adj"][edge_padding] = float("nan")
        with torch.no_grad():
            for key in KEYS:
                with self.subTest(key=key):
                    model = make_model(key).cuda().eval()
                    q = infer(model, self.b)
                    other = infer(model, altered)
                    torch.testing.assert_close(q, other, atol=0., rtol=0.)
                    torch.testing.assert_close(loss_terms(q, self.b)["total"],
                                               loss_terms(other, altered)["total"], atol=0., rtol=0.)

    def test_unobserved_feature_placeholders_are_ignored(self):
        altered = dict(self.b, x=self.b["x"].clone())
        altered["x"][..., :14].masked_fill_(self.b["x"][..., 14:] == 0, float("nan"))
        with torch.no_grad():
            for key in KEYS:
                with self.subTest(key=key):
                    model = make_model(key).cuda().eval()
                    torch.testing.assert_close(infer(model, self.b), infer(model, altered), atol=0., rtol=0.)

    def test_pretraining_targets_are_observed_features_not_labels(self):
        observable = {k:self.b[k] for k in ("x", "valid", "adj")}
        for key in ("DAE", "GMAE", "HDC"):
            with self.subTest(key=key):
                model = make_model(key).cuda()
                torch.manual_seed(442)
                a = model.pretraining_loss(self.b)
                torch.manual_seed(442)
                b = model.pretraining_loss(observable)
                self.assertEqual(a.shape, (4,))
                torch.testing.assert_close(a, b, atol=0., rtol=0.)
                self.assertTrue(torch.isfinite(a).all())
                a.mean().backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                self.assertIsNone(model.readout.weight.grad)

    def test_context_controls_match_parameters_and_gate_is_convex(self):
        context = make_model("CTX_FLOW").cuda()
        local = make_model("LOCAL_FLOW").cuda()
        local.load_state_dict(context.state_dict())
        self.assertEqual(sum(p.numel() for p in context.parameters()), sum(p.numel() for p in local.parameters()))
        _, local_context = local._encode(self.b)
        self.assertEqual(torch.count_nonzero(local_context).item(), 0)
        weights = make_model("SCR_FLOW").cuda().context_weights(self.b)
        torch.testing.assert_close(weights.sum(-1), torch.ones(4, device="cuda"))
        self.assertTrue((weights >= 0).all())

    def test_slot_permutation_equivariance_except_explicit_slot_key_hdc(self):
        order = torch.randperm(32, device="cuda")
        permuted = dict(self.b, x=self.b["x"][:, order], valid=self.b["valid"][:, order],
                        adj=self.b["adj"][:, order][:, :, order])
        with torch.no_grad():
            for key in KEYS:
                if key == "HDC":
                    continue
                with self.subTest(key=key):
                    model = make_model(key).cuda().eval()
                    torch.testing.assert_close(infer(model, permuted), infer(model, self.b)[:, order], atol=2e-6, rtol=2e-6)

    def test_variable_window_lengths_and_all_padding(self):
        with torch.no_grad():
            for key in KEYS:
                model = make_model(key).cuda().eval()
                for count in (32, 64):
                    with self.subTest(key=key, flows=count):
                        b = batch(count)
                        q = infer(model, b)
                        self.assertEqual(q.shape, (4, count))
                        b["valid"].zero_()
                        q = infer(model, b)
                        self.assertTrue(torch.isfinite(q).all())
                        self.assertEqual(torch.count_nonzero(q).item(), 0)
                        self.assertEqual(loss_terms(q, b)["total"].sum().item(), 0)

    def test_dense_has_no_channel_ten_prior_shortcut(self):
        for key in ("DENSE", "GMAE", "HDC"):
            with self.subTest(key=key), torch.no_grad():
                model = make_model(key).cuda()
                model.readout.weight.zero_()
                model.readout.bias.zero_()
                q = infer(model, self.b)
                self.assertTrue((q[self.b["valid"]] == .5).all())

    def test_hdc_slot_keys_reproduce_after_checkpoint_loading(self):
        with torch.no_grad():
            original = make_model("HDC").cuda().eval()
            prediction = infer(original, self.b)
            restored = make_model("HDC").cuda().eval()
            restored.load_state_dict(original.state_dict())
            self.assertNotIn("role_keys", original.state_dict())
            torch.testing.assert_close(infer(restored, self.b), prediction, atol=0., rtol=0.)
            self.assertTrue(torch.equal(restored.role_keys, original.role_keys))

    def test_cpu_research_inference_fails_clearly(self):
        cpu_batch = {k:v.cpu() for k,v in self.b.items()}
        for key in KEYS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    infer(make_model(key), cpu_batch)


if __name__ == "__main__":
    unittest.main()
