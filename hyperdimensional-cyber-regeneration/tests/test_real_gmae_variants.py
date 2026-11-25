"""Scientific invariants for the two exploratory CUDA GraphMAE variants."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.real_models import GraphFlow, infer, loss_terms, normalized_adjacency
from src.real_gmae_variants import (KEYS, make_model, reliability_edges,
    feature_masked_view, symmetric_latent_consistency, LATENT_WEIGHT)


def batch(flows=12):
    values = torch.randn(3, flows, 14, device="cuda")
    observed = (torch.rand(3, flows, 14, device="cuda") > .2).float()
    valid = torch.ones(3, flows, device="cuda", dtype=torch.bool)
    valid[0, -3:] = False
    adj = (torch.rand(3, flows, flows, device="cuda") > .5).float()
    adj = torch.maximum(adj, adj.transpose(1, 2))
    y = (torch.rand(3, flows, device="cuda") > .5).float()
    return dict(x=torch.cat((values*observed, observed), -1), valid=valid, adj=adj, y=y)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GraphMAEVariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73119)
        self.b = batch()

    def test_all_parameter_counts_match_graphmae_and_controls(self):
        original = GraphFlow("GMAE").cuda()
        for key in KEYS:
            with self.subTest(key=key):
                model = make_model(key).cuda()
                self.assertEqual(sum(p.numel() for p in model.parameters()), 121643)
                self.assertEqual(set(model.state_dict()), set(original.state_dict()))
                model.load_state_dict(original.state_dict())
        edge, control = make_model("EDGE_GMAE").cuda(), make_model("EDGE_DENSE").cuda()
        control.load_state_dict(edge.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(infer(edge, self.b), infer(control, self.b), atol=0., rtol=0.)

    def test_consistency_variants_have_exact_baseline_inference_architecture(self):
        original = GraphFlow("GMAE").cuda().eval()
        with torch.no_grad():
            expected = infer(original, self.b)
            for key in ("CONS_GMAE", "CONS_PRE"):
                model = make_model(key).cuda().eval()
                model.load_state_dict(original.state_dict())
                torch.testing.assert_close(infer(model, self.b), expected, atol=0., rtol=0.)

    def test_edge_gate_observable_formula_self_edges_symmetry_and_no_overlap(self):
        x = torch.zeros(1, 4, 28, device="cuda")
        x[0, :2, 14:] = 1.
        x[0, 1, :14] = 2.
        active = torch.tensor([[True, True, True, False]], device="cuda")
        adj = torch.ones(1, 4, 4, device="cuda")
        adj[0, 3, :] = float("nan"); adj[0, :, 3] = float("nan")
        x[0, 3] = float("nan")
        edge = reliability_edges(x, adj, active)
        expected = .1 + .9*torch.exp(torch.tensor(-2., device="cuda"))
        torch.testing.assert_close(edge[0, 0, 1], expected)
        self.assertEqual(edge[0, 0, 2].item(), torch.tensor(.1).item())
        torch.testing.assert_close(edge, edge.transpose(1, 2), atol=0., rtol=0.)
        torch.testing.assert_close(edge[0].diag(), torch.tensor([1.,1.,1.,0.], device="cuda"))
        self.assertTrue(torch.isfinite(edge).all())
        self.assertEqual(edge[0, 3].abs().sum().item(), 0.)
        normalized = normalized_adjacency(edge, active)
        torch.testing.assert_close(normalized.sum(-1), active.float())
        self_only = normalized_adjacency(reliability_edges(x, torch.zeros_like(adj), active), active)
        torch.testing.assert_close(self_only[0], torch.diag(active[0].float()))

    def test_masked_node_values_cannot_leak_through_edge_gate(self):
        model = make_model("EDGE_GMAE").cuda()
        chosen_seed = 9441
        torch.manual_seed(chosen_seed)
        selected = (torch.rand(self.b["valid"].shape, device="cuda") < .3) & self.b["valid"]
        self.assertTrue(selected.any())
        altered = dict(self.b, x=self.b["x"].clone())
        altered["x"][..., :14][selected] += 100.
        original_encode = model.encode
        captures = []
        def captured(*args, **kwargs):
            h, graph = original_encode(*args, **kwargs)
            captures.append((h.detach().clone(), graph.detach().clone(), kwargs['observed'].clone()))
            return h, graph
        with patch.object(model, "encode", side_effect=captured):
            torch.manual_seed(chosen_seed); model.pretraining_loss(self.b)
            torch.manual_seed(chosen_seed); model.pretraining_loss(altered)
        for a, b in zip(captures[0], captures[1]):
            torch.testing.assert_close(a, b, atol=0., rtol=0.)
        self.assertEqual(captures[0][2][selected].count_nonzero().item(), 0)

    def test_hidden_labels_and_metadata_are_not_prediction_or_pretraining_inputs(self):
        visible = {k:self.b[k] for k in ("x", "adj", "valid")}
        altered = dict(visible, y=1-self.b["y"], clean_truth=torch.randn_like(self.b["x"]), scenario=999)
        for key in KEYS:
            with self.subTest(key=key):
                model = make_model(key).cuda()
                torch.testing.assert_close(infer(model, visible), infer(model, altered), atol=0., rtol=0.)
                torch.manual_seed(812); first = model.pretraining_loss(visible)
                torch.manual_seed(812); second = model.pretraining_loss(altered)
                torch.testing.assert_close(first, second, atol=0., rtol=0.)
                p = infer(model, visible)
                torch.manual_seed(551); first = model.supervised_regularizer(visible, p)
                torch.manual_seed(551); second = model.supervised_regularizer(altered, p)
                torch.testing.assert_close(first, second, atol=0., rtol=0.)

    def test_padding_and_missing_nan_placeholders_leave_predictions_and_losses_unchanged(self):
        altered = {k:v.clone() for k,v in self.b.items()}
        inactive = ~self.b["valid"]
        altered["x"][inactive] = float("nan")
        altered["y"][inactive] = float("nan")
        altered["adj"][inactive[:, :, None] | inactive[:, None, :]] = float("nan")
        missing = (self.b["x"][..., 14:] == 0) | inactive[..., None]
        altered["x"][..., :14].masked_fill_(missing, float("nan"))
        for key in KEYS:
            with self.subTest(key=key):
                model = make_model(key).cuda()
                p, other = infer(model, self.b), infer(model, altered)
                torch.testing.assert_close(p, other, atol=0., rtol=0.)
                torch.manual_seed(112); loss = model.pretraining_loss(self.b)
                torch.manual_seed(112); other_loss = model.pretraining_loss(altered)
                torch.testing.assert_close(loss, other_loss, atol=0., rtol=0.)
                torch.manual_seed(113); reg = model.supervised_regularizer(self.b, p)
                torch.manual_seed(113); other_reg = model.supervised_regularizer(altered, other)
                torch.testing.assert_close(reg, other_reg, atol=0., rtol=0.)

    def test_pretraining_and_supervised_gradients_are_finite_under_autocast(self):
        for key in KEYS:
            with self.subTest(key=key):
                model = make_model(key).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pretrain = model.pretraining_loss(self.b)
                self.assertEqual(pretrain.shape, (3,))
                self.assertTrue(torch.isfinite(pretrain).all())
                pretrain.mean().backward()
                self.assertIsNone(model.readout.weight.grad)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(model.feature_decoder[-1].weight.grad.abs().sum().item(), 0.)
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    q = infer(model, self.b)
                    reg = model.supervised_regularizer(self.b, q)
                objective = loss_terms(q, self.b)["total"] + reg
                objective.mean().backward()
                self.assertTrue(torch.isfinite(objective).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(model.readout.weight.grad.abs().sum().item(), 0.)
                if key != "CONS_GMAE":
                    self.assertEqual(reg.count_nonzero().item(), 0)

    def test_feature_masking_only_removes_observations_and_never_uses_labels(self):
        torch.manual_seed(91); view = feature_masked_view(self.b)
        self.assertEqual(set(view), {"x", "adj", "valid"})
        self.assertTrue((view["x"][..., 14:] <= self.b["x"][..., 14:]).all())
        kept = view["x"][..., 14:] > 0
        torch.testing.assert_close(view["x"][..., :14][kept], self.b["x"][..., :14][kept])
        self.assertEqual(view["x"][..., :14][~kept].count_nonzero().item(), 0)
        self.assertGreater((self.b["x"][..., 14:]*self.b["valid"][...,None]-view["x"][..., 14:]).sum().item(), 0)
        torch.manual_seed(91); visible_view = feature_masked_view({k:self.b[k] for k in ("x", "adj", "valid")})
        torch.testing.assert_close(view["x"], visible_view["x"], atol=0., rtol=0.)

    def test_latent_consistency_is_symmetric_and_trains_both_views(self):
        first = torch.randn(3, 12, 8, device="cuda", requires_grad=True)
        second = torch.randn(3, 12, 8, device="cuda", requires_grad=True)
        a = symmetric_latent_consistency(first, second, self.b["valid"])
        b = symmetric_latent_consistency(second, first, self.b["valid"])
        torch.testing.assert_close(a, b, atol=0., rtol=0.)
        self.assertTrue(((a >= 0) & (a <= 2)).all())
        a.mean().backward()
        self.assertGreater(first.grad.abs().sum().item(), 0.)
        self.assertGreater(second.grad.abs().sum().item(), 0.)
        self.assertEqual(first.grad[~self.b["valid"]].count_nonzero().item(), 0)
        zero = symmetric_latent_consistency(first.detach(), first.detach(), self.b["valid"])
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=2e-7, rtol=0.)

    def test_two_view_terms_share_two_encodes_and_reconstruction_anchors_alignment(self):
        model = make_model("CONS_PRE").cuda()
        with patch.object(model, "encode", wraps=model.encode) as encoder:
            terms = model.pretraining_terms(self.b)
        self.assertEqual(encoder.call_count, 2)
        torch.testing.assert_close(terms["total"], terms["reconstruction"]+LATENT_WEIGHT*terms["consistency"])
        self.assertTrue((terms["reconstruction"] > 0).all())
        # Both independent masks are observable in the two encoder inputs.
        self.assertFalse(torch.equal(encoder.call_args_list[0].args[0], encoder.call_args_list[1].args[0]))

    def test_prediction_consistency_stops_original_target_gradient(self):
        model = make_model("CONS_GMAE").cuda()
        target = torch.full(self.b["valid"].shape, .2, device="cuda", requires_grad=True)
        regularizer = model.supervised_regularizer(self.b, target)
        self.assertTrue((regularizer > 0).all())
        regularizer.mean().backward()
        self.assertIsNone(target.grad)
        self.assertGreater(model.readout.weight.grad.abs().sum().item(), 0.)

    def test_slot_permutation_equivariance_and_all_padding(self):
        order = torch.randperm(12, device="cuda")
        permuted = dict(self.b, x=self.b["x"][:, order], valid=self.b["valid"][:, order],
                        adj=self.b["adj"][:, order][:, :, order])
        for key in KEYS:
            with self.subTest(key=key):
                model = make_model(key).cuda()
                torch.testing.assert_close(infer(model, permuted), infer(model, self.b)[:, order], atol=2e-6, rtol=2e-6)
                empty = {k:v.clone() for k,v in self.b.items()}; empty["valid"].zero_()
                q = infer(model, empty)
                self.assertTrue(torch.isfinite(q).all())
                self.assertEqual(q.count_nonzero().item(), 0)
                self.assertEqual(model.pretraining_loss(empty).count_nonzero().item(), 0)
                self.assertEqual(model.supervised_regularizer(empty, q).count_nonzero().item(), 0)

    def test_cpu_research_fails_instead_of_falling_back(self):
        cpu = {k:v.cpu() for k,v in self.b.items()}
        for key in KEYS:
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "CUDA"):
                infer(make_model(key), cpu)


if __name__ == "__main__":
    unittest.main()
