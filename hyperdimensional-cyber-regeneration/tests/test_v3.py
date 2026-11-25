"""CUDA scientific checks for direct candidates and cheap reconstruction."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.regen import generate
from src.regen_v2 import independent_probability
from src.regen_v3 import (KEYS, CANDIDATE_KEYS, make_model, infer, loss_terms,
                          observable_features, canonical_requirements)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CandidateAndCheapModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(707)
        self.batch = generate(8, 72027, "cuda", facts=16)

    def test_all_methods_finite_outputs_and_supervised_gradients(self):
        for key in KEYS:
            with self.subTest(key=key):
                model = make_model(key).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p, q = infer(model, self.batch)
                self.assertEqual(q.shape, (8, 32))
                self.assertEqual(q.dtype, torch.float32)
                self.assertTrue(torch.isfinite(q).all())
                self.assertTrue(((q >= 0) & (q <= 1)).all())
                self.assertEqual(p is None, key in CANDIDATE_KEYS)
                terms = loss_terms(model, p, q, self.batch)
                self.assertEqual(terms["total"].shape, (8,))
                terms["total"].mean().backward()
                grads = [v.grad for v in model.parameters() if v.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.)
                if key == "SCR":
                    self.assertGreater(model.gate[-1].weight.grad.abs().sum().item(), 0.)

    def test_inference_reads_only_observable_fields(self):
        changed = dict(self.batch, y=1-self.batch["y"],
                       previous_y=1-self.batch["previous_y"],
                       mask=~self.batch["mask"], transitions=~self.batch["transitions"],
                       group=self.batch["group"] + 100,
                       observation_group=torch.full((8,), 99, device="cuda"),
                       values=self.batch["values"] + 10., costs=self.batch["costs"] + 10.)
        observable = {k:self.batch[k] for k in ("x", "parent", "req")}
        for key in KEYS:
            with self.subTest(key=key), torch.no_grad():
                model = make_model(key).cuda().eval()
                first = infer(model, self.batch)
                for other in (changed, observable):
                    second = infer(model, other)
                    self.assertTrue(torch.equal(first[1], second[1]))
                    if first[0] is not None:
                        self.assertTrue(torch.equal(first[0], second[0]))

    def test_candidate_permutation_equivariance(self):
        permutation = torch.randperm(32, device="cuda")
        changed = dict(self.batch, req=self.batch["req"][:, permutation])
        for key in KEYS:
            with self.subTest(key=key), torch.no_grad():
                model = make_model(key).cuda().eval()
                p, q = infer(model, self.batch)
                p2, q2 = infer(model, changed)
                torch.testing.assert_close(q2, q[:, permutation], atol=1e-6, rtol=1e-6)
                if p is not None:
                    torch.testing.assert_close(p2, p, atol=0., rtol=0.)

    def test_joint_fact_renumbering(self):
        permutation = torch.randperm(16, device="cuda")
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(16, device="cuda")
        changed = dict(self.batch, x=self.batch["x"][:, permutation],
                       parent=inverse[self.batch["parent"][:, permutation]],
                       req=self.batch["req"][:, :, permutation])
        for key in KEYS:
            with self.subTest(key=key), torch.no_grad():
                model = make_model(key).cuda().eval()
                p, q = infer(model, self.batch)
                p2, q2 = infer(model, changed)
                torch.testing.assert_close(q2, q, atol=2e-6, rtol=2e-6)
                if p is not None:
                    torch.testing.assert_close(p2, p[:, permutation], atol=2e-6, rtol=2e-6)

    def test_canonical_ancestor_redundancy_and_empty_conjunction(self):
        parent = torch.tensor([[0, 1, 0, 0, 1, 1, 0, 1]], device="cuda")
        req = torch.zeros(1, 5, 8, device="cuda")
        req[0, 0, 2] = 1
        req[0, 1, [0, 2]] = 1
        req[0, 2, [2, 3]] = 1
        req[0, 3, [0, 2, 3]] = 1
        x = torch.rand(1, 8, 14, device="cuda")
        children, roots = canonical_requirements(req, parent)
        self.assertTrue(torch.equal(children[:, 0], children[:, 1]))
        self.assertTrue(torch.equal(roots[:, 0], roots[:, 1]))
        self.assertEqual(roots[0, 2].sum().item(), 1.)
        for key in CANDIDATE_KEYS:
            with self.subTest(key=key), torch.no_grad():
                model = make_model(key).cuda().eval()
                q = model(x, parent, req)
                self.assertEqual(q[0, 0].item(), q[0, 1].item())
                self.assertEqual(q[0, 2].item(), q[0, 3].item())
                self.assertEqual(q[0, 4].item(), 1.)
        for key in ("LOGIT", "DAE"):
            with self.subTest(key=key), torch.no_grad():
                _, q = infer(make_model(key).cuda(), dict(x=x, parent=parent, req=req))
                self.assertEqual(q[0, 4].item(), 1.)

    def test_candidate_context_weights_and_matched_controls(self):
        model = make_model("SCR").cuda()
        weights = model.context_weights(self.batch)
        self.assertEqual(weights.shape, (8, 4))
        torch.testing.assert_close(weights.sum(-1), torch.ones(8, device="cuda"))
        self.assertTrue((weights >= 0).all())
        context = make_model("CAND_CTX").cuda()
        local = make_model("CAND_LOCAL").cuda()
        self.assertEqual(sum(p.numel() for p in context.parameters()),
                         sum(p.numel() for p in local.parameters()))
        # Identical weights and removing global context is the only difference.
        local.load_state_dict(context.state_dict())
        _, local_context = local._encode(self.batch["x"], self.batch["parent"])
        self.assertEqual(torch.count_nonzero(local_context).item(), 0)

    def test_dae_pretraining_reconstructs_observed_features_without_truth(self):
        model = make_model("DAE").cuda()
        observed = {k:self.batch[k] for k in ("x", "parent")}
        self.assertEqual(observable_features(**observed).shape, (8, 16, 56))
        torch.manual_seed(188)
        first = model.pretraining_loss(self.batch)
        torch.manual_seed(188)
        second = model.pretraining_loss(observed)
        torch.testing.assert_close(first, second, atol=0., rtol=0.)
        self.assertEqual(first.shape, (8,))
        first.mean().backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.decoder[-1].weight.grad.abs().sum().item(), 0.)
        self.assertIsNone(model.readout.weight.grad)

    def test_cheap_methods_use_independent_candidate_product(self):
        for key in ("LOGIT", "DAE"):
            with self.subTest(key=key), torch.no_grad():
                p, q = infer(make_model(key).cuda(), self.batch)
                torch.testing.assert_close(q, independent_probability(p, self.batch["req"]))
        self.assertEqual(sum(p.numel() for p in make_model("LOGIT").parameters()), 57)

    def test_same_models_support_all_study_sizes(self):
        models = {key:make_model(key).cuda().eval() for key in KEYS}
        with torch.no_grad():
            for facts in (64, 128, 256):
                batch = generate(2, 72028, "cuda", facts=facts)
                for key, model in models.items():
                    with self.subTest(key=key, facts=facts):
                        p, q = infer(model, batch)
                        self.assertEqual(q.shape, (2, 32))
                        self.assertTrue(torch.isfinite(q).all())
                        if p is not None:
                            self.assertEqual(p.shape, (2, facts))

    def test_cpu_research_inference_fails_clearly(self):
        observable = {k:self.batch[k].cpu() for k in ("x", "parent", "req")}
        for key in KEYS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    infer(make_model(key), observable)


if __name__ == "__main__":
    unittest.main()
