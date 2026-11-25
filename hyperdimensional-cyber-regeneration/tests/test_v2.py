"""CUDA scientific checks for dependency-aware full regeneration."""
import itertools
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.regen import generate
from src.regen_v2 import (RegeneratorV2, infer, marginal_probabilities,
                          joint_probability, independent_probability, loss_terms, consistency_penalty)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DependencyRegenerationTests(unittest.TestCase):
    def setUp(self):
        self.parent = torch.tensor([[0, 1, 0, 0]], device="cuda")
        self.factors = torch.tensor([[.6, .4, .8, .7]], device="cuda")
        self.requirements = torch.tensor([[[0., 0., 1., 1.], [1., 0., 1., 1.],
                                          [0., 1., 1., 0.], [0., 0., 0., 0.]]], device="cuda")

    def test_joint_probability_matches_enumerated_valid_worlds(self):
        worlds = torch.tensor(list(itertools.product([0., 1.], repeat=4)), device="cuda")
        roots = worlds[:, :2]
        children = worlds[:, 2:]
        root_mass = torch.where(roots.bool(), self.factors[0, :2], 1-self.factors[0, :2]).prod(-1)
        child_mass = torch.where(children.bool(), self.factors[0, 2:], 1-self.factors[0, 2:]).prod(-1)
        valid = (children <= roots[:, :1]).all(-1)
        # Conditional children are deterministically zero when root zero.
        mass = root_mass * torch.where(roots[:, 0].bool(), child_mass,
                                        torch.ones_like(child_mass)) * valid
        self.assertAlmostEqual(mass.sum().item(), 1., places=6)
        feasible = ((1-self.requirements[0, None]) + self.requirements[0, None] * worlds[:, None]).prod(-1)
        enumerated = (feasible * mass[:, None]).sum(0)
        actual = joint_probability(self.factors, self.requirements, self.parent)[0]
        torch.testing.assert_close(actual, enumerated, atol=1e-6, rtol=1e-6)

    def test_shared_root_is_counted_once(self):
        actual = joint_probability(self.factors, self.requirements, self.parent)[0]
        self.assertAlmostEqual(actual[0].item(), .6*.8*.7, places=6)
        self.assertAlmostEqual(actual[1].item(), actual[0].item(), places=6)
        marginals = marginal_probabilities(self.factors, self.parent)
        old = independent_probability(marginals, self.requirements)[0]
        self.assertAlmostEqual(old[0].item(), (.6*.8)*(.6*.7), places=6)
        self.assertGreater(actual[0].item(), old[0].item())

    def test_coherent_marginals_and_hard_semantics(self):
        p = marginal_probabilities(self.factors, self.parent)
        self.assertTrue((p <= p.gather(1, self.parent)).all())
        for bits in itertools.product([0., 1.], repeat=4):
            f = torch.tensor([bits], device="cuda")
            hard = marginal_probabilities(f, self.parent)
            actual = joint_probability(f, self.requirements, self.parent)
            expected = independent_probability(hard, self.requirements)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual[0, -1].item(), 1.)

    def test_consistency_penalty_respects_joint_semantics(self):
        p = marginal_probabilities(self.factors, self.parent)
        self.assertEqual(consistency_penalty(p, self.parent).item(), 0.)
        violated = torch.tensor([[.2, .4, .8, .5]], device="cuda")
        self.assertGreater(consistency_penalty(violated, self.parent).item(), 0.)

    def test_all_variants_have_finite_nonzero_gradients(self):
        batch = generate(8, 7817, "cuda", facts=16)
        for variant in ["JH", "JD", "IH", "ID", "JH_ERM"]:
            with self.subTest(variant=variant):
                model = RegeneratorV2(variant, hidden=16, dim=128).cuda()
                p, s = infer(model, batch)
                terms = loss_terms(p, s, batch)
                self.assertEqual(terms["total"].shape, (8,))
                terms["total"].mean().backward()
                grads = [v.grad for v in model.parameters() if v.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(model.belief.weight.grad.abs().sum().item(), 0.)

    def test_inference_ignores_hidden_labels_and_masks(self):
        batch = generate(8, 8817, "cuda", facts=16)
        for variant in ["JH", "JD", "IH", "ID", "JH_ERM"]:
            model = RegeneratorV2(variant, hidden=16, dim=128).cuda().eval()
            with torch.no_grad():
                model.belief.weight.normal_(0, .1)
                for override in [False, True]:
                    first = infer(model, batch, product_override=override)
                    changed = dict(batch, y=1-batch["y"], previous_y=1-batch["previous_y"],
                                   mask=~batch["mask"], transitions=~batch["transitions"])
                    second = infer(model, changed, product_override=override)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_product_ablation_preserves_marginals(self):
        batch = generate(8, 9817, "cuda", facts=16)
        model = RegeneratorV2("JH", hidden=16, dim=128).cuda().eval()
        with torch.no_grad():
            p, joint = infer(model, batch)
            p2, product = infer(model, batch, product_override=True)
        self.assertTrue(torch.equal(p, p2))
        torch.testing.assert_close(product, independent_probability(p, batch["req"]))
        self.assertTrue((joint + 1e-6 >= product).all())


if __name__ == "__main__":
    unittest.main()
