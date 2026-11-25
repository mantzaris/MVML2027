"""Scientific correctness checks for the CUDA benchmark; run through run_gpu.py."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.regen import (generate,Model,utilities,regret,teacher_delta,topk_gate,
                       apply_gate,semantic_loss,masked_feature_loss,serialize_reports)

@unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.batch=generate(8,1717,"cuda",facts=16)
    def test_worlds_and_candidates_are_paired_across_conditions(self):
        other=generate(8,1717,"cuda",facts=16,corruption=.4,missing=.5,budget=0,attack="coherent")
        for key in ("y","previous_y","parent","adj","req","values","costs","group"):
            self.assertTrue(torch.equal(self.batch[key],other[key]),key)
    def test_inference_cannot_read_hidden_labels(self):
        model=Model(kind="dense",hidden=16).cuda().eval()
        with torch.no_grad():
            first=model(self.batch["x"],self.batch["adj"])[0]
            self.batch["y"]=1-self.batch["y"]
            self.batch["mask"]=~self.batch["mask"]
            second=model(self.batch["x"],self.batch["adj"])[0]
        self.assertTrue(torch.equal(first,second))
    def test_vectorized_teacher_matches_single_fact_interventions(self):
        batch=self.batch
        actual=teacher_delta(batch,tau=.1)
        base=regret(batch["prior"],batch,tau=.1)
        expected=[]
        for fact in range(batch["y"].shape[1]):
            repaired=batch["prior"].clone()
            repaired[:,fact]=batch["y"][:,fact]
            expected.append((base-regret(repaired,batch,tau=.1)).clamp_min(0))
        self.assertTrue(torch.allclose(actual,torch.stack(expected,-1),atol=2e-6,rtol=2e-5))
    def test_gate_budget_and_limiting_cases(self):
        prior=self.batch["prior"]
        repaired=torch.rand_like(prior)
        scores=torch.zeros_like(prior)
        self.assertTrue(torch.equal(apply_gate(prior,repaired,topk_gate(scores,0)),prior))
        self.assertTrue(torch.equal(apply_gate(prior,repaired,topk_gate(scores,16)),repaired))
        self.assertEqual(topk_gate(scores,4,positive_only=True).count_nonzero().item(),0)
        self.assertTrue((topk_gate(torch.randn_like(scores),4).sum(-1)<=4).all())
    def test_generated_truth_satisfies_symbolic_implications(self):
        y=self.batch["y"]
        self.assertTrue((y<=y.gather(1,self.batch["parent"])).all())
        self.assertAlmostEqual(semantic_loss(y,self.batch).item(),0,places=6)
    def test_hard_state_candidate_utility(self):
        ones=torch.ones_like(self.batch["y"])
        zeros=torch.zeros_like(ones)
        self.assertTrue(torch.equal(utilities(ones,self.batch),self.batch["values"]-self.batch["costs"]))
        self.assertTrue(torch.equal(utilities(zeros,self.batch),-self.batch["costs"]))
    def test_payload_budget_is_respected(self):
        for budget in (0,32,64,512):
            batch=generate(8,1717,"cuda",facts=16,budget=budget)
            self.assertTrue((batch["bytes"]<=2*budget).all())
    def test_report_serialization_matches_counted_bytes(self):
        for budget in (0,32,64,512):
            batch=generate(3,1717,"cuda",facts=16,budget=budget)
            for index in range(3):
                payloads=serialize_reports(batch,index)
                self.assertEqual(sum(map(len,payloads.values())),int(batch["bytes"][index]))
                self.assertTrue(all(len(payload)<=budget for payload in payloads.values()))
    def test_pretraining_and_hdc_forward_backward_are_finite(self):
        model=Model(kind="hdc",hidden=16,dim=128).cuda()
        mask=torch.rand_like(self.batch["prior"])<.3
        rec=model.feature_reconstruction(self.batch["x"],self.batch["adj"],mask)
        loss=masked_feature_loss(rec,self.batch["x"],mask)
        logits,gate=model(self.batch["x"],self.batch["adj"])
        loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(logits,self.batch["y"])+gate.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=="__main__":
    unittest.main()
