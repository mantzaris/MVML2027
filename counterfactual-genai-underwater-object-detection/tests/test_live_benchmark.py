"""Metadata and interruption safety of live benchmarking; no CUDA model execution."""
import json
import ast
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_live import planned_config, valid_receipt, commit, controller_file_hashes, equivalence_observation, heuristic_query_mask, heuristic_case
from uac.runtime import EventLedger, sha256_file

class LiveBenchmarkTests(unittest.TestCase):
    def setUp(self):
        directory=ROOT/'.local'/'benchmark_tests';directory.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=directory);self.path=Path(self.temp.name)
        self.ledger=EventLedger(self.path/'events.jsonl')
    def tearDown(self):self.temp.cleanup()
    def test_cohort_is_canonical_calibration_block_and_frozen(self):
        config=planned_config();saved=json.loads((ROOT/'configs/runtime_benchmark.json').read_text())
        self.assertEqual(config,saved);self.assertEqual(config['cohort_start']%32,0)
        self.assertEqual(len(config['cohort_ids']),32);self.assertEqual(len(set(config['cohort_ids'])),32)
        rows=[json.loads(line) for line in (ROOT/'data/manifests/planned_real.jsonl').read_text().splitlines()]
        calibration=[r for r in rows if r['split']=='calibration'];start=config['cohort_start']
        self.assertEqual(config['cohort_ids'],[r['id'] for r in calibration[start:start+32]])
    def test_actual_controller_weight_and_tokenizer_changes_change_digest(self):
        directory=self.path/'controller';directory.mkdir()
        for name in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json'):
            (directory/name).write_text('{}')
        (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'fixture':'model.safetensors'}}))
        weight=directory/'model.safetensors';weight.write_bytes(b'fixture-v1')
        first=controller_file_hashes(directory);weight.write_bytes(b'fixture-v2')
        second=controller_file_hashes(directory)
        self.assertNotEqual(first,second)
        (directory/'tokenizer.json').write_text('{"fixture":"changed"}')
        self.assertNotEqual(second,controller_file_hashes(directory))
    def test_parity_checks_penalty_and_final_calibrated_probability(self):
        identities=['fixture-'+str(i) for i in range(32)]
        records=[{'id':identity,'trace':[{'action':'verify','valid':True}],
          'views':['stft'],'verified':True,'intervention_valid':True,'base_logit':.3,
          'delta':[.1,.2],'penalty':.01} for identity in identities]
        path=self.path/'results/raw/policies/calibration/agent_seed17/all.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'records':records}));(path.parent/'index.json').write_text(json.dumps({'status':'complete','sha256':sha256_file(path)}))
        prediction=self.path/'results/raw/real_predictions_calibration_seed17.pt';prediction.write_bytes(b'fixture-mocked-loader')
        calibration=self.path/'manifests/calibration.json';calibration.parent.mkdir()
        calibration.write_text(json.dumps({'seeds':{'17':{'thresholds':{'proposed_agent':{'threshold':.5}}}}}))
        result={'ids':identities,'policy_records':json.loads(json.dumps(records)),'probabilities':[.2]*32,'decisions':[False]*32}
        archived={'ids':identities,'probabilities':{'proposed_agent':torch.tensor([.2]*32)}}
        with mock.patch('benchmark_live.PROJECT',self.path),mock.patch('benchmark_live.torch.load',return_value=archived):
            self.assertEqual(equivalence_observation('proposed_agent',result,{'cohort_start':0})['status'],'matched')
            result['policy_records'][0]['penalty']=.02
            audit=equivalence_observation('proposed_agent',result,{'cohort_start':0})
            self.assertEqual(audit['status'],'numerical_or_policy_difference');self.assertGreater(audit['max_penalty_abs_error'],.009)
            result['policy_records'][0]['penalty']=.01;result['probabilities'][0]=.3
            audit=equivalence_observation('proposed_agent',result,{'cohort_start':0})
            self.assertEqual(audit['status'],'numerical_or_policy_difference');self.assertGreater(audit['max_calibrated_probability_abs_error'],.09)
    def test_heuristic_queries_and_invalid_fallback_match_frozen_rule(self):
        stft=torch.tensor([-5.,0.,5.,0.,-5.,0.,5.,-5.])
        uncertain=heuristic_query_mask(stft)
        self.assertEqual(uncertain.tolist(),[False,True,False,True,False,True,False,False])
        component={'delta':torch.ones(8,4,3),'penalty':torch.zeros(8),
          'valid':torch.tensor([True,False,True,True,True,True,True,True])}
        case=heuristic_case(torch.full((8,),9.),stft,uncertain,{0:component},{'dispersion':.5,'penalty':.1})
        self.assertEqual(case['base_names'],['stft','main','stft','main','stft','main','stft','stft'])
        self.assertEqual(case['base'].tolist(),[-5.,9.,5.,9.,-5.,9.,5.,-5.])
        self.assertEqual(case['verify'].tolist(),[False,False,False,True,False,True,False,False])
        self.assertEqual(case['v'].tolist(),[0.,0.,0.,1.,0.,1.,0.,0.])
        # Valid unqueried companions cannot turn verification on; an invalid
        # queried clip still falls back to its main detector, not STFT.
    def test_timed_decision_keeps_float64_nextafter_threshold(self):
        # Execute the actual benchmark decision expression on two tiny CPU
        # values, independent of all model/GPU work. A naive FP32 comparison
        # rounds both nextafter boundaries down and fails these expectations.
        tree=ast.parse((ROOT/'scripts/benchmark_live.py').read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='timed_pass')
        assignment=next(node for node in ast.walk(function) if isinstance(node,ast.Assign)
          and any(isinstance(target,ast.Name) and target.id=='decisions' for target in node.targets))
        expression=compile(ast.Expression(assignment.value),'<timed_pass decision>','eval')
        probability=torch.tensor([.5,1.],dtype=torch.float32)
        for boundary in (.5,1.):
            threshold=math.nextafter(boundary,math.inf)
            with self.subTest(boundary=boundary):
                expected=[float(value)>=threshold for value in probability.tolist()]
                actual=eval(expression,{'torch':torch,'prob':probability,
                  'fit':{'thresholds':{'fixture':{'threshold':threshold}}},'method':'fixture'})
                self.assertEqual(actual,expected)
                self.assertNotEqual((probability>=threshold).tolist(),expected)
    def test_committed_batch_survives_restart(self):
        path=self.path/'batch.json';payload={'fixture':'not_research_results','step':2}
        commit(path,payload,'fixturehash')
        self.assertEqual(valid_receipt(path,'fixturehash',self.ledger),payload)
    def test_torn_payload_is_preserved_and_replayed(self):
        path=self.path/'batch.json';commit(path,{'fixture':True},'fixturehash');path.write_text('{')
        self.assertIsNone(valid_receipt(path,'fixturehash',self.ledger))
        self.assertFalse(path.exists());self.assertEqual(len(list(self.path.glob('*.corrupt-*'))),2)
    def test_power_loss_before_receipt_replays_uncommitted_unit(self):
        path=self.path/'batch.json';path.write_text('{"fixture": true}')
        self.assertIsNone(valid_receipt(path,'fixturehash',self.ledger));self.assertFalse(path.exists())
    def test_incompatible_model_config_fails_without_overwriting(self):
        path=self.path/'batch.json';commit(path,{'fixture':True},'old')
        with self.assertRaises(RuntimeError):valid_receipt(path,'new',self.ledger)
        self.assertTrue(path.exists())

if __name__=='__main__':unittest.main()
