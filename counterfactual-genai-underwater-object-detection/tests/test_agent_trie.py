"""Finite legal JSON enumeration, per-state trie routing and terminal EOS safety."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uac.agent import parse_action,REASONS
from uac.agent_batch import legal_action_texts,ActionTokenTrie,BatchedActionConstraints

class Tokenizer:
    eos_token_id=1;pad_token_id=0;all_special_ids=[0,1]
    def encode(self,text,add_special_tokens=False):return [ord(c)+10 for c in text]
    def decode(self,tokens,**kwargs):return ''.join(chr(token-10) for token in tokens)


def state():
    return {'candidates':[{'candidate_id':'candidate0','interval_seconds':[.75,2.25],'base_probability':.5,'uncertainty':1.}],
      'observations':{'candidate0':{'stft':{'detector_logit':0.},'wavelet':{'detector_logit':.2}}},
      'history':[{'action':'wavelet','candidate_id':'candidate0','valid':True}],
      'available_tools':['stft','fourier','wavelet','verify','stop'],'remaining_actions':2,'remaining_verifications':1}

class TrieTests(unittest.TestCase):
    def test_observed_and_historical_tools_excluded_even_if_available(self):
        texts=legal_action_texts(state());actions=[json.loads(t) for t in texts]
        self.assertEqual({a['action'] for a in actions},{'fourier','verify','stop'})
        self.assertEqual(len(actions),3*len(REASONS))
        self.assertTrue(all(parse_action(t,state())['valid'] for t in texts))
        self.assertTrue(all(a['candidate_id'] is None for a in actions if a['action']=='stop'))
    def test_verified_valid_and_invalid_states_cannot_repeat_verify(self):
        for valid in (True,False):
            s=state();s['observations']['candidate0']['verify']={'valid':valid};s['remaining_verifications']=0
            s['history'].append({'action':'verify','candidate_id':'candidate0','valid':True});s['remaining_actions']=1
            self.assertEqual({json.loads(t)['action'] for t in legal_action_texts(s)},{'fourier','stop'})
    def test_budget_zero_leaves_only_stop_with_actual_null(self):
        s=state();s['remaining_actions']=0
        self.assertTrue(all(json.loads(t)['action']=='stop' and json.loads(t)['candidate_id'] is None for t in legal_action_texts(s)))
    def test_per_candidate_observations_filter_locally(self):
        s=state();second=copy.deepcopy(s['candidates'][0]);second['candidate_id']='candidate1';s['candidates'].append(second)
        s['observations']['candidate1']={'fourier':{'value':1.}}
        actions=[json.loads(t) for t in legal_action_texts(s)]
        self.assertTrue(any(a['candidate_id']=='candidate1' and a['action']=='stft' for a in actions))
        self.assertFalse(any(a['candidate_id']=='candidate0' and a['action']=='stft' for a in actions))
        self.assertFalse(any(a['candidate_id']=='candidate1' and a['action']=='fourier' for a in actions))
    def test_all_completions_reachable_and_eos_required(self):
        tokenizer=Tokenizer();texts=legal_action_texts(state());trie=ActionTokenTrie(texts,tokenizer,max_output_tokens=128)
        for text in texts:
            tokens=tokenizer.encode(text);prefix=[]
            for token in tokens+[1]:
                self.assertIn(token,trie.allowed(prefix));prefix.append(token)
            self.assertFalse(trie.complete(tokens));self.assertTrue(trie.complete(prefix))
            self.assertEqual(trie.allowed(prefix),[0]);self.assertEqual(trie.allowed(prefix+[0,0]),[0])
            self.assertFalse(trie.complete(prefix+[99]))
    def test_shared_padded_width_and_row_specific_constraints(self):
        first=state();second=state();second['available_tools']=['stop']
        tokenizer=Tokenizer();batch=BatchedActionConstraints([first,second],tokenizer,prompt_width=7,max_output_tokens=128)
        text=next(t for t in legal_action_texts(first) if json.loads(t)['action']=='fourier')
        prefix=[0,0,42,43,44,45,46]+tokenizer.encode(text)
        self.assertEqual(batch(0,prefix),[1])
        with self.assertRaises(ValueError):batch(1,prefix)
        self.assertEqual(batch(0,prefix+[1,0]),[0])
        with self.assertRaises(ValueError):batch(2,[0]*7)
    def test_output_budget_includes_terminal_eos(self):
        tokenizer=Tokenizer();text=legal_action_texts(state())[0];length=len(tokenizer.encode(text))
        with self.assertRaises(ValueError):ActionTokenTrie([text],tokenizer,max_output_tokens=length)
        ActionTokenTrie([text],tokenizer,max_output_tokens=length+1)
    def test_special_tokens_and_roundtrip_fail_closed(self):
        tokenizer=Tokenizer();tokenizer.all_special_ids=[0,1,ord('{')+10]
        with self.assertRaises(ValueError):ActionTokenTrie(legal_action_texts(state()),tokenizer,max_output_tokens=128)
        class BadRoundtrip(Tokenizer):
            def decode(self,tokens,**kwargs):return 'wrong'
        with self.assertRaises(ValueError):ActionTokenTrie(legal_action_texts(state()),BadRoundtrip(),max_output_tokens=128)

if __name__=='__main__':unittest.main()
