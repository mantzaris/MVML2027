"""Bounded Qwen2.5-7B-Instruct planner adaptation, two calls maximum."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
from underwater_reasoning.runtime import budget_check
from run_generation import serialize_evidence
ROOT=Path(__file__).resolve().parents[1];REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['development','calibration','test'],required=True);p.add_argument('--batch-size',type=int,default=16);a=p.parse_args()
    if a.partition=='test' and not json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise ValueError('Test locked')
    torch.set_num_threads(8);path=(ROOT/'.cache/planner_path.txt').read_text().strip();tokenizer=AutoTokenizer.from_pretrained(path,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());prompt=(ROOT/'prompts/planner.txt').read_text();items=[];started=time.monotonic()
    class TokenCounter:
        def tokens(self,text):return len(tokenizer.encode(text,add_special_tokens=False))
    counter=TokenCounter()
    for uid in manifest['assessments'][a.partition]:
        out=ROOT/f'results/raw/planner/{a.partition}/{uid}.json'
        asset=json.loads((ROOT/f'data/derived/evidence/{uid}.json').read_text())
        key={'revision':REVISION,'prompt_sha256':canonical_hash(prompt),'waveform_sha256':asset['waveform_sha256'],
            'observations_sha256':canonical_hash([asset['initial_state'],asset['prior'],asset['compact']]),
            'family_token_cap':256,'text_token_cap':4096,'batch_size':a.batch_size,'max_new_tokens':32,'greedy':True,'precision':'bfloat16'}
        if out.exists():
            if json.loads(out.read_text()).get('key')!=key:raise ValueError('Planner cache content key mismatch')
            continue
        items.append({'id':uid,'asset':asset,'key':key,'out':out,'actions':[],'mask':0,'calls':[],'stopped':False})
    for start in range(0,len(items),a.batch_size):
        batch=items[start:start+a.batch_size]
        for step in range(2):
            budget_check();active=[i for i in batch if not i['stopped']];texts=[]
            if not active:break
            for item in active:
                evidence,counts,trims=serialize_evidence(counter,item['asset'],item['mask'])
                observed=[json.loads(line) for line in evidence.splitlines()];initial=item['asset']['initial_state']
                payload={'prior':item['asset']['prior'],'initial_observation':{'entropy':initial[5],'margin':initial[6],
                    'embedding_pca':initial[7:39],'digital_energy':initial[39],'zero_fraction':initial[40],'clipping_fraction':initial[41]},
                    'acquired_evidence':observed,'used_actions':['ABCD'[j] for j in item['actions']],'remaining_budget':2-len(item['actions'])}
                texts.append(tokenizer.apply_chat_template([{'role':'system','content':prompt},{'role':'user','content':json.dumps(payload,separators=(',',':'))}],tokenize=False,add_generation_prompt=True))
                if counter.tokens(texts[-1])>4096:raise ValueError('Planner text budget exceeded')
                item['step_family_tokens']=counts;item['step_family_trimmed']=trims
            inputs=tokenizer(texts,return_tensors='pt',padding=True).to(model.device);torch.cuda.synchronize();t=time.monotonic()
            with torch.inference_mode():outputs=model.generate(**inputs,max_new_tokens=32,do_sample=False,pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize();elapsed=time.monotonic()-t;raw=tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            for item,text in zip(active,raw):
                try:
                    cleaned=text.strip()
                    if cleaned.startswith('```'):cleaned=cleaned[cleaned.find('\n')+1:].rsplit('```',1)[0]
                    action=json.loads(cleaned).get('action');valid=action in ['A','B','C','D','STOP']
                except (ValueError,AttributeError):action=None;valid=False
                item['calls'].append({'raw':text,'valid':valid,'batch_seconds':elapsed,'amortized_seconds':elapsed/len(active),'step':step,
                    'input_tokens':len(tokenizer.encode(texts[active.index(item)])),'output_tokens':len(tokenizer.encode(text,add_special_tokens=False)),
                    'family_tokens':item['step_family_tokens'],'family_trimmed':item['step_family_trimmed']})
                if not valid or action=='STOP':item['stopped']=True;continue
                index='ABCD'.index(action)
                if index in item['actions']:item['stopped']=True;item['calls'][-1]['duplicate_action']=True;continue
                item['actions'].append(index);item['mask']|=1<<index
        for item in batch:
            atomic_json(item['out'],{'id':item['id'],'key':item['key'],'mask':item['mask'],'actions':item['actions'],'calls':item['calls'],
                'model':'Qwen/Qwen2.5-7B-Instruct','revision':REVISION,'precision':'bfloat16','prompt_sha256':file_hash(ROOT/'prompts/planner.txt'),
                'waveform_sha256':item['asset']['waveform_sha256'],'peak_vram_bytes':torch.cuda.max_memory_allocated(),'is_external_agent_reproduction':False})
        print('planner',a.partition,start+len(batch),len(items),flush=True)
    atomic_json(ROOT/f'results/manifests/planner_{a.partition}.json',{'clips':len(items),'seconds':time.monotonic()-started,'max_planner_calls_per_clip':2,'terminal_calls_additional':1,'model_revision':REVISION})
