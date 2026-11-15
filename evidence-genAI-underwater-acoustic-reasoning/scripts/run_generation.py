"""Content-keyed terminal or blinded AI review inference; one call per record."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import soundfile as sf
from underwater_reasoning.omni import Omni,REVISION
from underwater_reasoning.reports import parse_report
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
from underwater_reasoning.runtime import budget_check
ROOT=Path(__file__).resolve().parents[1]

def serialize_evidence(runner,assets,mask):
    rendered=[];counts={};trimmed={}
    for a,family in enumerate('ABCD'):
        if not mask&(1<<a):continue
        card=json.loads(json.dumps(assets['compact'][family]));text=json.dumps(card,separators=(',',':'));trimmed[family]=False
        while runner.tokens(text)>256:
            field='tones' if family=='A' else 'modulation' if family=='B' else 'neighbors' if family=='D' else None
            if field and len(card[field])>1:card[field].pop();trimmed[family]=True
            else:raise ValueError(f'Evidence family {family} exceeds 256 tokens')
            text=json.dumps(card,separators=(',',':'))
        counts[family]=runner.tokens(text);rendered.append(text)
    return '\n'.join(rendered),counts,trimmed

def review_contract():
    return '''Return only JSON with keys band, tone, modulation, trend. Each is an object with identifiable (boolean), value (allowed value or null), time_s (two numbers), and note (one short sentence). Tone also has frequency_hz (number or null); modulation also has rate_hz (number or null). Band options are "20-100","100-500","500-2000","2000-8000". Tone and modulation options are "present" or null; no detection does not prove absence. Trend options are "increase","decrease","steady" or null. Do not infer class, vessel identity, mechanism, source motion, or human validation. Treat independent numeric diagnostics as fallible. Unresolved attributes have identifiable=false and value=null. You have not seen another reviewer or the report generator's tools or outputs.'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['policy_fit','development','calibration','test','all'],required=True)
    p.add_argument('--mode',choices=['terminal','review'],default='terminal');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--limit',type=int);p.add_argument('--masks',default='all');p.add_argument('--tag',default='main');a=p.parse_args()
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());lookup={r['id']:r for r in manifest['records']}
    parts=['policy_fit','development','calibration','test'] if a.partition=='all' else [a.partition]
    if a.mode=='terminal' and 'test' in parts:
        unlock=json.loads((ROOT/'protocol/test_unlock_log.json').read_text())
        if not unlock.get('unlocked',False):raise ValueError('Test terminal evaluation is locked')
    prompt=(ROOT/'prompts/terminal_v1.txt').read_text();runner=Omni(ROOT);jobs=[];started=time.monotonic()
    masks=list(range(16))+[-1] if a.masks=='all' else [int(v) for v in a.masks.split(',')]
    for part in parts:
        ids=manifest['assessments'][part][:a.limit]
        for uid in ids:
            if a.mode=='terminal':assets=json.loads((ROOT/f'data/derived/evidence/{uid}.json').read_text())
            else:assets=json.loads((ROOT/f'data/derived/references/{uid}.json').read_text())
            for condition in (masks if a.mode=='terminal' else [0,1]):
                extra={}
                if a.mode=='terminal':
                    evidence,counts,trims=serialize_evidence(runner,assets,max(condition,0));prior='' if condition==-1 else json.dumps(assets['prior'],separators=(',',':'))
                    text=prompt+'\nAvailable evidence: W0 raw audio.\n'+prior+'\n'+evidence+'\nProduce the required JSON report.'
                    extra={'mask':condition,'family_tokens':counts,'family_trimmed':trims,'classifier_sha256':assets['classifier_sha256']}
                else:
                    role='Assess measurable acoustic attributes carefully.' if condition==0 else 'Act as a skeptical second AI assessor. Reject over-specific claims and prefer uncertainty when evidence is weak.'
                    text='You are an AI acoustic reviewer, not a human. '+role+'\n'+review_contract()+'\nIndependent diagnostics:\n'+json.dumps(assets['reference'],separators=(',',':'))
                    extra={'reviewer_role':condition,'reference_sha256':canonical_hash(assets['reference'])}
                key={'waveform_sha256':assets['waveform_sha256'],'model':'Qwen/Qwen2.5-Omni-7B','revision':REVISION,'precision':'bfloat16','attention':'sdpa',
                    'prompt_sha256':canonical_hash(text),'mode':a.mode,'condition':condition,'decoding':{'greedy':True,'max_new_tokens':768,'eos_token_id':151645},'batch_size':a.batch_size,**extra}
                out=ROOT/f'results/raw/{a.mode}/{a.tag}/{part}/{uid}_{condition}.json'
                if out.exists():
                    old=json.loads(out.read_text())
                    if old['key']!=key:raise ValueError('Existing result belongs to a different content key; use a new tag')
                    continue
                jobs.append({'id':uid,'partition':part,'text':text,'key':key,'out':out})
    for start in range(0,len(jobs),a.batch_size):
        budget_check();batch=jobs[start:start+a.batch_size]
        for job in batch:job['audio'],job['sample_rate']=sf.read(ROOT/lookup[job['id']]['audio_path'],dtype='float32')
        results=runner.generate(batch)
        for job,result in zip(batch,results):
            if a.mode=='terminal':parsed,errors=parse_report(result['raw'])
            else:
                try:
                    raw=result['raw'].strip()
                    if raw.startswith('```'):raw=raw[raw.find('\n')+1:].rsplit('```',1)[0].strip()
                    parsed=json.loads(raw);errors=[]
                    for slot in ('band','tone','modulation','trend'):
                        if not isinstance(parsed.get(slot),dict) or not isinstance(parsed[slot].get('identifiable'),bool):errors.append(slot+'_contract')
                except (ValueError,TypeError,AttributeError):parsed=None;errors=['review_json_parse']
            atomic_json(job['out'],{'id':job['id'],'partition':job['partition'],'key':job['key'],'result':{**result,'parsed':parsed,'errors':errors}})
            del job['audio']
        print(a.mode,a.partition,start+len(batch),'/',len(jobs),'seconds',round(time.monotonic()-started),flush=True)
    atomic_json(ROOT/f'results/manifests/generation_{a.mode}_{a.tag}_{a.partition}.json',{'new_reports':len(jobs),'seconds':time.monotonic()-started,
        'batch_size':a.batch_size,'mode':a.mode,'partitions':parts,'tag':a.tag,'prompt_sha256':file_hash(ROOT/'prompts/terminal_v1.txt'),'reviewer_human_count':0})

if __name__=='__main__':main()
