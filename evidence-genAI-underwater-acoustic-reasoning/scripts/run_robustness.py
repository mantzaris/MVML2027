"""Frozen-policy stress evaluation with fresh AI reviews and terminal reports."""
import json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from underwater_reasoning.omni import Omni,REVISION
from underwater_reasoning.controller import Ensemble,method_mask
from underwater_reasoning.reports import parse_report
from underwater_reasoning.scoring import score,deterministic_report
from underwater_reasoning.io import atomic_json,canonical_hash
from underwater_reasoning.runtime import budget_check
from underwater_reasoning.report_schema import terminal_schema
from run_generation import serialize_evidence,review_contract
from analyze_experiment import metrics,cluster_contrast,classifier_metrics
ROOT=Path(__file__).resolve().parents[1]

def parse_review(raw):
    try:
        text=raw.strip()
        if text.startswith('```'):text=text[text.find('\n')+1:].rsplit('```',1)[0]
        parsed=json.loads(text);errors=[]
        for s in ('band','tone','modulation','trend'):
            if not isinstance(parsed.get(s),dict) or not isinstance(parsed[s].get('identifiable'),bool):errors.append(s+'_contract')
        return parsed,errors
    except (ValueError,AttributeError):return None,['review_json_parse']

def main():
    if not json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise ValueError('Test locked')
    started=time.monotonic();manifest=json.loads((ROOT/'data/private/robustness_manifest.json').read_text());rows=manifest['records'];labels=manifest['labels'];settings=json.loads((ROOT/'configs/locked_controller.json').read_text())
    ensemble=Ensemble(torch.load(ROOT/'checkpoints/controller_loss_17.pt',weights_only=False));runner=Omni(ROOT);prompt=(ROOT/'prompts/terminal_v1.txt').read_text();jobs=[];items=[]
    methods=['P',settings['locked_comparator'],'B5_all','B8_deterministic']
    for row in rows:
        uid=row['id'];asset=json.loads((ROOT/f'data/derived/evidence/{uid}.json').read_text());ref=json.loads((ROOT/f'data/derived/references/{uid}.json').read_text())['reference']
        item={'row':row,'asset':asset,'reference':ref,'choices':{}}
        for method in methods:item['choices'][method]=method_mask(method,item,settings,ensemble,ensemble)
        items.append(item)
        for role in (0,1):
            text='You are an AI acoustic reviewer, not a human. '+('Assess measurable acoustic attributes carefully.' if role==0 else 'Act as a skeptical second AI assessor. Reject over-specific claims and prefer uncertainty when evidence is weak.')+'\n'+review_contract()+'\nIndependent diagnostics:\n'+json.dumps(ref,separators=(',',':'))
            jobs.append({'id':uid,'row':row,'text':text,'mode':'review','condition':role,'out':ROOT/f'results/raw/review/robustness/{uid}_{role}.json','waveform_sha256':asset['waveform_sha256']})
        for mask in sorted({item['choices'][m][0] for m in methods if m!='B8_deterministic'}):
            evidence,counts,trims=serialize_evidence(runner,asset,mask);text=prompt+'\nAvailable evidence: W0 raw audio.\n'+json.dumps(asset['prior'],separators=(',',':'))+'\n'+evidence+'\nProduce the required JSON report.'
            jobs.append({'id':uid,'row':row,'text':text,'mode':'terminal','condition':mask,'out':ROOT/f'results/raw/terminal/robustness/{uid}_{mask}.json','waveform_sha256':asset['waveform_sha256']})
    remaining=[]
    for job in jobs:
        job['key']={'model_revision':REVISION,'waveform_sha256':job['waveform_sha256'],'prompt_sha256':canonical_hash(job['text']),'mode':job['mode'],'condition':job['condition'],'precision':'bfloat16','max_new_tokens':1024 if job['mode']=='terminal' else 768,'greedy':True,'batch_size':16}
        if job['mode']=='terminal':job['key']['format_schema_sha256']=canonical_hash(terminal_schema());job['key']['format_enforcer']='xgrammar==0.2.9'
        if job['out'].exists():
            if json.loads(job['out'].read_text())['key']!=job['key']:raise ValueError('Stress cache mismatch')
        else:remaining.append(job)
    batches=[]
    for mode in ('review','terminal'):
        subset=[j for j in remaining if j['mode']==mode];batches.extend(subset[start:start+16] for start in range(0,len(subset),16))
    completed=0
    for batch in batches:
        budget_check()
        for job in batch:job['audio'],job['sample_rate']=sf.read(ROOT/job['row']['audio_path'],dtype='float32')
        outputs=runner.generate(batch,report_schema=batch[0]['mode']=='terminal')
        for job,result in zip(batch,outputs):
            parsed,errors=parse_report(result['raw']) if job['mode']=='terminal' else parse_review(result['raw'])
            atomic_json(job['out'],{'id':job['id'],'key':job['key'],'result':{**result,'parsed':parsed,'errors':errors}});del job['audio']
        completed+=len(batch);print('stress reports',completed,len(remaining),'seconds',round(time.monotonic()-started),flush=True)
    scored={variant:{m:[] for m in methods} for variant in manifest['variants']};review_valid=0
    for item in items:
        row=item['row'];uid=row['id'];reviews=[]
        for role in (0,1):
            result=json.loads((ROOT/f'results/raw/review/robustness/{uid}_{role}.json').read_text())['result'];reviews.append(result['parsed'] if not result['errors'] else None);review_valid+=not result['errors']
        for method in methods:
            mask,trace=item['choices'][method];risk=ensemble.predict(item['asset'],mask)[2]
            if method=='B8_deterministic':
                report=deterministic_report(item['asset'],mask,labels);errors=[];seconds=0;tokens=0;calls=0
            else:
                result=json.loads((ROOT/f'results/raw/terminal/robustness/{uid}_{mask}.json').read_text())['result'];report=result['parsed'];errors=result['errors'];seconds=result['amortized_seconds'];tokens=result['output_tokens'];calls=1
            value=score(report,row['label'],item['reference'],item['asset'],mask,reviews,errors);label=report['class']['value'] if report and value['valid'] and value['class_asserted'] else None
            scored[row['variant']][method].append({'id':uid,'original_id':row['original_id'],'source_group':row['source_group'],'date':row['date'],'label_index':row['label_index'],
                'prediction':labels.index(label) if label in labels else -1,'mask':mask,'actions':trace,'risk':risk,'retained':risk<=settings['thresholds'][method],
                'score':value,'tool_calls':mask.bit_count(),'generator_calls':calls,'output_tokens':tokens,'reconstructed_seconds':seconds+sum(item['asset']['tool_seconds']['ABCD'[a]] for a in trace)})
    summary={};prediction=np.load(ROOT/'data/derived/robustness_classifier.npz')
    for variant,method_records in scored.items():
        idx=[i for i,r in enumerate(rows) if r['variant']==variant];y=np.array([rows[i]['label_index'] for i in idx])
        summary[variant]={'methods':{m:metrics(v) for m,v in method_records.items()},'P_vs_locked_comparator':cluster_contrast(method_records['P'],method_records[settings['locked_comparator']]),
            'classifier':classifier_metrics(y,prediction['logits'][idx],float(prediction['temperature']))}
    atomic_json(ROOT/'results/tables/robustness_results.json',{'conditions':summary,'source_clips':manifest['source_clips'],'transformed_clips':len(rows),'reviewer_calls':2*len(rows),'valid_reviews':int(review_valid),'new_model_calls':len(remaining),'seconds':time.monotonic()-started,'exploratory':True,'thresholds_reused_from_clean_calibration':True})
    atomic_json(ROOT/'results/raw/scored_robustness_records.json',scored)

if __name__=='__main__':main()
