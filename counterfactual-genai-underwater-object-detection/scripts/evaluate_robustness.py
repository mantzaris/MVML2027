#!/usr/bin/env python3
"""Prospective frozen-calibration real-waveform noise robustness, seed 17 only.

120 test parents: 12 AIS positives and 12 backgrounds per collection day. Clean
and four Gaussian-noise conditions share parents/noise/flow latent seeds. Noise
SNR is relative to the whole recorded waveform, not an isolated vessel signal.
The proposed agent makes actual bounded Qwen calls; no probabilities are supplied
by the LLM. Numeric scores retain the exact frozen calibration/fallback logic.

--write-config writes only the prospective config; no GPU or test waveform use.
run_agent_policy and calibrate_cases are import-safe helpers for the separately
owned live benchmark. The helper consumes supplied feature/intervention evidence;
its timing alone never claims end-to-end adaptive acquisition cost.
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import torch
from uac.agent import BoundedQwenController
from uac.agent_batch import select_actions
from uac.dsp import extract_features,PRIMARY_GROUPS
from uac.metrics import apply_calibrator,binary_metrics
from uac.verification import (load_real_cache,load_flow,DetectorBank,generate_intervention_features,
 intervention_components,verification_score,SUBSET_NAMES)
from uac.training import save_torch
from uac.protocol import require_frozen_test
from uac.runtime import (RunLock,StopRequested,EventLedger,atomic_bytes,atomic_json,
 fingerprint,sha256_file,source_fingerprint,project_path,utc_now)
from synthetic_common import safe_gpu

CONFIG={'schema_version':1,'model_seeds':[17],'selection_seed':17,'noise_seed':17,
 'sample_rate':16000,'clip_seconds':3,'batch_size':8,'expected_days':5,'per_day_per_label':12,
 'expected_clips':120,'conditions':[{'name':'clean','snr_db':None},{'name':'white_noise_20dB','snr_db':20},
  {'name':'white_noise_10dB','snr_db':10},{'name':'white_noise_0dB','snr_db':0},{'name':'white_noise_minus5dB','snr_db':-5}],
 'noise_definition':'CUDA standard Gaussian draw, independently RMS-normalized per clip; same draw shared across SNR conditions and methods',
 'snr_definition':'whole recorded waveform RMS to added-noise RMS; not clean vessel-source SNR',
 'methods':['main','stft','fixed_verification','proposed_agent'],'groups':list(PRIMARY_GROUPS),
 'counterfactual_samples':4,'flow_steps':8,'flow_seed_namespace':8400000,'probe_seconds':[.75,2.25],
 'dispersion':.5,'boundary_penalty_coefficient':.1,'max_actions':3,'max_verifications':1,
 'selection':'hash-sort within UTC day and AIS label; first12 each; interleave background/positive in each day',
 'score_fitting':'reuse frozen natural-real calibration for every corruption, no refit or threshold search',
 'independence':'same120 parents across conditions; five collection days form one connected day-vessel component; no independent-component confidence interval',
 'model_seed_scope':'seed17-only robustness for actual-controller compute budget; other studies report three detector seeds',
 'parent_manifest':'data/manifests/real_test.jsonl','test_feature_index':'data/features/real/test/index.json',
 'timing_scope':'shared evidence work plus live batched controller/scoring; not an isolated per-method latency benchmark'}
CODE_PATHS=['scripts/evaluate_robustness.py','src/uac/verification.py','src/uac/agent.py','src/uac/agent_batch.py',
 'src/uac/dsp.py','src/uac/models.py','src/uac/metrics.py','src/uac/protocol.py','scripts/extract_real_features.py',
 'scripts/run_policies.py','scripts/analyze_real.py']
REQUIRED_FROZEN=['configs/robustness.json','scripts/evaluate_robustness.py','configs/research.json',
 'manifests/calibration.json','manifests/detector_index.json','checkpoints/background_flow/final.pt',
 'manifests/model_lock.json','data/features/real/train/index.json']

def code_fingerprint():return source_fingerprint(CODE_PATHS)

def check_gate():
    freeze=require_frozen_test()
    values={'robustness_config_sha256':sha256_file('configs/robustness.json'),
      'robustness_code_fingerprint':code_fingerprint(),'robustness_parent_manifest_sha256':sha256_file(CONFIG['parent_manifest'])}
    for field,value in values.items():
        if freeze.get(field)!=value:raise RuntimeError(f'Robustness test locked: mismatched/missing {field}')
    for name in REQUIRED_FROZEN:
        if freeze.get('artifacts',{}).get(name,{}).get('sha256')!=sha256_file(name):
            raise RuntimeError(f'Robustness requires pinned artifact {name}')
    return freeze,sha256_file('runs/test_freeze.json')

def select_subset(rows):
    days=sorted({r['day'] for r in rows})
    if len(days)!=5 or any(r['split']!='test' or r['origin']!='real' for r in rows):raise ValueError('Expected only the five frozen real test days')
    selected=[]
    for day in days:
        by_label={label:sorted([r for r in rows if r['day']==day and r['label']==label],
          key=lambda r:(fingerprint({'seed':17,'day':day,'label':label,'id':r['id']}),r['id'])) for label in (0,1)}
        if any(len(values)<12 for values in by_label.values()):raise ValueError('Insufficient frozen class/day capacity')
        for j in range(12):selected.extend([by_label[0][j],by_label[1][j]])
    if len(selected)!=120 or len({r['id'] for r in selected})!=120:raise ValueError('Invalid deterministic subset')
    return selected

def day_vessel_components(rows):
    parent={}
    def find(value):
        parent.setdefault(value,value)
        if parent[value]!=value:parent[value]=find(parent[value])
        return parent[value]
    for row in rows:
        day='day:'+row['day'];find(day)
        if row['label']:
            a,b=find(day),find(row['source_id']);parent[b]=a
    return len({find('day:'+row['day']) for row in rows})

def calibrate_cases(cases,fit):
    """Exact acquired-subset base fallback and optional valid-verification fit."""
    results={}
    for name,case in cases.items():
        probability=torch.empty_like(case['base'])
        for base_name in sorted(set(case['base_names'])):
            mask=torch.tensor([v==base_name for v in case['base_names']],device='cuda')
            probability[mask]=apply_calibrator(case['base'][mask,None],fit['base'][base_name])
        fitted=fit.get('verification',{}).get(name)
        if fitted is not None:
            pair=torch.stack([case['base'],case['v']],-1)
            probability=torch.where(case['verify'],apply_calibrator(pair,fitted),probability)
        results[name]=probability
    return results

def run_agent_policy(controller,bank,features,components,ids,*,seed=17):
    """Same live bounded state machine as run_policies.py, without true labels.

    Precomputed evidence is supplied explicitly. Callers wishing to measure the
    entire pipeline must time its construction too and disclose shared work.
    """
    if features.device.type!='cuda' or len(ids)!=len(features):raise ValueError('CUDA features and matching opaque IDs required')
    view_logits=bank.view_logits(features);n=len(ids)
    states=[];traces=[[] for _ in ids];views=[['stft'] for _ in ids];verified=[False]*n;active=list(range(n))
    for i in range(n):
        probability=float(torch.sigmoid(view_logits[i,1]))
        states.append({'candidates':[{'candidate_id':'candidate0','interval_seconds':[.75,2.25],'base_probability':probability,'uncertainty':1-2*abs(probability-.5)}],
          'observations':{'candidate0':{'stft':{'detector_logit':float(view_logits[i,1]),'presence_probability':probability}}},
          'history':[],'available_tools':['fourier','wavelet','stop','verify'],'remaining_actions':3,'remaining_verifications':1})
    for _ in range(3):
        if not active:break
        actions=select_actions(controller,[states[i] for i in active]);next_active=[]
        for local,action in zip(active,actions):
            traces[local].append(action);state=states[local];tool=action['action']
            if tool=='stop' or not action['valid']:continue
            state['history'].append({'action':tool,'candidate_id':'candidate0','valid':True});state['remaining_actions']-=1
            state['available_tools'].remove(tool)
            if tool=='verify':
                verified[local]=True;state['remaining_verifications']=0
                observed={view:float(components['delta'][local,:,PRIMARY_GROUPS.index(view)].median()) for view in views[local]}
                state['observations']['candidate0']['verify']={'valid':bool(components['valid'][local]),'view_delta_medians':observed,'background_penalty':float(components['penalty'][local])}
            else:
                views[local].append(tool);position=PRIMARY_GROUPS.index(tool)
                state['observations']['candidate0'][tool]={'detector_logit':float(view_logits[local,position]),'presence_probability':float(torch.sigmoid(view_logits[local,position]))}
                if verified[local]:state['observations']['candidate0']['verify']['view_delta_medians'][tool]=float(components['delta'][local,:,position].median())
            current=float(torch.sigmoid(bank.subset_logit(features[local:local+1],views[local])[0]))
            state['candidates'][0]['base_probability']=current;state['candidates'][0]['uncertainty']=1-2*abs(current-.5)
            if state['remaining_actions']>0:next_active.append(local)
        active=next_active
    records=[]
    for i,identity in enumerate(ids):
        selected=views[i];positions=[PRIMARY_GROUPS.index(view) for view in selected]
        base=float(bank.subset_logit(features[i:i+1],selected)[0]);delta=components['delta'][i,:,positions].flatten()
        records.append({'id':identity,'seed':seed,'mode':'agent','views':selected,'subset_method':SUBSET_NAMES[frozenset(selected)],
          'verified':verified[i],'intervention_valid':bool(components['valid'][i]),'base_logit':base,
          'delta':delta.cpu().tolist() if verified[i] else [],'penalty':float(components['penalty'][i]) if verified[i] else 0.,
          'trace':traces[i],'amortized_llm_seconds':sum(t['amortized_seconds'] for t in traces[i]),
          'measurement_scope':'supplied acoustic/intervention evidence and actual Qwen calls; not standalone pipeline latency'})
    return records

def make_cases(bank,features,components,policy,calibration):
    n=len(features);zero=torch.zeros(n,device='cuda');false=torch.zeros(n,dtype=torch.bool,device='cuda')
    main=bank.logits('main',features);stft=bank.logits('stft',features)
    dispersion=calibration['dispersion'];penalty=calibration['penalty']
    cases={'main':{'base':main,'v':zero,'verify':false,'base_names':['main']*n},
      'stft':{'base':stft,'v':zero,'verify':false,'base_names':['stft']*n},
      'fixed_verification':{'base':main,'v':verification_score(components,PRIMARY_GROUPS,dispersion,penalty),
        'verify':components['valid'],'base_names':['main']*n}}
    scores=[]
    for row in policy:
        values=torch.tensor(row['delta'],device='cuda');median=values.median() if len(values) else torch.tensor(0.,device='cuda')
        score=median-dispersion*(values-median).abs().median()-penalty*row['penalty'] if len(values) else median
        scores.append(float(score))
    cases['proposed_agent']={'base':torch.tensor([r['base_logit'] for r in policy],device='cuda'),
      'v':torch.tensor(scores,device='cuda'),'verify':torch.tensor([r['verified'] and r['intervention_valid'] for r in policy],device='cuda'),
      'base_names':[r['subset_method'] for r in policy]}
    return cases

def receipt_valid(path,sidecar,expected):
    if not sidecar.is_file():return None
    receipt=json.loads(sidecar.read_text())
    if any(receipt.get(k)!=v for k,v in expected.items()):raise RuntimeError('Incompatible robustness artifact; do not overwrite')
    if path.is_file() and path.stat().st_size==receipt.get('payload_bytes') and sha256_file(path)==receipt.get('payload_sha256'):return receipt
    stamp=f'.corrupt-{time.time_ns()}'
    for item in (path,sidecar):
        if item.exists():item.rename(item.with_name(item.name+stamp))
    return None

def commit_payload(path,sidecar,payload,expected,*,tensor=False,measurements=None):
    if tensor:save_torch(path,payload)
    else:atomic_json(path,payload)
    receipt={**expected,'payload':str(path.relative_to(PROJECT)),'payload_sha256':sha256_file(path),
      'payload_bytes':path.stat().st_size,'measurements':measurements or {},'completed_utc':utc_now()}
    atomic_json(sidecar,receipt);return receipt

def train_background_statistics():
    index=json.loads((PROJECT/'data/features/real/train/index.json').read_text())
    if index.get('status')!='complete':raise RuntimeError('Training feature cache incomplete')
    features=[];labels=[]
    for receipt in index['shards']:
        if sha256_file(receipt['payload'])!=receipt['payload_sha256']:raise RuntimeError('Training feature checksum mismatch')
        part=torch.load(project_path(receipt['payload']),map_location='cpu',weights_only=False)
        features.append(part['features'][:,:32].to('cuda'));labels.append(part['labels'].to('cuda'))
    background=torch.cat(features)[torch.cat(labels)==0]
    return {'mean':background.mean(0),'scale':background.std(0).clamp_min(1e-4)}

def condition_summary(predictions,condition,fit):
    output=[];days=[]
    for day in [None]+sorted({r['day'] for r in predictions}):
        rows=predictions if day is None else [r for r in predictions if r['day']==day]
        for method in CONFIG['methods']:
            metric=binary_metrics([r['label'] for r in rows],[r['probabilities'][method] for r in rows],threshold=fit['thresholds'][method]['threshold'])
            record={'method':method,'seed':17,'split':'test','origin':'natural_real_subset' if condition['snr_db'] is None else 'corrupted_real',
              'condition':condition['name'],'snr_db':condition['snr_db'],'noise_snr_scope':CONFIG['snr_definition'],
              'group_count':len({r['day'] for r in rows}),'day':day,'cluster_interpretation':'collection-day descriptive grouping, not independent population units',**metric}
            (output if day is None else days).append(record)
    return output,days

def write_report(common):
    records=[];day_records=[];indices=[];times=[]
    for condition in CONFIG['conditions']:
        path=PROJECT/'results/raw/robustness'/condition['name']/'index.json'
        if not path.exists():continue
        index=json.loads(path.read_text())
        if index.get('status')!='complete':continue
        if any(index.get(k)!=v for k,v in common.items()):raise RuntimeError('Robustness summary provenance mismatch')
        for receipt in index['prediction_batches']:
            if sha256_file(receipt['payload'])!=receipt['payload_sha256']:raise RuntimeError('Robustness prediction checksum mismatch')
            times.append({'condition':condition['name'],'begin':receipt['begin'],**receipt['measurements']})
        records.extend(index['records']);day_records.extend(index['day_records']);indices.append({'path':str(path.relative_to(PROJECT)),'sha256':sha256_file(path)})
    clean={r['method']:r['ap'] for r in records if r['condition']=='clean'}
    for row in records:row['ap_change_from_clean']=row['ap']-clean[row['method']] if row['method'] in clean else None
    output={**common,'status':'complete' if len(indices)==len(CONFIG['conditions']) else 'partial',
      'config':CONFIG,'records':records,'robustness':records,'day_records':day_records,'indices':indices,
      'completed_conditions':len(indices),'expected_conditions':len(CONFIG['conditions']),
      'timing_observations':times,'timing_isolated':False,'model_seed_scope':CONFIG['model_seed_scope'],
      'limitations':['120clips and five collection-day clusters only.','All five days connect through repeated vessel identities; no independent-component confidence interval is estimable.',
       'Noise SNR refers to whole recorded waveform, including background.','AIS labels retained; audibility under corruption is not relabelled.',
       'No test-corruption score recalibration or model adaptation.','Shared evidence construction and actual Qwen timings do not give isolated per-method runtime.'],
      'created_utc':utc_now()}
    atomic_json('results/tables/robustness.json',output);return output

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-config',action='store_true');parser.add_argument('--max-batches',type=int)
    args=parser.parse_args();config_path=PROJECT/'configs/robustness.json'
    if args.max_batches is not None and args.max_batches<1:parser.error('Positive batch cap required')
    if args.write_config:
        if config_path.exists() and json.loads(config_path.read_text())!=CONFIG:raise RuntimeError('Different prospective robustness config exists')
        atomic_json(config_path,CONFIG)
        print(json.dumps({'robustness_config_sha256':sha256_file(config_path),'robustness_code_fingerprint':code_fingerprint(),
          'parent_manifest_to_pin':CONFIG['parent_manifest'],'required_parent_hash_field':'robustness_parent_manifest_sha256',
          'required_frozen_artifacts':REQUIRED_FROZEN,'gpu_used':False,'test_waveforms_accessed':False},indent=2));return 0
    if not config_path.is_file() or json.loads(config_path.read_text())!=CONFIG:raise RuntimeError('Write and freeze prospective robustness config first')
    freeze,freeze_hash=check_gate();safe_gpu()
    calibration=json.loads((PROJECT/'manifests/calibration.json').read_text())
    if (calibration.get('fit_split')!='calibration' or calibration.get('data_fingerprint')!=freeze['data_fingerprint']
       or calibration['dispersion']!=CONFIG['dispersion'] or calibration['penalty']!=CONFIG['boundary_penalty_coefficient']):raise RuntimeError('Frozen calibration semantics mismatch')
    research=json.loads((PROJECT/'configs/research.json').read_text())
    if research['counterfactual_samples']!=4 or research['flow_sampling_steps']!=8:raise RuntimeError('Frozen intervention budget mismatch')
    with RunLock('runs/training.lock'),StopRequested() as stop:
        ledger=EventLedger();native_rows=[json.loads(l) for l in project_path(CONFIG['parent_manifest']).read_text().splitlines()]
        selected=select_subset(native_rows);dataset=load_real_cache('test')
        if (dataset['rows']!=native_rows or dataset['index'].get('rows_fingerprint')!=fingerprint(native_rows)
            or dataset['index'].get('split_manifest_sha256')!=sha256_file(CONFIG['parent_manifest'])
            or dataset['index'].get('data_fingerprint')!=freeze['data_fingerprint']):raise RuntimeError('Test cache does not match frozen native parents')
        positions={identity:i for i,identity in enumerate(dataset['ids'])}
        chosen=torch.tensor([positions[r['id']] for r in selected],device='cuda');audio=dataset['audio'][chosen].clone();del dataset
        if audio.shape!=(120,48000) or not bool(torch.isfinite(audio).all()):raise RuntimeError('Invalid selected waveforms')
        rms=audio.square().mean(-1,keepdim=True).sqrt()
        if bool((rms<1e-12).any()):raise RuntimeError('Frozen selected recording has undefined noise SNR; do not substitute outcome-selected parents')
        generator=torch.Generator(device='cuda').manual_seed(CONFIG['noise_seed'])
        white=torch.randn(audio.shape,device='cuda',generator=generator);white=white/white.square().mean(-1,keepdim=True).sqrt()
        selection_path=PROJECT/'data/manifests/robustness_subset.jsonl'
        selection_bytes=b''.join((json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode() for r in selected)
        if selection_path.exists() and selection_path.read_bytes()!=selection_bytes:raise RuntimeError('Frozen robustness subset changed')
        if not selection_path.exists():atomic_bytes(selection_path,selection_bytes)
        bgstats=train_background_statistics();flow=load_flow();bank=DetectorBank(17);controller=BoundedQwenController()
        fit=calibration['seeds']['17'];common={'schema_version':1,'config_hash':fingerprint(CONFIG),'config_sha256':sha256_file(config_path),
          'code_fingerprint':code_fingerprint(),'freeze_sha256':freeze_hash,'data_fingerprint':freeze['data_fingerprint'],
          'parent_manifest_sha256':sha256_file(CONFIG['parent_manifest']),'test_feature_index_sha256':sha256_file(CONFIG['test_feature_index']),
          'subset_manifest_sha256':sha256_file(selection_path),'calibration_sha256':sha256_file('manifests/calibration.json'),
          'detector_index_sha256':sha256_file('manifests/detector_index.json'),'generator_sha256':sha256_file('checkpoints/background_flow/final.pt'),
          'model_revision':controller.revision,'prompt_sha256':controller.prompt_sha256,'model_seed':17,
          'selected_day_vessel_components':day_vessel_components(selected),'full_test_day_vessel_components':day_vessel_components(native_rows)}
        fresh=0
        try:
            for condition in CONFIG['conditions']:
                directory=PROJECT/'results/raw/robustness'/condition['name'];directory.mkdir(parents=True,exist_ok=True)
                expected={**common,'condition':condition};predictions=[];evidence_receipts=[];prediction_receipts=[]
                index_path=directory/'index.json'
                if index_path.is_file():
                    old=json.loads(index_path.read_text())
                    if any(old.get(k)!=v for k,v in expected.items()):raise RuntimeError('Incompatible robustness condition')
                    if old.get('status')=='complete' and old.get('completed_rows')==120 and all(project_path(r['payload']).is_file() and sha256_file(r['payload'])==r['payload_sha256'] for r in old['prediction_batches']+old['evidence_batches']):continue
                for begin in range(0,120,8):
                    end=min(begin+8,120);rows=selected[begin:end];ids=[r['id'] for r in rows]
                    wanted={**expected,'begin':begin,'count':len(rows),'ids':ids}
                    evidence_path=directory/f'evidence-{begin:04d}.pt';evidence_sidecar=directory/f'evidence-{begin:04d}.json'
                    pred_path=directory/f'predictions-{begin:04d}.json';pred_sidecar=directory/f'predictions-{begin:04d}.receipt.json'
                    evidence_receipt=receipt_valid(evidence_path,evidence_sidecar,wanted)
                    pred_receipt=receipt_valid(pred_path,pred_sidecar,wanted)
                    if pred_receipt is None and (stop.requested or (args.max_batches is not None and fresh>=args.max_batches)):
                        return 75 if stop.requested else 0
                    if evidence_receipt is None:
                        torch.cuda.synchronize();started=time.perf_counter()
                        original=audio[begin:end];snr=condition['snr_db']
                        noise=torch.zeros_like(original) if snr is None else white[begin:end]*rms[begin:end]*(10.**(-snr/20.))
                        corrupted=original+noise
                        error=0. if snr is None else float((20*torch.log10(rms[begin:end].squeeze(1)/noise.square().mean(-1).sqrt())-snr).abs().max())
                        if error>1e-3:raise RuntimeError('Gaussian corruption failed exact RMS SNR check')
                        with torch.inference_mode():
                            features=extract_features(corrupted,groups=PRIMARY_GROUPS)['features']
                            cache=generate_intervention_features(flow,corrupted,seed=CONFIG['flow_seed_namespace']+begin,k=4,steps=8,probe='central')
                        torch.cuda.synchronize();measured={'live_shared_evidence_seconds':time.perf_counter()-started,'max_snr_error_db':error,'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2}
                        payload={'ids':ids,'features':features.cpu(),'interventions':{k:v.cpu() for k,v in cache.items()},'waveform_rms':rms[begin:end].cpu(),'condition':condition}
                        evidence_receipt=commit_payload(evidence_path,evidence_sidecar,payload,wanted,tensor=True,measurements=measured)
                    else:payload=torch.load(evidence_path,map_location='cuda',weights_only=False)
                    if pred_receipt is not None and pred_receipt.get('evidence_payload_sha256')!=evidence_receipt['payload_sha256']:
                        raise RuntimeError('Prediction refers to different intervention evidence; preserve before replay')
                    if pred_receipt is None:
                        if stop.requested:return 75
                        features=payload['features'].to('cuda');cache={k:v.to('cuda') for k,v in payload['interventions'].items()}
                        torch.cuda.synchronize();started=time.perf_counter()
                        with torch.inference_mode():
                            components=intervention_components(bank,cache,bgstats)
                            policy=run_agent_policy(controller,bank,features,components,ids,seed=17)
                            probabilities=calibrate_cases(make_cases(bank,features,components,policy,calibration),fit)
                        torch.cuda.synchronize();wall=time.perf_counter()-started
                        batch_records=[{'id':row['id'],'day':row['day'],'group':row['group'],'label':row['label'],
                          'native_sha256':row['sha256'],'source_id':row['source_id'],'condition':condition['name'],
                          'probabilities':{method:float(probabilities[method][i]) for method in CONFIG['methods']},'policy':policy[i]} for i,row in enumerate(rows)]
                        measurements={'live_scoring_and_controller_seconds':wall,'summed_amortized_llm_seconds':sum(r['amortized_llm_seconds'] for r in policy),
                          'shared_evidence_seconds':evidence_receipt['measurements']['live_shared_evidence_seconds'],'isolated_method_benchmark':False,
                          'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,'intervention_valid_fraction':float(components['valid'].float().mean())}
                        pred_receipt=commit_payload(pred_path,pred_sidecar,{'records':batch_records},{**wanted,'evidence_payload_sha256':evidence_receipt['payload_sha256']},measurements=measurements);fresh+=1
                        ledger.append('robustness_batch_complete',condition=condition['name'],begin=begin,count=len(rows),sha256=pred_receipt['payload_sha256'])
                        print(json.dumps({'stage':'robustness','condition':condition['name'],'through':end,'total':120}),flush=True)
                    else:batch_records=json.loads(pred_path.read_text())['records']
                    predictions.extend(batch_records);evidence_receipts.append(evidence_receipt);prediction_receipts.append(pred_receipt)
                    atomic_json(index_path,{**expected,'status':'running','completed_rows':len(predictions),'evidence_batches':evidence_receipts,'prediction_batches':prediction_receipts})
                records,day_records=condition_summary(predictions,condition,fit)
                atomic_json(index_path,{**expected,'status':'complete','completed_rows':120,'evidence_batches':evidence_receipts,'prediction_batches':prediction_receipts,
                  'records':records,'day_records':day_records,'completed_utc':utc_now()})
                write_report(common)
            report=write_report(common);print(json.dumps({'status':report['status'],'report':'results/tables/robustness.json'}),flush=True)
        finally:controller.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
