#!/usr/bin/env python3
"""Calibration/development lock, then frozen real test analysis with day intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from uac.verification import load_real_cache,DetectorBank,intervention_components,verification_score,PRIMARY_GROUPS
from uac.models import FeatureDetector
from uac.metrics import fit_monotonic_calibrator,apply_calibrator,binary_metrics,binary_average_precision,threshold_for_fpr,paired_group_bootstrap
from uac.runtime import RunLock,EventLedger,atomic_json,sha256_file,fingerprint
from uac.training import save_torch

p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibrate','test'],required=True);args=p.parse_args()
DISPERSION=.5;PENALTY=.1

def interventions(split,ids,full=False):
    index=json.loads((Path('data/interventions_full' if full else 'data/interventions')/split/'index.json').read_text())
    if index['status']!='complete':raise RuntimeError('Incomplete generated evidence')
    chunks=[]
    for row in index['shards']:
        if sha256_file(row['path'])!=row['sha256']:raise RuntimeError('Generated evidence checksum mismatch')
        chunks.append(torch.load(row['path'],map_location='cuda',weights_only=False))
    if [x for c in chunks for x in c['ids']]!=ids:raise RuntimeError('Evidence identity mismatch')
    return {key:torch.cat([c[key] for c in chunks]) for key in chunks[0] if key!='ids'}

def foundation_features(split,ids):
    index=json.loads((Path('data/features/foundation')/split/'index.json').read_text())
    if index['status']!='complete':raise RuntimeError('Incomplete foundation features')
    parts=[]
    for row in index['shards']:
        if sha256_file(row['payload'])!=row['payload_sha256']:raise RuntimeError('Foundation checksum mismatch')
        parts.append(torch.load(row['payload'],map_location='cuda',weights_only=False))
    if [x for r in parts for x in r['ids']]!=ids:raise RuntimeError('Foundation ID mismatch')
    return torch.cat([r['features'] for r in parts])

def policy_rows(split,seed,mode,ids):
    path=Path('results/raw/policies')/split/(mode+'_seed'+str(seed))/'all.json'
    index=json.loads((path.parent/'index.json').read_text())
    if index.get('status')!='complete' or index.get('sha256')!=sha256_file(path):raise RuntimeError('Incomplete or corrupt policy summary')
    obj=json.loads(path.read_text());rows=obj['records']
    if obj['config']['detectors_hash']!=sha256_file('manifests/detector_index.json') or obj['config']['split']!=split or obj['config']['seed']!=seed or obj['config']['mode']!=mode:raise RuntimeError('Policy provenance mismatch')
    if [r['id'] for r in rows]!=ids:raise RuntimeError('Policy row identity mismatch')
    return rows

def base_logits(bank,dataset,foundation):
    scores={}
    with torch.inference_mode():
        for name,model in bank.models.items():
            if name=='cnn':scores[name]=torch.cat([model(dataset['audio'][b:b+32]) for b in range(0,len(dataset['ids']),32)])
            else:scores[name]=bank.logits(name,dataset['features'])
        record=bank.records['foundation'];head=FeatureDetector(record['input_dim']).cuda();head.load_state_dict(torch.load(record['checkpoint'],map_location='cuda',weights_only=False)['model']);head.eval();scores['foundation']=head(foundation);del head
        scores['classical_energy']=dataset['features'][:,:24].mean(-1)
    return scores

def cases_for(split,seed,bank,dataset,scores,components,full_components):
    n=len(dataset['ids']);cases={}
    zero=torch.zeros(n,device='cuda');false=torch.zeros(n,dtype=torch.bool,device='cuda')
    for name,logit in scores.items():cases[name]={'base':logit,'v':zero,'verify':false,'base_names':[name]*n}
    base=scores['main'];v=verification_score(components,PRIMARY_GROUPS,DISPERSION,PENALTY)
    cases['fixed_verification']={'base':base,'v':v,'verify':components['valid'],'base_names':['main']*n}
    cases['full_record_null']={'base':base,'v':verification_score(full_components,PRIMARY_GROUPS,DISPERSION,PENALTY),'verify':full_components['valid'],'base_names':['main']*n}
    cases['no_dispersion']={'base':base,'v':verification_score(components,PRIMARY_GROUPS,0.,PENALTY),'verify':components['valid'],'base_names':['main']*n}
    cases['no_penalty']={'base':base,'v':verification_score(components,PRIMARY_GROUPS,DISPERSION,0.),'verify':components['valid'],'base_names':['main']*n}
    original=bank.view_logits(dataset['features'])[:,None,:]
    raw=(original-components['minus_logits']).flatten(1);median=raw.median(1).values
    cases['no_null_adjustment']={'base':base,'v':median-DISPERSION*(raw-median[:,None]).abs().median(1).values-PENALTY*components['penalty'],'verify':components['valid'],'base_names':['main']*n}
    uncertain=(torch.sigmoid(scores['stft'])-.5).abs()<.3
    cases['heuristic_verification']={'base':torch.where(uncertain,base,scores['stft']),'v':torch.where(uncertain,v,zero),'verify':uncertain&components['valid'],'base_names':['main' if bool(t) else 'stft' for t in uncertain]}
    for mode in ('agent','agent_no_generation'):
        rows=policy_rows(split,seed,mode,dataset['ids']);bs=[];vs=[];valid=[]
        for row in rows:
            values=torch.tensor(row['delta'],device='cuda')
            med=values.median() if len(values) else torch.tensor(0.,device='cuda')
            score=med-DISPERSION*(values-med).abs().median()-PENALTY*row['penalty'] if len(values) else med
            bs.append(row['base_logit']);vs.append(float(score));valid.append(row['verified'] and row['intervention_valid'])
        name='proposed_agent' if mode=='agent' else 'agent_no_generation'
        cases[name]={'base':torch.tensor(bs,device='cuda'),'v':torch.tensor(vs,device='cuda'),'verify':torch.tensor(valid,device='cuda'),'base_names':[r['subset_method'] for r in rows],'policy_rows':rows}
    for original,ablation in [('fixed_verification','fixed_validity_only'),('proposed_agent','agent_validity_only')]:
        cases[ablation]={**cases[original],'v':zero}
    return cases

def calibrated_predictions(cases,fit):
    result={}
    for name,case in cases.items():
        baseprob=torch.empty_like(case['base'])
        for base_name in sorted(set(case['base_names'])):
            mask=torch.tensor([x==base_name for x in case['base_names']],device='cuda')
            baseprob[mask]=apply_calibrator(case['base'][mask,None],fit['base'][base_name])
        if name in fit['verification'] and fit['verification'][name] is not None:
            mask=case['verify'];pair=torch.stack([case['base'],case['v']],-1)
            baseprob=torch.where(mask,apply_calibrator(pair,fit['verification'][name]),baseprob)
        result[name]=baseprob
    return result

with RunLock('runs/training.lock'):
    config=json.loads(Path('configs/research.json').read_text());seeds=config['seeds'];ledger=EventLedger()
    train=load_real_cache('train');bg=train['features'][train['labels']==0,:32]
    bgstats={'mean':bg.mean(0),'scale':bg.std(0).clamp_min(1e-4)};del bg,train
    if args.stage=='calibrate':
        splits=['calibration','development']
    else:
        frozen=json.loads(Path('runs/test_freeze.json').read_text())
        for path,entry in frozen['artifacts'].items():
            if sha256_file(path)!=entry['sha256']:raise RuntimeError('Frozen artifact changed: '+path)
        splits=['test']
    datasets={s:load_real_cache(s) for s in splits};foundations={s:foundation_features(s,datasets[s]['ids']) for s in splits};caches={s:interventions(s,datasets[s]['ids']) for s in splits};full_caches={s:interventions(s,datasets[s]['ids'],full=True) for s in splits}
    all_records=[];predictions={};mechanism=[];policy_usage=[]
    calibration_path=Path('manifests/calibration.json');calibration=json.loads(calibration_path.read_text()) if args.stage=='test' else {'seeds':{},'dispersion':DISPERSION,'penalty':PENALTY,'fit_split':'calibration','operating_fpr_target':.01,'invalid_or_unqueried_verification':'return calibrated acquired-subset base detector','data_fingerprint':datasets['calibration']['index']['data_fingerprint']}
    for seed in seeds:
        bank=DetectorBank(seed);cases={}
        for split in splits:
            scores=base_logits(bank,datasets[split],foundations[split]);comp=intervention_components(bank,caches[split],bgstats)
            full_comp=intervention_components(bank,full_caches[split],bgstats)
            cases[split]=cases_for(split,seed,bank,datasets[split],scores,comp,full_comp)
            if split=='test':
                for label,condition in [(0,'AIS_background'),(1,'AIS_vessel_present')]:
                    selected=datasets[split]['labels']==label;accepted=selected&comp['valid']
                    mechanism.append({'condition':condition,'seed':seed,'delta_mean':float(comp['delta'][accepted].mean()) if bool(accepted.any()) else None,'valid_fraction':float(comp['valid'][selected].float().mean()),'n_valid':int(accepted.sum()),'n_total':int(selected.sum()),'boundary_penalty_mean':float(comp['penalty'][accepted].mean()) if bool(accepted.any()) else None,'origin':'valid_same_support_model_sensitivity_not_physical_causality'})
                for method in ('proposed_agent','agent_no_generation'):
                    rows=cases[split][method]['policy_rows'];actions=[t for row in rows for t in row['trace']]
                    policy_usage.append({'method':method,'seed':seed,'n':len(rows),'verification_fraction':sum(r['verified'] for r in rows)/len(rows),'mean_acquired_views':float(np.mean([len(r['views']) for r in rows])),'mean_actions':len(actions)/len(rows),'valid_action_fraction':sum(t['valid'] for t in actions)/max(len(actions),1),'actions':{tool:sum(t['action']==tool for t in actions) for tool in ('fourier','wavelet','verify','stop')}})
        if args.stage=='calibrate':
            fit={'base':{},'verification':{},'thresholds':{}}
            for name,case in cases['calibration'].items():
                if name in bank.records or name=='classical_energy':fit['base'][name]=fit_monotonic_calibrator(case['base'][:,None],datasets['calibration']['labels'])
            for name,case in cases['calibration'].items():
                count=int(case['verify'].sum())
                if count:
                    mask=case['verify'];labels=datasets['calibration']['labels'][mask]
                    fit['verification'][name]=fit_monotonic_calibrator(torch.stack([case['base'][mask],case['v'][mask]],-1),labels) if count>=30 and len(labels.unique())==2 else None
            cal_predictions=calibrated_predictions(cases['calibration'],fit)
            for name,scores in cal_predictions.items():fit['thresholds'][name]=threshold_for_fpr(datasets['calibration']['labels'].cpu().numpy(),scores.cpu().numpy(),.01)
            calibration['seeds'][str(seed)]=fit;atomic_json(calibration_path,calibration)
        else:fit=calibration['seeds'][str(seed)]
        for split in splits:
            pred=calibrated_predictions(cases[split],fit);labels=datasets[split]['labels'].cpu().numpy();predictions[(split,seed)]={name:value.cpu().numpy() for name,value in pred.items()}
            for name,value in pred.items():
                metrics=binary_metrics(labels,value.cpu().numpy(),threshold=fit['thresholds'][name]['threshold'])
                all_records.append({'method':name,'seed':seed,'split':split,'group_count':len(set(r['day'] for r in datasets[split]['rows'])),**metrics})
            save_torch(Path('results/raw')/f'real_predictions_{split}_seed{seed}.pt',{'ids':datasets[split]['ids'],'labels':torch.tensor(labels),'rows':datasets[split]['rows'],'probabilities':{name:torch.tensor(value) for name,value in predictions[(split,seed)].items()}})
        del bank
    if args.stage=='calibrate':
        eligible=[name for name in predictions[('development',seeds[0])] if name not in ('proposed_agent','hht_exploratory','no_dispersion','no_penalty','no_null_adjustment','full_record_null','fixed_validity_only','agent_validity_only')]
        ranking=sorted([{'method':name,'ensemble_ap':binary_average_precision(datasets['development']['labels'].cpu().numpy(),np.mean([predictions[('development',seed)][name] for seed in seeds],axis=0)),'mean_seed_ap':float(np.mean([r['ap'] for r in all_records if r['method']==name and r['split']=='development']))} for name in eligible],key=lambda r:(-r['ensemble_ap'],r['method']))
        atomic_json('manifests/final_development_selection.json',{'ranked_baselines':ranking,'locked_comparator':ranking[0]['method'],'selection':'development AP of mean three-seed calibrated probabilities, matching the primary test endpoint; lexical tie break; calibration-only score fitting; test unopened','records':all_records,'dispersion':DISPERSION,'penalty':PENALTY})
        ledger.append('calibration_and_development_selection_complete',comparator=ranking[0]['method'])
    else:
        baseline=json.loads(Path('manifests/final_development_selection.json').read_text())['locked_comparator']
        labels=datasets['test']['labels'].cpu().numpy();groups=[r['day'] for r in datasets['test']['rows']]
        mean_predictions={name:np.mean([predictions[('test',seed)][name] for seed in seeds],axis=0) for name in predictions[('test',seeds[0])]}
        stats=paired_group_bootstrap(labels,mean_predictions['proposed_agent'],mean_predictions[baseline],groups,n_bootstrap=2000,seed=17)
        ci=stats.get('confidence_interval',stats.get('ci'))
        if ci is None:ci=[stats.get('ci_low'),stats.get('ci_high')]
        comparison={'method_a':'proposed_agent','method_b':baseline,'metric':'ap_of_mean_seed_probability','difference_a_minus_b':binary_average_precision(labels,mean_predictions['proposed_agent'])-binary_average_precision(labels,mean_predictions[baseline]),'ci':None,'independent_groups':1,'collection_days':len(set(groups)),'seed':17,'day_cluster_sensitivity':{'ci':ci,'details':stats,'interpretation':'Nonconfirmatory day resampling; vessel identities recur across days'},'limitation':'All five test days form one connected day-vessel component. Independent-component uncertainty is not estimable. Primary AP difference is descriptive; seed mean AP is separately tabulated.'}
        mechanism_summary=[]
        for condition in sorted({r['condition'] for r in mechanism}):
            rows=[r for r in mechanism if r['condition']==condition];valid_rows=[r for r in rows if r['delta_mean'] is not None]
            if valid_rows:mechanism_summary.append({'condition':condition,'delta_mean':float(np.mean([r['delta_mean'] for r in valid_rows])),'valid_fraction':float(np.mean([r['valid_fraction'] for r in rows])),'origin':'mean_across_detector_seeds_of_valid_null_adjusted_logit_contrasts','n_valid':rows[0]['n_valid'],'n_total':rows[0]['n_total'],'confidence_interval':None})
        score_contrasts=[]
        for a,b in [('fixed_verification','fixed_validity_only'),('proposed_agent','agent_validity_only')]:
            score_contrasts.append({'method_a':a,'method_b':b,'metric':'ap_of_mean_seed_probability','difference_a_minus_b':binary_average_precision(labels,mean_predictions[a])-binary_average_precision(labels,mean_predictions[b]),'per_seed_differences':{str(seed):binary_average_precision(labels,predictions[('test',seed)][a])-binary_average_precision(labels,predictions[('test',seed)][b]) for seed in seeds},'ci':None,'independent_components':1,'scope':'Paired descriptive continuous-score contribution with identical queries, views, validity and calibration protocol'})
        output={'status':'core_test_complete_extensions_pending','records':all_records,'primary_comparison':comparison,'ablations':[r for r in all_records if r['method'] in ('no_null_adjustment','no_dispersion','no_penalty','agent_no_generation','fixed_verification','heuristic_verification','full_record_null','main','proposed_agent','fixed_validity_only','agent_validity_only')], 'continuous_score_contrasts':score_contrasts,'mechanism':mechanism_summary,'mechanism_seed_records':mechanism,'policy_usage':policy_usage,'robustness':[],'cost':[],'data_scope':'PORTIA custom AIS-presence labels; five held-out days; vessel IDs disjoint from train/development/calibration but recurring within test, one connected day-vessel component; no human audibility labels','generator_seed':17,'detector_seeds':seeds}
        atomic_json('results/tables/real_results.json',output)
        ledger.append('real_primary_analysis_complete',**comparison)
    print(json.dumps({'stage':args.stage,'records':len(all_records)}),flush=True)
