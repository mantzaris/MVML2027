"""Locked real-data analysis with paired source-cluster uncertainty."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix
from underwater_reasoning.controller import Ensemble,load_study,method_mask,deterministic_score
from underwater_reasoning.scoring import deterministic_report
from underwater_reasoning.evaluation import holm_adjust
from underwater_reasoning.io import atomic_json,file_hash
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
METHODS=['B1_raw','B2_prior','B3_fixed','B4_retrieval','B4_retrieval_dsp','B5_all','B6_planner','B7_class_only','B8_deterministic','B9_entropy','P','A1_one_shot','A2_no_penalty','A3_no_stop']

def classifier_metrics(labels,logits,temperature):
    p=softmax(logits/temperature,axis=1);pred=p.argmax(1);confidence=p.max(1);correct=pred==labels
    bins=np.array_split(np.argsort(confidence),10);ece=sum(len(b)*abs(float(correct[b].mean()-confidence[b].mean())) for b in bins if len(b))/len(labels)
    return {'n':len(labels),'accuracy':float(accuracy_score(labels,pred)),
        'macro_f1':float(f1_score(labels,pred,labels=range(5),average='macro',zero_division=0)),
        'balanced_accuracy':float(recall_score(labels,pred,labels=range(5),average='macro',zero_division=0)),
        'per_class_recall':recall_score(labels,pred,labels=range(5),average=None,zero_division=0).tolist(),
        'confusion_matrix':confusion_matrix(labels,pred,labels=range(5)).tolist(),
        'brier':float(np.mean(np.sum((p-np.eye(5)[labels])**2,axis=1))),'ece_equal_frequency_10':float(ece),'temperature':float(temperature)}

def metrics(records):
    kept=[r for r in records if r['retained']];n=len(kept);a=sum(r['score']['assertions'] for r in kept);g=sum(r['score']['supported'] for r in kept);u=sum(r['score']['unsupported'] for r in kept)
    ratio=lambda x,y:float(x/y) if y else None
    y=[r['label_index'] for r in kept];pred=[r['prediction'] for r in kept]
    claims=[s for r in kept for s in r['score']['physical']]
    ambiguous=[c for c in claims if not c['identifiable']]
    return {'total_reports':len(records),'retained_reports':n,'report_coverage':ratio(n,len(records)),'supported_coverage':ratio(g,4*n),
        'whole_test_useful_coverage':ratio(g,4*len(records)),'assertion_coverage':ratio(a,4*n),'unsupported_rate':ratio(u,a),
        'assertions':a,'supported_assertions':g,'wrong_or_unsupported_assertions':u,'consensus_supported_coverage':ratio(sum(r['score']['consensus_supported'] for r in kept),4*n),
        'mean_loss':float(np.mean([r['score']['loss'] for r in kept])) if kept else None,
        'class_accuracy':ratio(sum(r['score']['class_correct'] for r in kept),n),
        'class_macro_f1':float(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)) if n else None,
        'class_abstention_rate':ratio(sum(not r['score']['class_asserted'] for r in kept),n),
        'factual_error_rate_on_assertions':ratio(sum(r['score']['factual_errors'] for r in kept),a),
        'citation_fidelity_on_assertions':ratio(sum(c['citation_supported'] for c in claims),a),
        'reference_identifiability':ratio(sum(r['score']['identifiable'] for r in kept),4*n),
        'appropriate_uncertainty_on_ambiguous':ratio(sum(c['appropriate_uncertainty'] for c in ambiguous),len(ambiguous)),
        'schema_validity_all':ratio(sum(r['score']['valid'] for r in records),len(records)),
        'full_format_validity_all':ratio(sum(r['score']['format_valid'] for r in records),len(records)),
        'mean_tools_all':float(np.mean([r['tool_calls'] for r in records])),
        'mean_generator_calls_all':float(np.mean([r['generator_calls'] for r in records])),
        'mean_terminal_output_tokens_all':float(np.mean([r['output_tokens'] for r in records])),
        'mean_reconstructed_seconds_all':float(np.mean([r['reconstructed_seconds'] for r in records]))}

def rows_for(method,items,settings,ensemble,class_ensemble,labels,threshold=None,budget=2):
    out=[]
    for item in items:
        mask,trace=method_mask(method,item,settings,ensemble,class_ensemble,budget);risk=ensemble.predict(item['asset'],max(mask,0))[2]
        if method=='B8_deterministic':
            scored=deterministic_score(item,mask,labels);report=deterministic_report(item['asset'],mask,labels);terminal_seconds=0;tokens=0;calls=0
        else:
            scored=item['scores'][mask];generation=item['reports'][mask]['result'];report=generation['parsed'];terminal_seconds=generation['amortized_seconds'];tokens=generation['output_tokens'];calls=1
        if method=='B6_planner':calls+=len(item['planner']['calls']);terminal_seconds+=sum(c['amortized_seconds'] for c in item['planner']['calls'])
        value=report['class']['value'] if report and scored['valid'] and scored['class_asserted'] else None;prediction=labels.index(value) if value in labels else -1
        out.append({'id':item['row']['id'],'source_group':item['row']['source_group'],'date':item['row']['date'],'label_index':item['row']['label_index'],
            'prediction':prediction,'mask':mask,'actions':trace,'risk':risk,'retained':threshold is None or risk<=threshold,'score':scored,
            'tool_calls':max(mask,0).bit_count(),'generator_calls':calls,'output_tokens':tokens,
            'reconstructed_seconds':terminal_seconds+sum(item['asset']['tool_seconds']['ABCD'[a]] for a in trace)})
    return out

def cluster_contrast(left,right,field='source_group',resamples=2000,seed=17):
    if [r['id'] for r in left]!=[r['id'] for r in right]:raise ValueError('Unpaired test records')
    groups=sorted({r[field] for r in left});index={g:np.array([i for i,r in enumerate(left) if r[field]==g]) for g in groups}
    l=np.array([[r['score']['supported'] if r['retained'] else 0,4*int(r['retained'])] for r in left],dtype=float)
    r=np.array([[v['score']['supported'] if v['retained'] else 0,4*int(v['retained'])] for v in right],dtype=float)
    def stat(a):return a[:,0].sum()/a[:,1].sum() if a[:,1].sum() else np.nan
    observed=stat(l)-stat(r);rng=np.random.default_rng(seed);values=[]
    for _ in range(resamples):
        chosen=rng.choice(groups,len(groups),replace=True);idx=np.concatenate([index[g] for g in chosen]);values.append(stat(l[idx])-stat(r[idx]))
    values=np.array(values);values=values[np.isfinite(values)]
    p=(1+np.sum(np.abs(values-observed)>=abs(observed)))/(len(values)+1)
    return {'difference':float(observed),'ci95':np.quantile(values,[.025,.975]).tolist(),'p_two_sided_centered_bootstrap':float(p),'clusters':len(groups),'cluster_field':field,'resamples':len(values),'seed':seed}

def main():
    if not json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise ValueError('Test analysis is locked')
    manifest,data=load_study(ROOT,['calibration','test']);settings=json.loads((ROOT/'configs/locked_controller.json').read_text());labels=manifest['labels']
    for part in data:
        for item in data[part]:item['planner']=json.loads((ROOT/f'results/raw/planner/{part}/{item["row"]["id"]}.json').read_text())
    ensemble=Ensemble(torch.load(ROOT/'checkpoints/controller_loss_17.pt',weights_only=False));class_ensemble=Ensemble(torch.load(ROOT/'checkpoints/controller_class_loss_17.pt',weights_only=False))
    method_rows={};table={};curves={}
    for method in METHODS:
        threshold=settings['thresholds'][method]
        rows=rows_for(method,data['test'],settings,ensemble,class_ensemble,labels,threshold);method_rows[method]=rows;table[method]=metrics(rows)
        ordered=sorted(rows,key=lambda r:(r['risk'],r['id']));curve=[]
        for coverage in np.linspace(.1,1,19):
            n=max(1,int(np.ceil(coverage*len(rows))));selected=[{**r,'retained':True} for r in ordered[:n]]
            value=metrics(selected);curve.append({'coverage':n/len(rows),'supported_coverage':value['supported_coverage'],'unsupported_rate':value['unsupported_rate'],'loss':value['mean_loss']})
        curves[method]=curve
    contrast_names=[settings['locked_comparator'],'B7_class_only','A1_one_shot'];contrasts={name:cluster_contrast(method_rows['P'],method_rows[name]) for name in contrast_names}
    adjusted=holm_adjust([contrasts[n]['p_two_sided_centered_bootstrap'] for n in contrast_names])
    for n,p in zip(contrast_names,adjusted):contrasts[n]['p_holm']=p;contrasts[n]['utc_day_sensitivity']=cluster_contrast(method_rows['P'],method_rows[n],'date')
    classifier={};all_rows=manifest['records'];test_index=[i for i,r in enumerate(all_rows) if r['partition']=='test'];y=np.array([all_rows[i]['label_index'] for i in test_index])
    for seed in (17,29,43):
        pred=np.load(ROOT/f'data/derived/classifier_seed{seed}.npz');classifier[str(seed)]=classifier_metrics(y,pred['logits'][test_index],float(pred['temperature']))
    seed_results={};budget_results={}
    for seed in (17,29,43):
        ens=Ensemble(torch.load(ROOT/f'checkpoints/controller_loss_{seed}.pt',weights_only=False))
        cal=rows_for('P',data['calibration'],settings,ens,class_ensemble,labels);threshold=float(np.sort([r['risk'] for r in cal])[int(np.ceil(.7*len(cal)))-1])
        seed_results[str(seed)]=metrics(rows_for('P',data['test'],settings,ens,class_ensemble,labels,threshold))
    for budget in (1,2,3):
        cal=rows_for('P',data['calibration'],settings,ensemble,class_ensemble,labels,budget=budget);threshold=float(np.sort([r['risk'] for r in cal])[int(np.ceil(.7*len(cal)))-1])
        budget_results[str(budget)]=metrics(rows_for('P',data['test'],settings,ensemble,class_ensemble,labels,threshold,budget))
    p=table['P'];base=table[settings['locked_comparator']];primary=contrasts[settings['locked_comparator']]
    success=bool(primary['difference']>=.05 and primary['ci95'][0]>0 and p['unsupported_rate'] is not None and base['unsupported_rate'] is not None and p['unsupported_rate']<=base['unsupported_rate'] and p['class_macro_f1']>=base['class_macro_f1']-.02 and all(v['difference']>0 and v['p_holm']<.05 for v in contrasts.values()))
    summary={'labels':labels,'assessment_clips':len(data['test']),'assessment_source_groups':len({i['row']['source_group'] for i in data['test']}),
        'assessment_utc_days':len({i['row']['date'] for i in data['test']}),'locked_comparator':settings['locked_comparator'],'methods':table,'primary_contrasts':contrasts,
        'classifier_all_test':classifier,'controller_seeds':seed_results,'tool_budget_sensitivity':budget_results,'preregistered_success_criteria_met':success,
        'human_reviewers':0,'endpoint':'automated proxy supported coverage','latency_is_reconstructed_not_online_benchmark':True}
    atomic_json(ROOT/'results/tables/main_results.json',summary);atomic_json(ROOT/'results/tables/risk_coverage_curves.json',curves)
    atomic_json(ROOT/'results/raw/scored_test_records.json',method_rows)
    with (ROOT/'results/tables/main_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['method']+list(next(iter(table.values()))));writer.writeheader()
        for name,values in table.items():writer.writerow({'method':name,**values})
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
