"""Real-report paraphrase, misleading-card interventions, and blinded AI prose audit."""
from __future__ import annotations
import copy,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from underwater_reasoning.omni import Omni,REVISION
from underwater_reasoning.controller import Ensemble,load_study,method_mask
from underwater_reasoning.scoring import score,deterministic_report
from underwater_reasoning.reports import parse_report
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
from underwater_reasoning.runtime import budget_check
from underwater_reasoning.report_schema import terminal_schema
from run_generation import serialize_evidence
from analyze_experiment import cluster_contrast
ROOT=Path(__file__).resolve().parents[1]

def main():
    if not json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise ValueError('Test locked')
    started=time.monotonic();manifest,data=load_study(ROOT,['test']);items=data['test'];settings=json.loads((ROOT/'configs/locked_controller.json').read_text());labels=manifest['labels']
    ensemble=Ensemble(torch.load(ROOT/'checkpoints/controller_loss_17.pt',weights_only=False));runner=Omni(ROOT);base_prompt=(ROOT/'prompts/terminal_v1.txt').read_text();paraphrase=(ROOT/'prompts/terminal_paraphrase.txt').read_text();jobs=[]
    def terminal(item,asset,mask,prompt,kind,condition,metadata=None):
        evidence,counts,trims=serialize_evidence(runner,asset,mask);text=prompt+'\nAvailable evidence: W0 raw audio.\n'+json.dumps(asset['prior'],separators=(',',':'))+'\n'+evidence+'\nProduce the required JSON report.'
        jobs.append({'item':item,'asset':asset,'mask':mask,'mode':'terminal','kind':kind,'condition':condition,'text':text,
            'metadata':metadata or {},'out':ROOT/f'results/raw/report_audits/{kind}/{item["row"]["id"]}_{condition}.json'})
    for item in items:
        mask,_=ensemble.select(item['asset'],settings['beta'],settings['lambda'])
        terminal(item,item['asset'],mask,paraphrase,'paraphrase',mask)
        terminal(item,item['asset'],mask,(ROOT/'prompts/terminal_paraphrase_2.txt').read_text(),'paraphrase_2',mask)
    intervention_items=sorted(items,key=lambda i:canonical_hash([17,'intervention',i['row']['id']]))[:25]
    for index,item in enumerate(intervention_items):
        for condition in ('missing_A','swapped_A','false_A','missing_B'):
            asset=copy.deepcopy(item['asset']);family='B' if condition=='missing_B' else 'A'
            if condition.startswith('missing'):
                asset['cards'][family]={'unavailable':True,'record_hash':asset['waveform_sha256']}
                asset['compact'][family]={'id':family+'0','status':'unavailable','message':'No measurements returned; mark dependent attributes uncertain.'}
            elif condition=='swapped_A':
                other=intervention_items[(index+1)%len(intervention_items)]['asset'];asset['cards']['A']=copy.deepcopy(other['cards']['A']);asset['compact']['A']=copy.deepcopy(other['compact']['A'])
            else:
                card=asset['cards']['A']['measurements'];card['dominant_band_hz']=[2000,8000] if item['reference']['band']['value']!='2000-8000' else [20,100]
                asset['compact']['A']['dominant_band_hz']=card['dominant_band_hz']
                if card['narrowband_candidates']:
                    candidates=item['reference']['tone']['candidates']
                    false_hz=next(float(f) for f in [1777,1234,653,97,*range(31,1990,17)] if not any(abs(f-c['frequency_hz'])<=max(3,.03*c['frequency_hz']) for c in candidates))
                    card['narrowband_candidates'][0]['frequency_hz']=false_hz;card['narrowband_candidates'][0]['frequency_interval_hz']=[false_hz-1,false_hz+1]
                    asset['compact']['A']['tones'][0]['hz']=false_hz
            terminal(item,asset,7,base_prompt,'intervention',condition)
        asset=copy.deepcopy(item['asset'])
        for neighbor in asset['compact']['D']['neighbors']:
            neighbor['label']=labels[(labels.index(neighbor['label'])+1)%len(labels)]
        terminal(item,asset,15,base_prompt,'intervention','corrupt_D_labels')
    removal_eligibility={}
    for condition,want_cited in [('remove_cited',True),('remove_uncited',False)]:
        eligible=[]
        for item in sorted(items,key=lambda i:canonical_hash([17,condition,i['row']['id']])):
            report=item['reports'][15]['result']['parsed']
            if report is None:continue
            citations={citation for slot in ('class','band','tone','modulation','trend') for citation in report[slot]['evidence']}
            families=[f for f in 'ABCD' if (f+'0' in citations)==want_cited]
            if families:eligible.append((item,families[0]))
        removal_eligibility[condition]={'eligible_clips':len(eligible),'selected_clips':min(25,len(eligible))}
        for item,family in eligible[:25]:
            asset=copy.deepcopy(item['asset']);asset['cards'][family]={'unavailable':True,'record_hash':asset['waveform_sha256']}
            asset['compact'][family]={'id':family+'0','status':'unavailable','message':'No measurements returned; mark dependent attributes uncertain.'}
            terminal(item,asset,15,base_prompt,'intervention',condition,{'removed_family':family,'cited_in_clean_report':want_cited})
    prose_items=[]
    for label in labels:
        candidates=[i for i in items if i['row']['label']==label];prose_items+=sorted(candidates,key=lambda i:canonical_hash([17,'prose',i['row']['id']]))[:6]
    for item in prose_items:
        for method in ['P',settings['locked_comparator'],'B5_all','B8_deterministic']:
            mask,_=method_mask(method,item,settings,ensemble,ensemble)
            report=deterministic_report(item['asset'],mask,labels) if method=='B8_deterministic' else item['reports'][mask]['result']['parsed']
            text='You are an AI auditor, not a human expert. Assess this anonymous report against the raw audio and independent acoustic diagnostics. Do not infer the true vessel class. Do not assume another model is correct. Return only JSON with boolean fields unsupported_mechanical_claim, new_unsupported_physical_claim, slot_rationale_inconsistency, overstated_certainty, and note (one short sentence). A received energy trend does not establish motion. Exact machinery, speed, identity, position, or source causation is unsupported. Explicit uncertainty is not itself an error.\nIndependent diagnostics:\n'+json.dumps(item['reference'],separators=(',',':'))+'\nAnonymous report:\n'+json.dumps(report,separators=(',',':'))
            jobs.append({'item':item,'asset':item['asset'],'mask':mask,'mode':'prose','kind':'prose','condition':method,'text':text,
                'out':ROOT/f'results/raw/report_audits/prose/{item["row"]["id"]}_{method}.json'})
    remaining=[]
    for job in jobs:
        job['key']={'model_revision':REVISION,'waveform_sha256':job['asset']['waveform_sha256'],'prompt_sha256':canonical_hash(job['text']),
            'kind':job['kind'],'condition':job['condition'],'presented_evidence_sha256':canonical_hash(job['asset']['compact']),
            'precision':'bfloat16','max_new_tokens':1024 if job['mode']=='terminal' else 768,'greedy':True,'batch_size':16}
        if job['mode']=='terminal':job['key']['format_schema_sha256']=canonical_hash(terminal_schema());job['key']['format_enforcer']='xgrammar==0.2.9'
        if job['out'].exists():
            if json.loads(job['out'].read_text())['key']!=job['key']:raise ValueError('Audit cache mismatch')
        else:remaining.append(job)
    batches=[]
    for mode in ('terminal','prose'):
        subset=[j for j in remaining if j['mode']==mode];batches.extend(subset[start:start+16] for start in range(0,len(subset),16))
    completed=0
    for batch in batches:
        budget_check()
        for job in batch:job['audio'],job['sample_rate']=sf.read(ROOT/job['item']['row']['audio_path'],dtype='float32')
        results=runner.generate(batch,report_schema=batch[0]['mode']=='terminal')
        for job,result in zip(batch,results):
            if job['mode']=='terminal':parsed,errors=parse_report(result['raw'])
            else:
                try:
                    raw=result['raw'].strip()
                    if raw.startswith('```'):raw=raw[raw.find('\n')+1:].rsplit('```',1)[0]
                    parsed=json.loads(raw);errors=[]
                    for field in ('unsupported_mechanical_claim','new_unsupported_physical_claim','slot_rationale_inconsistency','overstated_certainty'):
                        if not isinstance(parsed.get(field),bool):errors.append(field+'_contract')
                except (ValueError,AttributeError):parsed=None;errors=['audit_parse']
            atomic_json(job['out'],{'id':job['item']['row']['id'],'key':job['key'],'result':{**result,'parsed':parsed,'errors':errors}});del job['audio']
        completed+=len(batch);print('report audits',completed,len(remaining),round(time.monotonic()-started),flush=True)
    collected={'paraphrase':[],'paraphrase_2':[],'intervention':[],'prose':[]}
    for job in jobs:
        record=json.loads(job['out'].read_text());result=record['result'];item=job['item']
        if job['mode']=='terminal':
            current=score(result['parsed'],item['row']['label'],item['reference'],job['asset'],job['mask'],item['reviews'],result['errors'])
            baseline=item['scores'][job['mask']]
            collected[job['kind']].append({'id':item['row']['id'],'source_group':item['row']['source_group'],'condition':job['condition'],'metadata':job['metadata'],'score':current,'clean_score':baseline})
        else:collected['prose'].append({'id':item['row']['id'],'source_group':item['row']['source_group'],'method':job['condition'],'parsed':result['parsed'],'errors':result['errors']})
    def comparison(rows):
        a=sum(r['score']['assertions'] for r in rows);base=sum(r['clean_score']['assertions'] for r in rows)
        paired_left=[{'id':r['id'],'source_group':r['source_group'],'retained':True,'score':r['score']} for r in rows]
        paired_right=[{**r,'score':original['clean_score']} for r,original in zip(paired_left,rows)]
        return {'n':len(rows),'paired_source_supported_difference':cluster_contrast(paired_left,paired_right),'schema_validity':float(np.mean([r['score']['valid'] for r in rows])),
            'supported_coverage':float(np.mean([r['score']['supported']/4 for r in rows])),
            'clean_supported_coverage':float(np.mean([r['clean_score']['supported']/4 for r in rows])),
            'unsupported_rate':sum(r['score']['unsupported'] for r in rows)/a if a else None,
            'clean_unsupported_rate':sum(r['clean_score']['unsupported'] for r in rows)/base if base else None,
            'assertion_coverage':a/(4*len(rows)),'clean_assertion_coverage':base/(4*len(rows)),
            'class_accuracy':float(np.mean([r['score']['class_correct'] for r in rows])),
            'clean_class_accuracy':float(np.mean([r['clean_score']['class_correct'] for r in rows])),
            'citation_fidelity_on_assertions':sum(c['citation_supported'] for r in rows for c in r['score']['physical'])/a if a else None,
            'appropriate_uncertainty_count':sum(c['appropriate_uncertainty'] for r in rows for c in r['score']['physical']),
            'clean_appropriate_uncertainty_count':sum(c['appropriate_uncertainty'] for r in rows for c in r['clean_score']['physical']),
            'mean_loss_change':float(np.mean([r['score']['loss']-r['clean_score']['loss'] for r in rows]))}
    prose={}
    for method in ['P',settings['locked_comparator'],'B5_all','B8_deterministic']:
        rows=[r for r in collected['prose'] if r['method']==method];valid=[r for r in rows if not r['errors']]
        prose[method]={'requested':len(rows),'valid_audits':len(valid),'flag_rates':{field:float(np.mean([r['parsed'][field] for r in valid])) if valid else None for field in ('unsupported_mechanical_claim','new_unsupported_physical_claim','slot_rationale_inconsistency','overstated_certainty')}}
    atomic_json(ROOT/'results/tables/report_audits.json',{'paraphrase':comparison(collected['paraphrase']),'paraphrase_2':comparison(collected['paraphrase_2']),
        'evidence_interventions':{condition:comparison([r for r in collected['intervention'] if r['condition']==condition]) for condition in ('missing_A','swapped_A','false_A','missing_B','corrupt_D_labels','remove_cited','remove_uncited') if any(r['condition']==condition for r in collected['intervention'])},
        'removal_eligibility':removal_eligibility,'intervention_scope':'Fixed acquired evidence; no post-corruption policy rerun or action-change claim.',
        'prose':prose,'seconds':time.monotonic()-started,'new_calls':len(remaining),'human_reviewers':0,'exploratory':True,
        'caveat':'The AI auditor shares the generator checkpoint. Swapped-card provenance is retained privately but is not visible as a source hash in the bounded prompt.'})
    atomic_json(ROOT/'results/raw/scored_report_audits.json',collected)

if __name__=='__main__':main()
