"""Independent-reference correctness and cited-card fidelity, kept separate."""
from __future__ import annotations
import copy
import numpy as np
from .reports import SLOTS,parse_report

PHYSICAL=('band','tone','modulation','trend')

def interval_covered(span,support,tolerance=.02):
    return isinstance(span,(list,tuple)) and isinstance(support,(list,tuple)) and len(span)==len(support)==2 and all(isinstance(x,(int,float)) for x in [*span,*support]) and support[0]-tolerance<=span[0]<span[1]<=support[1]+tolerance

def close_number(a,b,absolute,relative):
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and abs(a-b)<=max(absolute,relative*abs(b))

def agrees(slot,claim,ref):
    if not ref.get('identifiable') or claim.get('value')!=ref.get('value'):return False
    if slot in ('band','trend'):return True
    key='frequency_hz' if slot=='tone' else 'rate_hz';absolute,relative=(3,.03) if slot=='tone' else (1,.10)
    span=claim.get('time_s',[0,5])
    for candidate in ref.get('candidates',[]):
        if not close_number(claim.get(key),candidate[key],absolute,relative):continue
        if slot=='tone':
            duration=span[1]-span[0];overlap=max(0,min(span[1],candidate['time_s'][1])-max(span[0],candidate['time_s'][0]))
            if duration>=2.5 and overlap>=.8*duration:return True
        elif interval_covered(span,candidate['time_s']):return True
    return False

def reviewer_agrees(slot,claim,reviews):
    if len(reviews)!=2:return False
    for review in reviews:
        if not review or not isinstance(review.get(slot),dict):return False
        value=review[slot]
        if value.get('identifiable') is not True or value.get('value')!=claim.get('value'):return False
        if not interval_covered(claim.get('time_s'),value.get('time_s')):return False
        if slot=='tone' and not close_number(claim.get('frequency_hz'),value.get('frequency_hz'),3,.03):return False
        if slot=='modulation' and not close_number(claim.get('rate_hz'),value.get('rate_hz'),1,.10):return False
    return True

def citation_support(slot,claim,assets,mask,reference,reviews):
    cites=claim.get('evidence',[])
    if not cites or len(cites)!=len(set(cites)):return False
    available={'W0'}|({'P0'} if mask!=-1 else set())|{f+'0' for a,f in enumerate('ABCD') if max(mask,0)&(1<<a)}
    if any(c not in available for c in cites):return False
    span=claim.get('time_s',[0,5]);valid=[]
    for cite in cites:
        if slot=='class':valid.append(cite in {'W0','P0','D0'});continue
        if cite=='W0':valid.append(slot in ('band','trend') and agrees(slot,claim,reference[slot]) and reviewer_agrees(slot,claim,reviews));continue
        if cite in {'A0','B0','C0'}:
            card=assets['cards'].get(cite[0])
            if not card or card.get('unavailable') or card.get('record_hash')!=assets.get('waveform_sha256'):
                valid.append(False);continue
        if cite=='A0' and slot=='band':
            band=assets['cards']['A']['measurements']['dominant_band_hz'];valid.append(band is not None and claim.get('value')==f'{band[0]}-{band[1]}' and interval_covered(span,[0,5]))
        elif cite=='A0' and slot=='tone':
            valid.append(any(close_number(claim.get('frequency_hz'),v['frequency_hz'],1.01,0) and any(interval_covered(span,s) for s in v['support_spans_seconds']) and span[1]-span[0]>=2.5 for v in assets['cards']['A']['measurements']['narrowband_candidates']))
        elif cite=='B0' and slot=='modulation':
            valid.append(any(close_number(claim.get('rate_hz'),v['rate_hz'],.51,0) and interval_covered(span,v['time_span_seconds']) for v in assets['cards']['B']['measurements']['modulation_candidates']))
        elif cite=='C0' and slot=='trend':
            valid.append(claim.get('value')==assets['cards']['C']['measurements']['energy_trend_candidate'] and interval_covered(span,[0,5]))
        else:valid.append(False)
    return bool(all(valid))

def score(report,label,reference,assets,mask,reviews=(),errors=()):
    structural=[e for e in errors if e!='interpretation_word_count']
    if report is None or structural:
        return {'loss':1.,'class_loss':1.,'physical_loss':1.,'assertions':0,'supported':0,'unsupported':0,'consensus_supported':0,
            'factual_errors':0,'identifiable':sum(reference[s]['identifiable'] for s in PHYSICAL),'valid':False,'format_valid':False,'class_correct':False,'class_asserted':False,'physical':[]}
    class_claim=report['class'];class_asserted=class_claim['status']=='asserted';class_correct=class_asserted and class_claim['value']==label
    class_loss=0. if class_correct else 1. if class_asserted else .25
    records=[]
    for slot in PHYSICAL:
        claim=report[slot];ref=reference[slot];asserted=claim['status']=='asserted';ident=bool(ref['identifiable'])
        correct=asserted and agrees(slot,claim,ref);cited=asserted and citation_support(slot,claim,assets,mask,reference,reviews)
        supported=bool(correct and cited);consensus=bool(supported and reviewer_agrees(slot,claim,reviews))
        loss=(0. if supported else 1.) if asserted else (.25 if ident else 0.)
        numeric_error=None;matched_error=None
        if asserted and ident and slot in ('tone','modulation'):
            key='frequency_hz' if slot=='tone' else 'rate_hz'
            if isinstance(claim.get(key),(int,float)) and ref.get('candidates'):
                numeric_error=min(abs(claim[key]-v[key]) for v in ref['candidates'])
                matches=[v for v in ref['candidates'] if agrees(slot,claim,{**ref,'candidates':[v]})]
                if matches:matched_error=min(abs(claim[key]-v[key]) for v in matches)
        records.append({'slot':slot,'asserted':asserted,'identifiable':ident,'correct':bool(correct),'citation_supported':bool(cited),'has_citation':bool(asserted and claim.get('evidence')),
            'supported':supported,'consensus_supported':consensus,'factual_error':bool(asserted and ident and not correct),'loss':loss,
            'appropriate_uncertainty':bool(not asserted and not ident),'numeric_absolute_error':numeric_error,'numeric_matched_absolute_error':matched_error})
    physical_loss=float(np.mean([r['loss'] for r in records]));assertions=sum(r['asserted'] for r in records);supported=sum(r['supported'] for r in records)
    return {'loss':.5*class_loss+.5*physical_loss,'class_loss':class_loss,'physical_loss':physical_loss,'assertions':assertions,'supported':supported,
        'unsupported':assertions-supported,'consensus_supported':sum(r['consensus_supported'] for r in records),'factual_errors':sum(r['factual_error'] for r in records),
        'identifiable':sum(r['identifiable'] for r in records),'valid':True,'format_valid':not errors,'class_correct':bool(class_correct),'class_asserted':class_asserted,'physical':records}

def deterministic_report(assets,mask,labels):
    prior=assets['prior']['class_probabilities'];label=max(prior,key=prior.get)
    report={s:{'status':'uncertain','value':None,'evidence':[],'time_s':[0,5]} for s in SLOTS}
    report['tone']['frequency_hz']=None;report['modulation']['rate_hz']=None
    report['class'].update(status='asserted',value=label,evidence=['P0'])
    if mask&1:
        a=assets['cards']['A']['measurements'];band=a['dominant_band_hz']
        if band:report['band'].update(status='asserted',value=f'{band[0]}-{band[1]}',evidence=['A0'])
        if a['narrowband_candidates']:
            v=a['narrowband_candidates'][0];span=max(v['support_spans_seconds'],key=lambda s:s[1]-s[0])
            report['tone'].update(status='asserted',value='present',frequency_hz=v['frequency_hz'],time_s=span,evidence=['A0'])
    if mask&2 and assets['cards']['B']['measurements']['modulation_candidates']:
        v=assets['cards']['B']['measurements']['modulation_candidates'][0]
        report['modulation'].update(status='asserted',value='present',rate_hz=v['rate_hz'],time_s=v['time_span_seconds'],evidence=['B0'])
    if mask&4:
        v=assets['cards']['C']['measurements']['energy_trend_candidate']
        if v!='uncertain':report['trend'].update(status='asserted',value=v,evidence=['C0'])
    report['interpretation']='The vessel category follows the supplied acoustic classifier prior and remains an estimate. The physical fields reproduce only measurements made by the selected analytical tools. Unavailable measurements remain uncertain, and a missing candidate does not establish that a feature is absent. These digital recordings do not provide calibrated sound pressure, source position, vessel identity, mechanical speed, or motion. Independent acoustic checks are still needed before interpreting these measurements as physical facts.'
    return report
