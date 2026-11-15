"""Aggregate blinded AI assessments without exposing source identities."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from underwater_reasoning.io import atomic_json
from underwater_reasoning.scoring import close_number

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text())
    counts=Counter();by_partition={};slots={s:Counter() for s in ('band','tone','modulation','trend')}
    differences={s:[] for s in slots};interval_differences={s:[] for s in slots}
    for part,ids in manifest['assessments'].items():
        pc=Counter()
        for uid in ids:
            reviews=[]
            for role in (0,1):
                record=json.loads((ROOT/f'results/raw/review/main/{part}/{uid}_{role}.json').read_text())
                result=record['result'];valid=not result['errors'];pc['requested']+=1;pc['valid']+=valid
                counts['truncated']+=result['hit_output_limit'];reviews.append(result['parsed'] if valid else None)
                counts['output_tokens']+=result['output_tokens']
            if not all(reviews):continue
            for slot,c in slots.items():
                a,b=[v[slot] for v in reviews];c['paired_valid']+=1
                c['identifiability_agreement']+=a['identifiable']==b['identifiable']
                c['both_identifiable']+=a['identifiable'] and b['identifiable']
                c['role0_identifiable']+=a['identifiable'];c['role1_identifiable']+=b['identifiable']
                if a['identifiable'] and b['identifiable']:
                    match=a.get('value')==b.get('value')
                    if slot in ('tone','modulation'):
                        key='frequency_hz' if slot=='tone' else 'rate_hz'
                        absolute,relative=(3,.03) if slot=='tone' else (1,.1)
                        match=match and close_number(a.get(key),b.get(key),absolute,relative)
                        if isinstance(a.get(key),(int,float)) and isinstance(b.get(key),(int,float)):
                            differences[slot].append(abs(a[key]-b[key]))
                    c['both_identifiable_value_agreement']+=bool(match)
                    av,bv=a.get('time_s'),b.get('time_s')
                    if isinstance(av,list) and isinstance(bv,list) and len(av)==len(bv)==2 and all(isinstance(x,(int,float)) for x in av+bv):
                        interval_differences[slot].append(float(np.mean(np.abs(np.array(av)-bv))))
        counts.update(pc);by_partition[part]=dict(pc)
    slot_summary={}
    for slot,c in slots.items():
        n=c['paired_valid'];a=c['role0_identifiable']/n;b=c['role1_identifiable']/n;chance=a*b+(1-a)*(1-b)
        slot_summary[slot]={**dict(c),'identifiability_kappa':(c['identifiability_agreement']/n-chance)/(1-chance) if chance<1 else None,
            'paired_numeric_count':len(differences[slot]),'paired_numeric_median_absolute_difference_hz':float(np.median(differences[slot])) if differences[slot] else None,
            'paired_interval_count':len(interval_differences[slot]),'mean_interval_endpoint_difference_seconds':float(np.mean(interval_differences[slot])) if interval_differences[slot] else None}
    summary={'human_reviewers':0,'checkpoint_shared_with_generator':True,'reviewer_roles':2,
        'counts':dict(counts),'partitions':by_partition,'slots':slot_summary,
        'caveat':'Blinded, correlated same-checkpoint AI proxy assessments; agreement is not accuracy or human validation.'}
    atomic_json(ROOT/'results/tables/reviewer_quality.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
