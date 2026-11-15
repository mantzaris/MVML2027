"""Verify a uniform reference revision without inspecting test model outputs."""
import json
from collections import Counter
from pathlib import Path
from underwater_reasoning.io import atomic_json,canonical_hash
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text())
summary={};pairs=[]
for part,ids in manifest['assessments'].items():
    counts=Counter()
    for uid in ids:
        old=json.loads((ROOT/f'data/derived/references_v1/{uid}.json').read_text())
        new=json.loads((ROOT/f'data/derived/references/{uid}.json').read_text())
        assert old['waveform_sha256']==new['waveform_sha256']
        assert old['reference']['version']=='independent-proxy-1.0' and new['reference']['version']=='independent-proxy-1.2'
        for slot in ('band','tone','trend'):assert old['reference'][slot]==new['reference'][slot]
        a,b=old['reference']['modulation'],new['reference']['modulation']
        counts['clips']+=1;counts['old_identifiable']+=a['identifiable'];counts['new_identifiable']+=b['identifiable']
        counts['identifiability_changed']+=a['identifiable']!=b['identifiable'];counts['modulation_record_changed']+=a!=b
        pairs.append([uid,canonical_hash(old),canonical_hash(new)])
    summary[part]=dict(counts)
atomic_json(ROOT/'results/manifests/reference_revision_audit.json',{'partitions':summary,'paired_reference_digest':canonical_hash(pairs),
    'other_physical_slots_unchanged':True,'waveforms_unchanged':True,'test_terminal_outcomes_used':False})
print(json.dumps(summary,indent=2))
