"""Separate slot stability from prose stability in the saved repeated-input smoke."""
import json
from pathlib import Path
from underwater_reasoning.io import atomic_json,file_hash

ROOT=Path(__file__).resolve().parents[1];path=ROOT/'results/raw/grammar_repeat.json';reports=json.loads(path.read_text())['reports']
fields=('class','band','tone','modulation','trend','interpretation');groups=[]
for uid in sorted({r['id'] for r in reports}):
    values=[r['parsed'] for r in reports if r['id']==uid]
    groups.append({'training_probe':uid,'repeats':len(values),'exact_fields':{key:all(v[key]==values[0][key] for v in values) for key in fields}})
record={'reports':len(reports),'unique_training_probes':len(groups),'comparison':'Three copies of each training probe within one 15-item BF16 greedy batch; not independent generator replications.',
    'all_five_slots_exact':all(all(g['exact_fields'][k] for k in fields[:5]) for g in groups),
    'prose_exact_groups':sum(g['exact_fields']['interpretation'] for g in groups),
    'raw_probe_sha256':file_hash(path),'groups':groups,
    'interpretation':'Structured observations agree on these probes, while wording can vary with batch position. This does not guarantee bitwise reproducibility or stability on every test clip.'}
atomic_json(ROOT/'results/manifests/repeat_stability.json',record);print(json.dumps(record,indent=2))
