"""Audit complete saved terminal matrices, including reused and failed outputs."""
import json
from collections import Counter
from pathlib import Path
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
from underwater_reasoning.reports import parse_report
from underwater_reasoning.report_schema import terminal_schema

ROOT=Path(__file__).resolve().parents[1]
if not json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise SystemExit('Test locked')
manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());schema_hash=canonical_hash(terminal_schema())
summary={};keys=[]
for part,ids in manifest['assessments'].items():
    counts=Counter();errors=Counter();tokens=0;peak=0
    for uid in ids:
        asset=json.loads((ROOT/f'data/derived/evidence/{uid}.json').read_text())
        for mask in list(range(16))+[-1]:
            path=ROOT/f'results/raw/terminal/main/{part}/{uid}_{mask}.json';record=json.loads(path.read_text());key=record['key'];result=record['result']
            if key['waveform_sha256']!=asset['waveform_sha256'] or key['format_schema_sha256']!=schema_hash or key['format_enforcer']!='xgrammar==0.2.9':raise ValueError('Incompatible terminal cache')
            parsed,found=parse_report(result['raw'])
            if parsed!=result['parsed'] or found!=result['errors']:raise ValueError('Saved parse differs from current deterministic parser')
            counts['reports']+=1;counts['schema_valid']+=not [e for e in found if e!='interpretation_word_count'];counts['format_valid']+=not found
            counts['output_limit_hits']+=bool(result.get('hit_output_limit'));tokens+=result['output_tokens'];peak=max(peak,result['peak_vram_bytes']);errors.update(found)
            keys.append([part,uid,mask,canonical_hash(key),file_hash(path)])
    summary[part]={**dict(counts),'clips':len(ids),'output_tokens':tokens,'peak_vram_bytes':peak,'errors':dict(errors)}
atomic_json(ROOT/'results/manifests/final_cache_audit.json',{'partitions':summary,'terminal_reports':sum(v['reports'] for v in summary.values()),
    'terminal_cache_digest':canonical_hash(keys),'format_schema_sha256':schema_hash,'failed_reports_retained':True,
    'independent_semantic_correctness_is_separate':True})
print(json.dumps(summary,indent=2))
