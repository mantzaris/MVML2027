#!/usr/bin/env python3
"""Combine complete empirical studies, preserving every frozen analysis table."""
import json
from pathlib import Path
from uac.runtime import atomic_json,sha256_file,utc_now
from uac.protocol import require_frozen_test

def load(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise RuntimeError(message)
def grid(rows,fields,expected,name):
    observed=[tuple(r[k] for k in fields) for r in rows]
    require(len(observed)==len(set(observed)) and set(observed)==set(expected),'Incomplete/duplicate '+name+' grid')
def index_integrity(document,name):
    for row in document.get('indices',[]):
        require(sha256_file(row['path'])==row['sha256'],name+' index checksum mismatch')
        index=load(row['path']);require(index['status']=='complete',name+' index incomplete')
        require(index.get('freeze_sha256')==sha256_file('runs/test_freeze.json'),name+' index from another protocol')

freeze=require_frozen_test()
real=load('results/tables/real_results.json');robustness=load('results/tables/robustness.json')
cost=load('results/tables/runtime_benchmark.json');semi=load('results/tables/semisynthetic_results.json')
synthetic=load('reports/synthetic_results.json');views=load('results/tables/supplemental_views.json')
cal=load('manifests/calibration.json');sc=load('configs/synthetic_temporal.json');vc=load('configs/supplemental.json');rc=load('configs/robustness.json');mc=load('configs/semisynthetic.json')
require(real['status']=='core_test_complete_extensions_pending','Core real analysis incomplete')
grid(real['records'],['method','seed'],[(m,int(s)) for s,fit in cal['seeds'].items() for m in fit['thresholds']],'real test')
require(all(r['split']=='test' and r['n']==1200 for r in real['records']),'Wrong real test denominator')
require(synthetic['completed_conditions']==synthetic['expected_test_conditions']==len(sc['seeds'])*len(sc['test_families'])*(1+len(sc['test_snr_db'])),'Synthetic condition horizon incomplete')
conditions=['mixed_snr']+[f'snr_{s:+03d}' for s in sc['test_snr_db']]
grid(synthetic['records'],['seed','family','condition','iou_threshold'],[(s,f,c,i) for s in sc['seeds'] for f in sc['test_families'] for c in conditions for i in sc['temporal_iou_thresholds']],'synthetic temporal')
require(all(r['n_scenes']==sc['test_scenes'] and r['split']=='test' for r in synthetic['records']),'Wrong synthetic denominator')
require(views['status']=='complete','Supplemental view study incomplete')
grid(views['view_ablation'],['method','seed','split'],[(m,s,t) for m in vc['methods'] for s in vc['seeds'] for t in vc['evaluation_splits']],'supplemental view')
require(len(views['regime_ablation'])==len(views['view_ablation'])*len(vc['regimes']),'Supplemental regimes incomplete')
require(len(views['missing_view'])==sum(len(v) for v in vc['methods'].values())*len(vc['seeds'])*len(vc['evaluation_splits']),'Missing-view horizon incomplete')
require(semi['status']=='complete' and semi['completed_conditions']==semi['expected_conditions']==len(mc['seeds'])*len(mc['component_snr_db']),'Semi-synthetic horizon incomplete')
require(semi['base_scenes']==330 and semi['condition_scenes']==990,'Semi-synthetic denominator changed')
grid(semi['records'],['seed','component_snr_db','iou_threshold'],[(s,c,i) for s in mc['seeds'] for c in mc['component_snr_db'] for i in sc['temporal_iou_thresholds']],'semi-synthetic')
require(robustness['status']=='complete' and robustness['completed_conditions']==robustness['expected_conditions']==len(rc['conditions']),'Robustness horizon incomplete')
grid(robustness['records'],['method','condition'],[(m,c['name']) for m in rc['methods'] for c in rc['conditions']],'robustness')
require(all(r['n']==120 and r['seed']==17 for r in robustness['records']),'Wrong robustness denominator/seed')
require(cost['status']=='complete','Live runtime benchmark incomplete')
require(set(r['method'] for r in cost['records'])=={'main','fixed_verification','proposed_agent','agent_no_generation','heuristic_verification'},'Runtime method coverage incomplete')
require(all(r['mode']=='live' and r['milliseconds_per_clip']>0 and r['peak_memory_mib']>0 for r in cost['records']),'Invalid or replay-only runtime result')
for name,document in [('synthetic',synthetic),('semi-synthetic',semi),('robustness',robustness)]:index_integrity(document,name)
real['robustness']=robustness['records'];real['cost']=cost['records'];real['semisynthetic_summary']=semi
real['status']='complete';real['assembled_utc']=utc_now()
paths=['results/tables/real_results.json','results/tables/robustness.json','results/tables/runtime_benchmark.json','results/tables/semisynthetic_results.json','reports/synthetic_results.json','results/tables/supplemental_views.json']
real['source_artifacts']={p:sha256_file(p) for p in paths}
atomic_json('results/tables/final_real_results.json',real)
print(json.dumps({'status':'complete','real_records':len(real['records']),'synthetic_records':len(synthetic['records']),'supplemental_records':len(views['view_ablation']),'semisynthetic_records':len(semi['records']),'robustness_records':len(real['robustness']),'runtime_records':len(real['cost'])}),flush=True)
