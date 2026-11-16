"""Development-only precision diagnostic, before test terminal outcomes."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from underwater_reasoning.controller import Ensemble,load_study,method_mask
from underwater_reasoning.io import atomic_json

ROOT=Path(__file__).resolve().parents[1]
if json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise SystemExit('This diagnostic must precede test unlock')
manifest,data=load_study(ROOT,['development']);items=data['development'];settings=json.loads((ROOT/'configs/locked_controller.json').read_text())
formats={}
for part in ('policy_fit','development','calibration'):
    results=[json.loads(path.read_text())['result'] for path in (ROOT/f'results/raw/terminal/main/{part}').glob('*.json')]
    if len(results)!=17*len(manifest['assessments'][part]):raise ValueError('Incomplete pre-test matrix')
    formats[part]={'reports':len(results),'schema_valid':sum(not [e for e in r['errors'] if e!='interpretation_word_count'] for r in results),
        'format_valid':sum(not r['errors'] for r in results),'output_limit_hits':sum(bool(r.get('hit_output_limit')) for r in results)}
    if formats[part]['schema_valid']/len(results)<.98:raise ValueError('Pre-test matrix schema validity below 98%')
ens=Ensemble(torch.load(ROOT/'checkpoints/controller_loss_17.pt',weights_only=False));base=settings['locked_comparator'];group_differences={}
for item in items:
    p,_=method_mask('P',item,settings,ens,ens);b,_=method_mask(base,item,settings,ens,ens)
    group_differences.setdefault(item['row']['source_group'],[]).append((item['scores'][p]['supported']-item['scores'][b]['supported'])/4)
means=np.array([np.mean(v) for v in group_differences.values()]);sd=float(means.std(ddof=1))
lookup={r['id']:r for r in manifest['records']};test_groups=len({lookup[uid]['source_group'] for uid in manifest['assessments']['test']})
record={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'test_terminal_outcomes_examined':False,'development_clips':len(items),
    'development_groups':len(means),'planned_test_groups':test_groups,'locked_comparator':base,
    'pretest_format_by_partition':formats,'invalid_reports_retained_as_failures':True,
    'development_mean_source_difference_before_rejection':float(means.mean()),'development_source_difference_sd':sd,
    'rough_unfiltered_95pct_half_width_at_test_group_count':1.96*sd/np.sqrt(test_groups),
    'rough_unfiltered_80pct_power_effect_at_test_group_count':(1.96+.84)*sd/np.sqrt(test_groups),
    'scope':'Optimistic normal-approximation planning diagnostic from development group means before rejection. It is not power for the calibrated conditional endpoint or Holm family; tuning on development and source/day dependence can invalidate transfer.',
    'decision':'Retain the prospectively capped 100-clip test assessment and report its actual intervals, including inconclusive or negative outcomes. Do not increase sample size after seeing test results.'}
atomic_json(ROOT/'results/manifests/pretest_design_audit.json',record);print(json.dumps(record,indent=2))
