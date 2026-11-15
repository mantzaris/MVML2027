"""Pre-scale development gate; never reads test terminal results."""
import json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from underwater_reasoning.controller import load_study
from underwater_reasoning.io import atomic_json,file_hash
from underwater_reasoning.runtime import DEADLINE
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    manifest,data=load_study(ROOT,['development']);items=data['development'];pairs=[m for m in range(16) if m.bit_count()==2]
    per_pair={m:float(np.mean([i['scores'][m]['loss'] for i in items])) for m in pairs};best=min(per_pair,key=per_pair.get)
    oracle=float(np.mean([min(i['scores'][m]['loss'] for m in pairs) for i in items]));heterogeneity=Counter(min(pairs,key=lambda m:i['scores'][m]['loss']) for i in items)
    reports=[i['reports'][m]['result'] for i in items for m in range(16)];valid=np.mean([i['scores'][m]['valid'] for i in items for m in range(16)])
    training=json.loads((ROOT/'results/manifests/classifier_training.json').read_text());recovery=all(r['restore_check']['rng_exact'] and r['restore_check']['dev_logits_exact'] for r in training['seeds'])
    seconds=float(np.mean([r['amortized_seconds'] for r in reports]));remaining=(DEADLINE-datetime.now(timezone.utc)).total_seconds()
    result={'development_clips':len(items),'reports':len(reports),'schema_validity':float(valid),'full_format_validity':float(np.mean([not r['errors'] for r in reports])),
        'best_fixed_pair':best,'best_pair_loss':per_pair[best],'pair_losses':per_pair,'oracle_pair_loss':oracle,'oracle_pair_advantage':per_pair[best]-oracle,
        'best_pair_distribution':dict(heterogeneity),'checkpoint_recovery_passed':recovery,'mean_seconds_per_terminal_report':seconds,
        'peak_vram_bytes':max(r['peak_vram_bytes'] for r in reports),'remaining_wall_clock_seconds':remaining,
        'projected_remaining_core_terminal_seconds':seconds*(100+37+100)*17,'dollar_rate_unknown':True,'test_outcomes_examined':False,
        'terminal_prompt_sha256':file_hash(ROOT/'prompts/terminal_v1.txt'),'scoring_sha256':file_hash(ROOT/'src/underwater_reasoning/scoring.py'),
        'passed':bool(valid>=.98 and recovery and per_pair[best]-oracle>.005 and len(heterogeneity)>1)}
    atomic_json(ROOT/'results/manifests/pilot_gate.json',result);print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['passed'] else 2)
