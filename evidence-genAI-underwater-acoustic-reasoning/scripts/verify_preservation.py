"""Index worker artifacts, then verify their independent local copies.

Worker: python scripts/verify_preservation.py --write-worker-index
Local, after artifact sync: python scripts/verify_preservation.py --verify-local
The detailed path index stays outside Git; the public audit contains aggregates.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash

ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'results/raw/preservation_index.json'

def paths():
    patterns={
        'checkpoints':['checkpoints/*.pt'],
        'classifier_caches':['data/derived/classifier*.npz','data/derived/robustness_classifier.npz'],
        'references':['data/derived/references*/*.json'],
        'evidence':['data/derived/evidence/*.json'],
        'audio':['data/raw/experiment_audio/**/*.wav','data/raw/robustness/*.wav','data/raw/release_probe/*.wav'],
        'private_manifests':['data/private/*.json'],
        'predictions_and_audits':['results/raw/**/*.json','results/raw/**/*.py'],
        'job_records':['results/manifests/jobs/*.json'],
        'worker_logs':['results/logs/worker/*.log'],
        'locked_protocol':['configs/locked_controller.json','protocol/test_unlock_log.json'],
    }
    selected={}
    for category,globs in patterns.items():
        for pattern in globs:
            for path in ROOT.glob(pattern):
                if path.is_file() and path!=INDEX and not path.name.startswith('checkpoint-'):
                    selected[str(path.relative_to(ROOT))]=(category,path)
    return selected

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write-worker-index',action='store_true');group.add_argument('--verify-local',action='store_true');args=parser.parse_args()
    if args.write_worker_index:
        records=[{'path':relative,'category':category,'bytes':path.stat().st_size,'sha256':file_hash(path)} for relative,(category,path) in sorted(paths().items())]
        atomic_json(INDEX,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'records':records})
        print('Indexed',len(records),'worker files; detailed index remains outside Git');return
    records=json.loads(INDEX.read_text())['records'];counts=defaultdict(lambda:{'files':0,'bytes':0});mismatches=[]
    for record in records:
        path=ROOT/record['path'];category=counts[record['category']]
        category['files']+=1;category['bytes']+=record['bytes']
        if not path.is_file() or path.stat().st_size!=record['bytes'] or file_hash(path)!=record['sha256']:mismatches.append(record['path'])
    audit={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'all_match':not mismatches,'mismatch_count':len(mismatches),
        'files':len(records),'bytes':sum(r['bytes'] for r in records),'categories':dict(counts),
        'worker_index_sha256':file_hash(INDEX),'content_digest':canonical_hash(records),
        'private_mismatch_record':'results/raw/preservation_mismatches.json' if mismatches else None,
        'excluded_recomputable_assets':['pretrained model download caches','mel-image feature caches','Python environments'],
        'trained_checkpoints':[r for r in records if r['category']=='checkpoints']}
    atomic_json(ROOT/'results/manifests/final_preservation.json',audit)
    if mismatches:atomic_json(ROOT/'results/raw/preservation_mismatches.json',mismatches)
    print(json.dumps({k:v for k,v in audit.items() if k!='trained_checkpoints'},indent=2))
    if mismatches:raise SystemExit('Local preservation mismatch; do not finalize')

if __name__=='__main__':main()
