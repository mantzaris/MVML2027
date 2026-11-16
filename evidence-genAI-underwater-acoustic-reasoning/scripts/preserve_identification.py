"""Index the follow-up's worker artifacts and verify their local preservation."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from underwater_reasoning.io import atomic_json, file_hash

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker-index',action='store_true')
    parser.add_argument('--verify-local',action='store_true')
    args=parser.parse_args()
    index_path=ROOT/'data/private/identification_preservation_index.json'
    if args.worker_index:
        paths=set()
        for manifest in ['data/private/identification_manifest.json','data/private/identification_evaluation_manifest.json','data/private/untrained_categories_manifest.json']:
            paths.add(manifest)
            for row in json.loads((ROOT/manifest).read_text())['records']:paths.add(row['audio_path'])
        for pattern in ['data/derived/identification_*.npz','data/private/*identification*.json',
                        'data/private/untrained_categories*.json','results/raw/identification/**/*',
                        'results/manifests/jobs/identification*.json','results/manifests/jobs/untrained_categories*.json']:
            paths.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file() and p!=index_path)
        paths.update(['checkpoints/identification_fusion.joblib','.cache/identification_followup.log',
                      '.cache/identification_omni_pilot.log','.cache/untrained_categories_evaluation.log'])
        records=[{'path':path,'bytes':(ROOT/path).stat().st_size,'sha256':file_hash(ROOT/path)} for path in sorted(paths)]
        atomic_json(index_path,{'recorded_utc':datetime.now(timezone.utc).isoformat(),'files':records})
        (ROOT/'.cache/identification_preservation_paths.txt').write_text('\n'.join(sorted(paths))+'\n')
        print('Worker preservation index',len(records),'files',sum(r['bytes'] for r in records),'bytes')
    if args.verify_local:
        index=json.loads(index_path.read_text());mismatches=[]
        for record in index['files']:
            path=ROOT/record['path']
            if not path.exists() or path.stat().st_size!=record['bytes'] or file_hash(path)!=record['sha256']:
                mismatches.append(record['path'])
        summary={'verified_utc':datetime.now(timezone.utc).isoformat(),'files':len(index['files']),
                 'bytes':sum(r['bytes'] for r in index['files']),'all_match':not mismatches,
                 'mismatches':mismatches,'index_sha256':file_hash(index_path),
                 'scope':'fresh and stress audio, numerical arrays, raw generation, trained follow-up model, execution records',
                 'regenerable_mel_image_caches_included':False,'large_artifacts_excluded_from_git':True}
        atomic_json(ROOT/'results/manifests/identification_preservation.json',summary)
        print(json.dumps(summary,indent=2))
        if mismatches:raise ValueError('Preservation mismatches')


if __name__=='__main__':main()
