"""Pinned public model acquisition; CPU and disk only, no paid APIs."""
import argparse,json,time
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'omni':('Qwen/Qwen2.5-Omni-7B','ae9e1690543ffd5c0221dc27f79834d0294cba00'),
 'planner':('Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28')}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);a=p.parse_args()
    repo,revision=MODELS[a.model];t=time.monotonic()
    path=snapshot_download(repo,revision=revision,cache_dir=str(ROOT/'.cache/huggingface'),
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE','README.md'],max_workers=4)
    (ROOT/'.cache'/f'{a.model}_path.txt').write_text(path)
    print(json.dumps({'model':repo,'revision':revision,'path':path,'seconds':time.monotonic()-t}),flush=True)
