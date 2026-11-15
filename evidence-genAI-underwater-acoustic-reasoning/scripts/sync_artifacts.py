"""Copy remote result shards and checkpoints locally without pushing Git.

Run with explicit SSH options. This keeps ephemeral-worker outputs recoverable.
It never deletes either side and never copies model downloads or raw logs.
"""
import argparse,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--ssh',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        result=subprocess.run(['rsync','-az','--update','--prune-empty-dirs','-e',a.ssh,
            '--exclude=.writing-*','--exclude=checkpoints/checkpoint-*.pt',
            '--include=*/','--include=results/***','--include=checkpoints/***','--include=data/derived/classifier*.npz',
            '--include=data/derived/evidence/***','--include=data/derived/references*/***',
            '--include=data/derived/robustness_classifier.npz','--include=data/private/*.json',
            '--include=data/raw/robustness/***','--exclude=*',
            f'{a.host}:/workspace/evidence-genAI-underwater-acoustic-reasoning/',str(ROOT)+'/'])
        print('artifact_sync_exit',result.returncode,flush=True)
        if a.once:break
        time.sleep(120)
