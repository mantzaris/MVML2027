#!/usr/bin/env python3
"""Pin public model revisions and download only weights/tokenizers inside project."""
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
from uac.runtime import atomic_json, EventLedger
ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'manifests/model_lock.json'
models=['Qwen/Qwen3-4B-Instruct-2507','facebook/wav2vec2-base-960h']
lock=json.loads(LOCK.read_text()) if LOCK.exists() else {}
api=HfApi()
for name in models:
    if name not in lock:
        info=api.model_info(name)
        lock[name]={'revision':info.sha,'source':'https://huggingface.co/'+name,'license':str(getattr(info.card_data,'license',None)), 'path':'checkpoints/pretrained/'+name.replace('/','--')}
        atomic_json(LOCK,lock)
    entry=lock[name]
    destination=ROOT/entry['path']
    snapshot_download(name,revision=entry['revision'],local_dir=destination,allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.md'],max_workers=4)
    entry['download_complete']=True
    atomic_json(LOCK,lock)
    EventLedger('runs/events.jsonl').append('model_download_complete',model=name,revision=entry['revision'],path=entry['path'])
    print(name,entry['revision'],flush=True)
