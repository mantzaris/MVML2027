#!/usr/bin/env python3
"""Bounded prepared-evidence wall timing for v3 candidate and reconstruction models.

Includes packet packing/unpacking, CPU orchestration, H2D evidence/metadata,
GPU feature fusion, belief inference, candidate ranking and D2H top-three IDs.
Excludes sensing, bandwidth admission, topology/catalogue construction, network
transport delays, cold model loading and complete operational cyber planning.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from regen import Model,generate
from regen_v2 import RegeneratorV2
from run_study import subset
from run_v3 import NEW_KEYS,predictions,candidate_utilities
from regen_v3 import make_model
from measure_pipeline import prepare


def main():
    manifest=ROOT/'results/v3/manifest.json'
    if not manifest.is_file():raise RuntimeError('Complete v3 study first')
    if not torch.cuda.is_available() or '5000 Ada' not in torch.cuda.get_device_name():raise RuntimeError('Attached Ada required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True)
    models={};checkpoints={}
    for key in NEW_KEYS+['ID','JH']:
        if key in NEW_KEYS:
            model=make_model(key).cuda();path=ROOT/'results/checkpoints/v3'/f'37_{key}.pt'
        else:
            model=RegeneratorV2(key).cuda();path=ROOT/'results/checkpoints/v2'/f'37_{key}.pt'
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));models[key]=model.eval()
        checkpoints[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    thresholds=json.loads((ROOT/'results/selected_thresholds.json').read_text())['rows']
    threshold={r['method']:r['threshold'] for r in thresholds if r['seed']==37}
    reference=subset(generate(4000,62027,'cuda',family='test'),slice(0,1))
    host={k:reference[k].cpu() for k in ['x','reports','observed','adj','req','values','costs','parent']}
    rows=[]
    with torch.inference_mode():
        restored,payload_bytes=prepare(host)
        torch.testing.assert_close(restored['x'],reference['x'],atol=1e-6,rtol=0)
        for method in NEW_KEYS+['ID','JH']:
            p,q=predictions(method,models,restored,threshold=threshold.get(method,0.))
            pe,qe=predictions(method,models,reference,threshold=threshold.get(method,0.))
            if p is not None:torch.testing.assert_close(p,pe,atol=1e-6,rtol=0)
            torch.testing.assert_close(q,qe,atol=1e-6,rtol=0)
            durations=[]
            for rep in range(55):
                torch.cuda.synchronize();start=time.perf_counter()
                b,_=prepare(host);p,q=predictions(method,models,b,threshold=threshold.get(method,0.))
                top=candidate_utilities(q,b).topk(3,-1).indices.cpu().tolist()
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                if rep>=5:durations.append(elapsed)
            rows.append(dict(method=method,median_ms=float(np.median(durations)),p10_ms=float(np.quantile(durations,.1)),
                             p95_ms=float(np.quantile(durations,.95)),repeats=50,top3=top[0]))
    result=dict(seed=37,test_seed=62027,snapshot=0,facts=64,payload_bytes=payload_bytes,rows=rows,
        hardware=torch.cuda.get_device_name(),scope=__doc__.strip(),warmups=5,
        prediction_roundtrip='All seven candidate probability paths and four available marginal paths match prepared inputs at atol1e-6',
        checkpoint_hashes=checkpoints,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'scripts/measure_pipeline.py']})
    path=ROOT/'results/v3/pipeline_timing.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(path),methods=len(rows),payload_bytes=payload_bytes)))


if __name__=='__main__':main()
