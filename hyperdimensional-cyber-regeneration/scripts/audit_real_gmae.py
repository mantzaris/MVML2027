#!/usr/bin/env python3
"""Same-session Ada costs and checkpoint replay for the GraphMAE follow-up.

Wall timing includes decoded CPU tensors, CUDA transfers/normalization/endpoint
adjacency/inference and CPU probabilities. It excludes CSV/flow extraction,
waiting for completed windows, model loading and network transmission. GPU event
timing separately covers warmed model inference. Neither is operational delay.
"""
import csv
import json
import time
import numpy as np
import torch
import measure_real_pipeline as preparation
from run_real_gmae import base_evidence
from real_gmae_variants import KEYS,make_model as make_new
from real_models import make_model as make_old,infer
from real_metrics import GaussianNB,metrics
from real_data import ROOT,CONDITIONS,TEST,load_capture,windows,condition_batch,input_hash,subset,sha
from run_real import collect

SEED=37
REPLAY=list(KEYS)+['GMAE']


def load_model(method,manifest):
    record=next(r for r in manifest['checkpoints'] if r['method']==method and r['seed']==SEED)
    path=ROOT/record['checkpoint_path'];assert sha(path)==record['checkpoint_sha256']
    model=(make_new(method) if method in KEYS else GaussianNB() if method=='NB' else make_old(method)).cuda()
    model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));return model.eval(),record


def probability(model,b):return model.probabilities(b) if model.key=='NB' else infer(model,b)


@torch.inference_mode()
def main():
    path=ROOT/'results/real_gmae/manifest.json';manifest=json.loads(path.read_text())
    assert not manifest['settings']['pilot'] and len(manifest['checkpoints'])==65 and len(manifest['raw_hashes'])==975
    assert len(manifest['thresholds'])==65 and len(manifest['case_hashes'])==15
    for relative,digest in manifest['source_hashes'].items():assert sha(ROOT/relative)==digest,relative
    base_path,old=base_evidence();assert sha(base_path)==manifest['base_manifest_sha256']
    preparation.configure_cuda();prep=preparation.frozen_preprocessing(manifest)
    reference=windows(load_capture(10),10,prep);assert input_hash(reference)==manifest['case_hashes']['clean_10']
    host=preparation.cpu_first_window(10);restored=preparation.prepare(host,prep);direct=subset(reference,slice(0,1))
    for key in ['x','adj','valid']:torch.testing.assert_close(restored[key],direct[key],atol=0,rtol=0)
    timing=[];records={}
    for method in manifest['methods']:
        model,record=load_model(method,manifest);records[method]=record
        torch.testing.assert_close(probability(model,restored),probability(model,direct),atol=1e-6,rtol=0)
        origin=manifest['method_origins'][method]
        raw_path=ROOT/'results/runs'/origin/str(SEED)/f'{method}_clean_10.npz'
        assert sha(raw_path)==manifest['raw_hashes'][str(raw_path.relative_to(ROOT))]
        original=probability(model,subset(reference,slice(0,manifest['settings']['batch'])))[0].cpu().numpy()
        with np.load(raw_path,allow_pickle=False) as saved:
            assert np.max(np.abs(original-saved['prob'][:32]))<=1e-5
        wall=[]
        for repetition in range(55):
            torch.cuda.synchronize();start=time.perf_counter()
            b=preparation.prepare(host,prep);returned=probability(model,b).cpu().numpy();torch.cuda.synchronize()
            if repetition>=5:wall.append((time.perf_counter()-start)*1000)
        for _ in range(5):probability(model,direct)
        torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();cuda=[]
        for _ in range(30):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();probability(model,direct);end.record();end.synchronize();cuda.append(start.elapsed_time(end))
        timing.append(dict(method=method,wall_median_ms=float(np.median(wall)),wall_p10_ms=float(np.quantile(wall,.1)),
            wall_p95_ms=float(np.quantile(wall,.95)),wall_repeats=50,cuda_median_ms=float(np.median(cuda)),cuda_p95_ms=float(np.quantile(cuda,.95)),
            cuda_repeats=30,peak_vram_mb=(torch.cuda.max_memory_allocated()-resident)/2**20,parameters=record['parameters'],
            checkpoint_path=record['checkpoint_path'],checkpoint_sha256=record['checkpoint_sha256']))
        del model
    summary_path=ROOT/'results/real_gmae/summary.csv'
    with summary_path.open() as f:summary={(int(r['seed']),r['method'],r['condition'],int(r['scenario'])):r for r in csv.DictReader(f)}
    thresholds={r['method']:r['threshold'] for r in manifest['thresholds'] if r['seed']==SEED}
    models={m:load_model(m,manifest)[0] for m in REPLAY};replay=[]
    for scenario in TEST:
        clean=windows(load_capture(scenario),scenario,prep)
        for condition in CONDITIONS:
            b=condition_batch(clean,condition,prep);digest=input_hash(b);assert digest==manifest['case_hashes'][f'{condition}_{scenario}']
            valid=b['valid'];labels=b['y'][valid]
            expected=dict(labels=labels.cpu().numpy(),row_id=b['row_id'][valid].cpu().numpy(),
                window=b['window'][:,None].expand_as(valid)[valid].cpu().numpy(),block=b['block'][:,None].expand_as(valid)[valid].cpu().numpy())
            for method,model in models.items():
                q=collect(model,b,manifest['settings']['batch'])[valid];q_cpu=q.cpu().numpy()
                origin=manifest['method_origins'][method];raw_path=ROOT/'results/runs'/origin/str(SEED)/f'{method}_{condition}_{scenario}.npz'
                raw_digest=sha(raw_path);assert raw_digest==manifest['raw_hashes'][str(raw_path.relative_to(ROOT))]
                with np.load(raw_path,allow_pickle=False) as saved:
                    for key,value in expected.items():assert np.array_equal(value,saved[key]),(method,key)
                    error=float(np.max(np.abs(q_cpu-saved['prob'])))
                assert np.isfinite(error) and error<=1e-5
                current=metrics(labels,q,thresholds[method]);saved_summary=summary[(SEED,method,condition,scenario)]
                errors={k:abs(current[k]-float(saved_summary[k])) for k in ['ap','brier']}
                assert all(np.isfinite(v) and v<=1e-8 for v in errors.values()),errors
                record=records[method]
                replay.append(dict(method=method,condition=condition,scenario=scenario,flows=len(q_cpu),probability_error=error,
                    ap_error=errors['ap'],brier_error=errors['brier'],raw_path=str(raw_path.relative_to(ROOT)),raw_sha256=raw_digest,
                    checkpoint_path=record['checkpoint_path'],checkpoint_sha256=record['checkpoint_sha256'],input_sha256=digest))
    assert len(replay)==75 and sum(r['flows'] for r in replay)==3570325
    source_paths=['scripts/audit_real_gmae.py','scripts/measure_real_pipeline.py','scripts/run_real_gmae.py','scripts/run_real.py','src/real_gmae_variants.py','src/real_models.py','src/real_data.py','src/real_metrics.py']
    output=dict(passed=True,manifest_sha256=sha(path),summary_sha256=sha(summary_path),base_manifest_sha256=sha(base_path),
        source_hashes={p:sha(ROOT/p) for p in source_paths},hardware=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        seed=SEED,timing_rows=timing,input_bytes=sum(v.numel()*v.element_size() for v in host.values()),output_bytes=int(returned.nbytes),
        warmups=5,timing_scope=__doc__.strip(),labels_in_timing_inputs=False,model_and_statistics_resident=True,
        replay_rows=replay,model_flow_evaluations=sum(r['flows'] for r in replay),
        maximum_errors={'prob':max(r['probability_error'] for r in replay),'ap':max(r['ap_error'] for r in replay),'brier':max(r['brier_error'] for r in replay)})
    (ROOT/'results/real_gmae/gpu_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['passed','hardware','input_bytes','output_bytes','model_flow_evaluations','maximum_errors']}))


if __name__=='__main__':main()
