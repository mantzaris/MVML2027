#!/usr/bin/env python3
"""CUDA feature caches for synthetic augmentation and separate view supplements.

3 s scenes have their own seed namespace, distinct from the 10 s activity study.
The cache is compatible with extract_foundation.load_index/load_shard. Every
feature operation runs on Ada; CPU storage/metadata/reporting only. Test and
heldout_test are inaccessible until the parent protocol freeze pins this file's
config in synthetic_features_config_sha256. No real-excerpt mixtures here.
"""
from __future__ import annotations
import argparse,io,json,math,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import torch
from uac.synthetic import generate_batch
from uac.dsp import extract_features,ALL_GROUPS,feature_names,GROUP_DIMS
from uac.runtime import (atomic_bytes,atomic_json,RunLock,StopRequested,EventLedger,
  fingerprint,sha256_file,source_fingerprint,utc_now)
from synthetic_common import safe_gpu
from extract_real_features import _receipt_if_valid

CONFIG={'schema_version':1,'sample_rate':16000,'seconds':3,'batch_size':16,'shard_size':128,
 'counts':{'train':20000,'development':2000,'test':2000,'heldout_test':2000},
 'namespaces':{'train':41000000,'development':51000000,'test':61000000,'heldout_test':61000000},
 'families':{'train':'train','development':'train','test':'train','heldout_test':'heldout'},
 'groups':list(ALL_GROUPS),'primary_indices':list(range(152)),'group_dimensions':GROUP_DIMS,
 'source_types':{'0':'harmonic_AM','1':'chirp_dominant','2':'decaying_AM'},
 'label_definition':'zero versus one/two generated acoustic-like sources; no real vessel category',
 'origin':'fully_synthetic','hht_status':'bounded piecewise-linear approximation; exploratory',
 'heldout_claim':'joint source/background/propagation shift, not isolated causal ablation'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',nargs='+',choices=list(CONFIG['counts']),default=['train','development'])
    parser.add_argument('--max-shards',type=int);args=parser.parse_args()
    if len(set(args.split))!=len(args.split) or (args.max_shards is not None and args.max_shards<1):parser.error('Unique splits and positive limit required')
    safe_gpu();config_path=PROJECT/'configs/synthetic_features.json'
    if config_path.is_file() and json.loads(config_path.read_text())!=CONFIG:raise RuntimeError('Synthetic feature settings already frozen differently')
    atomic_json(config_path,CONFIG);data_hash=sha256_file(config_path)
    if any('test' in split for split in args.split):
        freeze=PROJECT/'runs/test_freeze.json'
        if not freeze.is_file() or json.loads(freeze.read_text()).get('synthetic_features_config_sha256')!=data_hash:
            raise RuntimeError('Synthetic test features locked until matching parent protocol freeze')
    code_hash=source_fingerprint(['scripts/generate_synthetic_features.py','src/uac/synthetic.py','src/uac/dsp.py'])
    config_hash=fingerprint(CONFIG);new=0
    with RunLock('runs/training.lock'),StopRequested() as stop:
        ledger=EventLedger()
        for split in args.split:
            total=CONFIG['counts'][split];family=CONFIG['families'][split]
            directory=PROJECT/'data/features/synthetic'/split;directory.mkdir(parents=True,exist_ok=True)
            prior_path=directory/'index.json'
            if prior_path.is_file():
                prior=json.loads(prior_path.read_text())
                if (prior.get('status')=='complete' and prior.get('config_hash')==config_hash
                    and prior.get('code_fingerprint')==code_hash and prior.get('data_fingerprint')==data_hash
                    and prior.get('completed_rows')==total and prior.get('expected_rows')==total):
                    valid=all((PROJECT/r['payload']).is_file()
                        and (PROJECT/r['payload']).stat().st_size==r['payload_bytes']
                        and sha256_file(r['payload'])==r['payload_sha256'] for r in prior['shards'])
                    if valid and sum(r['count'] for r in prior['shards'])==total:
                        print(json.dumps({'status':'already_complete','split':split,'rows':total}),flush=True)
                        continue
            index={'schema_version':1,'status':'running','split':split,'family':family,'config':CONFIG,'config_hash':config_hash,
              'code_fingerprint':code_hash,'data_fingerprint':data_hash,'expected_rows':total,'completed_rows':0,'shards':[],
              'feature_names':feature_names(ALL_GROUPS),'primary_indices':list(range(152)),
              'group_slices':{'fourier':[0,32],'stft':[32,96],'wavelet':[96,152],'hht':[152,184]}}
            for begin in range(0,total,CONFIG['shard_size']):
                end=min(begin+CONFIG['shard_size'],total);ids=[f'synfeat-{split}-{i:06d}' for i in range(begin,end)]
                stem=f'shard-{begin:06d}-{end:06d}';path=directory/(stem+'.pt');sidecar=directory/(stem+'.json')
                expected={'schema_version':1,'split':split,'begin':begin,'count':end-begin,'ids':ids,'config_hash':config_hash,
                   'code_fingerprint':code_hash,'data_fingerprint':data_hash,'payload':str(path.relative_to(PROJECT))}
                receipt=_receipt_if_valid(path,sidecar,expected,ledger)
                if receipt is None:
                    if stop.requested or (args.max_shards is not None and new>=args.max_shards):
                        index['status']='paused' if stop.requested else 'pilot_limit';atomic_json(directory/'index.json',index);return 75 if stop.requested else 0
                    features=[];labels=[];types=[];counts=[];actives=[];snrs=[];diags=[];rows=[]
                    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():
                        for start in range(begin,end,CONFIG['batch_size']):
                            count=min(CONFIG['batch_size'],end-start)
                            batch=generate_batch(count,num_samples=48000,sample_rate=16000,seed=CONFIG['namespaces'][split]+start//CONFIG['batch_size'],family=family)
                            result=extract_features(batch['audio'],16000,groups=ALL_GROUPS)
                            features.append(result['features'].cpu());labels.append(batch['labels'].cpu());types.append(batch['source_types'].cpu())
                            counts.append(batch['source_count'].cpu());actives.append(batch['source_active'].cpu());snrs.append(batch['snr_db'].cpu())
                            diags.append({k:v.cpu() for k,v in result['diagnostics']['hht'].items()})
                            for offset,(label,source_count,source_types) in enumerate(zip(labels[-1].tolist(),counts[-1].tolist(),types[-1].tolist())):
                                scene_id=f'synfeat-{split}-{start+offset:06d}'
                                rows.append({'id':scene_id,'label':int(label),'source_count':source_count,'source_types':source_types,'family':family,'split':split,'origin':'fully_synthetic','group':scene_id})
                            if stop.requested:return 75
                    torch.cuda.synchronize()
                    hht={key:torch.cat([d[key] for d in diags]) for key in diags[0]}
                    payload={'schema_version':1,'features':torch.cat(features),'labels':torch.cat(labels),
                      'source_types':torch.cat(types),'source_count':torch.cat(counts),'source_active':torch.cat(actives),
                      'per_source_snr_db':torch.cat(snrs),'ids':ids,'rows':rows,'family':family,'origin':'fully_synthetic',
                      'sample_rate':16000,'seconds':3,'hht_diagnostics':hht,'feature_names':index['feature_names'],
                      'group_slices':index['group_slices'],'primary_indices':index['primary_indices'],
                      'config_hash':config_hash,'code_fingerprint':code_hash,'data_fingerprint':data_hash}
                    payload['x']=payload['features'];payload['y']=payload['labels']
                    buffer=io.BytesIO();torch.save(payload,buffer);atomic_bytes(path,buffer.getvalue())
                    measured={'seconds_including_transfers':time.perf_counter()-started,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                       'hht_flagged_mode_fraction':float(hht['unconverged_or_extrema_failed'].float().mean())}
                    receipt={**expected,'payload_bytes':path.stat().st_size,'payload_sha256':sha256_file(path),'measurements':measured,'completed_utc':utc_now()}
                    atomic_json(sidecar,receipt);new+=1
                    ledger.append('synthetic_feature_shard_committed',split=split,begin=begin,count=end-begin,payload_sha256=receipt['payload_sha256'],**measured)
                    print(json.dumps({'stage':'synthetic_features','split':split,'through':end,'total':total,**measured}),flush=True)
                index['shards'].append(receipt);index['completed_rows']+=end-begin;atomic_json(directory/'index.json',index)
            index['status']='complete';index['completed_utc']=utc_now();atomic_json(directory/'index.json',index)
            ledger.append('synthetic_feature_split_complete',split=split,count=total,data_fingerprint=data_hash)
    return 0
if __name__=='__main__':raise SystemExit(main())
