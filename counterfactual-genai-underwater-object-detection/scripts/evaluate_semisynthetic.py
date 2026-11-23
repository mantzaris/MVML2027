#!/usr/bin/env python3
"""Locked CUDA evaluation of known real-excerpt insertion activity.

This is a semi-synthetic construction, never annotation of natural real events.
All source excerpts retain their recording noise. Component SNR describes the
whole inserted tapered excerpt relative to the constructed background, not an
isolated vessel. The synthetic-trained temporal CNN is applied without fitting.

--write-config creates only the prospective protocol (no test data or GPU use).
Evaluation requires the parent freeze to pin semisynthetic_config_sha256,
semisynthetic_code_fingerprint, semisynthetic_parent_manifest_sha256, and the
existing synthetic_config_sha256/synthetic_checkpoints. Every prediction shard
is atomic and checksummed; completed shards are reused after restart.
"""
from __future__ import annotations
import argparse, collections, json, math, sys, time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import torch
from uac.models import SpectralDetector
from uac.metrics import binary_metrics, temporal_average_precision
from uac.protocol import require_frozen_test
from uac.runtime import (RunLock,StopRequested,EventLedger,atomic_bytes,atomic_json,
  fingerprint,sha256_file,source_fingerprint,project_path,utc_now)
from synthetic_common import safe_gpu,prediction_rows

CLASS='known_excerpt_insertion_activity'
CONFIG={
 'schema_version':1,'origin':'semi_synthetic','split':'test',
 'sample_rate':16000,'seconds':10,'native_excerpt_seconds':3,'batch_size':16,'shard_size':64,
 'seeds':[17,29,43],'selection_seed':20261123,'expected_test_days':5,'base_scenes_per_day':66,
 'source_count_cycle':[0,1,2],'component_snr_db':[-10,0,10],'total_condition_scenes':990,
 'background_crossfade_seconds':0.1,'background_loop_hop_seconds':2.9,
 'background_crossfade':'complementary linear weights; same three-second parent repeated; no normalization',
 'source_taper_seconds':0.05,'source_taper':'sin-squared, half-sample-centered edges; positive over all inserted samples',
 'source_onset_range_seconds':[0.25,6.75],
 'parent_selection':'hash-sorted metadata; unique background and source windows within day/base-scene set; two-source scenes use distinct vessel identities',
 'source_snr_definition':'whole noisy tapered excerpt RMS divided by constructed-background RMS over its insertion support; overlapping components each scaled separately',
 'truth_definition':'union of insertion supports; not natural event truth or isolated vessel audibility',
 'scene_pairing':'same parents, offsets and taper across component-SNR conditions and detector seeds; zero-insertion controls repeat unchanged across SNR',
 'independence':'group all derived scenes by original UTC collection day; repeated SNR/seeds and vessels are dependent; five days form one connected day-vessel component; no independent-component confidence interval',
 'family':'PORTIA_same_day_real_excerpt_insertion',
 'model_training':'fully synthetic temporal CNN; no semi-synthetic fitting, threshold selection or test-driven modification',
 'test_parent_manifest':'data/manifests/real_test.jsonl',
 'real_feature_index':'data/features/real/test/index.json',
 'prediction_decoding':'exactly frozen synthetic_temporal.json proposal thresholds, NMS and approximate 128ms grid',
}
CODE_PATHS=['scripts/evaluate_semisynthetic.py','scripts/synthetic_common.py',
 'src/uac/models.py','src/uac/metrics.py','src/uac/protocol.py','scripts/extract_real_features.py']
LIMITATIONS=[
 'Insertion labels are known by construction; no natural real event timing or clean-source truth exists.',
 'AIS-positive source excerpts contain ambient noise and may not contain an audible target; component SNR is not clean vessel SNR.',
 'AIS-negative backgrounds may contain untracked or distant sources; zero inserted excerpts does not imply acoustic silence.',
 'Looped three-second backgrounds are artificial and contain deterministic crossfade structure.',
 'All conditions reuse 330 base scenes and five collection days; 990 condition-scenes are not independent encounters.',
 'Shared vessel identities connect the five test days; no independent-component confidence interval is estimable.',
 'Two-source insertion supports are unioned; overlapping physical sources are not separated.',
 'Transferred synthetic detector is evaluated without adaptation; this does not establish improvement on natural recordings.',
]

def code_fingerprint():return source_fingerprint(CODE_PATHS)

def check_gate(config_path,checkpoint_paths):
    # Verify the global research gate before parsing any held-out parent/cache rows.
    freeze=require_frozen_test()
    required={'semisynthetic_config_sha256':sha256_file(config_path),
      'semisynthetic_code_fingerprint':code_fingerprint(),
      'semisynthetic_parent_manifest_sha256':sha256_file(CONFIG['test_parent_manifest']),
      'synthetic_config_sha256':sha256_file('configs/synthetic_temporal.json')}
    for field,value in required.items():
        if freeze.get(field)!=value:raise RuntimeError(f'Semi-synthetic test locked: mismatched/missing {field}')
    for path in checkpoint_paths:
        name=str(path.relative_to(PROJECT))
        if freeze.get('synthetic_checkpoints',{}).get(name)!=sha256_file(path):
            raise RuntimeError(f'Semi-synthetic test locked: unpinned checkpoint {name}')
    return freeze,sha256_file('runs/test_freeze.json')

def _rank(kind,day,identity):
    return fingerprint({'selection_seed':CONFIG['selection_seed'],'kind':kind,'day':day,'identity':identity})

def build_base_plan(rows):
    """Metadata-only deterministic construction; no waveform inspection/selection."""
    if any(r['split']!='test' or r['origin']!='real' for r in rows):raise ValueError('Only held-out real parents are allowed')
    days=sorted({r['day'] for r in rows})
    if len(days)!=CONFIG['expected_test_days']:raise ValueError('Frozen protocol requires exactly five test days')
    plan=[];rate=CONFIG['sample_rate']
    for day in days:
        bg=sorted([r for r in rows if r['day']==day and r['label']==0],key=lambda r:(_rank('background',day,r['id']),r['id']))
        positive=sorted([r for r in rows if r['day']==day and r['label']==1],key=lambda r:(_rank('source',day,r['id']),r['id']))
        if len(bg)<66 or len(positive)<66:raise ValueError(f'Insufficient fixed parent capacity on {day}; no adaptive replacement')
        available=list(positive)
        for j in range(CONFIG['base_scenes_per_day']):
            count=CONFIG['source_count_cycle'][j%3];sources=[];onsets=[]
            for slot in range(count):
                options=[i for i,r in enumerate(available) if all(r['source_id']!=s['source_id'] for s in sources)]
                if not options:raise ValueError('Cannot obtain distinct source identities; do not alter frozen selection')
                source=available.pop(options[0]);sources.append(source)
                lo,hi=[int(t*rate) for t in CONFIG['source_onset_range_seconds']]
                value=int(_rank('onset',day,f'{j}:{slot}')[:16],16)
                onsets.append(lo+value%(hi-lo+1))
            record={'base_id':f'semireal-{day}-{j:03d}','group':f'day:{day}','day':day,
              'source_count':count,'background_parent_id':bg[j]['id'],
              'source_parent_ids':[r['id'] for r in sources],
              'source_ids':[r['source_id'] for r in sources],'onset_samples':onsets,
              'insertion_samples':[[onset,onset+3*rate] for onset in onsets],
              'overlap_samples':max(0,min(onsets)+3*rate-max(onsets)) if count==2 else 0}
            plan.append(record)
    if len(plan)!=330 or len({r['background_parent_id'] for r in plan})!=330:raise ValueError('Background reuse in fixed base-scene plan')
    source_ids=[p for r in plan for p in r['source_parent_ids']]
    if len(source_ids)!=330 or len(set(source_ids))!=330:raise ValueError('Unexpected source-parent reuse')
    return plan

def load_parents(freeze):
    """Read only verified derivatives after check_gate; original hashes retained."""
    manifest_path=project_path(CONFIG['test_parent_manifest'])
    rows=[json.loads(line) for line in manifest_path.read_text().splitlines()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate native test parent identity')
    native={r['id']:r for r in rows}
    index_path=project_path(CONFIG['real_feature_index']);index=json.loads(index_path.read_text())
    if (index.get('status')!='complete' or index.get('completed_rows')!=len(rows)
        or index.get('expected_rows')!=len(rows) or index.get('data_fingerprint')!=freeze['data_fingerprint']
        or index.get('split_manifest_sha256')!=sha256_file(manifest_path)
        or index.get('rows_fingerprint')!=fingerprint(rows)):
        raise RuntimeError('Real test derivative index does not match the frozen native parents')
    audio_parts=[];ordered=[];seen=set()
    for receipt in index['shards']:
        path=project_path(receipt['payload'])
        if path.stat().st_size!=receipt['payload_bytes'] or sha256_file(path)!=receipt['payload_sha256']:
            raise RuntimeError('Real test derivative checksum failed')
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['ids']!=receipt['ids'] or data['sample_rate']!=16000 or data['audio'].shape!=(receipt['count'],48000):
            raise RuntimeError('Unexpected derivative identities/rate/duration')
        for row in data['rows']:
            if row['id'] in seen or row!=native.get(row['id']):raise RuntimeError('Derivative parent provenance mismatch')
            if row['split']!='test' or not row.get('header_verified') or not row.get('sha256'):raise RuntimeError('Unverified parent')
            seen.add(row['id']);ordered.append(row)
        audio_parts.append(data['audio'].to(device='cuda',dtype=torch.float32))
    if seen!=set(native):raise RuntimeError('Missing or extra parent derivatives')
    audio=torch.cat(audio_parts);del audio_parts
    if not bool(torch.isfinite(audio).all()):raise RuntimeError('Nonfinite real derivative')
    positions={r['id']:i for i,r in enumerate(ordered)}
    return audio,ordered,positions,index

def loop_background(parent,seconds=10,crossfade_seconds=.1):
    """Complementary linear overlap-add of a native three-second excerpt."""
    if parent.device.type!='cuda':raise RuntimeError('Background construction requires Ada CUDA')
    rate=CONFIG['sample_rate'];length=parent.shape[-1];n=int(seconds*rate);fade=int(crossfade_seconds*rate);hop=length-fade
    output=parent.new_zeros((len(parent),n));denominator=parent.new_zeros(n)
    ramp=torch.arange(fade,device='cuda',dtype=parent.dtype)/fade
    for start in range(0,n,hop):
        count=min(length,n-start);weight=parent.new_ones(length)
        if start:weight[:fade]=ramp
        if start+length<n:weight[-fade:]=1-ramp
        output[:,start:start+count]+=parent[:,:count]*weight[:count]
        denominator[start:start+count]+=weight[:count]
    if not torch.allclose(denominator,torch.ones_like(denominator),atol=1e-6,rtol=0):raise RuntimeError('Invalid background crossfade weight sum')
    return output

def construct_batch(base_rows,parent_audio,positions,snr):
    """CUDA-only construction with exact inserted-component supports and gains."""
    rate=CONFIG['sample_rate'];n=rate*CONFIG['seconds'];length=rate*CONFIG['native_excerpt_seconds'];b=len(base_rows)
    bg_index=torch.tensor([positions[r['background_parent_id']] for r in base_rows],device='cuda')
    background=loop_background(parent_audio[bg_index],CONFIG['seconds'],CONFIG['background_crossfade_seconds'])
    counts=torch.tensor([r['source_count'] for r in base_rows],device='cuda')
    active=torch.arange(2,device='cuda')[None,:]<counts[:,None]
    source_index=torch.tensor([[positions[r['source_parent_ids'][j]] if j<r['source_count'] else 0 for j in range(2)] for r in base_rows],device='cuda')
    onsets=torch.tensor([[r['onset_samples'][j] if j<r['source_count'] else 0 for j in range(2)] for r in base_rows],device='cuda')
    t=torch.arange(n,device='cuda')[None,None,:];local=t-onsets[:,:,None]
    masks=(local>=0)&(local<length)&active[:,:,None]
    pos=torch.arange(length,device='cuda',dtype=torch.float32)
    fade=CONFIG['source_taper_seconds']*rate
    edge=torch.minimum((pos+.5)/fade,(length-.5-pos)/fade).clamp(0,1)
    taper=torch.sin(.5*torch.pi*edge).square()
    excerpts=parent_audio[source_index]*taper
    stems=excerpts.gather(-1,local.clamp(0,length-1))*masks
    denominator=masks.sum(-1).clamp_min(1)
    source_rms=((stems.square()*masks).sum(-1)/denominator).sqrt()
    background_rms=((background[:,None].square()*masks).sum(-1)/denominator).sqrt()
    if bool(((source_rms<1e-10)|(background_rms<1e-10))[active].any()):
        raise RuntimeError('Frozen selected parent has negligible RMS; do not select replacements based on outcomes')
    gains=torch.where(active,background_rms/source_rms.clamp_min(1e-10)*(10.**(float(snr)/20.)),torch.zeros_like(source_rms))
    stems=stems*gains[:,:,None];audio=background+stems.sum(1)
    measured=20*torch.log10((((stems.square()*masks).sum(-1)/denominator).sqrt()/background_rms.clamp_min(1e-10)).clamp_min(1e-20))
    max_snr_error=float((measured[active]-float(snr)).abs().max()) if bool(active.any()) else 0.
    outside_error=float(((audio-background)*(~masks.any(1))).abs().max())
    if max_snr_error>1e-3 or outside_error>1e-7 or not bool(torch.isfinite(audio).all()):raise RuntimeError('Constructed scene failed SNR/support integrity')
    return {'audio':audio,'background':background,'stems':stems,'event_masks':masks,
      'source_count':counts,'labels':(counts>0).float(),'source_active':active,
      'snr_db':torch.full((b,2),float(snr),device='cuda'),'source_gains':gains,
      'source_rms_before_gain':source_rms,'background_rms_on_source_support':background_rms,
      'ids':[f"{r['base_id']}-snr{snr:+03d}" for r in base_rows],
      'max_component_snr_error_db':max_snr_error,'outside_insertion_max_error':outside_error}

def provenance(row,native):
    parents=[row['background_parent_id']]+row['source_parent_ids']
    return {key:{field:native[key].get(field) for field in ('sha256','decoded_channel_sha256','source_id','label','day','start_utc','start_seconds','path','sample_rate','num_samples','analysis_channel')} for key in parents}

def summarize(rows,temporal,*,seed,snr,day=None):
    truth=[e for r in rows for e in r['truth']];pred=[e for r in rows for e in r['predictions']]
    presence=binary_metrics([r['label'] for r in rows],[r['probability'] for r in rows]);records=[]
    for iou in temporal['temporal_iou_thresholds']:
        metric=temporal_average_precision(pred,truth,iou_threshold=iou)
        operated=temporal_average_precision([p for p in pred if p['score']>=temporal['event_f1_score_threshold']],truth,iou_threshold=iou)['classes'].get(CLASS,{})
        tp=operated.get('true_positive_events',0);fp=operated.get('false_positive_events',0);fn=operated.get('missed_events',len(truth))
        records.append({'method':'synthetic_supervised_temporal_cnn','seed':seed,'origin':'semi_synthetic','split':'test',
          'family':CONFIG['family'],'condition':f'component_snr_{snr:+03d}','per_source_snr_db':snr,'component_snr_db':snr,
          'iou_threshold':iou,'temporal_ap':metric['map'],'event_f1':2*tp/max(2*tp+fp+fn,1),
          'presence_ap':presence['ap'],'presence_auroc':presence['auroc'],'n_scenes':len(rows),'n_events':len(truth),
          'n_predictions':len(pred),'true_positive_events':tp,'false_positive_events':fp,'missed_events':fn,
          'day_cluster_count':len({r['day'] for r in rows}),'day':day,
          'truth_scope':'known real-excerpt insertion support, not natural real event truth'})
    return records

def valid_receipt(path,sidecar,expected):
    if not sidecar.exists():return None
    receipt=json.loads(sidecar.read_text())
    if any(receipt.get(k)!=v for k,v in expected.items()):raise RuntimeError('Incompatible semi-synthetic shard; do not overwrite')
    if path.is_file() and path.stat().st_size==receipt.get('payload_bytes') and sha256_file(path)==receipt.get('payload_sha256'):return receipt
    stamp=f'.corrupt-{time.time_ns()}'
    for item in (path,sidecar):
        if item.exists():item.rename(item.with_name(item.name+stamp))
    return None

def evaluate_condition(model,seed,snr,base_plan,parent_audio,positions,native,temporal,common,stop,ledger,budget):
    directory=PROJECT/'predictions/semisynthetic'/f'seed_{seed}'/f'snr_{snr:+03d}';directory.mkdir(parents=True,exist_ok=True)
    expected={**common,'seed':seed,'component_snr_db':snr};receipts=[];rows=[]
    index_path=directory/'index.json'
    if index_path.is_file():
        old=json.loads(index_path.read_text())
        if any(old.get(k)!=v for k,v in expected.items()):raise RuntimeError('Incompatible semi-synthetic condition index')
        if old.get('status')=='complete' and old.get('completed_rows')==len(base_plan):
            if all(project_path(r['payload']).is_file() and sha256_file(r['payload'])==r['payload_sha256'] for r in old['shards']):return old
    for begin in range(0,len(base_plan),CONFIG['shard_size']):
        end=min(begin+CONFIG['shard_size'],len(base_plan));subset=base_plan[begin:end]
        path=directory/f'shard-{begin:05d}-{end:05d}.jsonl';sidecar=path.with_suffix('.json')
        ids=[f"{r['base_id']}-snr{snr:+03d}" for r in subset]
        wanted={**expected,'begin':begin,'count':len(subset),'ids':ids}
        receipt=valid_receipt(path,sidecar,wanted)
        if receipt is None:
            if stop.requested or budget[0]==0:return None
            shard_rows=[];started=time.monotonic();snr_errors=[];outside_errors=[]
            with torch.inference_mode():
                for b in range(0,len(subset),CONFIG['batch_size']):
                    selected=subset[b:b+CONFIG['batch_size']];batch=construct_batch(selected,parent_audio,positions,snr)
                    clip,frames=model(batch['audio'],return_frames=True)
                    predicted=prediction_rows(batch,clip,frames,temporal)
                    gains=batch['source_gains'].cpu().tolist();rms=batch['source_rms_before_gain'].cpu().tolist();bg_rms=batch['background_rms_on_source_support'].cpu().tolist()
                    for i,(record,base) in enumerate(zip(predicted,selected)):
                        for event in record['truth']+record['predictions']:event['class_id']=CLASS
                        record.update(base_scene=base,base_id=base['base_id'],group=base['group'],day=base['day'],origin='semi_synthetic',
                          parent_provenance=provenance(base,native),source_gains=gains[i][:base['source_count']],
                          source_rms_before_gain=rms[i][:base['source_count']],background_rms_on_source_support=bg_rms[i][:base['source_count']],
                          component_snr_db=snr,truth_scope=CONFIG['truth_definition'])
                    shard_rows.extend(predicted);snr_errors.append(batch['max_component_snr_error_db']);outside_errors.append(batch['outside_insertion_max_error'])
                    if stop.requested:return None
            atomic_bytes(path,b''.join((json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode() for r in shard_rows))
            receipt={**wanted,'payload':str(path.relative_to(PROJECT)),'payload_bytes':path.stat().st_size,'payload_sha256':sha256_file(path),
              'seconds':time.monotonic()-started,'max_component_snr_error_db':max(snr_errors),'outside_insertion_max_error':max(outside_errors),'completed_utc':utc_now()}
            atomic_json(sidecar,receipt);budget[0]-=1
            ledger.append('semisynthetic_prediction_shard',seed=seed,snr=snr,begin=begin,count=len(subset),payload_sha256=receipt['payload_sha256'])
            print(json.dumps({'stage':'semisynthetic_prediction','seed':seed,'snr':snr,'through':end,'total':len(base_plan)}),flush=True)
        else:shard_rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows.extend(shard_rows);receipts.append(receipt)
        atomic_json(index_path,{**expected,'status':'running','completed_rows':len(rows),'expected_rows':len(base_plan),'shards':receipts})
    records=summarize(rows,temporal,seed=seed,snr=snr)
    day_records=[record for day in sorted({r['day'] for r in rows}) for record in summarize([r for r in rows if r['day']==day],temporal,seed=seed,snr=snr,day=day)]
    index={**expected,'status':'complete','completed_rows':len(rows),'expected_rows':len(base_plan),'shards':receipts,
      'records':records,'day_records':day_records,'completed_utc':utc_now()}
    atomic_json(index_path,index);return index

def day_vessel_components(base_plan):
    parent={}
    def find(value):
        parent.setdefault(value,value)
        if parent[value]!=value:parent[value]=find(parent[value])
        return parent[value]
    for row in base_plan:
        day='day:'+row['day'];find(day)
        for source in row['source_ids']:
            a,b=find(day),find(source);parent[b]=a
    return len({find('day:'+row['day']) for row in base_plan})

def write_report(common,base_plan):
    records=[];day_records=[];indices=[]
    for path in sorted((PROJECT/'predictions/semisynthetic').glob('seed_*/snr_*/index.json')):
        index=json.loads(path.read_text())
        if index.get('status')!='complete':continue
        # Model hashes differ across seeds, other provenance must agree.
        for key,value in common.items():
            if key!='checkpoint_sha256' and index.get(key)!=value:raise RuntimeError('Cannot aggregate incompatible semi-synthetic runs')
        if index['seed'] not in CONFIG['seeds'] or index['component_snr_db'] not in CONFIG['component_snr_db']:raise RuntimeError('Unplanned condition')
        for receipt in index['shards']:
            if sha256_file(receipt['payload'])!=receipt['payload_sha256']:raise RuntimeError('Semi-synthetic prediction checksum mismatch')
        records.extend(index['records']);day_records.extend(index['day_records']);indices.append({'path':str(path.relative_to(PROJECT)),'sha256':sha256_file(path)})
    expected=len(CONFIG['seeds'])*len(CONFIG['component_snr_db'])
    selected=set(r['background_parent_id'] for r in base_plan)|{p for r in base_plan for p in r['source_parent_ids']}
    report={**{k:v for k,v in common.items() if k!='checkpoint_sha256'},'status':'complete' if len(indices)==expected else 'partial',
      'origin':'semi_synthetic','records':records,'day_records':day_records,'indices':indices,'config':CONFIG,
      'completed_conditions':len(indices),'expected_conditions':expected,'base_scenes':len(base_plan),
      'condition_scenes':len(base_plan)*len(CONFIG['component_snr_db']),'unique_selected_native_parent_windows':len(selected),
      'day_groups':sorted({r['group'] for r in base_plan}),'selected_day_vessel_components':day_vessel_components(base_plan),'source_count_counts_per_snr':dict(collections.Counter(r['source_count'] for r in base_plan)),
      'two_source_overlapping_base_scenes':sum(r['source_count']==2 and r['overlap_samples']>0 for r in base_plan),
      'zero_insertion_controls_repeated_across_snr':sum(r['source_count']==0 for r in base_plan),
      'limitations':LIMITATIONS,'created_utc':utc_now()}
    atomic_json('results/tables/semisynthetic_results.json',report)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-config',action='store_true')
    parser.add_argument('--seeds',nargs='+',type=int,default=CONFIG['seeds']);parser.add_argument('--max-shards',type=int)
    args=parser.parse_args();config_path=PROJECT/'configs/semisynthetic.json'
    if args.max_shards is not None and args.max_shards<1:parser.error('Positive shard limit required')
    if len(set(args.seeds))!=len(args.seeds) or any(s not in CONFIG['seeds'] for s in args.seeds):parser.error('Unknown or duplicate model seed')
    if args.write_config:
        if config_path.exists() and json.loads(config_path.read_text())!=CONFIG:raise RuntimeError('Do not overwrite a different prospective protocol')
        atomic_json(config_path,CONFIG)
        info={'semisynthetic_config_sha256':sha256_file(config_path),'semisynthetic_code_fingerprint':code_fingerprint(),
          'parent_manifest_to_pin':CONFIG['test_parent_manifest'],'required_parent_hash_field':'semisynthetic_parent_manifest_sha256',
          'test_accessed':False,'gpu_used':False}
        atomic_json('manifests/semisynthetic_protocol.json',info);print(json.dumps(info,indent=2));return 0
    if not config_path.is_file() or json.loads(config_path.read_text())!=CONFIG:raise RuntimeError('Write and freeze the prospective semi-synthetic config first')
    paths=[PROJECT/'checkpoints/synthetic_temporal'/f'seed_{s}'/'best.pt' for s in args.seeds]
    freeze,freeze_hash=check_gate(config_path,paths)
    safe_gpu();temporal=json.loads((PROJECT/'configs/synthetic_temporal.json').read_text())
    with RunLock('runs/training.lock'),StopRequested() as stop:
        ledger=EventLedger();audio,parents,positions,real_index=load_parents(freeze)
        base_plan=build_base_plan(parents);native={r['id']:r for r in parents}
        plan_records=[{**r,'parent_provenance':provenance(r,native)} for r in base_plan]
        plan_path=PROJECT/'data/manifests/semisynthetic_test_scenes.jsonl'
        plan_bytes=b''.join((json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode() for r in plan_records)
        if plan_path.exists() and plan_path.read_bytes()!=plan_bytes:raise RuntimeError('Immutable semi-synthetic scene plan changed')
        if not plan_path.exists():atomic_bytes(plan_path,plan_bytes)
        common={'schema_version':1,'config_hash':fingerprint(CONFIG),'config_sha256':sha256_file(config_path),
          'code_fingerprint':code_fingerprint(),'freeze_sha256':freeze_hash,'data_fingerprint':freeze['data_fingerprint'],
          'parent_manifest_sha256':sha256_file(CONFIG['test_parent_manifest']),'real_feature_index_sha256':sha256_file(CONFIG['real_feature_index']),
          'scene_manifest_sha256':sha256_file(plan_path),'synthetic_temporal_config_sha256':sha256_file('configs/synthetic_temporal.json')}
        budget=[args.max_shards if args.max_shards is not None else math.inf]
        model=SpectralDetector().cuda().eval()
        for seed,path in zip(args.seeds,paths):
            checkpoint=torch.load(path,map_location='cuda',weights_only=False)
            if checkpoint['config']!=temporal:raise RuntimeError('Temporal checkpoint configuration changed')
            model.load_state_dict(checkpoint['model']);common['checkpoint_sha256']=sha256_file(path)
            for snr in CONFIG['component_snr_db']:
                index=evaluate_condition(model,seed,snr,base_plan,audio,positions,native,temporal,common,stop,ledger,budget)
                if index is None:return 75 if stop.requested else 0
                report=write_report(common,base_plan)
                ledger.append('semisynthetic_condition_complete',seed=seed,snr=snr,records=index['records'])
        print(json.dumps({'status':report['status'],'report':'results/tables/semisynthetic_results.json','origin':'semi_synthetic'}),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
