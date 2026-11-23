"""Fixed synthetic activity protocol shared by training and evaluation.

Only controlled acoustic-like scenes: no claim of validated ocean/vessel physics.
Overlapping received source supports are unioned into one activity label. Host
operations below serialize metadata or calculate reporting metrics only.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import torch
import torch.nn.functional as F
from uac.runtime import PROJECT_ROOT, fingerprint, sha256_file, source_fingerprint
from uac.synthetic import generate_batch
from uac.metrics import binary_metrics, temporal_average_precision, temporal_iou

CLASS = 'received_acoustic_activity'
DEFAULTS = {'schema_version':1,'sample_rate':16000,'seconds':10,'batch_size':16,
 'train_scenes':20000,'development_scenes':2000,'test_scenes':2000,'epochs':3,
 'seeds':[17,29,43],'learning_rate':0.0003,'weight_decay':0.001,'frame_loss_weight':0.5,
 'selection':'minimum development clip BCE plus 0.5 frame BCE; all three epochs',
 'data_namespaces':{'train':11000000,'development':21000000,'test':31000000},
 'frame_grid':'equal-duration bins covering the full scene; occupancy-fraction targets',
 'event_definition':'connected components of the exact union of received source masks',
 'proposal_thresholds':[0.3,0.5,0.7],'proposal_min_seconds':0.128,'proposal_nms_iou':0.5,
 'event_f1_score_threshold':0.5,'temporal_iou_thresholds':[0.3,0.5,0.7],
 'test_families':['train','heldout'],'test_snr_db':[-15,-10,-5,0,5,10,20],
 'snr_definition':'per-source received RMS SNR over its own support, not total mixture SNR',
 'origin':'fully_synthetic','source_claim':'harmonic/AM/chirp acoustic-like sources, not real vessel classes',
 'heldout_claim':'joint background/source/propagation shift, not an isolated causal family ablation',
 'checkpoint_steps':100,'checkpoint_seconds':120}

def safe_gpu():
    cfg=json.loads((PROJECT_ROOT/'configs/gpu.json').read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid'] or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Use run_on_ada.py; exactly the pinned Ada CUDA device is required')
    prop=torch.cuda.get_device_properties(0)
    if prop.name!=cfg['gpu_name'] or prop.total_memory<cfg['minimum_memory_mib']*1024**2:
        raise RuntimeError('Wrong CUDA device or insufficient Ada memory')
    return cfg

def code_hash():
    return source_fingerprint(['src/uac/models.py','src/uac/synthetic.py','src/uac/metrics.py',
      'scripts/synthetic_common.py','scripts/train_synthetic_detector.py','scripts/evaluate_synthetic.py'])

def scene_fingerprint(config):
    return fingerprint({k:config[k] for k in ('sample_rate','seconds','batch_size','train_scenes','development_scenes','test_scenes','data_namespaces')})

def make_batch(config, split, batch_index, family='train', snr=None):
    size=config['batch_size']; start=batch_index*size
    count=min(size,config[split+'_scenes']-start)
    if count<=0: raise ValueError('Batch starts beyond frozen scene count')
    seed=config['data_namespaces'][split]+batch_index
    # A model seed never changes the scenes. Conditions reuse the same generated
    # scene inside each family. Heldout random draw structure differs, so the
    # train-family versus heldout-family comparison is not exactly paired.
    batch=generate_batch(count,num_samples=config['sample_rate']*config['seconds'],
      sample_rate=config['sample_rate'],seed=seed,family=family)
    if snr is not None:
        gain=torch.pow(10.,(float(snr)-batch['snr_db'])/20.)*batch['source_active']
        batch['audio']=batch['background']+(batch['stems']*gain[...,None]).sum(1)
        batch['snr_db']=torch.where(batch['source_active'],torch.full_like(batch['snr_db'],float(snr)),batch['snr_db'])
    batch['ids']=[f'syn-{split}-{family}-{i:06d}' for i in range(start,start+count)]
    return batch

def targets(batch,frames):
    union=batch['event_masks'].any(1)
    return F.adaptive_avg_pool1d(union[:,None].float(),frames.shape[-1]).squeeze(1)

def objective(batch,clip,frames,config):
    clip_loss=F.binary_cross_entropy_with_logits(clip,batch['labels'])
    frame_loss=F.binary_cross_entropy_with_logits(frames,targets(batch,frames))
    return clip_loss+config['frame_loss_weight']*frame_loss

def _runs(values):
    start=None
    for i,value in enumerate(values+[False]):
        if value and start is None:start=i
        elif not value and start is not None:
            yield start,i;start=None

def prediction_rows(batch,clip,frames,config):
    """GPU inference outputs to compact per-scene metadata, never raw CPU DSP."""
    union=batch['event_masks'].any(1)
    padded=F.pad(union.to(torch.int8),(1,1))
    boundaries=(padded[:,1:]-padded[:,:-1]).nonzero().cpu().tolist()
    cuts={i:[] for i in range(len(batch['ids']))}
    for row,pos in boundaries:cuts[row].append(pos)
    probabilities=frames.sigmoid().float().cpu().tolist()
    clips=clip.sigmoid().float().cpu().tolist()
    counts=batch['source_count'].cpu().tolist();labels=batch['labels'].cpu().tolist()
    snrs=batch['snr_db'].cpu().tolist();active=batch['source_active'].cpu().tolist()
    result=[]
    for i,recording_id in enumerate(batch['ids']):
        truth=[{'recording_id':recording_id,'class_id':CLASS,'onset':a/config['sample_rate'],'offset':b/config['sample_rate']}
               for a,b in zip(cuts[i][::2],cuts[i][1::2])]
        p=probabilities[i]; width=config['seconds']/len(p); proposals=[]
        for threshold in config['proposal_thresholds']:
            for a,b in _runs([v>=threshold for v in p]):
                if (b-a)*width+1e-8<config['proposal_min_seconds']:continue
                proposals.append({'recording_id':recording_id,'class_id':CLASS,'onset':a*width,'offset':b*width,'score':sum(p[a:b])/(b-a)})
        selected=[]
        for prop in sorted(proposals,key=lambda q:(-q['score'],q['onset'],q['offset'])):
            if not any(temporal_iou((prop['onset'],prop['offset']),(old['onset'],old['offset']))>=config['proposal_nms_iou'] for old in selected):selected.append(prop)
        result.append({'id':recording_id,'label':int(labels[i]),'probability':clips[i],
          'source_count':counts[i],'per_source_snr_db':[s for s,a in zip(snrs[i],active[i]) if a],
          'truth':truth,'predictions':selected,'frame_probabilities':p})
    return result

def summarize_rows(rows,config,*,seed,split,family,condition,snr=None):
    truth=[q for row in rows for q in row['truth']];pred=[q for row in rows for q in row['predictions']]
    presence=binary_metrics([r['label'] for r in rows],[r['probability'] for r in rows])
    records=[]
    for threshold in config['temporal_iou_thresholds']:
        metrics=temporal_average_precision(pred,truth,iou_threshold=threshold)
        op=temporal_average_precision([p for p in pred if p['score']>=config['event_f1_score_threshold']],truth,iou_threshold=threshold)
        c=op['classes'].get(CLASS,{})
        tp=c.get('true_positive_events',0);fp=c.get('false_positive_events',0);fn=c.get('missed_events',len(truth))
        records.append({'method':'synthetic_supervised_temporal_cnn','seed':seed,'origin':'fully_synthetic',
          'split':split,'family':family,'condition':condition,'per_source_snr_db':snr,
          'iou_threshold':threshold,'temporal_ap':metrics['map'],'event_f1':2*tp/max(2*tp+fp+fn,1),
          'n_events':len(truth),'n_scenes':len(rows),'n_predictions':len(pred),'true_positive_events':tp,
          'false_positive_events':fp,'missed_events':fn,'presence_ap':presence['ap'],'presence_auroc':presence['auroc']})
    return records
