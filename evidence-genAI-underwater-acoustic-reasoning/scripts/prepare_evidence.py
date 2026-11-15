"""Build independent proxies and proposed cards in separate, auditable records."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.special import softmax
from underwater_reasoning import dsp
from underwater_reasoning.references import estimate
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
ROOT=Path(__file__).resolve().parents[1]

def compact(card):
    m=card['measurements'];family=card['tool'];base={'id':family+'0','seconds':[0,5],'candidate_only':True}
    if family=='A':
        base.update({'dominant_band_hz':m['dominant_band_hz'],'band_power_fraction':np.round(np.array(m['integrated_band_power'])/max(sum(m['integrated_band_power']),1e-30),4).tolist(),
            'tones':[{'hz':c['frequency_hz'],'span_s':max(c['support_spans_seconds'],key=lambda s:s[1]-s[0]),'contrast_db':round(c['peak_to_local_floor_db'],1)} for c in m['narrowband_candidates'][:2]]})
    elif family=='B':base.update({'modulation':[{'hz':c['rate_hz'],'carrier_hz':c['carrier_band_hz'],'span_s':c['time_span_seconds'],'contrast_db':round(c['peak_to_floor_db'],1)} for c in m['modulation_candidates']]})
    elif family=='C':base.update({'first_last_change_db':None if m['energy_change_db'] is None else round(m['energy_change_db'],3),
        'trend_3db':m['energy_trend_candidate'],'rms_quartiles':np.round(np.quantile(m['rms'],[0,.25,.5,.75,1]),7).tolist(),
        'zero_fraction':round(m['fraction_zero'],6),'clipped_fraction':round(m['fraction_at_digital_full_scale'],6)})
    return base

def vectors(cards,neighbors,labels):
    a,b,c=[cards[f]['measurements'] for f in 'ABC'];v=[]
    power=np.array(a['integrated_band_power']);av=list(power/max(power.sum(),1e-30))
    for tone in a['narrowband_candidates'][:3]:av.extend([tone['frequency_hz']/2000,tone['longest_persistence_seconds']/5,tone['peak_to_local_floor_db']/40])
    bv=[]
    for m in b['modulation_candidates']:bv.extend([m['rate_hz']/80,m['peak_to_floor_db']/40,m['relative_envelope_std']])
    rms=np.array(c['rms']);cv=[(c['energy_change_db'] or 0)/20,*np.quantile(rms,[0,.25,.5,.75,1]).tolist(),c['fraction_zero'],c['fraction_at_digital_full_scale']]
    dv=[sum(n['training_label']==label for n in neighbors)/max(len(neighbors),1) for label in labels]+[n['cosine_distance'] for n in neighbors]
    for values in (av,bv,cv,dv):v.append((values+[0.]*16)[:16])
    return v

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['references','cards'],default='references');args=p.parse_args()
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());lookup={r['id']:r for r in manifest['records']};labels=manifest['labels']
    ids=[uid for part in ('policy_fit','development','calibration','test') for uid in manifest['assessments'][part]]
    if args.stage=='cards':
        pred=np.load(ROOT/'data/derived/classifier_seed17.npz');positions={uid:i for i,uid in enumerate(pred['ids'])}
        train_ids=[r['id'] for r in manifest['records'] if r['partition']=='acoustic_fit'];bank=pred['embeddings'][[positions[uid] for uid in train_ids]]
        metadata=[{**lookup[uid],'anonymous_id':'T'+canonical_hash(uid)[:8]} for uid in train_ids]
    started=time.monotonic()
    for i,uid in enumerate(ids):
        row=lookup[uid];x,fs=sf.read(ROOT/row['audio_path'],dtype='float32');wavehash=dsp.record_hash(x,fs)
        if args.stage=='references':
            out=ROOT/f'data/derived/references/{uid}.json'
            if not out.exists():atomic_json(out,{'id':uid,'waveform_sha256':wavehash,'reference':estimate(x,fs)})
        else:
            cards={};times={}
            for f,func in zip('ABC',(dsp.spectrum,dsp.modulation,dsp.temporal)):
                t=time.perf_counter();cards[f]=func(x,fs);times[f]=time.perf_counter()-t
            j=positions[uid];t=time.perf_counter();neighbors=dsp.retrieve(pred['embeddings'][j],bank,metadata,row['source_group']);times['D']=time.perf_counter()-t
            compacted={f:compact(cards[f]) for f in 'ABC'}
            compacted['D']={'id':'D0','class_analogies_only':True,'neighbors':[{'id':n['exemplar_id'],'label':n['training_label'],'distance':round(n['cosine_distance'],4)} for n in neighbors]}
            probs=softmax(pred['logits'][j]/float(pred['temperature']));sort=np.sort(probs)
            initial=[*probs.tolist(),float(-np.dot(probs,np.log(np.maximum(probs,1e-12)))),float(sort[-1]-sort[-2]),*pred['pca'][j].tolist(),float(np.mean(x*x)),float(np.mean(x==0)),float(np.mean(abs(x)>=1))]
            atomic_json(ROOT/f'data/derived/evidence/{uid}.json',{'id':uid,'waveform_sha256':wavehash,'cards':cards,'compact':compacted,'neighbors':neighbors,
                'prior':{'id':'P0','class_probabilities':{label:round(float(p),5) for label,p in zip(labels,probs)}},
                'initial_state':initial,'tool_vectors':vectors(cards,neighbors,labels),'tool_seconds':times,
                'classifier_sha256':file_hash(ROOT/'data/derived/classifier_seed17.npz') if i==0 else classifier_hash})
            if i==0:classifier_hash=file_hash(ROOT/'data/derived/classifier_seed17.npz')
        if (i+1)%25==0:print(args.stage,i+1,len(ids),round(time.monotonic()-started,1),flush=True)
    atomic_json(ROOT/f'results/manifests/{args.stage}_preparation.json',{'clips':len(ids),'seconds':time.monotonic()-started,'is_human_validation':False})

if __name__=='__main__':main()
