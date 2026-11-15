"""Blinded AI-reviewer checks on prospectively generated known-signal controls."""
import json,time
from pathlib import Path
import numpy as np
from scipy import signal
from underwater_reasoning.references import estimate
from underwater_reasoning.omni import Omni,REVISION
from underwater_reasoning.io import atomic_json,canonical_hash
from run_generation import review_contract
from underwater_reasoning.control_scoring import score_controls
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    fs=32000;t=np.arange(5*fs)/fs;rng=np.random.default_rng(20261115);signals=[]
    for frequency,band in [(60,'20-100'),(250,'100-500'),(800,'500-2000'),(1500,'500-2000')]:
        signals.append((.2*np.sin(2*np.pi*frequency*t),{'kind':'tone','band':band,'tone':frequency,'modulation':None,'trend':'steady'}))
    for rate in (2,6,70,80):
        signals.append((.2*(1+.6*np.sin(2*np.pi*rate*t))*np.sin(2*np.pi*600*t),{'kind':'am','band':'500-2000','tone':600,'modulation':rate,'trend':'steady'}))
    for low,high,band in [(30,70,'20-100'),(150,350,'100-500'),(700,1500,'500-2000'),(3000,6000,'2000-8000')]:
        x=signal.sosfiltfilt(signal.butter(5,[low,high],fs=fs,btype='bandpass',output='sos'),rng.normal(0,.1,len(t)))
        signals.append((x,{'kind':'band_noise','band':band,'tone':None,'modulation':None,'trend':'steady'}))
    for direction in (-1,1):
        for frequency,band in [(250,'100-500'),(800,'500-2000')]:
            x=.1*np.exp(direction*t*.2)*np.sin(2*np.pi*frequency*t)
            signals.append((x,{'kind':'gain_ramp','band':band,'tone':frequency,'modulation':None,'trend':'increase' if direction>0 else 'decrease'}))
    for _ in range(4):signals.append((np.zeros(len(t)),{'kind':'silence','band':None,'tone':None,'modulation':None,'trend':None}))
    runner=Omni(ROOT);jobs=[]
    for i,(x,truth) in enumerate(signals):
        ref=estimate(x,fs)
        for role in (0,1):
            text='You are an AI acoustic reviewer, not a human. '+('Assess measurable acoustic attributes carefully.' if role==0 else 'Act as a skeptical second AI assessor. Reject over-specific claims and prefer uncertainty when evidence is weak.')+'\n'+review_contract()+'\nIndependent diagnostics:\n'+json.dumps(ref,separators=(',',':'))
            jobs.append({'id':i,'role':role,'audio':x.astype('float32'),'sample_rate':fs,'text':text,'truth':truth,'reference':ref})
    results=[];start=time.monotonic()
    for b in range(0,len(jobs),16):
        batch=jobs[b:b+16];outputs=runner.generate(batch)
        for job,output in zip(batch,outputs):
            try:
                raw=output['raw'].strip()
                if raw.startswith('```'):raw=raw[raw.find('\n')+1:].rsplit('```',1)[0]
                parsed=json.loads(raw)
            except ValueError:parsed=None
            valid=isinstance(parsed,dict) and all(isinstance(parsed.get(s),dict) and isinstance(parsed[s].get('identifiable'),bool) for s in ('band','tone','modulation','trend'))
            results.append({'id':job['id'],'role':job['role'],'truth':job['truth'],'reference':job['reference'],'parsed':parsed,
                'contract_valid':valid,'prompt_sha256':canonical_hash(job['text']),'model_revision':REVISION,**output})
    valid=sum(r['contract_valid'] for r in results)
    scores=score_controls(results)
    atomic_json(ROOT/'results/raw/reviewer_controls.json',results)
    atomic_json(ROOT/'results/manifests/reviewer_controls.json',{'unique_constructed_signals':20,'reviewer_calls':40,'schema_valid':valid,'slot_results':scores,
        'elapsed_seconds':time.monotonic()-start,'human_reviewers':0,'model':'Qwen2.5-Omni-7B','shared_model_roles_are_correlated':True,
        'tone_control_truth':'AM carrier and both spectral sidebands are accepted','model_revision':REVISION,'reference_version':results[0]['reference']['version'],'am_control_rates_hz':[2,6,70,80],
        'interpretation':'Known-signal agreement characterizes these controls only, not human or real-ocean truth.'})
    print(json.dumps(scores,indent=2),flush=True)
