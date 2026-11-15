"""Independent automated acoustic proxies, never human ground truth.

This module intentionally imports no proposed DSP tool. Multitaper spectra,
non-overlapping-frame persistence, and squared-envelope autocorrelation use
different estimators. Silence/weak evidence stays ambiguous; no-detection is not
a verified negative physical label. Thresholds are fixed before test outcomes.
"""
from functools import lru_cache
import numpy as np
from scipy import signal,ndimage

BANDS=((20,100),(100,500),(500,2000),(2000,8000))
VERSION='independent-proxy-1.2'

@lru_cache(maxsize=4)
def tapers(n,nw,k):return signal.windows.dpss(n,NW=nw,Kmax=k)

def estimate(audio,fs):
    x=np.asarray(audio,dtype=np.float64);n=len(x)
    if x.ndim!=1 or n!=5*fs or not np.isfinite(x).all():raise ValueError('Reference expects finite five-second mono audio')
    f=np.fft.rfftfreq(n,1/fs)
    psd=np.mean(abs(np.fft.rfft(tapers(n,3,5)*(x-x.mean()),axis=1))**2,axis=0)/fs
    psd[1:-1]*=2
    power=np.array([psd[(f>=lo)&(f<hi)].sum() for lo,hi in BANDS])
    order=np.argsort(power);margin=float(10*np.log10(max(power[order[-1]],1e-30)/max(power[order[-2]],1e-30)))
    energy=float(np.mean(x*x));band_ok=energy>1e-20 and margin>=1
    band={'identifiable':bool(band_ok),'value':f'{BANDS[order[-1]][0]}-{BANDS[order[-1]][1]}' if band_ok else None,'margin_db':margin,'time_s':[0,5]}
    floor=ndimage.median_filter(psd,size=151,mode='reflect')
    contrast=10*np.log10(np.maximum(psd,1e-30)/np.maximum(floor,1e-30))
    peaks,_=signal.find_peaks(contrast,height=12,distance=20)
    # Five independent non-overlapping 1-second multitaper frames.
    frames=x[:5*fs].reshape(5,fs);frames=frames-frames.mean(1,keepdims=True)
    frame_psd=np.mean(abs(np.fft.rfft(frames[:,None,:]*tapers(fs,2,3)[None,:,:],axis=2))**2,axis=1)
    frame_floor=ndimage.median_filter(frame_psd,size=(1,31),mode='reflect')
    fc=10*np.log10(np.maximum(frame_psd,1e-30)/np.maximum(frame_floor,1e-30))
    tones=[]
    for i in sorted(peaks,key=lambda i:psd[i],reverse=True):
        if not 20<=f[i]<=2000 or psd[i]<1e-20:continue
        center=int(round(f[i]));lo=max(0,center-2);hi=center+3
        active=fc[:,lo:hi].max(1)>=9
        runs=np.diff(np.r_[False,active,False].astype(int));spans=[(a,b) for a,b in zip(np.where(runs==1)[0],np.where(runs==-1)[0]) if b-a>=3]
        if spans:
            a,b=max(spans,key=lambda p:p[1]-p[0]);tones.append({'frequency_hz':float(f[i]),'time_s':[int(a),int(b)],'contrast_db':float(contrast[i])})
        if len(tones)==5:break
    modulation=[]
    for low,high in ((100,1000),(1000,8000)):
        y=signal.sosfiltfilt(signal.butter(3,[low,min(high,fs/2-1)],fs=fs,btype='bandpass',output='sos'),x)
        # Attenuated out-of-band tones must not define a normalized envelope
        # reference merely because the absolute floating-point energy is nonzero.
        if np.mean(y*y)<.01*max(energy,1e-20):continue
        envelope_power=signal.sosfiltfilt(signal.butter(3,100,fs=fs,output='sos'),y*y)
        env=np.sqrt(np.maximum(envelope_power,0))[fs//4:-fs//4:max(1,fs//1000)]
        relative=float(np.std(env)/max(float(np.mean(env)),1e-30));env-=env.mean()
        ac=signal.correlate(env,env,mode='full',method='fft')[len(env)-1:];ac/=max(float(ac[0]),1e-30)
        hz=fs/max(1,fs//1000)
        # Find peaks before cutting the permitted lag interval: slicing first
        # discards boundary peaks and can select a half-rate subharmonic.
        peaks,_=signal.find_peaks(ac,height=.30,prominence=.08)
        candidates=[]
        for lag in peaks[:1]:
            # A faster dominant period cannot be relabeled by choosing its
            # second or later autocorrelation peak inside the task range.
            if not max(1,int(np.floor(hz/80)))<=lag<=int(np.ceil(hz/2)):continue
            curvature=ac[lag-1]-2*ac[lag]+ac[lag+1]
            offset=.5*(ac[lag-1]-ac[lag+1])/curvature if curvature else 0.
            refined=lag+float(np.clip(offset,-.5,.5));rate=hz/refined
            # Tiny endpoint tolerance covers finite-window interpolation bias.
            if 1.99<=rate<=80.1:candidates.append((lag,float(np.clip(rate,2,80))))
        if candidates and relative>=.03 and np.mean(y*y)>1e-20:
            # Earliest prominent repeating period prevents selecting a later subharmonic.
            lag,rate=candidates[0]
            repeat=2*lag<len(ac) and ac[2*lag]>=.15
            if repeat:modulation.append({'rate_hz':rate,'time_s':[.25,4.75],'carrier_hz':[low,high],'autocorrelation':float(ac[lag])})
    first=float(np.dot(x[:fs],x[:fs])/fs);last=float(np.dot(x[-fs:],x[-fs:])/fs)
    change=float(10*np.log10(last/first)) if min(first,last)>1e-20 else None
    trend_ok=change is not None and abs(abs(change)-3)>.5
    trend=None if not trend_ok else 'increase' if change>3 else 'decrease' if change< -3 else 'steady'
    return {'version':VERSION,'band':band,'tone':{'identifiable':bool(tones),'value':'present' if tones else None,'candidates':tones},
        'modulation':{'identifiable':bool(modulation),'value':'present' if modulation else None,'candidates':modulation},
        'trend':{'identifiable':bool(trend_ok),'value':trend,'change_db':change,'time_s':[0,5]},
        'sample_rate':fs,'seconds':n/fs,'is_human_label':False}
