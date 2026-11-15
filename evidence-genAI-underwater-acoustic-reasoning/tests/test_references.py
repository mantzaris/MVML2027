import numpy as np
from underwater_reasoning.references import estimate

def test_independent_reference_known_tone_and_modulation():
    fs=16000;t=np.arange(5*fs)/fs;x=(1+.6*np.sin(2*np.pi*6*t))*np.sin(2*np.pi*400*t)
    ref=estimate(x,fs)
    assert ref['band']['value']=='100-500'
    assert any(abs(c['frequency_hz']-400)<3 for c in ref['tone']['candidates'])
    assert any(abs(c['rate_hz']-6)<1 for c in ref['modulation']['candidates'])
    assert ref['trend']['value']=='steady'

def test_silence_never_becomes_confirmed_physical_absence():
    ref=estimate(np.zeros(160000),32000)
    assert all(not ref[s]['identifiable'] for s in ('band','tone','modulation','trend'))
