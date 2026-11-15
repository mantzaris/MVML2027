import numpy as np
import pytest
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

@pytest.mark.parametrize('fs',[16000,32000])
@pytest.mark.parametrize('rate',[2.,6.,30.,70.,80.])
def test_modulation_boundaries_and_high_rates_do_not_select_subharmonics(fs,rate):
    t=np.arange(5*fs)/fs
    x=(1+.6*np.cos(2*np.pi*rate*t))*np.sin(2*np.pi*600*t)
    candidates=estimate(x,fs)['modulation']['candidates']
    assert candidates
    assert all(abs(c['rate_hz']-rate)<.2 for c in candidates)

@pytest.mark.parametrize('fs',[16000,32000])
@pytest.mark.parametrize('frequency',[20.,60.,90.,250.,1500.])
def test_unmodulated_tones_do_not_turn_filter_leakage_into_modulation(fs,frequency):
    t=np.arange(5*fs)/fs
    reference=estimate(np.sin(2*np.pi*frequency*t),fs)
    assert not reference['modulation']['identifiable']

def test_out_of_range_modulation_is_not_relabelled_as_half_rate():
    fs=32000;t=np.arange(5*fs)/fs
    x=(1+.6*np.cos(2*np.pi*120*t))*np.sin(2*np.pi*600*t)
    assert not estimate(x,fs)['modulation']['identifiable']
