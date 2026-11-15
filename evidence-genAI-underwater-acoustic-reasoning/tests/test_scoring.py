from underwater_reasoning.scoring import score,citation_support

def fixture():
    reference={s:{'identifiable':True,'value':v} for s,v in [('band','100-500'),('trend','steady')]}
    reference.update(tone={'identifiable':False,'value':None,'candidates':[]},modulation={'identifiable':False,'value':None,'candidates':[]})
    assets={'waveform_sha256':'query','cards':{'A':{'record_hash':'query','measurements':{'dominant_band_hz':[100,500]}}}}
    report={s:{'status':'uncertain','value':None,'evidence':[],'time_s':[0,5]} for s in ['class','band','tone','modulation','trend']}
    report['band'].update(status='asserted',value='100-500',evidence=['A0'])
    return reference,assets,report

def test_valid_citation_does_not_excuse_false_measurement():
    ref,assets,report=fixture();good=score(report,'cargo',ref,assets,1)
    assert good['supported']==1
    ref['band']['value']='500-2000';bad=score(report,'cargo',ref,assets,1)
    assert bad['supported']==0 and bad['unsupported']==1
    assert bad['physical'][0]['citation_supported']

def test_extra_invalid_citation_rejects_support():
    ref,assets,report=fixture();report['band']['evidence']=['A0','D0']
    assert score(report,'cargo',ref,assets,9)['supported']==0

def test_swapped_card_retains_foreign_provenance_and_is_rejected():
    ref,assets,report=fixture();assets['cards']['A']['record_hash']='another-recording'
    assert score(report,'cargo',ref,assets,1)['supported']==0

def test_uncertainty_not_positive_and_word_count_not_hidden_repair():
    ref,assets,report=fixture();report['band']['status']='uncertain';report['band']['value']=None;report['band']['evidence']=[]
    result=score(report,'cargo',ref,assets,0,errors=['interpretation_word_count'])
    assert result['valid'] and not result['format_valid'] and result['supported']==0 and result['assertions']==0

def test_matched_numeric_error_uses_a_reference_in_the_claimed_interval():
    ref,assets,report=fixture()
    ref['tone']={'identifiable':True,'value':'present','candidates':[
        {'frequency_hz':100,'time_s':[2,5]}, {'frequency_hz':102,'time_s':[0,3]}]}
    report['tone'].update(status='asserted',value='present',frequency_hz=100,time_s=[0,3])
    value=next(v for v in score(report,'cargo',ref,assets,0)['physical'] if v['slot']=='tone')
    assert value['correct'] and value['numeric_absolute_error']==0
    assert value['numeric_matched_absolute_error']==2
