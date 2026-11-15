from underwater_reasoning.control_scoring import score_controls

def test_am_sidebands_are_true_persistent_components():
    truth={'kind':'am','band':'500-2000','tone':600,'modulation':70,'trend':'steady'}
    parsed={s:{'identifiable':True,'value':truth[s]} for s in ('band','tone','modulation','trend')}
    parsed['tone'].update(value='present',frequency_hz=530)
    parsed['modulation'].update(value='present',rate_hz=70)
    assert score_controls([{'truth':truth,'parsed':parsed}])['tone']['known_positive_correct']==1

def test_invalid_review_does_not_receive_silence_agreement_credit():
    truth={'kind':'silence','band':None,'tone':None,'modulation':None,'trend':None}
    scores=score_controls([{'truth':truth,'parsed':None}])
    assert all(v['agreement_with_constructed_control']==0 for v in scores.values())
