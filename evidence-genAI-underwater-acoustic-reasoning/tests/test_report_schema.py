import json
import jsonschema
from underwater_reasoning.report_schema import terminal_schema

def accepts(report):
    try:jsonschema.validate(report,terminal_schema());return True
    except jsonschema.ValidationError:return False

def report():
    value={s:{'status':'uncertain','value':None,'evidence':[],'time_s':[0,5]} for s in ('class','band','tone','modulation','trend')}
    value['tone']['frequency_hz']=None;value['modulation']['rate_hz']=None
    value['interpretation']=' '.join(['word']*60)
    return value

def test_grammar_rejects_conflicting_uncertainty_and_measurement_citations():
    value=report();assert accepts(value)
    value['tone']['value']='present';assert not accepts(value)
    value=report();value['band'].update(status='asserted',value='100-500',evidence=[{'hz':100}]);assert not accepts(value)

def test_grammar_enforces_word_count_without_choosing_evidence_values():
    value=report();value['interpretation']=' '.join(['word']*59);assert not accepts(value)
    value['interpretation']=' '.join(['word']*100);assert accepts(value)
    value['interpretation']=' '.join(['word']*101);assert not accepts(value)
    value=report();value['tone'].update(status='asserted',value='present',frequency_hz=1777,evidence=['invented']);assert accepts(value)
