import json
import pytest
from underwater_reasoning.report_schema import terminal_schema

lmfe=pytest.importorskip('lmformatenforcer')

def accepts(report):
    parser=lmfe.JsonSchemaParser(terminal_schema())
    for char in json.dumps(report,separators=(',',':')):
        if char not in parser.get_allowed_characters():return False
        parser=parser.add_character(char)
    return parser.can_end()

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
