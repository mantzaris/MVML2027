"""Exercise the actual compiled backend when the GPU environment is installed."""
import json
import pytest
from underwater_reasoning.report_schema import terminal_schema
xgr=pytest.importorskip('xgrammar')

@pytest.fixture(scope='module')
def compiled():
    tokenizer=xgr.TokenizerInfo(['<eos>']+[chr(i) for i in range(32,127)],stop_token_ids=[0])
    return xgr.GrammarCompiler(tokenizer).compile_json_schema(terminal_schema())

def report():
    result={}
    for slot in ('class','band','tone','modulation','trend'):
        claim={'status':'uncertain','value':None}
        if slot=='tone':claim['frequency_hz']=None
        if slot=='modulation':claim['rate_hz']=None
        claim.update(evidence=[],time_s=[0,5]);result[slot]=claim
    result['interpretation']=' '.join(['word']*60)
    return result

def accepts(compiled,value):
    matcher=xgr.GrammarMatcher(compiled)
    return matcher.accept_string(json.dumps(value,separators=(',',':'))) and matcher.is_completed()

def test_compiled_decoder_blocks_schema_and_length_failures(compiled):
    value=report();assert accepts(compiled,value)
    value['interpretation']=' '.join(['word']*59);assert not accepts(compiled,value)
    value=report();value['tone']['value']='present';assert not accepts(compiled,value)
    value=report();value['band'].update(status='asserted',value='100-500',evidence=[{'hz':100}]);assert not accepts(compiled,value)

def test_compiled_decoder_does_not_supply_semantic_truth(compiled):
    value=report();value['tone'].update(status='asserted',value='present',frequency_hz=1777,evidence=['invented'])
    assert accepts(compiled,value)
    value['interpretation']=' '.join(['word']*101);assert not accepts(compiled,value)
