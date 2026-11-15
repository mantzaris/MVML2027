"""Common format-only grammar; no query evidence or reference values enter it."""
from .reports import VALUES,SLOTS
VERSION='report-json-grammar-1.0'

def terminal_schema():
    span={'type':'array','items':{'type':'number','minimum':0,'maximum':5},'minItems':2,'maxItems':2}
    def branch(slot,asserted):
        props={'status':{'type':'string','enum':['asserted' if asserted else 'uncertain']},
            'value':{'type':'string','enum':sorted(VALUES[slot])} if asserted else {'type':'null'}}
        numeric='frequency_hz' if slot=='tone' else 'rate_hz' if slot=='modulation' else None
        if numeric:props[numeric]={'type':'number','minimum':20 if slot=='tone' else 2,'maximum':2000 if slot=='tone' else 80} if asserted else {'type':'null'}
        props['evidence']={'type':'array','items':{'type':'string','minLength':1,'maxLength':64},'minItems':0,'maxItems':6 if asserted else 0}
        props['time_s']=span
        return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
    props={slot:{'anyOf':[branch(slot,True),branch(slot,False)]} for slot in SLOTS}
    props['interpretation']={'type':'string','pattern':r'^\S+(?: \S+){59,99}$'}
    return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
