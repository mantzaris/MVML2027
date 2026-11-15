"""Common terminal report parsing and contract, with one deterministic cleanup."""
import json,re
SLOTS=('class','band','tone','modulation','trend')
VALUES={'class':{'background','cargo','passenger','tanker','tug_tow'},'band':{'20-100','100-500','500-2000','2000-8000'},
 'tone':{'present'},'modulation':{'present'},'trend':{'increase','decrease','steady'}}

def parse_report(raw):
    # One cleanup pass removes an enclosing Markdown fence only. No repair call.
    cleaned=raw.strip()
    if cleaned.startswith('```'):
        cleaned=re.sub(r'^```(?:json)?\s*','',cleaned);cleaned=re.sub(r'\s*```$','',cleaned)
    try:report=json.loads(cleaned)
    except (ValueError,TypeError):return None,['json_parse']
    errors=[]
    if not isinstance(report,dict):return None,['not_object']
    for slot in SLOTS:
        claim=report.get(slot)
        if not isinstance(claim,dict):errors.append(slot+'_missing');continue
        status=claim.get('status');value=claim.get('value')
        if status not in {'asserted','uncertain'}:errors.append(slot+'_status')
        if status=='asserted' and value not in VALUES[slot]:errors.append(slot+'_value')
        if status=='uncertain' and value is not None:errors.append(slot+'_uncertain_value')
        if not isinstance(claim.get('evidence'),list) or not all(isinstance(v,str) for v in claim.get('evidence',[])):errors.append(slot+'_evidence')
        span=claim.get('time_s')
        if not isinstance(span,list) or len(span)!=2 or not all(isinstance(v,(int,float)) for v in span) or not 0<=span[0]<span[1]<=5:errors.append(slot+'_time')
        numeric='frequency_hz' if slot=='tone' else 'rate_hz' if slot=='modulation' else None
        if numeric and status=='asserted':
            number=claim.get(numeric)
            if not isinstance(number,(int,float)) or not (20<=number<=2000 if slot=='tone' else 2<=number<=80):errors.append(slot+'_numeric')
    words=len(str(report.get('interpretation','')).split())
    if not 60<=words<=100:errors.append('interpretation_word_count')
    return report,errors
