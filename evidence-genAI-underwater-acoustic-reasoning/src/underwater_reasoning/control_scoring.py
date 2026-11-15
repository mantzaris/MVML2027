"""Known-signal reviewer scoring, including all AM spectral components."""
def score_controls(results):
    scores={}
    for slot in ('band','tone','modulation','trend'):
        correct=asserted=false_positive=known_positive=positive_correct=0
        for row in results:
            truth=row['truth'][slot];known_positive+=truth is not None
            parsed=row['parsed'];claim=parsed.get(slot,{}) if isinstance(parsed,dict) else {}
            valid=isinstance(parsed,dict) and all(isinstance(parsed.get(s),dict) and isinstance(parsed[s].get('identifiable'),bool) for s in ('band','tone','modulation','trend'))
            yes=claim.get('identifiable') is True;asserted+=yes
            if truth is None:ok=not yes;false_positive+=yes
            elif slot in ('band','trend'):ok=yes and claim.get('value')==truth
            else:
                value=claim.get('frequency_hz' if slot=='tone' else 'rate_hz')
                expected=[truth]
                if slot=='tone' and row['truth']['kind']=='am':
                    rate=row['truth']['modulation'];expected=[truth-rate,truth,truth+rate]
                ok=yes and isinstance(value,(int,float)) and any(abs(value-v)<=max(3,.03*v) if slot=='tone' else abs(value-v)<=max(1,.1*v) for v in expected)
            ok=bool(ok and valid);correct+=ok;positive_correct+=bool(ok and truth is not None)
        scores[slot]={'agreement_with_constructed_control':correct/len(results),'assertions':int(asserted),
            'false_positive_count':int(false_positive),'known_positive_judgments':int(known_positive),'known_positive_correct':int(positive_correct)}
    return scores
