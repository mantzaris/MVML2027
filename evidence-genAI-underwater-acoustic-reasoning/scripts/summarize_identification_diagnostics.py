"""Descriptive diagnostics; these do not alter the frozen primary analysis."""
import json
from pathlib import Path
import numpy as np
from underwater_reasoning.io import atomic_json
from analyze_identification import bootstrap

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=json.loads((ROOT/'data/private/identification_manifest.json').read_text())['records']
    labels=np.array([r['label_index'] for r in rows])
    raw=np.load(ROOT/'results/raw/identification/evaluation_predictions.npz')
    positions={uid:i for i,uid in enumerate(raw['ids'])};ix=np.array([positions[r['id']] for r in rows])
    names=['background','cargo','passenger','tanker','tug_tow']
    diagnostics={}
    for variant,prior in [('acoustic_prior','anchor'),('fused_evidence','fusion')]:
        records=[json.loads((ROOT/f'results/raw/identification/omni/test/{variant}/{r["id"]}.json').read_text()) for r in rows]
        predicted=np.array([names.index(r['parsed']['label']) if r['parsed'] and r['parsed']['label'] in names else -1 for r in records])
        correct=raw[prior][ix].argmax(1)==labels;generated_correct=predicted==labels;issued=predicted>=0
        k=int(issued.sum());order=np.argsort(-raw['score_'+prior][ix],kind='stable')[:k]
        diagnostics[variant]={
            'numeric_correct_to_wrong_label':int((correct & issued & ~generated_correct).sum()),
            'numeric_correct_to_withheld':int((correct & ~issued).sum()),
            'numeric_wrong_corrected':int((~correct & generated_correct).sum()),
            'generator_issued':k,'generator_correct':int(generated_correct.sum()),
            'numeric_correct_on_same_generator_issued_records':int((correct & issued).sum()),
            'numeric_correct_at_same_number_issued':int(correct[order].sum()),
            'matched_count_is_test_ranked_descriptive':True}
    contrasts={}
    for comparator in ['anchor','acoustic_refit','no_physical','no_prototypes']:
        _,contrasts[comparator]=bootstrap(rows,labels,{'fusion':raw['fusion'][ix],comparator:raw[comparator][ix]},comparator,2000)
    classwise={}
    for comparator in ['ensemble','acoustic_refit']:
        classwise[comparator]={}
        for label in range(5):
            selected=[i for i,r in enumerate(rows) if r['label_index']==label]
            groups=sorted({rows[i]['source_group'] for i in selected})
            totals=[]
            for group in groups:
                indices=[i for i in selected if rows[i]['source_group']==group]
                a=(raw['fusion'][ix[indices]].argmax(1)==label).sum()
                b=(raw[comparator][ix[indices]].argmax(1)==label).sum()
                totals.append([a-b,len(indices)])
            totals=np.array(totals);rng=np.random.default_rng(1729)
            chosen=totals[rng.integers(0,len(groups),(2000,len(groups)))].sum(1)
            changes=chosen[:,0]/chosen[:,1]
            classwise[comparator][names[label]]={'recall_difference':float(totals[:,0].sum()/totals[:,1].sum()),
                'ci95':np.quantile(changes,[.025,.975]).tolist(),'source_groups':len(groups)}
    atomic_json(ROOT/'results/tables/identification_diagnostics.json',{
        'analysis_status':'exploratory descriptive follow-up; primary criterion unchanged',
        'intervals_are_pointwise_not_simultaneous':True,'generator_overrides':diagnostics,
        'additional_paired_contrasts':contrasts,'class_recall_contrasts':classwise})
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':main()
