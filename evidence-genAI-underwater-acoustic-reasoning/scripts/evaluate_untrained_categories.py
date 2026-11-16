"""Evaluate fixed referral rules on fresh, untrained dataset vessel categories."""
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import soundfile as sf
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
from underwater_reasoning.io import atomic_json, file_hash
from underwater_reasoning.identification import physical_features, prototype_evidence, review_score
from underwater_reasoning.runtime import budget_check
from train_classifier import features, AudioImages, new_model, infer
from train_identification import inputs, calibrated, HEADS

ROOT=Path(__file__).resolve().parents[1]


def main():
    for record in ['protocol/identification_freeze.json','protocol/untrained_categories_freeze.json']:
        for path,expected in json.loads((ROOT/record).read_text())['files'].items():
            if file_hash(ROOT/path)!=expected:raise ValueError('Freeze mismatch: '+path)
    manifest=json.loads((ROOT/'data/private/untrained_categories_manifest.json').read_text());rows=manifest['records']
    model=joblib.load(ROOT/'checkpoints/identification_fusion.joblib')
    feature_path=features(rows,cache_name='untrained_categories')
    loader=DataLoader(AudioImages(feature_path,rows,list(range(len(rows)))),batch_size=64,num_workers=4,pin_memory=True)
    trained=json.loads((ROOT/'results/manifests/classifier_training.json').read_text())['seeds']
    vectors=[];posteriors=[]
    for info in trained:
        budget_check(reserve_seconds=600)
        backbone=new_model();backbone.load_state_dict(torch.load(ROOT/info['checkpoint'],weights_only=False)['model'])
        logits,_,idx,emb=infer(backbone,loader,embedding=True)
        assert np.array_equal(idx,np.arange(len(rows)))
        original=np.load(ROOT/f'data/derived/classifier_seed{info["seed"]}.npz')
        posteriors.append(softmax(logits/float(original['temperature']),axis=1))
        if info['seed']==17:
            vectors=emb;pca=(emb-original['pca_mean'])@original['pca_components'].T
        del backbone;torch.cuda.empty_cache()
    physical=np.array([physical_features(*sf.read(ROOT/r['audio_path'],dtype='float32')) for r in rows])
    exemplar,proto=prototype_evidence(vectors,model['prototypes'],model['prototype_classes'],
                                      [r['source_group'] for r in rows],model['prototype_groups'])
    xx=inputs(np.c_[pca,np.log(np.clip(posteriors[0],1e-8,1))],physical,exemplar)
    predictions={name:calibrated(model['heads'][name],xx[name]) for name in HEADS}
    predictions.update(anchor=posteriors[0],ensemble=np.mean(posteriors,axis=0),prototype_vote=proto)
    predictions['fusion_maxprob']=predictions['fusion']
    scores={name:p.max(1) for name,p in predictions.items()}
    scores['fusion']=review_score(predictions['fusion'],predictions['physical_only'],proto,model['penalty'])
    methods={}
    for name,p in predictions.items():
        issued=scores[name]>=model['thresholds'][name]
        vessel=1-p[:,0]>=model['binary_thresholds'][name]
        def report(use):
            return {'n':int(use.sum()),'trained_category_issued':int(issued[use].sum()),'referred':int((~issued[use]).sum()),
                    'trained_category_issue_rate':float(issued[use].mean()),'broad_vessel_recognition':float(vessel[use].mean()),
                    'issued_background_labels':int((issued[use] & (p[use].argmax(1)==0)).sum()),
                    'issued_label_counts':np.bincount(p[use][issued[use]].argmax(1),minlength=5).tolist() if issued[use].any() else [0]*5}
        methods[name]={'overall':report(np.ones(len(rows),dtype=bool)),
                       'by_untrained_category':{label:report(np.array([r['untrained_category']==label for r in rows])) for label in manifest['labels']}}
    out=ROOT/'results/raw/identification/untrained_categories.npz'
    np.savez_compressed(out,ids=np.array([r['id'] for r in rows]),**predictions,**{'score_'+name:s for name,s in scores.items()})
    summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'clips':len(rows),
             'source_groups':len({r['source_group'] for r in rows}),'methods':methods,
             'exploratory_after_primary_results':True,'no_refitting_or_threshold_change':True,
             'predictions_sha256':file_hash(out),'primary_success_unchanged':True}
    atomic_json(ROOT/'results/tables/untrained_categories_results.json',summary)
    print(json.dumps({name:v['overall'] for name,v in methods.items()},indent=2))


if __name__=='__main__':main()
