"""Check the vectorized paired bootstrap against direct resampling."""
import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('sklearn')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_identification import bootstrap
from sklearn.metrics import f1_score


def test_paired_source_bootstrap_matches_direct_resampling():
    labels=np.arange(35)%5
    rows=[{'source_group':f'g{i//(1+i%2):02d}'} for i in range(35)]
    left=np.eye(5)[labels]
    predicted=labels.copy();predicted[::3]=(predicted[::3]+1)%5
    right=np.eye(5)[predicted]
    intervals,contrast=bootstrap(rows,labels,{'fusion':left,'reference':right},'reference',200)
    groups=sorted({r['source_group'] for r in rows})
    members=[np.array([i for i,r in enumerate(rows) if r['source_group']==g]) for g in groups]
    rng=np.random.default_rng(1729);direct=[]
    for selected in rng.integers(0,len(groups),(200,len(groups))):
        ix=np.concatenate([members[g] for g in selected])
        direct.append(f1_score(labels[ix],labels[ix],labels=range(5),average='macro',zero_division=0)-
                      f1_score(labels[ix],predicted[ix],labels=range(5),average='macro',zero_division=0))
    assert np.allclose(np.quantile(direct,[.025,.975]),contrast['macro_f1']['ci95'])
    _,identical=bootstrap(rows,labels,{'fusion':right,'reference':right},'reference',200)
    assert identical['macro_f1']['difference']==0
    assert identical['macro_f1']['ci95']==[0,0]
