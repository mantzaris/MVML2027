"""Known-answer checks of the actual paired conditional-metric implementation."""
from pathlib import Path
import numpy as np
from analyze_experiment import cluster_contrast,classifier_metrics
from underwater_reasoning.io import atomic_json

def row(uid,group,retained,supported,assertions):
    return {'id':uid,'source_group':group,'retained':retained,
        'score':{'supported':supported,'assertions':assertions,'unsupported':assertions-supported}}

left=[row('a','x',True,2,3),row('b','x',False,4,4),row('c','y',True,2,3)]
right=[row('a','x',True,1,2),row('b','x',True,1,2),row('c','y',True,1,2)]
support=cluster_contrast(left,right,resamples=100)
assert support['difference']==.25 and support['ci95']==[.25,.25]
whole=cluster_contrast(left,right,resamples=100,metric='whole_test_useful_coverage')
assert np.isclose(whole['difference'],1/12)
unsupported=cluster_contrast(left,right,resamples=100,metric='unsupported_rate')
assert np.isclose(unsupported['difference'],-1/6)
same=cluster_contrast(left,left,resamples=100)
assert same['difference']==0 and same['ci95']==[0,0] and same['p_two_sided_centered_bootstrap']==1
empty=[{**r,'retained':False} for r in left]
assert cluster_contrast(empty,right,resamples=100)['difference'] is None
try:cluster_contrast(left,list(reversed(right)),resamples=100)
except ValueError:pass
else:raise AssertionError('Unpaired reports were accepted')
# Uniform genuine scores: five balanced classes, accuracy/F1 tie outcome and exact Brier/ECE.
prob=classifier_metrics(np.arange(5),np.zeros((5,5)),1.)
assert np.isclose(prob['brier'],.8) and np.isclose(prob['accuracy'],.2)
assert np.isclose(prob['ece_equal_frequency_10'],.32)
root=Path(__file__).resolve().parents[1]
atomic_json(root/'results/manifests/analysis_math_checks.json',{'checks_passed':8,
    'scope':['conditional denominators','whole-test denominator','unsupported ratio','paired identical methods','empty retention','pairing rejection','genuine-score Brier','equal-frequency ECE'],
    'test_outcomes_used':False,'small_sample_ece_caveat':'With five records and ten requested bins, the implementation uses five singleton nonempty bins.'})
print('Eight known-answer analysis checks passed')
