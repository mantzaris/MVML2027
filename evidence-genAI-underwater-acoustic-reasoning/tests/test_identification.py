import numpy as np
from underwater_reasoning.identification import (source_weights, make_prototypes, prototype_evidence,
    physical_features, coverage_threshold, identification_metrics, js_divergence)


def test_source_weighting_equalizes_classes_and_sources():
    labels = np.array([0, 0, 0, 1, 1, 1])
    groups = ['a', 'a', 'b', 'c', 'd', 'd']
    weights = source_weights(labels, groups)
    assert np.isclose(weights[:2].sum(), weights[2])
    assert np.isclose(weights[3], weights[4:].sum())
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())


def test_source_prototypes_are_invariant_to_repeated_identical_clips():
    vectors = np.eye(5).repeat(3, axis=0)
    labels = np.arange(5).repeat(3)
    groups = [str(i) for i in range(15)]
    a, classes, keys = make_prototypes(vectors, labels, groups)
    b, classes_b, keys_b = make_prototypes(np.vstack([vectors, vectors[0]]), np.r_[labels, labels[0]], groups+[groups[0]])
    assert np.allclose(a, b)
    features, probs = prototype_evidence(np.eye(5), a, classes)
    assert np.array_equal(probs.argmax(1), np.arange(5))
    assert features.shape == (5, 15)


def test_referrals_are_never_counted_as_correct_identifications():
    labels = np.arange(5)
    probabilities = np.eye(5)
    scores = np.array([.9, .8, .7, .6, .5])
    metrics = identification_metrics(labels, probabilities, scores, .7)
    assert metrics['correct_issued'] == 3
    assert metrics['wrong_issued'] == 0
    assert metrics['referred'] == 2
    assert metrics['correct_per_100'] == 60
    assert metrics['macro_f1'] == 1


def test_threshold_ties_and_disagreement():
    assert coverage_threshold([.9, .9, .5, .2], .25) == .9
    p = np.array([[.5, .5], [1., 0.]])
    assert np.allclose(js_divergence(p, p), 0)
    assert np.isclose(js_divergence(np.array([[1., 0.]]), np.array([[0., 1.]]))[0], 1)


def test_physical_features_gain_invariance_and_known_band():
    t = np.arange(160000)/32000
    x = .1*np.sin(2*np.pi*300*t)
    a, b = physical_features(x, 32000), physical_features(.25*x, 32000)
    assert a.shape == (62,)
    assert np.allclose(a, b, atol=1e-5)
    assert np.argmax(a[32:36]) == 1
    assert np.isfinite(physical_features(np.zeros_like(x), 32000)).all()
