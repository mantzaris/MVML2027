"""Numerical evidence and source-balanced identification, without evaluation labels.

Physical features are statistical predictors, not proofs of vessel identity.
No source/date/position metadata is exposed as a classifier input.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from scipy import signal
from scipy.special import softmax


def source_weights(labels, groups):
    """Equal mass per class, then per recorded source within each class."""
    labels = np.asarray(labels)
    counts = Counter(zip(labels.tolist(), groups))
    class_groups = {c: len({g for y, g in counts if y == c}) for c in set(labels)}
    weights = np.array([1.0/(class_groups[y]*counts[y, g]) for y, g in zip(labels, groups)])
    return weights*len(weights)/weights.sum()


def unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def make_prototypes(embeddings, labels, groups):
    """One normalized mean per acoustic-fit source, not per audio clip."""
    vectors = unit_rows(embeddings)
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    means, classes, keys = [], [], []
    for group in sorted(members):
        indices = members[group]
        unique = set(np.asarray(labels)[indices])
        if len(unique) != 1:
            raise ValueError('A training source has inconsistent class labels')
        means.append(vectors[indices].mean(0))
        classes.append(unique.pop())
        keys.append(group)
    return unit_rows(means), np.array(classes, dtype=int), np.array(keys)


def prototype_evidence(embeddings, prototypes, classes, query_groups=None, bank_groups=None):
    similarities = unit_rows(embeddings)@unit_rows(prototypes).T
    if query_groups is not None:
        similarities[np.asarray(query_groups)[:, None] == np.asarray(bank_groups)[None, :]] = -2
    maxima, averages = [], []
    for label in range(5):
        scores = np.sort(similarities[:, np.asarray(classes) == label], axis=1)
        if scores.shape[1] < 3:
            raise ValueError('Need three distinct acoustic-fit sources per class')
        maxima.append(scores[:, -1])
        averages.append(scores[:, -3:].mean(1))
    maxima, averages = np.array(maxima).T, np.array(averages).T
    distribution = softmax(averages/0.1, axis=1)
    features = np.concatenate([maxima-maxima.mean(1, keepdims=True),
                               averages-averages.mean(1, keepdims=True), distribution], axis=1)
    return features, distribution


def physical_features(audio, fs):
    """Relative spectrum, envelope modulation and digital quality (62 values)."""
    x = np.asarray(audio, dtype=np.float64)
    if fs != 32000 or x.ndim != 1 or len(x) != 160000 or not np.isfinite(x).all():
        raise ValueError('Expected a finite native 32 kHz, five-second mono waveform')
    f, psd = signal.welch(x, fs, nperseg=8192, noverlap=4096, detrend='constant')
    region = (f >= 20) & (f < 8000)
    mass = float(psd[region].sum())
    normalized = psd/max(mass, 1e-30)
    edges = np.geomspace(20, 8000, 33)
    bands = [np.log10(max(normalized[(f >= low) & (f < high)].sum(), 1e-10))
             for low, high in zip(edges[:-1], edges[1:])]
    coarse = [normalized[(f >= low) & (f < high)].sum()
              for low, high in [(20, 100), (100, 500), (500, 2000), (2000, 8000)]]
    p = normalized[region]
    centroid = float(np.dot(f[region], p))
    spread = float(np.sqrt(np.dot((f[region]-centroid)**2, p)))
    entropy = float(-np.sum(p*np.log(np.maximum(p, 1e-30)))/np.log(len(p)))
    flatness = float(np.exp(np.log(np.maximum(psd[region], 1e-30)).mean())/max(psd[region].mean(), 1e-30))
    rms = float(np.sqrt(np.mean(x*x)))
    shape = [centroid/8000, spread/8000, entropy, flatness, float(np.sort(p)[-5:].sum()),
             np.log1p(np.max(abs(x))/max(rms, 1e-15))]
    first, last = np.mean(x[:fs]**2), np.mean(x[-fs:]**2)
    frames = np.sqrt(np.mean(x.reshape(20, -1)**2, axis=1))
    quality = [np.clip(10*np.log10(max(last, 1e-30)/max(first, 1e-30)), -60, 60)/60,
               np.std(frames)/max(np.mean(frames), 1e-15), np.mean(x == 0), np.mean(abs(x) >= 1)]
    reduced = signal.resample_poly(x, 1, 4)
    modulation = []
    for low, high in [(100, 1000), (1000, 3500)]:
        filtered = signal.sosfiltfilt(signal.butter(4, [low, high], fs=8000, btype='bandpass', output='sos'), reduced)
        envelope = abs(signal.hilbert(filtered))[2000:-2000]
        env = signal.resample_poly(envelope, 1, 20)
        ef, ep = signal.welch(env, 400, nperseg=800, noverlap=400, detrend='constant')
        denominator = max(float(ep[(ef >= 2) & (ef <= 80)].sum()), 1e-30)
        bounds = [2, 4, 8, 12, 20, 30, 40, 60, 80.5]
        modulation.extend(np.log10(max(float(ep[(ef >= lo) & (ef < hi)].sum())/denominator, 1e-10))
                          for lo, hi in zip(bounds[:-1], bounds[1:]))
    result = np.array(bands+coarse+shape+quality+modulation, dtype=np.float32)
    if result.shape != (62,) or not np.isfinite(result).all():
        raise ValueError('Invalid physical feature vector')
    return result


def js_divergence(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    middle = (p+q)/2
    return 0.5*(np.sum(p*np.log(p/middle), axis=1)+np.sum(q*np.log(q/middle), axis=1))/np.log(2)


def review_score(probabilities, physical_probabilities, prototype_probabilities, penalty):
    disagreement = (js_divergence(probabilities, physical_probabilities)+
                    js_divergence(probabilities, prototype_probabilities))/2
    return probabilities.max(1)-penalty*disagreement


def coverage_threshold(scores, coverage):
    if not 0 < coverage <= 1 or len(scores) == 0:
        raise ValueError('Invalid calibration coverage')
    # Include ties, exactly as specified before fresh test evaluation.
    k = max(1, int(np.ceil(coverage*len(scores))))
    return float(np.sort(np.asarray(scores))[::-1][k-1])


def confusion(labels, predictions):
    matrix = np.zeros((5, 5), dtype=int)
    np.add.at(matrix, (np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)), 1)
    return matrix


def macro_f1(matrix):
    matrix = np.asarray(matrix, dtype=float)
    denominator = matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix), denominator, out=np.zeros(5), where=denominator > 0).mean())


def identification_metrics(labels, probabilities, scores=None, threshold=None, binary_threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    prediction = np.argmax(probabilities, axis=1)
    matrix = confusion(labels, prediction)
    accepted = np.ones(len(labels), dtype=bool) if scores is None else np.asarray(scores) >= threshold
    correct = prediction == labels
    true_vessel = labels != 0
    predicted_vessel = 1-probabilities[:, 0] >= binary_threshold
    recalls = np.divide(np.diag(matrix), matrix.sum(1), out=np.zeros(5), where=matrix.sum(1) > 0)
    binary_recalls = [float((predicted_vessel[true_vessel == c] == c).mean()) if np.any(true_vessel == c) else None for c in [False, True]]
    result = {'n': len(labels), 'accuracy': float(correct.mean()), 'macro_f1': macro_f1(matrix),
              'confusion_matrix': matrix.tolist(), 'per_class_recall': recalls.tolist(),
              'class_counts': matrix.sum(1).tolist(), 'issued': int(accepted.sum()),
              'correct_issued': int((accepted & correct).sum()), 'wrong_issued': int((accepted & ~correct).sum()),
              'referred': int((~accepted).sum()), 'coverage': float(accepted.mean()),
              'conditional_accuracy': float(correct[accepted].mean()) if accepted.any() else None,
              'correct_per_100': float(100*(accepted & correct).mean()),
              'wrong_per_100': float(100*(accepted & ~correct).mean()),
              'referred_per_100': float(100*(~accepted).mean()),
              'vessel_background_balanced_accuracy': float(np.mean(binary_recalls)) if None not in binary_recalls else None,
              'vessel_background_recalls': binary_recalls, 'binary_threshold': binary_threshold}
    assert result['correct_issued']+result['wrong_issued']+result['referred'] == len(labels)
    return result
