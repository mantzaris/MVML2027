from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest
import soundfile as sf

from underwater_reasoning.data import (source_group, assign_splits, audit_splits,
                                      purge_temporal_conflicts, sample_annotations, inspect_manifest)
from underwater_reasoning.io import ResultStore, safe_relative
from underwater_reasoning.acquisition import observed_state, subset_targets, choose_action
from underwater_reasoning.evaluation import (ScoredClaim, terminal_loss, physical_metrics,
                                             paired_cluster_bootstrap, structural_citation_check, holm_adjust)
from underwater_reasoning.dsp import spectrum, modulation, temporal, retrieve


def row(i, partition="acoustic_fit", label="cargo", time=None):
    r = {"clip_id": str(i), "label": label, "official_split": "train", "mmsi": str(100000000 + i),
         "deployment_id": "verified-deployment", "parent_recording_id": "parent-" + str(i),
         "start_utc": time or (datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat(),
         "duration_seconds": 5., "partition": partition, "waveform_sha256": "hash-" + str(i)}
    if label == "background": r["mmsi"] = ""
    r["source_group"] = source_group(r)
    return r


def test_missing_identity_is_blocker():
    r = row(0); r["mmsi"] = ""
    with pytest.raises(ValueError, match="MMSI"): source_group(r)
    r = row(0); r["deployment_id"] = ""
    with pytest.raises(ValueError, match="deployment"): source_group(r)


def test_background_groups_use_deployment_and_utc_day():
    a, b = row(0, label="background"), row(1, label="background", time="2020-01-01T04:00:00+00:00")
    assert source_group(a) == source_group(b)
    b["deployment_id"] = "other"
    assert source_group(a) != source_group(b)


def test_official_fold_preserved_and_subdivision_reproducible():
    rows = [row(i) for i in range(16)]
    for i, r in enumerate(rows): r["official_split"] = "train" if i < 8 else "validation" if i < 12 else "test"
    a, b = assign_splits(rows), assign_splits(list(reversed(rows)))
    assert a == b
    assert [r["clip_id"] for r in a if r["partition"] == "test"] == ["12", "13", "14", "15"]
    assert audit_splits(a)["passed"]


def test_conflicting_official_source_fails():
    a, b = row(0), row(1)
    b["source_group"] = a["source_group"]; b["official_split"] = "test"
    with pytest.raises(ValueError, match="Source spans"): assign_splits([a, b])


@pytest.mark.parametrize("field", ["source_group", "parent_recording_id", "waveform_sha256"])
def test_cross_partition_identity_and_duplicates_detected(field):
    a, b = row(0), row(1, "test")
    b[field] = a[field]
    report = audit_splits([a, b])
    assert not report["passed"]
    assert any(p["kind"] == "cross_partition_" + field for p in report["problems"])


def test_temporal_guard_keeps_test_and_records_exclusion():
    a, b = row(0), row(1, "test", time="2020-01-01T00:01:00+00:00")
    assert not audit_splits([a, b])["passed"]
    kept, removed = purge_temporal_conflicts([a, b])
    assert kept == [b] and removed[0]["clip_id"] == a["clip_id"]
    assert audit_splits(kept)["passed"]


def test_temporal_guard_exact_boundary():
    assert audit_splits([row(0), row(1, "test", time="2020-01-01T00:01:05+00:00")])["passed"]


def test_annotation_cap_cannot_be_silently_relaxed():
    with pytest.raises(ValueError, match="Insufficient"):
        sample_annotations([row(0, "test")], {"test": 240})


def test_safe_audio_paths(tmp_path):
    with pytest.raises(ValueError): safe_relative(tmp_path, "../outside.wav")
    with pytest.raises(ValueError): safe_relative(tmp_path, "/outside.wav")
    outside = tmp_path.parent / "outside.wav"
    (tmp_path / "alias").symlink_to(outside)
    with pytest.raises(ValueError): safe_relative(tmp_path, "alias")


def test_native_header_validation(tmp_path):
    import csv
    from underwater_reasoning.data import REQUIRED
    sf.write(tmp_path / "x.wav", np.zeros(160000), 32000, subtype="PCM_32")
    r = row(0); r["audio_path"] = "x.wav"
    with (tmp_path / "input.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED, extrasaction="ignore"); writer.writeheader(); writer.writerow(r)
    assert inspect_manifest(tmp_path / "input.csv", tmp_path)[0]["sample_rate"] == 32000
    sf.write(tmp_path / "x.wav", np.zeros(100000), 20000)
    with pytest.raises(ValueError, match="32 kHz"): inspect_manifest(tmp_path / "input.csv", tmp_path)


def test_unselected_values_are_never_read():
    class ObservedOnly(dict):
        def __getitem__(self, key):
            assert key == 0
            return [12., 13.]
    value = observed_state([1, 2], {0}, ObservedOnly(), 2)
    assert np.array_equal(value[-6:], np.zeros(6))


def test_subset_delta_direction_and_action_masks():
    losses = {s: 1 - s.bit_count() / 4 for s in range(16)}
    targets = subset_targets(losses)
    assert targets[0]["delta"] == [.25] * 4
    assert targets[1]["available"] == [False, True, True, True]
    assert targets[1]["delta"][0] == 0


def test_policy_stopping_and_masking():
    assert choose_action([0, 0, 0, 0], [0] * 4, set()) is None
    assert choose_action([1, .5, .3, 0], [0] * 4, {0}) == 1
    assert choose_action([1] * 4, [0] * 4, {0, 1}) is None
    assert choose_action([0] * 4, [0] * 4, set(), force=True) == 0


def test_resumption_reuses_success_and_recovers_failure(tmp_path):
    store = ResultStore(tmp_path)
    key = {"waveform": "h", "revision": "r", "prompt": "p", "subset": 0}
    store.write(key, {"error": "interrupted"}, "failed")
    assert store.read(key) is None
    store.write(key, {"raw": "valid"})
    assert store.read(key)["payload"] == {"raw": "valid"}
    with pytest.raises(ValueError, match="overwrite"): store.write(key, {"raw": "different"})
    assert store.read(dict(key, prompt="different")) is None


def test_uncertainty_and_empty_assertion_denominators():
    uncertain = ScoredClaim("uncertain", False, False, False)
    assert uncertain.loss() == 0
    assert terminal_loss(None, [uncertain] * 4) == .125
    metrics = physical_metrics([[uncertain] * 4])
    assert metrics["supported_coverage"] == 0
    assert metrics["unsupported_claim_rate"] is None
    assert metrics["appropriate_uncertainty"] == 1


def test_metrics_keep_omissions_in_denominator():
    good = ScoredClaim("asserted", True, True, True, True)
    missing = ScoredClaim("omitted", True, False, False)
    m = physical_metrics([[good, missing, missing, missing]], whole_test_n=2)
    assert m["supported_coverage"] == .25
    assert m["whole_test_useful_coverage"] == .125
    assert m["citation_completeness"] == 1


def test_cluster_bootstrap_preserves_pairing_and_duplicate_clusters():
    seen = []
    def stat(values):
        seen.append(tuple(values)); return np.mean(values)
    result = paired_cluster_bootstrap([1, 1, 1, 0], [1, 1, 1, 0], ["a", "a", "a", "b"], stat, 50)
    assert result["ci95"] == [0, 0]
    assert (1, 1, 1, 1, 1, 1) in seen and (0, 0) in seen


def test_holm_monotonic_adjustment():
    assert np.allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])


def test_valid_but_false_card_does_not_become_independent_truth():
    card = {"record_hash": "query", "tool": "A", "measurements": {"frequency_hz": 999}}
    assert structural_citation_check("narrowband", ["A0"], {"A0": card}, "query", numeric=True) == []
    independently_wrong = ScoredClaim("asserted", True, False, False, True)
    assert independently_wrong.loss() == 1


def test_null_audio_does_not_invent_measurements():
    x = np.zeros(100000)
    assert spectrum(x, 20000)["measurements"]["dominant_band_hz"] is None
    assert modulation(x, 20000)["measurements"]["modulation_candidates"] == []
    assert temporal(x, 20000)["measurements"]["energy_change_db"] is None


def test_known_tone_modulation_and_received_trend():
    fs = 20000; t = np.arange(5 * fs) / fs
    x = .2 * (1 + .6 * np.sin(2 * np.pi * 12 * t)) * np.sin(2 * np.pi * 450 * t)
    assert abs(spectrum(x, fs)["measurements"]["narrowband_candidates"][0]["frequency_hz"] - 450) <= 3
    assert abs(modulation(x, fs)["measurements"]["modulation_candidates"][0]["rate_hz"] - 12) <= 1
    assert temporal(x * np.exp(t / 2), fs)["measurements"]["energy_trend_candidate"] == "increase"


def test_retrieval_excludes_query_group_and_never_exposes_private_metadata():
    bank = np.array([[1., 0], [.9, .1], [.8, .2], [0, 1]])
    meta = [{"partition": "acoustic_fit", "source_group": g, "anonymous_id": str(i), "label": "cargo",
             "mmsi": "private", "deployment_id": "private"} for i, g in enumerate(["query", "a", "a", "b"])]
    result = retrieve([1, 0], bank, meta, "query")
    assert len(result) == 2
    assert all(set(r) == {"exemplar_id", "training_label", "cosine_distance"} for r in result)
    meta[0]["partition"] = "test"
    with pytest.raises(ValueError, match="non-acoustic-fit"): retrieve([1, 0], bank, meta, "query")
