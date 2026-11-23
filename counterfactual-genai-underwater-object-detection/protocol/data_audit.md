# Real-data access, provenance, and frozen cohort audit

Audit date: 23 November 2026. This amendment was selected from metadata before any acoustic model outcomes. The study uses a custom binary subset of **PORTIA v2**, not the official PORTIA benchmark split or a completed UniqueShip experiment.

## Sources and access

The [official PORTIA v2 release](https://zenodo.org/records/21409381), DOI 10.5281/zenodo.21409381, supplies the original audio and metadata. Its Zenodo API record declares **CC-BY-4.0**. Cite Fernández García, Belmonte-Hernández, Higón Sorribes et al., *PORTIA: A Large-Scale Port Acoustic Dataset with AIS-Grounded Annotations for AI-Based Vessel Monitoring*, Scientific Data (2026), [DOI 10.1038/s41597-026-08384-9](https://doi.org/10.1038/s41597-026-08384-9). Retain their attribution and the release DOI when redistributing any allowed derivative.

The locally pinned API response is `data/manifests/portia_release/record.json`. The Binary Detection, Single-Vessel Classification, and general metadata ZIPs were fully downloaded and matched to the provider's MD5 checksums. The original CSV bytes and their SHA-256 digests are retained. A ZIP64 directory was retrieved by exact HTTP byte ranges; audio acquisition retrieves selected members only, verifies uncompressed size and CRC32, and records native-file and decoded-channel SHA-256. **No whole 74.7 GB archive MD5 verification is claimed.**

UniqueShip was the initially preferred source. On this date, exact one-byte probes of both the 60 h and balanced five-class release parts returned Google Drive quota-exceeded HTML, not audio. No access control was bypassed. The sibling checkout has historical aggregate acquisition audits, but no local raw WAVs or private source manifests. Its original recordings were ONC observations from 2016–2023. PORTIA is a separately collected Port of Valencia dataset from May–June 2025. Its recording collection therefore differs in location and time; no sibling waveform, fitted model, or model outcome is reused. Exact cross-corpus vessel-identity overlap cannot be established from the missing sibling private identifiers. The new models are fitted from scratch on this cohort.

VTUAD rebuilt was considered as an alternative but was not acquired: it derives from ONC and has only 16 dedicated background parents. PORTIA provides an independently collected real evaluation with public native audio and explicit metadata. Earlier literature notes about unavailable dataset licenses do not override the verified PORTIA CC-BY-4.0 record.

## Labels and conflicts

All real labels are **AIS-derived co-occurrence**, not human judgments of acoustic audibility. A negative does not guarantee that no distant, untracked, or non-AIS source is audible. There are no human-validated real event intervals. This study reports presence discrimination against the available AIS labels; temporal detection claims belong to separately labelled controlled mixtures.

The official binary task contains 36,148 windows over 19 UTC days. Most positive windows contain multiple AIS-linked vessels, but only a primary MMSI is exposed, so their complete source identities cannot be kept disjoint. The primary cohort instead combines all 39,125 single-vessel-task positives with the 18,074 binary-task negatives, deduplicated by `window_id`.

**2,388 window IDs conflict across these two task definitions:** background in the binary task, vessel in the single-vessel task. Both memberships are excluded without guessing which label is correct. The complete exclusion list is `data/manifests/cross_task_conflicts.json`; SHA-256 `5c65b22bdee1e0bd6f62c3f05258c64770ff7784f8274fa8ffb93b95af478054`. These contradictions may reflect task-dependent annotation behavior or release defects. Exclusion does not establish that the remaining labels are acoustically correct. The custom source-task combination and induced sampling shift must be reported.

## Independence and sampling

Whole-day plus vessel graph merging is infeasible without data loss: the original binary metadata has one connected component covering 36,104 of 36,148 windows. A metadata-only amendment assigns days by deterministic SHA-256 ordering with seed 17, reserving five days for test, two for calibration, three for development, and the remaining nine for training. Several seeds were inspected for metadata feasibility in the initial smaller binary cohort; the first seed, 17, was retained. No model predictions were available or used.

Positive vessel IDs spanning allocated days are reserved to their highest-priority partition: test, calibration, development, then training. Conflicting lower-priority records are discarded. A 60-second global guard around each three-second window is enforced across partitions. No official benchmark fold is represented as preserved. Complete original continuous-recording IDs are unavailable; whole UTC day provides a conservative observable grouping, but clips crossing midnight cannot establish unknown parent boundaries. Deployment generalization is not claimed.

Sampling is round-robin over days and vessel groups, with deterministic clip order. Per-positive-source caps are 200/60/60/40 for train/development/calibration/test; background-day caps are 500/150/200/200. Target class caps were 2000/400/300/600 per class; genuine capacity shortfalls are retained.

| Partition | Vessel windows | Background windows | Vessel IDs | Retained days | Background days |
|---|---:|---:|---:|---:|---:|
| Train | 1,735 | 2,000 | 30 | 8 | 7 |
| Development | 400 | 335 | 16 | 3 | 3 |
| Calibration | 300 | 300 | 11 | 2 | 2 |
| Test | 600 | 600 | 42 | 5 | 5 |

The cohort totals **6,270 original three-second windows**. Test days are 2025-05-21, 2025-05-22, 2025-05-23, 2025-06-09 and 2025-06-12. `group` is the whole UTC day; `source_id` separately records vessel or background-day identity. Both are disjoint across splits. Five test days are far below the original 100-background-group target. Report day-level results, source counts and highly qualified cluster uncertainty; 1,200 clips are not 1,200 independent encounters. Do not claim robust population-level significance from five days alone.

The frozen planned manifest is `data/manifests/planned_real.jsonl`, SHA-256 `be62767fad35a8dc782f9f6c64a2f2ba372974b7170012e7a18ce37049456ced`. `split_audit.json` records source hashes, exclusions, quotas and achieved counts. No changes based on test outcomes are permitted.

## Native audio, integrity, and restart

The header pilot confirmed **32,000 Hz, 96,000 frames, four channels, PCM_32**, 1,536,044 bytes per WAV. This resolves inconsistent public prose that elsewhere mentions PCM16. Preserve all four channels in original WAVs. Analysis selects zero-based channel 0 consistently; channel 1 is unused. No best-channel selection is allowed after outcome inspection. Decode with `uac.data.decode_wav`; resample and compute all substantive transforms on the pinned Ada GPU.

Audio remains under `data/raw/portia/`. Acquisition receipts are atomic per-file JSON under `data/manifests/download_records/`; they contain file SHA-256, decoded channel SHA-256 and archive CRC. `real_<split>.jsonl` is published only when that split is completely verified and then remains immutable. `real.jsonl` is published only after the entire cohort passes duplicate and integrity checks. `audio_audit.json` states the actual completed status; this document alone does not assert all downloads have finished.

The initial six-worker download encountered HTTP 429 after 66 verified members and was stopped. The resumed downloader uses one combined header/payload range per member, shared pacing at at most 80 requests/minute, and shared server-requested Retry-After backoff. It prioritizes training, then development, calibration and test. Progress and rate-limit receipts are local. Resume exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/prepare_data.py --stage download --workers 3
.venv/bin/python scripts/prepare_data.py --stage audit
```

Successfully hashed native files are reused. Partial or failed acquisitions never count as completed samples. Keep all caches, metadata, audio and receipts inside this submission directory; the sibling submission remains unchanged.
