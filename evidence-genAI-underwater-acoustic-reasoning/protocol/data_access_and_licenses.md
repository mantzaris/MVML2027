# Release access and provenance audit — 2026-11-15

Primary sources: [UniqueShip release page](https://uniqueshipdata.org/), [the exact small-release folder](https://drive.google.com/drive/folders/1HxV8AGh8ckZS3JehAU_gY2fuCnscX-eV), [release README](https://drive.google.com/file/d/1Nzyqr6KJiYt96PIHLZMM0oCp9nAJJVyP/view), [ONC data policy](https://www.oceannetworks.ca/data/data-policy/), and [author manuscript](https://arxiv.org/abs/2609.13659).

The provider declares CC BY 4.0 for ONC-owned data. ONC explicitly distinguishes partner-owned/co-owned recordings, whose terms may differ. Retain the provider attribution to Ocean Networks Canada and the UniqueShip authors (Connor Hashemi, Trevor Stout, Anthony Hoogs, Jason Parham), and resolve recording-specific provenance before any redistribution. This repository distributes no raw recordings or identity tables.

## Preparation-stage acquisition

The linked archive is multipart: `allclass_5hours.z01` (10,737,418,240 bytes) and `allclass_5hours.zip` (7,059,560,361 bytes). The ZIP64 central directory contains 43,242 entries. HTTP range requests retrieved the directory, fold-1 train/validation/test CSVs, test-fold metadata for sampling feasibility, and one training waveform per retained class. Exact ranges, entry counts, uncompressed sizes, and member CRC32 were checked. SHA-256 values of acquired metadata and five samples are in `sources/release_metadata_audit.json`. The full archive has **not** been downloaded or checksummed. CRC is an integrity check, not proof of publisher authenticity.

Five sampled headers and all fold-1 metadata say 32,000 Hz. Sampled files are mono PCM_32 and five seconds. This contradicts the landing page's 20 kHz description. Native data must stay unchanged; use an explicitly documented anti-aliased derivative for the planned 20 kHz encoder.

| Official fold 1 | Selected-class clips | Passenger vessels | Background UTC days |
|---|---:|---:|---:|
| Train | 14,372 | 88 | 641 |
| Validation | 1,381 | 9 | 79 |
| Test | 2,250 | 13 | 82 |

Counts precede short-clip, provenance, cap, and overlap exclusions; they are not final experiment counts. The selected subset contains 18,003 clips. Two cargo and two tanker training clips, plus one tug test clip, are shorter than five seconds. The raw class names are `background`, `cargo`, `passengership`, `tanker`, and `tug`; their declared study mapping is preserved separately.

## Execution update

The frozen [execution amendment](execution_amendment.md) resolves the feasible recorded-group audit without inventing missing parent/deployment identities. The selected 7,418 five-second WAVs were then downloaded by exact archive-member ranges, with CRC checks and a decoded-waveform hash audit. Total waveform files occupy 4,747,846,392 bytes; there are no duplicate decoded waveforms or cross-partition waveform-hash collisions. This is complete acquisition of the experimental subset, not a checksum of the entire multipart archive. Native 32 kHz audio is retained locally and on the worker. Trained checkpoints, embeddings and detailed results are copied locally and checked; the remote overlay is never represented as persistent storage. See `results/manifests/audio_download_audit.json`, `experiment_split_audit.json`, `local_checkpoint_preservation.json`, and `execution_ledger.json`.

## Integrity findings and blockers

The available MMSI groups and background UTC-date groups do not overlap across the official fold. This establishes only the checkable portion of the requested independence protocol. Parent recording and deployment identifiers are absent. Hydrophone device IDs cannot automatically be treated as deployment IDs; background lacks even those device fields. The documented filenames encode clip timestamps, but distinct filenames do not prove independent recordings.

At preparation, evaluation required authoritative provenance mappings or a documented conservative grouping amendment. The executed amendment groups all classes by whole UTC days and adds the global time guard, source separation and waveform-hash checks. It does not resolve unavailable parent/deployment identifiers. Missing fields remain missing.

The intended balanced 240-clip test reference set also fails a metadata-only feasibility check. Passenger test vessel counts in folds 1–5 are 13, 5, 10, 9, and 10. Even before enforcing ten-minute separation, at most 39, 15, 30, 27, and 30 passenger clips can be selected under a three-per-source cap, below the requested 48. Record and freeze any sampling amendment before model outcomes. These constraints still apply when AI reviewers substitute for humans.

## Storage and compute

Local ignored data and source caches are retained inside this research directory. The connected Pod exposes a 50 GB overlay filesystem at `/workspace`; it is ephemeral. GPU hardware is RTX PRO 4500 Blackwell, 32,623 MiB, driver 595.91.07; installed PyTorch reports 2.8.0+cu128/CUDA 12.8. Persistent remote storage is unverified. Do not put the only checkpoint copy there.

The user authorizes at most 10 GPU-hours of research on the existing service and explicitly requests that the Pod remain running when finished. Actual rate and allocation start/idle history are unknown. No GPU research workload was started during this preparation stage; this does **not** mean the running Pod has no idle charges. The research ceiling is not a cap on subsequent idle billing when the Pod remains running. No new Pod, volume, external dataset request, or publication was created.
