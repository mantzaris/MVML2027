# Evidence-Grounded Generative AI for Underwater Acoustic Reasoning

Research preparation for Alexander V. Mantzaris's MVML 2027 project. All session work is contained in this directory. Commit on **main only**; the owner performs pushes manually.

## Current status

**Preparation stage; no classifier/controller training, generator experiments, AI review runs, or completed conference paper.** The user has explicitly requested AI reviewers in place of humans and approved a 10-hour GPU ceiling. Read the supplied execution prompt and 22-page specification alongside [the amended protocol draft](protocol/preregistered_plan.md). The supplied documents remain unchanged; subsequent user instructions take precedence.

Completed here: live source and conference checks; a CRC-checked partial UniqueShip release acquisition (fold-1 metadata and five training waveform samples); grouped-data audits and annotation feasibility checks; CPU analysis tools; independent review views and local annotation forms; policy-state/scoring primitives; meaningful tests; an incomplete manuscript layout using the official Word template.

The release check changed the preparation assumptions:

- The actual sampled WAV files are **32 kHz mono PCM_32**, while the landing page says 20 kHz. Native audio is preserved; a future classifier pipeline must explicitly derive anti-aliased 20 kHz audio.
- Fold 1 supplies 14,372 training, 1,381 validation, and 2,250 test clips for the five selected classes, before exclusions. Five selected-class clips are shorter than five seconds.
- Parent recording IDs and deployment IDs are absent; all 3,600 background clips lack hydrophone IDs. Date-only background groups and vessel IDs are disjoint across the official fold, but that does not complete the stronger provenance/overlap audit.
- No official fold supports 48 passenger test annotations with at most three clips per vessel: test passenger vessel counts range from 5 to 13. The annotation design needs a documented amendment before test outcomes.
- RunPod `/workspace` is on `overlay`; persistent storage has not been verified. The research GPU ceiling is 10 hours; actual hourly rate and prior allocated time are unknown. **Leave the Pod running when finished**, as explicitly requested. No human reviewers will be used. Automated references and AI judgments must be reported as proxy assessments, never human validation.

See [the acquisition audit](protocol/data_access_and_licenses.md), [machine-readable findings](sources/grouping_audit.json), and [fold feasibility](sources/fold_annotation_feasibility.json). Large raw archives, audio, identity metadata, review keys, and model weights are excluded from Git. No test model outcomes have been generated or examined.

## Reproduce CPU preparation

From the repository root:

```bash
cd evidence-genAI-underwater-acoustic-reasoning
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/validate_synthetic.py
.venv/bin/python scripts/check_readiness.py
```

`check_readiness.py` deliberately exits 2 while required inputs are pending. The lock describes the tested **CPU preparation** environment, Python 3.10.18/Linux x86_64. It is not a GPU training lock. Synthetic checks cover 500 known-signal fixtures and 100 structural evidence interventions; they do not evaluate a language model or establish real-data grounding quality.

Public-source rechecks (network access, no paid API):

```bash
.venv/bin/python scripts/inspect_sources.py
.venv/bin/python scripts/probe_release.py
.venv/bin/python scripts/audit_release_metadata.py
.venv/bin/python scripts/build_bibliography.py
```

The probe reads only byte ranges needed for archive metadata and five training samples. It verifies response ranges, ZIP entry counts, member sizes, and CRCs. This is explicitly **not a full archive download or full audio-integrity audit**. `sources/` records metadata hashes and findings; downloaded identities/audio and cached source HTML remain ignored locally.

After a verified release-to-canonical mapping supplies the missing identities and full audio is acquired:

```bash
.venv/bin/python scripts/prepare_data.py \
  --manifest data/private/canonical_fold1.csv \
  --audio-root data/raw/uniqueship \
  --output data/private/prepared
.venv/bin/python scripts/audit_splits.py data/private/prepared/manifest.json \
  --output data/private/prepared/independent_audit.json
.venv/bin/python scripts/create_annotation_pack.py \
  --manifest data/private/prepared/manifest.json \
  --audio-root data/raw/uniqueship \
  --output annotations/packs/review_v1 \
  --private-key annotations/private/review_v1_key.json
```

These commands reject missing provenance, unexpected native headers, source overlap, and infeasible annotation counts. The current public release cannot yet satisfy those gates. Do not fabricate the missing columns. The current 600-clip balanced request intentionally fails rather than silently raising the per-source cap.

To inspect the review interface without representing a human study:

```bash
.venv/bin/python scripts/create_demo_pack.py
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory .cache/annotation_demo
```

Open `http://127.0.0.1:8765`. The demo contains one synthetic AM tone; it is not part of the 600-clip study. The creation command refuses to overwrite an existing pack. Keep exported reviewer JSONs and the identity key separate, following [the annotation guide](protocol/annotation_guide.md).

## Layout and implementation limits

| Location | Contents |
|---|---|
| `configs/` | Explicit draft study and execution inputs; unknown values remain null |
| `protocol/` | Protocol, amendments, annotation instructions, and locked test status |
| `sources/` | Verified bibliographic metadata, access checks, and aggregate release audits |
| `src/underwater_reasoning/` | Data guards, DSP candidates, distinct-source retrieval, blinded review views, metrics, observed-state masks, immutable result shards |
| `tests/` | Leakage, known-signal, masking, scoring, retrieval, and resume checks |
| `results/manifests/` | Actual CPU validation results and artifact lineage |
| `manuscript/` | Verified bibliography and explicitly incomplete official-template draft |

The acoustic training loop, frozen Omni wrapper, learned controller fitting, full baseline matrix, automated-reference semantic evaluator, AI-review inference, cost instrumentation, and empirical result figures remain unimplemented. They are not replaced by placeholder result tables. The current structural checker only checks citation compatibility; a valid but numerically false card can pass. Separate reference estimation and AI prose review must be validated on known signals and reported with their limitations. The review interface is optional inspection tooling, not a request for human reviewers.

Next stage: resolve provenance and passenger sampling; configure the AI reviewers and verified artifact preservation; then freeze the amended protocol and implement/smoke-test the GPU pipeline within the 10-hour ceiling. A usable draft layout is not a submission-ready manuscript. No repository push, external message, submission, or publication was made. Git contains compact summaries only; detailed logs, per-fixture records, audio, caches, and weights stay ignored.
