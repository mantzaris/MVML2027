# Third-party sources

- The supplied planning PDF and execution prompt are user-provided inputs, preserved unchanged.
- The Word template is obtained from [MVML](https://mvml.org/documents/MVML27_Paper_Template.docx) for conference manuscript preparation. It is an official downloaded DOCX, not a recreated official LaTeX class. Template provenance/hash is recorded in the source audit.
- UniqueShip/ONC release terms and unresolved partner provenance are documented in `protocol/data_access_and_licenses.md`. No raw audio or identity metadata is distributed in Git.
- Qwen model metadata reports Apache-2.0 for the proposed Omni and planner checkpoints. Weights were not downloaded or redistributed. Full model configurations/revisions must be verified at the GPU smoke test.
- Python package versions are listed in `requirements.lock`; each dependency retains its own license. No third-party source code was copied into the implementation.
- External article HTML and download probes are retained only in ignored local caches. The repository stores citations, access hashes, concise research notes, and aggregate audit findings.
- Original project code/manuscript licensing has not been selected. See `LICENSE`; do not infer a project-wide license from a model, dataset, dependency, or article license.
