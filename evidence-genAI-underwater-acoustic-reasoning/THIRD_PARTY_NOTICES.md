# Third-party sources

- The supplied planning PDF and execution prompt are user-provided inputs, preserved unchanged.
- The Word template is obtained from [MVML](https://mvml.org/documents/MVML27_Paper_Template.docx) for conference manuscript preparation. It is an official downloaded DOCX, not a recreated official LaTeX class. Template provenance/hash is recorded in the source audit.
- UniqueShip/ONC release terms and unresolved partner provenance are documented in `protocol/data_access_and_licenses.md`. No raw audio or identity metadata is distributed in Git.
- Qwen model metadata reports Apache-2.0. The worker downloaded `Qwen/Qwen2.5-Omni-7B` at revision `ae9e1690543ffd5c0221dc27f79834d0294cba00` and `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28`. Their pretrained weight caches remain on the worker and are not redistributed in Git. Torchvision's Swin V2 Tiny ImageNet weights initialize the acoustic classifier; trained research checkpoints are preserved locally outside Git.
- Python package versions are listed in `requirements.lock`; each dependency retains its own license. No third-party source code was copied into the implementation.
- External article HTML and download probes are retained only in ignored local caches. The repository stores citations, access hashes, concise research notes, and aggregate audit findings.
- Original project code/manuscript licensing has not been selected. See `LICENSE`; do not infer a project-wide license from a model, dataset, dependency, or article license.
