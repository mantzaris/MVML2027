# Research data

Place split manifests and generator specifications here. Keep large generated traces in `generated/` and external public inputs in `external/`; both are ignored by Git.

Every dataset manifest must include generator commit, settings, topology/episode seeds, split hashes, observation schema, action schema, deception mechanisms, and label-access rules. Hidden truth and corruption labels must be inaccessible to inference code. Retain paired scenario IDs across methods.
