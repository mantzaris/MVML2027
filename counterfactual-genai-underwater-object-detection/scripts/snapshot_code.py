#!/usr/bin/env python3
"""Save a checksummed source snapshot within this project before a protocol freeze."""
import argparse,tarfile
from pathlib import Path
from uac.runtime import source_fingerprint,atomic_json,sha256_file
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
root=Path.cwd();target=root/'manifests/source_snapshots';target.mkdir(parents=True,exist_ok=True)
paths=[p for folder in ('src','scripts','configs','protocol') for p in sorted((root/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
paths += [root/'requirements-research.lock',root/'RESEARCH_PLAN.md']
archive=target/(a.name+'.tar.gz')
if archive.exists():raise SystemExit('Refusing to replace an existing source snapshot')
with tarfile.open(archive,'w:gz') as tar:
 for path in paths:tar.add(path,arcname=path.relative_to(root))
atomic_json(target/(a.name+'.json'),{'source_fingerprint':source_fingerprint(paths),'archive_sha256':sha256_file(archive),'files':{str(path.relative_to(root)):sha256_file(path) for path in paths}})
print(archive)
