"""Collect actual JUnit results and additional measured validation records."""
import json,platform,xml.etree.ElementTree as ET
from pathlib import Path
from underwater_reasoning.io import atomic_json,file_hash
ROOT=Path(__file__).resolve().parents[1]
checks={}
for name in ('local','remote'):
    path=ROOT/f'.cache/software_tests_{name}.xml';suite=ET.parse(path).getroot().find('testsuite')
    value={k:int(suite.get(k,'0')) for k in ('tests','errors','failures','skipped')}
    value['passed']=value['tests']-value['errors']-value['failures']-value['skipped']
    value.update(seconds=float(suite.get('time')),junit_sha256=file_hash(path))
    if value['errors'] or value['failures']:raise ValueError('Software validation has failures')
    checks[name]=value
checks['local']['python']=platform.python_version()
for key,path in [('analysis_math','analysis_math_checks.json'),('reference_boundaries','reference_boundary_pilot.json'),
                 ('checkpoint_process_resume','process_resume_test.json')]:
    checks[key]=json.loads((ROOT/'results/manifests'/path).read_text())
atomic_json(ROOT/'results/manifests/software_validation.json',checks)
print({k:v for k,v in checks.items() if k in ('local','remote')})
