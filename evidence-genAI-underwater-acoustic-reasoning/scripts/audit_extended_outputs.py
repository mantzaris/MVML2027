"""Reparse completed stress and intervention reports without new inference."""
import json
from collections import Counter
from pathlib import Path
from underwater_reasoning.io import atomic_json, canonical_hash, file_hash
from underwater_reasoning.report_schema import terminal_schema
from underwater_reasoning.reports import parse_report

ROOT = Path(__file__).resolve().parents[1]
audits = json.loads((ROOT / 'results/tables/report_audits.json').read_text())
stress = json.loads((ROOT / 'results/raw/scored_robustness_records.json').read_text())
expected_stress = {
    (row['id'], row['mask'])
    for methods in stress.values()
    for method, rows in methods.items() if method != 'B8_deterministic'
    for row in rows
}
groups = {
    'waveform_stress': ('terminal/robustness', len(expected_stress)),
    'paraphrase': ('report_audits/paraphrase', audits['paraphrase']['n']),
    'paraphrase_2': ('report_audits/paraphrase_2', audits['paraphrase_2']['n']),
    'evidence_interventions': ('report_audits/intervention', sum(v['n'] for v in audits['evidence_interventions'].values())),
}
schema_hash = canonical_hash(terminal_schema())
summary = {}
for name, (relative, expected) in groups.items():
    paths = sorted((ROOT / 'results/raw' / relative).glob('*.json'))
    assert len(paths) == expected, (name, len(paths), expected)
    counts = Counter(); errors = Counter(); digests = []
    if name == 'waveform_stress':
        assert {p.stem for p in paths} == {f'{uid}_{mask}' for uid, mask in expected_stress}
    for path in paths:
        record = json.loads(path.read_text()); result = record['result']; key = record['key']
        asset = json.loads((ROOT / 'data/derived/evidence' / f"{record['id']}.json").read_text())
        assert key['waveform_sha256'] == asset['waveform_sha256']
        assert key['format_schema_sha256'] == schema_hash
        assert key['format_enforcer'] == 'xgrammar==0.2.9'
        parsed, found = parse_report(result['raw'])
        assert parsed == result['parsed'] and found == result['errors'], path
        counts['reports'] += 1
        counts['schema_valid'] += not [e for e in found if e != 'interpretation_word_count']
        counts['format_valid'] += not found
        counts['output_limit_hits'] += bool(result.get('hit_output_limit'))
        errors.update(found); digests.append([str(path.relative_to(ROOT)), file_hash(path)])
    summary[name] = {**dict(counts), 'errors': dict(errors), 'cache_digest': canonical_hash(digests)}
atomic_json(ROOT / 'results/manifests/extended_cache_audit.json', {
    'groups': summary,
    'schema_hash': schema_hash,
    'saved_parse_matches_current_parser': True,
    'failed_reports_retained': True,
    'scope': 'Post-generation format and cache integrity audit; no semantic scores or selection rules changed.',
})
print(json.dumps(summary, indent=2))
