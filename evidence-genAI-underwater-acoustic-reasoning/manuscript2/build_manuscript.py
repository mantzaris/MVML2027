"""Rebuild manuscript2 from its retained DOCX and editable prose.json.

Requires Python 3.10+ and lxml. No experiment, model or network calls.
Preserves table values and figure image data, then applies MVML layout rules.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from lxml import etree as ET
from submission_format import apply as apply_submission_format

ROOT = Path(__file__).resolve().parent
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def paragraph_text(element):
    return ''.join(element.xpath('.//w:t/text() | .//m:t/text()', namespaces={**NS, 'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}))


def set_text(paragraph, text):
    """Patch a plain paragraph while retaining its paragraph and run formatting."""
    if paragraph.xpath('.//w:drawing | .//w:fldSimple | .//w:hyperlink', namespaces=NS):
        raise ValueError('Refusing a plain-text replacement of a complex paragraph')
    runs = paragraph.findall('w:r', NS)
    if not runs:
        raise ValueError('Expected a styled text run')
    run = deepcopy(runs[0])
    for child in list(run):
        if child.tag != W + 'rPr':
            run.remove(child)
    node = ET.SubElement(run, W + 't')
    node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    node.text = text
    for child in list(paragraph):
        if child.tag != W + 'pPr':
            paragraph.remove(child)
    paragraph.append(run)


def main():
    source = json.loads((ROOT / 'prose.json').read_text())
    template = ROOT / 'template' / 'manuscript_base.docx'
    if digest(template.read_bytes()) != source['template_sha256']:
        raise ValueError('Retained template changed')
    with zipfile.ZipFile(template) as archive:
        infos = archive.infolist()
        parts = {info.filename: archive.read(info.filename) for info in infos}
    document = ET.fromstring(parts['word/document.xml'])
    body = document.find('w:body', NS)
    paragraphs = body.findall('w:p', NS)
    heading = deepcopy(paragraphs[4])
    original_structures = {
        name: [ET.tostring(el) for el in document.findall(path, NS)]
        for name, path in [('tables', './/w:tbl'), ('drawings', './/w:drawing'), ('sections', './/w:sectPr')]
    }
    for block in source['replacements']:
        paragraph = paragraphs[block['slot']]
        if digest(paragraph_text(paragraph).encode()) != block['original_text_sha256']:
            raise ValueError(f"Original slot {block['slot']} no longer matches")
        position = body.index(paragraph)
        pattern = heading if block.get('kind') == 'heading' else paragraph
        for offset, text in enumerate(block['paragraphs']):
            replacement = deepcopy(pattern)
            set_text(replacement, text)
            body.insert(position + offset, replacement)
        body.remove(paragraph)
    for name, path in [('tables', './/w:tbl'), ('drawings', './/w:drawing'), ('sections', './/w:sectPr')]:
        if original_structures[name] != [ET.tostring(el) for el in document.findall(path, NS)]:
            raise ValueError('Unexpected change to ' + name)
    apply_submission_format(document, parts, json.loads((ROOT / 'submission.json').read_text()))
    parts['word/document.xml'] = ET.tostring(document, xml_declaration=True, encoding='UTF-8', standalone=True)
    settings = ET.fromstring(parts['word/settings.xml'])
    update = settings.find('w:updateFields', NS)
    if update is None:
        update = ET.SubElement(settings, W + 'updateFields')
    update.set(W + 'val', 'true')
    parts['word/settings.xml'] = ET.tostring(settings, xml_declaration=True, encoding='UTF-8', standalone=True)
    core = ET.fromstring(parts['docProps/core.xml'])
    modified = core.find('{http://purl.org/dc/terms/}modified')
    if modified is not None:
        modified.text = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    parts['docProps/core.xml'] = ET.tostring(core, xml_declaration=True, encoding='UTF-8', standalone=True)
    with zipfile.ZipFile(ROOT / 'manuscript.docx', 'w') as archive:
        for info in infos:
            archive.writestr(info, parts[info.filename])
    text = []
    for element in body:
        if element.tag == W + 'p':
            content = paragraph_text(element)
            if content:
                text.append(content)
        elif element.tag == W + 'tbl':
            rows = []
            for row in element.findall('w:tr', NS):
                rows.append(' | '.join(' '.join(paragraph_text(p) for p in cell.findall('w:p', NS))
                                       for cell in row.findall('w:tc', NS)))
            text.append('\n'.join(rows))
    (ROOT / 'manuscript.txt').write_text('\n\n'.join(text) + '\n')
    with zipfile.ZipFile(template) as baseline:
        changed = [info.filename for info in infos if baseline.read(info.filename) != parts[info.filename]]
    allowed = {'word/document.xml', 'word/_rels/document.xml.rels', 'word/settings.xml', 'docProps/core.xml'} | {n for n in parts if n.startswith(('word/header', 'word/footer')) and n.endswith('.xml')}
    if set(changed) - allowed:
        raise ValueError('Unexpected package change: ' + repr(changed))
    print(json.dumps({'words': len(' '.join(text).split()), 'replaced_slots': len(source['replacements']),
                      'changed_package_parts': changed, 'table_values_and_figure_images_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()
