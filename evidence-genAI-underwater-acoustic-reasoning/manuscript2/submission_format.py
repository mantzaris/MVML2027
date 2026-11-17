"""Apply the published MVML layout rules to the manuscript's OOXML package."""
from copy import deepcopy
import re
from lxml import etree as E

NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
}
W, M = '{'+NS['w']+'}', '{'+NS['m']+'}'
MARGIN = 1417  # 25 mm, rounded to the nearest twip
WIDTH = 12240 - 2 * MARGIN


def el(tag, **attrs):
    prefix, local = tag.split(':')
    node = E.Element('{'+NS[prefix]+'}'+local)
    for key, value in attrs.items():
        node.set('{'+NS[prefix]+'}'+key, str(value))
    return node


def child(parent, tag, **attrs):
    n = parent.find(tag, NS)
    if n is None:
        n = el(tag)
        parent.append(n)
    prefix = tag.split(':')[0]
    for k,v in attrs.items():
        n.set('{'+NS[prefix]+'}'+k, str(v))
    return n


def ptext(p):
    return ''.join(p.xpath('.//w:t/text() | .//m:t/text()', namespaces=NS))


def run(text, font='Times New Roman', size=11, bold=False, italic=False, superscript=False):
    r=el('w:r'); pr=E.SubElement(r,W+'rPr')
    child(pr,'w:rFonts',ascii=font,hAnsi=font,cs=font)
    child(pr,'w:sz',val=int(size*2)); child(pr,'w:szCs',val=int(size*2))
    if bold: child(pr,'w:b')
    if italic: child(pr,'w:i')
    if superscript: child(pr,'w:vertAlign',val='superscript')
    t=E.SubElement(r,W+'t'); t.text=text
    t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    return r


def para(text='', **kwargs):
    p=el('w:p'); p.append(run(text,**kwargs)); return p


def properties(p, align='both', indent=397, before=0, after=0, keep=False):
    pr=p.find('w:pPr',NS)
    if pr is None:
        pr=el('w:pPr');p.insert(0,pr)
    for name in ['contextualSpacing','mirrorIndents','tabs','pageBreakBefore']:
        for n in pr.findall('w:'+name,NS): pr.remove(n)
    child(pr,'w:ind',left=0,right=0,firstLine=indent)
    child(pr,'w:jc',val=align)
    child(pr,'w:spacing',before=before,after=after,line=240,lineRule='auto')
    child(pr,'w:widowControl',val='1')
    child(pr,'w:keepNext',val='1' if keep else '0')
    child(pr,'w:keepLines',val='0')
    return pr


def restyle(p, font='Times New Roman',size=11,bold=False):
    for r in p.findall('w:r',NS):
        if r.find('w:drawing',NS) is not None: continue
        pr=child(r,'w:rPr')
        child(pr,'w:rFonts',ascii=font,hAnsi=font,cs=font)
        child(pr,'w:sz',val=int(size*2));child(pr,'w:szCs',val=int(size*2))
        child(pr,'w:b',val='1' if bold else '0')
        child(pr,'w:color',val='000000')


def math_run(text, plain=False):
    r=el('m:r')
    if plain:
        pr=el('m:rPr');pr.append(el('m:sty',val='p'));r.append(pr)
    pr=el('w:rPr');child(pr,'w:rFonts',ascii='Cambria Math',hAnsi='Cambria Math')
    child(pr,'w:sz',val=22);r.append(pr)
    t=E.SubElement(r,M+'t');t.text=text
    return r


def script(base, value, sup=False):
    node=el('m:sSup' if sup else 'm:sSub')
    b=E.SubElement(node,M+'e');b.append(math_run(base))
    s=E.SubElement(node,M+('sup' if sup else 'sub'));s.append(math_run(value))
    return node


def fraction(top, bottom):
    f=el('m:f');n=E.SubElement(f,M+'num');d=E.SubElement(f,M+'den')
    for x in top:n.append(x)
    for x in bottom:d.append(x)
    return f


def equation(kind):
    p=el('w:p');pr=properties(p,align='left',indent=0,before=220,after=220)
    tabs=child(pr,'w:tabs')
    tabs.append(el('w:tab',val='center',pos=WIDTH//2));tabs.append(el('w:tab',val='right',pos=WIDTH))
    r=el('w:r');r.append(el('w:tab'));p.append(r)
    math=el('m:oMath')
    if kind=='weights':
        math.extend([script('w','i'),math_run(' ∝ '),fraction([math_run('1')],[script('G','c'),math_run(' '),script('n','g')]),math_run(',  i ∈ g')])
        num=1
    else:
        math.append(math_run('s = '));math.append(math_run('max',True));math.append(math_run('(p) − '))
        top=[math_run('γ['),math_run('JSD',True),math_run('(p,'),script('p','M',True),math_run(') + '),math_run('JSD',True),math_run('(p,'),script('p','R',True),math_run(')]')]
        math.append(fraction(top,[math_run('2')]))
        num=2
    p.append(math)
    r=el('w:r');r.append(el('w:tab'));p.append(r);p.append(run(f'({num})'))
    return p


def apply(document, parts, config):
    body=document.find('w:body',NS)
    section=body.find('w:sectPr',NS)
    child(section,'w:pgMar',top=MARGIN,bottom=MARGIN,left=MARGIN,right=MARGIN,header=567,footer=567,gutter=0)
    child(section,'w:pgSz',w=12240,h=15840)
    # Remove the draft header. The initial submission has no assigned paper code.
    for n in list(section.findall('w:headerReference',NS)):
        if n.get(W+'type')!='first': section.remove(n)
    for name in list(parts):
        if re.fullmatch(r'word/header\d+\.xml',name):
            header=E.fromstring(parts[name])
            for node in list(header):header.remove(node)
            if name=='word/header3.xml':
                for t in ['Proceedings of the 13th World Congress on Electrical Engineering and Computer Systems and Sciences (EECSS 2027)',
                          'August 15–17, 2027 – Prague, Czech Republic']:
                    p=para(t,size=9);properties(p,align='left',indent=0);header.append(p)
                if config.get('paper_code'):
                    p=para('Paper No. '+config['paper_code'],size=9);properties(p,align='left',indent=0);header.append(p)
            parts[name]=E.tostring(header,xml_declaration=True,encoding='UTF-8',standalone=True)
        if re.fullmatch(r'word/footer\d+\.xml',name):
            footer=E.fromstring(parts[name])
            for node in list(footer):footer.remove(node)
            p=el('w:p');properties(p,align='center',indent=0)
            if config.get('paper_code'): p.append(run(config['paper_code']+'-',size=10))
            field=el('w:fldSimple',instr=' PAGE ');field.append(run('1',size=10));p.append(field)
            footer.append(p);parts[name]=E.tostring(footer,xml_declaration=True,encoding='UTF-8',standalone=True)
    # Authors and affiliations are supplied explicitly in submission.json.
    first=body.findall('w:p',NS)
    author=first[1];pos=body.index(author);body.remove(author)
    author_lines=[]
    p=el('w:p')
    for i,a in enumerate(config['authors']):
        if i:p.append(run(', ',size=12,bold=True))
        p.append(run(a['name'],size=12,bold=True));p.append(run(str(a['affiliation']),size=12,bold=True,superscript=True))
    if not config['authors']:p.append(run('Author details pending',size=12,bold=True))
    author_lines.append(p)
    for i,a in enumerate(config['affiliations'],1):
        p=el('w:p');p.append(run(str(i),superscript=True));p.append(run(a['name']));author_lines.append(p)
        if a.get('address'):author_lines.append(para(a['address']))
    if config['authors']:
        author_lines.append(para(', '.join(a['email'] for a in config['authors'])))
    for i,p in enumerate(author_lines):
        properties(p,align='center',indent=0,keep=True)
        body.insert(pos+i,p)
    # Format paragraphs without changing their wording.
    for p in list(body.findall('w:p',NS)):
        text=ptext(p)
        if p in author_lines:continue
        if text.startswith('[[equation:'):
            kind=text.removeprefix('[[equation:').removesuffix(']]')
            pos=body.index(p);body.remove(p);body.insert(pos,equation(kind));continue
        if p.find('.//w:drawing',NS) is not None:
            properties(p,align='center',indent=0,keep=True)
            for extent in p.findall('.//wp:extent',NS):
                oldx,oldy=int(extent.get('cx')),int(extent.get('cy'))
                maxx=int(WIDTH/20*12700)
                ratio=min(1,maxx/oldx)
                extent.set('cx',str(round(oldx*ratio)));extent.set('cy',str(round(oldy*ratio)))
                for x in p.findall('.//a:xfrm/a:ext',NS):
                    x.set('cx',extent.get('cx'));x.set('cy',extent.get('cy'))
            continue
        if text.startswith('Evidence-Grounded Generative AI'):
            properties(p,align='center',indent=0,after=220,keep=True);restyle(p,'Arial',16,True)
        elif re.match(r'^\d+\. ',text) or text in ['References','Code Availability']:
            properties(p,align='left',indent=0,before=220,keep=True);restyle(p,'Arial',12,True)
        elif text.startswith('Abstract'):
            properties(p,indent=0,before=440,after=440)
            content=text.split('—',1)[1].strip()
            for r in list(p):
                if r.tag!=W+'pPr':p.remove(r)
            p.append(run('Abstract',font='Arial',size=10,bold=True,italic=True));p.append(run(' — '+content,size=10))
        elif text.startswith('Keywords:'):
            properties(p,indent=0)
            for r in list(p):
                if r.tag!=W+'pPr':p.remove(r)
            p.append(run('Keywords: ',font='Arial',size=10,bold=True));p.append(run(text.split(':',1)[1].strip()))
        elif text.startswith('Table ') or text.startswith('Fig. '):
            properties(p,align='center',indent=0,keep=text.startswith('Table '));restyle(p,size=10)
        elif text.startswith('['):
            properties(p,align='left',indent=0);restyle(p,size=11)
        else:
            properties(p);restyle(p,size=11)
            if text.startswith('Our approach combines learned sound patterns'):
                child(p.find('w:pPr',NS),'w:keepLines',val='1')
    # Use actual subscripts in the prose explanation of Equation (1).
    for p in body.findall('w:p', NS):
        for r in list(p.findall('w:r', NS)):
            t=r.find('w:t', NS)
            if t is None or not re.search(r'G_c|n_g', t.text or ''):continue
            pos=p.index(r);p.remove(r)
            for piece in re.split(r'(G_c|n_g)',t.text):
                if piece in ['G_c','n_g']:
                    base,sub=piece.split('_')
                    a=run(base,italic=True);b=run(sub,italic=True)
                    child(b.find('w:rPr',NS),'w:vertAlign',val='subscript')
                    p.insert(pos,a);p.insert(pos+1,b);pos+=2
                elif piece:
                    p.insert(pos,run(piece));pos+=1
    # Add a real external hyperlink, rather than relying on viewer autolinking.
    relpath='word/_rels/document.xml.rels'
    rels=E.fromstring(parts[relpath])
    relns='http://schemas.openxmlformats.org/package/2006/relationships'
    rid='rIdRepository'
    if any(n.get('Id')==rid for n in rels):raise ValueError('Repository relationship already exists')
    E.SubElement(rels,'{'+relns+'}Relationship',Id=rid,
                 Type=NS['r']+'/hyperlink',Target=config['repository'],TargetMode='External')
    linked=0
    for p in body.findall('w:p',NS):
        for r in list(p.findall('w:r',NS)):
            t=r.find('w:t',NS)
            if t is None or config['repository'] not in (t.text or ''):continue
            before,after=t.text.split(config['repository'],1)
            pos=p.index(r);p.remove(r)
            if before:p.insert(pos,run(before));pos+=1
            link=el('w:hyperlink');link.set('{'+NS['r']+'}id',rid)
            link.append(run(config['repository']));p.insert(pos,link);pos+=1
            if after:p.insert(pos,run(after))
            linked+=1
    if linked!=1:raise ValueError('Expected exactly one repository link')
    parts[relpath]=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
    # Fit tables to the usable page width; retain every cell value and border.
    for table in body.findall('w:tbl',NS):
        grid=table.find('w:tblGrid',NS)
        oldwidths=[int(c.get(W+'w')) for c in grid]
        widths=[round(WIDTH*w/sum(oldwidths)) for w in oldwidths];widths[-1]+=WIDTH-sum(widths)
        for c,w in zip(grid,widths):c.set(W+'w',str(w))
        pr=child(table,'w:tblPr');child(pr,'w:tblW',type='dxa',w=WIDTH);child(pr,'w:jc',val='center')
        child(pr,'w:tblLayout',type='fixed');child(pr,'w:tblInd',type='dxa',w=0)
        for ri,row in enumerate(table.findall('w:tr',NS)):
            rpr=child(row,'w:trPr');child(rpr,'w:cantSplit')
            if ri==0:child(rpr,'w:tblHeader')
            for cell,w in zip(row.findall('w:tc',NS),widths):
                child(child(cell,'w:tcPr'),'w:tcW',type='dxa',w=w)
                for p in cell.findall('w:p',NS):
                    existing=p.find('w:pPr/w:jc',NS)
                    alignment=existing.get(W+'val') if existing is not None else 'left'
                    properties(p,align=alignment,indent=0);restyle(p,size=11,bold=ri==0)
    core=E.fromstring(parts['docProps/core.xml'])
    creator=core.find('{http://purl.org/dc/elements/1.1/}creator')
    if creator is not None:creator.text=', '.join(a['name'] for a in config['authors'])
    parts['docProps/core.xml']=E.tostring(core,xml_declaration=True,encoding='UTF-8',standalone=True)
