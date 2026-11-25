#!/usr/bin/env python3
"""Build anonymous, editable conference drafts from the measured-study narrative.

Requires XeLaTeX, Tinos/Arimo fonts, and pypdf in the project environment.
Only document rendering is performed; this never fits models or evaluates data.
Compilation logs and page previews remain in ignored work/manuscript/.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/anonymous'
WORK = ROOT / 'work/manuscript/build'
TITLE = 'Observation-Aware Graph Learning for Corrupted Cyber Telemetry'
STYLE = r'''% Anonymous draft style; follows the MVML 2027 template's written instructions.
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{anonymous-draft}[2026/11/25 Anonymous scientific manuscript]
\usepackage[letterpaper,margin=25mm]{geometry}
\usepackage[no-math]{fontspec}
\setmainfont{Tinos}
\setsansfont{Arimo}
\usepackage{amsmath,amssymb}
\usepackage{newtxmath}
\SetSymbolFont{operators}{normal}{OT1}{ntxtlf}{m}{n}
\SetSymbolFont{operators}{bold}{OT1}{ntxtlf}{b}{n}
\SetMathAlphabet{\mathit}{normal}{OT1}{ntxtlf}{m}{it}
\SetMathAlphabet{\mathit}{bold}{OT1}{ntxtlf}{b}{it}
\SetMathAlphabet{\mathbf}{normal}{OT1}{ntxtlf}{b}{n}
\SetMathAlphabet{\mathbf}{bold}{OT1}{ntxtlf}{b}{n}
\usepackage{graphicx,booktabs,array,tabularx,float,placeins}
\usepackage{titlesec,caption,enumitem}
\usepackage[unicode,hidelinks]{hyperref}
\hypersetup{pdfauthor={},pdfcreator={XeLaTeX},pdfsubject={Anonymous research draft}}
\setlength{\parindent}{7mm}
\setlength{\parskip}{0pt}
\setlength{\textfloatsep}{9pt plus 2pt minus 1pt}
\setlength{\floatsep}{9pt plus 2pt minus 1pt}
\setlength{\intextsep}{8pt plus 2pt minus 1pt}
\setlength{\abovedisplayskip}{6pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{6pt plus 1pt minus 1pt}
\titleformat{\section}{\sffamily\bfseries\fontsize{12}{14}\selectfont}{\thesection.}{0.45em}{}
\titleformat{\subsection}{\sffamily\bfseries\fontsize{11}{13}\selectfont}{\thesubsection.}{0.45em}{}
\titlespacing*{\section}{0pt}{9pt plus 2pt minus 1pt}{4pt}
\titlespacing*{\subsection}{0pt}{7pt plus 1pt minus 1pt}{3pt}
\captionsetup{font={small},labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false,skip=4pt}
\renewcommand{\topfraction}{.90}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{.08}
\renewcommand{\floatpagefraction}{.78}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\setlength{\emergencystretch}{1.5em}
\widowpenalty=10000
\clubpenalty=10000
\raggedbottom
\urlstyle{same}
\AtBeginDocument{\fontsize{11}{12.65}\selectfont}
\newcommand{\drafttitle}[1]{\begin{center}{\sffamily\bfseries\fontsize{16}{18}\selectfont #1\par}\vspace{5pt}{\fontsize{10}{12}\selectfont Anonymous draft\par}\end{center}\vspace{2pt}}
'''

def esc(s: str) -> str:
    s=s.replace('\u2011','-').replace('\u2212','-')
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in s)

def inline(s: str) -> str:
    pattern = r'(\\\(.*?\\\)|\*\*.*?\*\*|`[^`]+`|https?://[^\s]+)'
    out=[]
    for part in re.split(pattern,s):
        if part.startswith(r'\('): out.append(part)
        elif part.startswith('**'): out.append(r'\textbf{'+inline(part[2:-2])+'}')
        elif part.startswith('`'): out.append(r'\texttt{'+esc(part[1:-1])+'}')
        elif part.startswith(('https://','http://')):
            url=part.rstrip('.,;')
            out.append(r'\url{'+url+'}'+part[len(url):])
        else: out.append(esc(part))
    return ''.join(out)

def figure(path: str, caption: str, supp: bool) -> str:
    caption=re.sub(r'^Figure S?\d+\.\s*','',caption)
    return '\n'.join([r'\begin{figure}['+('H' if supp or path == 'figures/fig4.pdf' else 'tbp')+']',r'\centering',r'\includegraphics[width=\linewidth]{'+path+'}',r'\caption{'+inline(caption)+'}',r'\end{figure}'])

def table(lines: list[str], caption: str | None, supp: bool) -> str:
    rows=[[x.strip() for x in l.strip('|').split('|')] for l in lines]
    rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',x) for x in r)]
    n=len(rows[0]); body=[]
    if not supp:
        body += [r'\begin{table}[tbp]',r'\centering',r'\caption{'+inline(re.sub(r'^Table \d+\.\s*','',caption or ''))+'}']
    else: body += [r'\begin{center}']
    body += [r'\fontsize{9.5}{11}\selectfont',r'\renewcommand{\arraystretch}{1.12}',r'\setlength{\tabcolsep}{4pt}']
    if n==6: spec='@{}lrrrrr@{}'; env='tabular*'; args=r'{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}'
    elif n==4: spec=r'@{}>{\hsize=1.25\hsize\raggedright\arraybackslash}Xrr>{\hsize=.75\hsize\raggedright\arraybackslash}X@{}'; env='tabularx'; args=r'{\linewidth}{'+spec+'}'
    else: spec=r'@{}>{\hsize=.84\hsize\raggedright\arraybackslash}X>{\hsize=1.16\hsize\raggedright\arraybackslash}X@{}'; env='tabularx'; args=r'{\linewidth}{'+spec+'}'
    body += [r'\begin{'+env+'}'+args,r'\toprule']
    for i,row in enumerate(rows):
        cells=[inline(x) for x in row]
        if i==0: cells=[r'\textbf{'+x+'}' for x in cells]
        body.append(' & '.join(cells)+r' \\')
        if i==0: body.append(r'\midrule')
    body += [r'\bottomrule',r'\end{'+env+'}',r'\end{center}' if supp else r'\end{table}']
    return '\n'.join(body)

def math_block(block: str) -> str:
    if r'\tag{2}' in block:
        block=block.replace(r'\bar A_{ij}=',r'\begin{aligned}\bar A_{ij}&=')
        block=block.replace('\\qquad\nP_{ij}=',r'\\[3pt]'+'\nP_{ij}&=')
        block=block.replace(r'\tag{2}',r'\end{aligned}'+'\n'+r'\tag{2}')
    elif r'\tag{3}' in block:
        block=block.replace('C=',r'\begin{aligned}C&=')
        block=block.replace('\\qquad\nL_{\\mathrm{pre}}=',r'\\[3pt]'+'\nL_{\\mathrm{pre}}&=')
        block=block.replace(r'\tag{3}',r'\end{aligned}'+'\n'+r'\tag{3}')
    return block

def convert(name: str) -> str:
    supp=name=='supplement'; lines=(OUT/(name+'.md')).read_text().splitlines(); out=[]; i=0; refs=False
    out += [r'\documentclass[11pt,letterpaper]{article}',r'\usepackage{anonymous-draft}',r'\hypersetup{pdftitle={'+TITLE+(' -- Supplementary Material' if supp else '')+'}}']
    if supp: out += [r'\renewcommand{\thefigure}{S\arabic{figure}}']
    out += [r'\begin{document}']
    while i<len(lines):
        l=lines[i].strip()
        if not l: i+=1; continue
        if l==r'\newpage': out.append(l); i+=1; continue
        if l.startswith('# '):
            title=l[2:]
            if supp: out.append(r'\drafttitle{Supplementary Material}')
            else: out.append(r'\drafttitle{'+inline(title)+'}')
            i+=1; continue
        if supp and l.startswith('## '):
            out.append(r'\begin{center}{\sffamily\bfseries\fontsize{12}{14}\selectfont '+inline(l[3:])+r'\par}\end{center}')
            i+=1;continue
        if l=='## Abstract':
            i+=1; chunks=[]
            while i<len(lines) and not lines[i].startswith('**Keywords:'):
                if lines[i].strip():chunks.append(lines[i].strip())
                i+=1
            out.append(r'\begingroup\fontsize{10}{11.5}\selectfont\setlength{\parindent}{0pt}\textbf{Abstract.} '+inline(' '.join(chunks))+r'\par\endgroup\vspace{5pt}')
            continue
        if l.startswith('## ') or l.startswith('### '):
            sub=l.startswith('### '); title=l[4:] if sub else l[3:]; title=re.sub(r'^\d+(?:\.\d+)*\.\s*','',title)
            if not supp and title in ('Resource costs and verification','Limitations and Scope'):
                out.append(r'\FloatBarrier')
            if title=='References':
                refs=True; out.append(r'\section*{References}\begin{enumerate}[label={[\arabic*]},leftmargin=6mm,labelsep=1.5mm,itemsep=3pt,parsep=0pt,topsep=0pt]')
            elif supp: out.append(r'\section*{'+inline(title)+'}')
            else: out.append(('\\subsection{' if sub else '\\section{')+inline(title)+'}')
            i+=1;continue
        if l==r'\[':
            block=[l];i+=1
            while i<len(lines):
                block.append(lines[i]);i+=1
                if block[-1].strip()==r'\]':break
            out.append(math_block('\n'.join(block)));continue
        if l.startswith('[FIGURE'):
            no=re.search(r'\d+',l).group();i+=1
            while not lines[i].strip():i+=1
            out.append(figure(f'figures/fig{no}.pdf',lines[i].strip(),False));i+=1;continue
        if l.startswith('!['):
            m=re.fullmatch(r'!\[(.*)\]\((.*)\)',l);assert m,l
            out.append(figure(m[2],m[1],supp));i+=1;continue
        if re.match(r'^Table \d+\.', l) or l.startswith('|'):
            caption=None
            if re.match(r'^Table \d+\.', l):
                caption=l;i+=1
                while not lines[i].strip():i+=1
            tbl=[]
            while i<len(lines) and lines[i].startswith('|'):tbl.append(lines[i]);i+=1
            assert tbl;out.append(table(tbl,caption,supp));continue
        para=[l];i+=1
        while i<len(lines) and lines[i].strip():
            para.append(lines[i].strip());i+=1
        text=' '.join(para)
        if refs: text=re.sub(r'^\[\d+\]\s*','',text);out.append(r'\item '+inline(text))
        elif text.startswith('**Keywords:'): out.append(r'\noindent '+inline(text)+r'\par')
        else: out.append(inline(text)+r'\par')
    if refs:out.append(r'\end{enumerate}')
    out.append(r'\end{document}')
    return '\n\n'.join(out)+'\n'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def check_pdf(path: Path, maximum: int):
    reader=PdfReader(path);text='\n'.join(p.extract_text() for p in reader.pages)
    assert len(reader.pages)<=maximum,(path,len(reader.pages))
    assert reader.metadata.get('/Author','')=='',reader.metadata
    links=[]
    for page in reader.pages:
        box=page.mediabox
        assert abs(float(box.width)-612)<.1 and abs(float(box.height)-792)<.1
        for ann in page.get('/Annots',[]):
            action=ann.get_object().get('/A',{})
            if action.get('/URI'):links.append(action['/URI'])
    expected = ['S1','S2','S3','S4'] if path.stem == 'supplement' else ['1','2','3','4']
    captions = re.findall(r'(?m)^Figure\s+(S?\d+)\.\s',text)
    assert captions == expected,(path,'figure captions',captions)
    content=(text+'\n'+str(reader.metadata)+'\n'+'\n'.join(links)).lower()
    for token in ('mantzaris','/home/','meow@','github.com/mantzaris','54437e6','[figure','affiliation'):
        assert token not in content,(path,token)
    return {'pages':len(reader.pages),'figures':captions,'page_size_points':[612,792],'metadata':dict(reader.metadata),'outbound_links':links,'sha256':digest(path)}

def main():
    WORK.mkdir(parents=True,exist_ok=True)
    (OUT/'anonymous-draft.sty').write_text(STYLE)
    results={}
    for name,limit in [('main',8),('supplement',4)]:
        (OUT/(name+'.tex')).write_text(convert(name))
        cmd=['xelatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape',f'-output-directory={WORK}',name+'.tex']
        for run in (1,2):
            proc=subprocess.run(cmd,cwd=OUT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (WORK/f'{name}.pass{run}.txt').write_text(proc.stdout)
            if proc.returncode:raise RuntimeError(proc.stdout[-7000:])
        shutil.copyfile(WORK/(name+'.pdf'),OUT/(name+'.pdf'))
        # Inspect the output immediately; page limits are reported but enforced below.
        pages=len(PdfReader(OUT/(name+'.pdf')).pages)
        print(f'{name}: {pages} pages; max {limit}',flush=True)
        results[name]=check_pdf(OUT/(name+'.pdf'),limit)
    archive=ROOT/'manuscript/anonymous-source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                info=zipfile.ZipInfo(str(path.relative_to(OUT)),(2026,11,25,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
                z.writestr(info,path.read_bytes())
    manifest={'description':'Internal provenance; excluded from anonymous archive. No new model experiments.',
              'pdf_checks':results,'sources':{str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
              'archive':{'path':str(archive.relative_to(ROOT)),'sha256':digest(archive)},
              'builder_sha256':digest(Path(__file__)),'figure_provenance_sha256':digest(ROOT/'manuscript/figure_provenance.json')}
    (ROOT/'manuscript/build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Anonymous source archive and internal build manifest written.')

if __name__=='__main__':main()
