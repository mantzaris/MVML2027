#!/usr/bin/env python3
"""Create the three-page external CTU-13 supplementary record."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

from make_real_results import ROOT, DATA, SUPP, NEW, METHODS, SEEDS, SCENARIOS, load, subset, estimate, sha


def main():
    manifest_path=DATA/'manifest.json';comparison_path=DATA/'comparisons.json'
    manifest=json.loads(manifest_path.read_text());comparison=json.loads(comparison_path.read_text())
    dataset_path=DATA/'ctu13_data_manifest.json';dataset=json.loads(dataset_path.read_text())
    totals=dataset['totals'];retained=sum(r['rows_retained'] for r in dataset['files']);original=sum(r['rows_total'] for r in dataset['files'])
    if sha(dataset_path)!=manifest['data_manifest_sha256'] or not dataset['temporal_split_check']['passed']:raise ValueError('Changed data manifest or failed temporal audit')
    if comparison['manifest_sha256']!=sha(manifest_path):raise ValueError('Stale comparison manifest')
    rows=load('summary.csv');clean=subset(rows,condition='clean');primary=comparison['primary']
    for stem in ['real_figS1_training','real_figS2_seed_operating_points']:
        record=json.loads((SUPP/f'{stem}.provenance.json').read_text())
        if sha(ROOT/record['script'])!=record['script_sha256']:raise ValueError('Stale figure script')
        for relative,digest in record['inputs'].items():
            if sha(ROOT/relative)!=digest:raise ValueError('Stale figure input '+relative)
        if sha(SUPP/f'{stem}.png')!=record['output_sha256']['png']:raise ValueError('Changed supplemental figure')
    font_dir=Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('RealSans',str(font_dir/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('RealSansBold',str(font_dir/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('RealSans',normal='RealSans',bold='RealSansBold',italic='RealSans',boldItalic='RealSansBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('TitleReal',fontName='RealSansBold',fontSize=16,leading=20,textColor=colors.HexColor('#193549'),spaceAfter=8))
    styles.add(ParagraphStyle('SubReal',fontName='RealSans',fontSize=9,leading=12,textColor=colors.HexColor('#506b80'),spaceAfter=10))
    styles.add(ParagraphStyle('BodyReal',fontName='RealSans',fontSize=8.6,leading=11.6,spaceAfter=7))
    styles.add(ParagraphStyle('HeadReal',fontName='RealSansBold',fontSize=10.5,leading=13,textColor=colors.HexColor('#226581'),spaceBefore=7,spaceAfter=5))
    styles.add(ParagraphStyle('CaptionReal',fontName='RealSans',fontSize=7.7,leading=10.3,textColor=colors.HexColor('#486379'),spaceAfter=7))
    styles.add(ParagraphStyle('SmallReal',fontName='RealSans',fontSize=7.3,leading=9.7,spaceAfter=6))
    story=[]
    def text(value,style='BodyReal'):story.append(Paragraph(value,styles[style]))
    def head(value):text(value,'HeadReal')
    def picture(stem):
        from PIL import Image as PILImage
        path=SUPP/f'{stem}.png'
        with PILImage.open(path) as img:width,height=img.size
        story.append(Image(str(path),width=499,height=499*height/width));story.append(Spacer(1,6))
    def table(data,widths):
        obj=Table(data,colWidths=widths,repeatRows=1)
        obj.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'RealSans'),('FONTNAME',(0,0),(-1,0),'RealSansBold'),
            ('FONTSIZE',(0,0),(-1,-1),7.7),('LEADING',(0,0),(-1,-1),10),
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff3')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f9fa')]),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#acc7d4'))]))
        story.append(obj);story.append(Spacer(1,6))

    text('External application: captured CTU-13 traffic','TitleReal')
    text('Supplementary evidence for the regeneration research project','SubReal')
    text('This extension replaces generated observations with publicly released network-flow measurements. CTU-13 combines normal university traffic with controlled executions of real botnets [1]. The task is binary flow classification; the dataset provides neither host-readiness truth nor actual course-of-action success outcomes.')
    head('S1  Data selection and temporal separation')
    text(f'The verified subset contains {retained:,} of {original:,} original flows ({100*retained/original:.2f}%): training {totals["train"]["rows_retained"]:,}, validation {totals["validation"]["rows_retained"]:,}, test {totals["test"]["rows_retained"]:,}. All retained training flows completed before validation begins, and all validation flows completed before test begins, using provider-local timestamps and recorded durations.','CaptionReal')
    text('Only author-labeled From-Normal and From-Botnet flows are retained. Uncertain background and other directions are excluded, not assumed benign. Raw addresses, absolute timestamps, capture IDs and label strings are excluded from model features. Addresses supply only observable endpoint-sharing graph edges. Hosts may recur across recordings, so the split is not host-disjoint.')
    table([['Partition','Capture IDs','Recorded capture period'],['Training','1-8 and 13','August 10-August 17 morning'],['Validation','9','August 17 midday/afternoon'],['Test','10, 11, 12','August 18-19, 2011']],[95,140,264])
    text('Scenario IDs are not chronological: capture 13 belongs in the training period. Test captures 10/11 contain Rbot; capture 12 contains NSIS.ay, absent from the training captures. Windows and graphs are constructed after verified-label filtering, not from the live unfiltered stream. The selected subset does not represent unknown background traffic. Hashes and per-capture exclusion counts are recorded separately.','CaptionReal')
    head('S2  Shared-context adaptation')
    text('For a fixed chronological window W, encode each observed flow x<sub>i</sub> as h<sub>i</sub>. Pool means and standard deviations over valid flow embeddings into g(W). The adapted mixture predicts attack probability:')
    equation=Table([[Paragraph('p<sub>i</sub> = Σ<sub>k=1..4</sub> softmax<sub>k</sub>(a(g(W))) · sigmoid(ρ<sub>k</sub>([h<sub>i</sub>, g(W)]))',styles['BodyReal'])]],colWidths=[499])
    equation.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eaf2f6')),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(equation);story.append(Spacer(1,7))
    text('SCR_FLOW retains shared latent gating and set context, while changing the output to flow labels. It is retrained, not zero-shot reuse of synthetic checkpoints. Fixed windows contain 32 consecutive retained flows; context is independent of training minibatch composition. Predictions require completed flow measurements in the window; they are retrospective, not packet-causal. No prerequisite implications are invented for flow labels.','CaptionReal')
    head('S3  Clean capture-macro results')
    data=[['Method','Attack AP','ROC-AUC','Brier','Recall','FPR']]
    for m in METHODS:
        chosen=subset(clean,method=m)
        data.append([m]+[f'{estimate(chosen,k)[0]:.4f}' for k in ['ap','auc','brier','recall','fpr']])
    table(data,[99,80,80,80,80,80])
    text('Table S1. Equal-weight means over three held-out captures and five training seeds. AP is attack-class average precision. Recall/FPR use thresholds selected on clean validation data and frozen across test conditions. Higher AP/AUC/recall and lower Brier/FPR are favorable, with operating-point tradeoffs.','CaptionReal')
    clean_ap={m:estimate(subset(clean,method=m),'ap')[0] for m in METHODS}
    text(f'DENSE has the largest descriptive clean macro AP ({clean_ap["DENSE"]:.4f}). The matched single-expert comparison favors context numerically: CTX_FLOW {clean_ap["CTX_FLOW"]:.4f} versus LOCAL_FLOW {clean_ap["LOCAL_FLOW"]:.4f}. The inexpensive NB reference reaches {clean_ap["NB"]:.4f}. These are descriptive comparisons, not additional superiority tests.','SmallReal')
    story.append(PageBreak())

    text('Training, adaptation and artificial stress','TitleReal')
    picture('real_figS1_training')
    text('Figure S1. Recorded supervised training and validation objectives. Epoch means use available seeds; stopped models do not contribute later epochs. Checkpoint metadata records actual pretraining, stopping and selection. Curves alone do not establish convergence.','CaptionReal')
    head('S4  Common inputs and controls')
    text('Fourteen observable features summarize duration, packets, bytes, source-byte fraction, packet size, ports, protocol and connection state; fourteen indicators record feature availability. Preprocessing is fitted on training captures only. No flow identity or hidden evaluation label enters prediction. Endpoint-sharing edges represent observed relationships, not logical implication or host readiness.')
    text('LOGIT is a linear classifier. DAE uses a small bottleneck and observed-feature denoising pretraining, followed by a supervised attack readout. DENSE and GMAE share the graph classifier while differing in masked pretraining. HDC augments the masked graph classifier with fixed hyperdimensional associative retrieval. SCR_FLOW mixes four heads through shared window context; CTX_FLOW uses one head with context, and LOCAL_FLOW zeros the global-context inputs. Gaussian naive Bayes supplies a conventional reference with 58 fitted statistics and no gradient-trained parameters; repeating its deterministic fit in five seed slots does not create stochastic training variation.')
    records=[r for r in manifest['checkpoints'] if r['method']!='NB']
    cap=int(manifest.get('settings',{}).get('epochs',40))
    at_cap=sum(r.get('epochs_run',0)==cap for r in records)
    text(f'All neural models use the common BCE-plus-Brier supervised objective, five training seeds, a maximum of {cap} supervised epochs, and validation-selected checkpoints. {at_cap} of {len(records)} neural runs reached the epoch cap. DAE/GMAE/HDC receive five pretraining epochs; the remaining neural methods receive none. Thus full-pipeline comparisons change pretraining effort and capacity. The DAE comparison does not isolate pretraining without a no-pretraining DAE control.')
    head('S5  Separating recorded traffic from constructed stress')
    text('The clean condition uses the recorded feature values. Missing20/Missing50 independently hide 20%/50% of features. Spoof20/Spoof40 replace 20%/40% of the first eight numeric feature coordinates with a training-only verified-benign centroid while preserving categorical flags and the observation mask. The observed endpoint graph remains unchanged. These are artificial interventions on real captures, not observed adversarial telemetry deception. Training includes clean, Missing20 and Spoof20 observations; stronger levels probe severity transfer.')
    head('S6  Compute and provenance')
    seconds=manifest.get('wall_seconds',0)
    text(f"All model computation uses {escape(str(manifest.get('hardware','the recorded CUDA device')))}. The study wall time is {seconds/60:.1f} minutes. Warm CUDA measurements concern one fixed 32-flow window; incremental peak allocation excludes resident models/data. The separate host-wall audit states its included transfers and preparation. Raw captures, processed arrays, checkpoints and logs remain ignored; compact measurements, source hashes and selected final figures are tracked.")
    story.append(PageBreak())

    text('Paired held-out evidence and its limits','TitleReal')
    picture('real_figS2_seed_operating_points')
    text('Figure S2. Left: clean SCR_FLOW-minus-GMAE AP by test capture and training seed. Positive differences favor SCR_FLOW. Right: validation-selected thresholds, retained unchanged across clean and stressed test conditions. Repeated predictions on the same flows do not create independent test samples.','CaptionReal')
    head('S7  Prespecified primary contrast')
    lo,hi=primary['percentile_95_ci']
    conclusion=('The interval favors SCR_FLOW on these fixed captures.' if lo>0 else 'The interval favors GMAE on these fixed captures.' if hi<0 else 'The interval includes zero and establishes neither a gain nor equivalence.')
    text(f"Clean capture-macro attack AP is {primary['means']['SCR_FLOW']:.5f} for SCR_FLOW and {primary['means']['GMAE']:.5f} for GMAE. Their paired difference is {primary['estimate']:+.5f}, with 95% interval [{lo:+.5f}, {hi:+.5f}]. {conclusion}")
    text(f"The interval uses {primary['bootstrap_repetitions']:,} accepted paired bootstrap draws, crossing training seeds with contiguous 64-window blocks independently resampled within each fixed test capture. Shared block multiplicities preserve method/seed pairing; each capture retains equal weight. Exact tie-aware weighted AP is recomputed. {primary['rejected_zero_positive_draws']} draws lacking positives in a resampled capture were rejected. The three captures supply {primary['blocks_per_capture']['10']}, {primary['blocks_per_capture']['11']} and {primary['blocks_per_capture']['12']} blocks; the last two populations are small. Captures themselves are not population-resampled.",'CaptionReal')
    head('S8  External validity and planning scope')
    text('The experiment supplies external evidence for adapted representation learning and flow classification. It does not validate a multi-step cyber planner, action execution, real mission utility or a hyperdimensional wire codec. The synthetic study and this captured-traffic task have different targets and feature semantics; their scores are not directly comparable. Author labels are partly host-derived, background is excluded, and recurring hosts or capture artifacts may remain informative even after raw identities are removed.')
    text('Only one dataset and three held-out captures are covered. The block interval conditions on these captures, the training data and the labeling policy; it does not establish performance across organizations or unknown attacks generally. Artificial missingness and centroid replacement cannot substitute for observed deception. Secondary model contrasts and seed-t figure intervals are descriptive and not multiplicity-adjusted.','SmallReal')
    head('Primary-source record')
    text('[1] García, Grill, Stiborek and Zunino. <link href="https://doi.org/10.1016/j.cose.2014.05.011" color="#226581">An empirical comparison of botnet detection methods</link>. Computers &amp; Security 45 (2014), 100-123. '
         '<link href="https://www.stratosphereips.org/datasets-ctu13" color="#226581">Official CTU-13 description</link>; <link href="https://www.stratosphereips.org/datasets-faq" color="#226581">author usage and citation guidance</link>. '
         'The shared-context adaptation builds on the project\'s documented set-learning and mixture-of-experts lineage; masked and denoising pretraining remain adapted established approaches.','SmallReal')
    text('Submission note. This separate record has the requested three-page length and two extra figures. Public MVML guidelines inspected for the project did not explicitly confirm a separate supplemental-upload category.','SmallReal')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#bacfd9'));canvas.line(48,40,547,40)
        canvas.setFont('RealSans',7.2);canvas.setFillColor(colors.HexColor('#506b80'))
        canvas.drawString(48,27,'Captured CTU-13 telemetry | External application supplementary record');canvas.drawRightString(547,27,str(doc.page));canvas.restoreState()
    path=SUPP/'Supplement_real.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=52,
        title='External application to captured CTU-13 traffic',author='Research project').build(story,onFirstPage=footer,onLaterPages=footer)
    count=len(PdfReader(path).pages)
    if count!=3:raise RuntimeError(f'Expected exactly three pages, produced {count}')
    inputs=[DATA/'summary.csv',manifest_path,comparison_path,dataset_path,SUPP/'real_figS1_training.png',SUPP/'real_figS2_seed_operating_points.png']
    if (DATA/'pipeline_timing.json').is_file():inputs.append(DATA/'pipeline_timing.json')
    record={'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'script':'scripts/make_real_supplement.py',
        'script_sha256':sha(__file__),'pdf_sha256':sha(path),'pages':count,'figures':['real_figS1_training','real_figS2_seed_operating_points'],
        'scope':'Captured telemetry classification with constructed feature-corruption stress; not real course-of-action outcomes.'}
    (SUPP/'Supplement_real.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Created {path.relative_to(ROOT)}: {count} pages, {path.stat().st_size} bytes')


if __name__=='__main__':main()
