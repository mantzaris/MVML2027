#!/usr/bin/env python3
"""Create and validate the three-page exploratory GraphMAE supplement."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from make_real_gmae_results import ROOT,DATA,SUPP,NEW,VARIANTS,METHODS,SEEDS,SCENARIOS,load,subset,estimate,sha


def main():
    paths={k:DATA/f'{k}.json' for k in ['manifest','comparisons','gpu_audit']}
    manifest=json.loads(paths['manifest'].read_text());comparison=json.loads(paths['comparisons'].read_text());gpu=json.loads(paths['gpu_audit'].read_text())
    dataset_path=ROOT/'results/real/ctu13_data_manifest.json';dataset=json.loads(dataset_path.read_text())
    if sha(dataset_path)!=manifest['data_manifest_sha256'] or not dataset['temporal_split_check']['passed']:raise ValueError('Changed/invalid dataset audit')
    if comparison['manifest_sha256']!=sha(paths['manifest']) or gpu['manifest_sha256']!=sha(paths['manifest']) or not gpu['passed']:raise ValueError('Stale or failed study/audit')
    rows=load('summary.csv');clean=subset(rows,condition='clean');primary=comparison['primary'];design=comparison['bootstrap']
    extras=['gmae_figS1_training','gmae_figS2_seed_effects']
    for stem in extras:
        record=json.loads((SUPP/f'{stem}.provenance.json').read_text())
        if sha(ROOT/record['script'])!=record['script_sha256']:raise ValueError('Stale figure source')
        for relative,digest in record['inputs'].items():
            if sha(ROOT/relative)!=digest:raise ValueError('Stale figure input')
        if sha(SUPP/f'{stem}.png')!=record['output_sha256']['png']:raise ValueError('Changed supplemental figure')
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('GmaeSans',str(fonts/'DejaVuSans.ttf')));pdfmetrics.registerFont(TTFont('GmaeBold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('GmaeSans',normal='GmaeSans',bold='GmaeBold',italic='GmaeSans',boldItalic='GmaeBold')
    styles=getSampleStyleSheet()
    for name,font,size,leading,color,after,before in [
        ('TitleG','GmaeBold',16,20,'#193549',8,0),('SubG','GmaeSans',9,12,'#506b80',9,0),
        ('BodyG','GmaeSans',8.5,11.4,'#17252c',7,0),('HeadG','GmaeBold',10.5,13,'#226581',5,7),
        ('CaptionG','GmaeSans',7.6,10.1,'#486379',7,0),('SmallG','GmaeSans',7.2,9.6,'#253b4a',6,0)]:
        styles.add(ParagraphStyle(name,fontName=font,fontSize=size,leading=leading,textColor=colors.HexColor(color),spaceAfter=after,spaceBefore=before))
    story=[]
    def text(value,style='BodyG'):story.append(Paragraph(value,styles[style]))
    def head(value):text(value,'HeadG')
    def picture(stem):
        from PIL import Image as PILImage
        path=SUPP/f'{stem}.png'
        with PILImage.open(path) as p:w,h=p.size
        story.append(Image(str(path),width=499,height=499*h/w));story.append(Spacer(1,5))
    def table(data,widths,size=7.7):
        item=Table(data,colWidths=widths,repeatRows=1)
        item.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'GmaeSans'),('FONTNAME',(0,0),(-1,0),'GmaeBold'),
        ('FONTSIZE',(0,0),(-1,-1),size),('LEADING',(0,0),(-1,-1),10),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff3')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f9fa')]),('TOPPADDING',(0,0),(-1,-1),4),
        ('BOTTOMPADDING',(0,0),(-1,-1),4),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#acc7d4'))]))
        story.append(item);story.append(Spacer(1,6))
    def equation(value):
        item=Table([[Paragraph(value,styles['BodyG'])]],colWidths=[499]);item.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eaf2f6')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        story.append(item);story.append(Spacer(1,6))

    text('Two GraphMAE variants on captured telemetry','TitleG')
    text('Exploratory CTU-13 follow-up | Supplementary research evidence','SubG')
    text('The previous test results motivated this follow-up. New methods and settings were frozen before their new test predictions, with validation-only selection. However, the test captures were already inspected: these are exploratory results, not independent confirmation. Nominal two-comparison adjustment does not remove adaptive dataset reuse.')
    head('S1  Fixed recorded-data task')
    text('CTU-13 records university traffic with controlled botnet executions [3]. The selected From-Normal/From-Botnet subset contains 799,777 of 19,976,700 flows (4.00%): 442,084 training, 214,880 validation and 142,813 test. Background is excluded. All training flows complete before validation starts, and validation completes before test starts. Hosts can recur. Fixed 32-flow windows and endpoint graphs are formed after label filtering; predictions use completed-flow statistics.','CaptionG')
    head('S2  Two controlled changes to GraphMAE')
    text('<b>EDGE_GMAE</b> reweights existing endpoint edges using shared observed coordinates. With overlap fraction r<sub>ij</sub> and mean squared discrepancy d<sub>ij</sub> over that overlap:')
    equation('Ā<sub>ij</sub> = A<sub>ij</sub> [0.1 + 0.9 r<sub>ij</sub> exp(−d<sub>ij</sub>/2)] for i ≠ j; Ā<sub>ii</sub> = 1. Normalize rows.')
    text('The gate uses only the masked view available to each encoder; self loops remain and padding is excluded. It adds no trainable parameters. EDGE_DENSE removes pretraining. Compatibility is a heuristic, not truth probability; useful heterogeneous neighbors can be downweighted.','CaptionG')
    text('<b>CONS_GMAE</b> keeps the GMAE inference architecture. Two independent 30% node-mask views reconstruct observed coordinates and align latent representations over valid nodes using symmetric stopped-target cosine loss C:')
    equation('L<sub>pre</sub> = ½(L<sub>rec,1</sub> + L<sub>rec,2</sub>) + 0.5 C(h<sub>1</sub>,h<sub>2</sub>);<br/>L<sub>sup</sub> = L<sub>BCE+Brier</sub> + 0.5 mean[(p<sub>20% feature-masked</sub> − stopgrad(p<sub>original</sub>))²].')
    text('CONS_PRE retains the two-view pretraining but omits supervised consistency. The original observations may already be corrupted; no oracle-clean target, family label, EMA teacher or extra inference head is used. These combine established ideas [1,2], without claiming a first-ever architecture.','CaptionG')
    head('S3  Clean capture-macro comparisons')
    shown=['NB','DAE','DENSE','GMAE','HDC']+NEW
    values=[['Method','Attack AP','ROC-AUC','Brier','Recall','FPR']]
    for m in shown:values.append([m]+[f'{estimate(subset(clean,method=m),k)[0]:.4f}' for k in ['ap','auc','brier','recall','fpr']])
    table(values,[103,78,78,80,80,80])
    text('Table S1. Selected comparators among all 13 recorded methods; full results accompany the main figures. Equal means over three fixed test captures and five seeds. AP denotes attack average precision. Recall/FPR use frozen clean-validation F1 thresholds. Nine earlier baselines are imported unchanged.','CaptionG')
    scores={m:estimate(subset(clean,method=m),'ap')[0] for m in METHODS}
    text(f'EDGE_DENSE has the numerically highest clean AP ({scores["EDGE_DENSE"]:.4f}), compared with EDGE_GMAE {scores["EDGE_GMAE"]:.4f}. The gated family\'s gain cannot be attributed to masked pretraining alone. CONS_GMAE reaches {scores["CONS_GMAE"]:.4f}; its paired uncertainty is reported on page 3.','SmallG')
    story.append(PageBreak())

    text('Controls, optimization and measured cost','TitleG')
    picture(extras[0])
    text('Figure S1. Means over available seeds at each epoch; stopped runs do not contribute later epochs. Training curves use the common base BCE-plus-Brier component, excluding the extra CONS regularizer. Validation uses the same base criterion for every method. Curves and selected epochs do not prove convergence.','CaptionG')
    head('S4  What the controls identify')
    text('EDGE_GMAE versus GMAE changes aggregation at matched pretraining/objectives; EDGE_DENSE versus DENSE checks aggregation without pretraining. EDGE_GMAE versus EDGE_DENSE adds masked pretraining within the gated representation. CONS_GMAE versus CONS_PRE adds supervised prediction consistency at fixed pretraining. CONS_PRE versus GMAE changes both reconstruction-view count and latent alignment; it does not isolate the cosine term alone. There is no two-view, zero-alignment control.')
    head('S5  Common training and observation contract')
    records=[r for r in manifest['checkpoints'] if r['method'] in NEW];at_cap=sum(r['epochs_run']==40 for r in records)
    parameters={r['parameters'] for r in records}
    text(f'All 20 new runs use seeds 11/23/37/51/71, AdamW learning rate 0.001, weight decay 0.0001, batch 256, maximum 40 supervised epochs and patience 8. {at_cap} of 20 reach the epoch cap. All four variants/controls have {int(next(iter(parameters))):,} trainable parameters, matching GMAE. EDGE_GMAE/CONS_GMAE/CONS_PRE receive five pretraining epochs; EDGE_DENSE receives none. The CONS objectives require extra forward passes, so equal capacity/epochs do not mean equal computation.')
    text('All 20 new training runs and validation-selected thresholds finish before new test inference. The fixed training mixture assigns windows to clean, 20% missing or 20% replacement observations. Early stopping averages the common validation loss over those three views. Thresholds maximize F1 on clean validation and remain unchanged for all test conditions. Fourteen flow features and 14 availability indicators use training-only scaling; raw identities/timestamps are not numeric inputs.','CaptionG')
    head('S6  Artificial stress and resource accounting')
    text('The recorded clean condition is distinguished from independent 20%/50% feature removal and 20%/40% replacement of the first eight numeric coordinates by a training-only normal centroid. The endpoint graph remains unchanged, although EDGE reweights it. These feature interventions are not observed adversarial deception, packet-valid adaptive attacks or corrupted graph metadata.')
    hardware=escape(str(manifest['hardware']));seconds=manifest['wall_seconds']
    text(f'New training/evaluation used {hardware}; follow-up run wall time was {seconds/60:.1f} minutes. Historical baseline training times remain labeled as imported. A common-session audit remeasures all 13 inference architectures: decoded CPU measurements to CUDA normalization, endpoint graph, model and CPU probabilities. It excludes CSV/packet parsing, waiting for flow completion, network transit and cold loading. Input/output transfer is {gpu["input_bytes"]:,}/{gpu["output_bytes"]:,} bytes per 32-flow window; this is not a compression codec.')
    timing={r['method']:r for r in gpu['timing_rows']}
    text(f'Common-session median host-wall times are GMAE {timing["GMAE"]["wall_median_ms"]:.3f} ms, EDGE_GMAE {timing["EDGE_GMAE"]["wall_median_ms"]:.3f} ms and CONS_GMAE {timing["CONS_GMAE"]["wall_median_ms"]:.3f} ms. EDGE adds pairwise gating cost; CONS retains the original inference architecture. These are bounded measurement-loop costs.','CaptionG')
    text('Warm CUDA timing and incremental peak allocation exclude resident data/model memory. Host timing quantiles describe 50 repeats, not confidence intervals or operational response latency. Raw captures, prepared data, predictions, weights and logs remain ignored by Git.','CaptionG')
    story.append(PageBreak())

    text('Exploratory paired effects and limitations','TitleG')
    picture(extras[1])
    text('Figure S2. Variant-minus-GMAE clean AP by training seed and fixed test capture. Positive values favor the variant. Capture 12 contains NSIS.ay, absent from training; captures 10/11 contain Rbot. These capture differences do not isolate a causal malware-family effect.','CaptionG')
    head('S7  Two planned contrasts and nominal adjustment')
    values=[['Variant','AP difference','95% interval','97.5% interval']]
    for m in VARIANTS:
        p=primary[m];lo,hi=p['percentile_95_ci'];blo,bhi=p['bonferroni_97_5_ci']
        values.append([m,f'{p["estimate"]:+.5f}',f'[{lo:+.5f}, {hi:+.5f}]',f'[{blo:+.5f}, {bhi:+.5f}]'])
    table(values,[100,88,155,156],size=7.5)
    for m in VARIANTS:
        p=primary[m];lo,hi=p['bonferroni_97_5_ci']
        conclusion=('has an adjusted interval above zero on these fixed captures; this remains exploratory evidence' if lo>0 else 'has an adjusted interval below zero, favoring GMAE on these fixed captures' if hi<0 else 'has an adjusted interval containing zero, establishing neither improvement nor equivalence')
        text(f'{m} {conclusion}.','CaptionG')
    text(f'The two contrasts share {design["bootstrap_repetitions"]:,} paired bootstrap draws crossing training seeds with contiguous 64-window blocks within each fixed capture. Captures 10/11/12 provide 60/6/5 blocks; the latter populations are small. Exact tie-aware weighted AP is recomputed with equal capture weights; {design["rejected_zero_positive_draws"]} zero-positive draws were rejected. The 97.5% marginal intervals use 1.25%/98.75% quantiles, a nominal Bonferroni target for two contrasts. Bootstrap approximation, uncertain block independence and earlier adaptive choices limit coverage interpretation.')
    head('S8  External validity and proposal connection')
    text('This single-dataset follow-up tests representation aggregation, reconstruction and robustness at measured compute cost. It does not establish generalization to unseen organizations, unknown background traffic or arbitrary malware. Host-derived labels and recurring hosts can leave dataset artifacts. Feature similarity can suppress useful heterogeneous edges, and consistency can preserve incorrect original predictions. New reserved data is required for independent confirmation.')
    text('The proposal connection is a perception component for incomplete/corrupted evidence. Real course-of-action outcomes, action execution, mission utility, logical implication guarantees, distributed replanning and HDC wire compression remain untested. The captured-flow target differs from earlier synthetic planning targets.','CaptionG')
    head('Primary-source references')
    text('[1] Hou et al. <link href="https://arxiv.org/abs/2205.10803" color="#226581">GraphMAE</link>, KDD 2022. [2] Hou et al. <link href="https://arxiv.org/abs/2304.04779" color="#226581">GraphMAE2</link>, WWW 2023. '
         '[3] García et al. <link href="https://doi.org/10.1016/j.cose.2014.05.011" color="#226581">An empirical comparison of botnet detection methods</link>, Computers &amp; Security 45 (2014), 100-123; <link href="https://www.stratosphereips.org/datasets-ctu13" color="#226581">official CTU-13 description</link>. Additional graph-weighting/reliability/context-consistency prior art is documented in the method specification.','SmallG')
    text('Submission note. This record has the requested three-page length and two additional figures; the project\'s inspected public conference guidelines did not explicitly confirm a separate supplemental-upload category.','SmallG')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#bacfd9'));canvas.line(48,40,547,40)
        canvas.setFont('GmaeSans',7.2);canvas.setFillColor(colors.HexColor('#506b80'))
        canvas.drawString(48,27,'GraphMAE variants | Exploratory captured-traffic follow-up');canvas.drawRightString(547,27,str(doc.page));canvas.restoreState()
    path=SUPP/'Supplement_real_gmae.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=52,
        title='Two exploratory GraphMAE variants on captured telemetry',author='Research project').build(story,onFirstPage=footer,onLaterPages=footer)
    count=len(PdfReader(path).pages)
    if count!=3:raise RuntimeError(f'Expected3pages, produced{count}')
    inputs=[DATA/'summary.csv',*paths.values(),dataset_path,*[SUPP/f'{stem}.png' for stem in extras],ROOT/'scripts/make_real_gmae_results.py',ROOT/'scripts/analyze_real_gmae.py',ROOT/'scripts/analyze_real.py']
    record=dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},script='scripts/make_real_gmae_supplement.py',script_sha256=sha(__file__),
        dependency_sha256={'scripts/make_real_gmae_results.py':sha(ROOT/'scripts/make_real_gmae_results.py'),'scripts/analyze_real_gmae.py':sha(ROOT/'scripts/analyze_real_gmae.py')},
        pdf_sha256=sha(path),pages=count,figures=extras,scope='Exploratory reuse of CTU-13 captures; artificial corruption; no operational COA outcomes.')
    (SUPP/'Supplement_real_gmae.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Created {path.relative_to(ROOT)}: {count}pages,{path.stat().st_size}bytes')


if __name__=='__main__':main()
