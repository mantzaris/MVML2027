#!/usr/bin/env python3
"""Publication-size figures from recorded GPU results; never train or infer.

Render vector PDFs at their intended 6.53-inch manuscript width. PNG previews
and rendering configuration stay in ignored work/. Source/output hashes are
kept in internal provenance, never in the anonymized PDF metadata.
"""
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

OUT = ROOT / 'manuscript/anonymous/figures'
REVIEW = ROOT / 'work/manuscript/figure_review'
REAL = ROOT / 'results/real_gmae'
SEEDS = [11, 23, 37, 51, 71]
CAPTURES = [10, 11, 12]
CONDITIONS = ['clean', 'missing20', 'missing50', 'spoof20', 'spoof40']
IMPORTED = ['NB', 'LOGIT', 'DAE', 'DENSE', 'GMAE', 'HDC', 'SCR_FLOW', 'CTX_FLOW', 'LOCAL_FLOW']
NEW = ['EDGE_GMAE', 'EDGE_DENSE', 'CONS_GMAE', 'CONS_PRE']
METHODS = IMPORTED + NEW
FOCUS = ['GMAE', 'DENSE'] + NEW
PR_FOCUS = ['GMAE', 'DENSE', 'HDC', 'DAE', 'EDGE_GMAE', 'CONS_GMAE']
SHORT = {'NB':'NB', 'LOGIT':'LOGIT', 'DAE':'DAE', 'DENSE':'DENSE', 'GMAE':'GMAE', 'HDC':'HDC',
         'SCR_FLOW':'SCR', 'CTX_FLOW':'CTX', 'LOCAL_FLOW':'LOCAL', 'EDGE_GMAE':'EDGE',
         'EDGE_DENSE':'EDGE-0', 'CONS_GMAE':'CONS', 'CONS_PRE':'CONS-P'}
COLORS = {'NB':'#7b8794', 'LOGIT':'#967149', 'DAE':'#9365b8', 'DENSE':'#254d66', 'GMAE':'#0072b2',
          'HDC':'#a09549', 'SCR_FLOW':'#a45c7d', 'CTX_FLOW':'#777b5d', 'LOCAL_FLOW':'#b3a992',
          'EDGE_GMAE':'#d55e00', 'EDGE_DENSE':'#edaa69', 'CONS_GMAE':'#009e73', 'CONS_PRE':'#80bcaa',
          'JH':'#007f72', 'JD':'#b05a9d', 'IH':'#56b4e9', 'ID':'#0072b2', 'JH_ERM':'#947133',
          'JH_product':'#e69f00', 'SCR':'#cc3366', 'CAND_CTX':'#d55e00', 'CAND_LOCAL':'#df9500'}
MARKERS = {m:k for m,k in zip(FOCUS,['s','D','o','^','P','v'])}
MARKERS.update({'NB':'X','DAE':'D','HDC':'^'})
COND_LABEL = ['Recorded', '20%\nmissing', '50%\nmissing', '20%\nreplaced', '40%\nreplaced']
PROVENANCE = {}
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def load_json(path):
    return json.loads(record(path).read_text())


def load_csv(path):
    with record(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return rows


def subset(rows, **filters):
    return [r for r in rows if all(r.get(k) == v for k,v in filters.items())]


def seed_values(rows, metric):
    values = {}
    for seed in SEEDS:
        v = [r[metric] for r in rows if int(r['seed']) == seed and np.isfinite(r[metric])]
        if v:
            values[seed] = float(np.mean(v))
    return values


def estimate(rows, metric):
    values = np.array(list(seed_values(rows, metric).values()))
    if not len(values):
        raise ValueError('No finite observations for ' + metric)
    critical = {2:12.706, 3:4.303, 4:3.182, 5:2.776445}.get(len(values))
    return float(values.mean()), (float(critical * values.std(ddof=1) / np.sqrt(len(values)))
                                 if critical is not None else 0.)


def paired(rows, first, second, metric='ap'):
    a = seed_values(subset(rows, method=first), metric)
    b = seed_values(subset(rows, method=second), metric)
    if set(a) != set(SEEDS) or set(b) != set(SEEDS):
        raise ValueError('Incomplete five-seed paired contrast')
    return np.array([a[s] - b[s] for s in SEEDS])


def core(rows):
    return [r for r in rows if r['missing'] == .2 and r['facts'] == 64 and r['budget'] == 2048
            and (r['corruption'] == 0 or (r['corruption'] == .2 and r['attack'] in ['targeted','random','coherent']))]


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.titlesize':9,
        'axes.titleweight':'bold', 'axes.labelsize':8, 'xtick.labelsize':7.5, 'ytick.labelsize':7.5,
        'legend.fontsize':7.5, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.grid':True, 'grid.alpha':.16, 'axes.axisbelow':True, 'pdf.fonttype':42,
        'ps.fonttype':42, 'savefig.facecolor':'white', 'lines.linewidth':1.1,
        'figure.constrained_layout.h_pad':.035, 'figure.constrained_layout.w_pad':.035})


def panel(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc='left', pad=7)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def canvas(height=3.1, **kwargs):
    return plt.subplots(1, 2, figsize=(6.53, height), layout='constrained', **kwargs)


def save(fig, name, sources, description, aggregation):
    # Fixed PDF media box, not bbox_inches='tight': no shrink-to-fit surprises.
    fig.canvas.draw()
    visible_text = [a for a in fig.findobj(matplotlib.text.Text) if a.get_visible() and a.get_text()]
    if any(a.get_fontsize() < 7.5 for a in visible_text):
        raise ValueError('Figure text below 7.5pt')
    metadata = {'Title':name + ': ' + description, 'Author':None, 'Subject':None,
                'Keywords':None, 'Creator':'', 'Producer':'', 'CreationDate':None, 'ModDate':None}
    fig.savefig(OUT / (name + '.pdf'), metadata=metadata)
    fig.savefig(REVIEW / (name + '.png'), dpi=180, metadata={'Software':'Anonymous scientific figure'})
    PROVENANCE[name] = {'description':description, 'size_inches':[6.53,float(fig.get_figheight())],
        'minimum_visible_font_pt':min(a.get_fontsize() for a in visible_text),
        'inputs':{str(p.relative_to(ROOT)):sha(p) for p in sources}, 'aggregation':aggregation,
        'pdf_sha256':sha(OUT / (name + '.pdf')), 'preview_sha256':sha(REVIEW / (name + '.png'))}
    plt.close(fig)


def audit(manifest, comparisons, directory):
    for key, filename in [('manifest_sha256','manifest.json'), ('summary_sha256','summary.csv')]:
        if comparisons[key] != sha(directory / filename):
            raise ValueError('Stale comparison: ' + str(directory / filename))
    for relative, expected in manifest['source_hashes'].items():
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError('Changed scientific source: ' + relative)
        record(path)


def synthetic_figure(series, methods, metric, label, description):
    directory = ROOT / 'results' / series
    rows = load_csv(directory / 'summary.csv')
    manifest = load_json(directory / 'manifest.json')
    comparison = load_json(directory / 'comparisons.json')
    audit(manifest, comparison, directory)
    rows = core(rows)
    for method in methods:
        if len(subset(rows, method=method)) != 20:
            raise ValueError('Four conditions per seed required: ' + method)
    fig, axs = canvas(3.0, sharey=True)
    y = np.arange(len(methods))
    for i, method in enumerate(methods):
        selected = subset(rows, method=method)
        mean, error = estimate(selected, 'regret')
        axs[0].errorbar(mean, i, xerr=error, fmt='o', color=COLORS[method], capsize=3, ms=5, zorder=3)
        values = list(seed_values(selected, 'regret').values())
        axs[0].scatter(values, i + np.linspace(-.12,.12,5), color=COLORS[method], alpha=.5, s=11)
        value, error = estimate(selected, metric)
        axs[1].barh(i, value, xerr=error, color=COLORS[method], capsize=3, height=.62)
        axs[1].text(value + error + (.001 if series == 'v2' else .008), i,
                    f'{value:.3f}', va='center', fontsize=7.5)
    axs[0].set_yticks(y, methods); axs[0].invert_yaxis()
    axs[0].xaxis.set_major_locator(MaxNLocator(4)); axs[1].xaxis.set_major_locator(MaxNLocator(4))
    largest = max(sum(estimate(subset(rows, method=m), metric)) for m in methods)
    axs[1].set_xlim(0, largest * 1.26)
    panel(axs[0], 'a  Four-condition mean regret', 'Regret (lower is better)')
    panel(axs[1], 'b  Probability consistency', label)
    name = 'figS1' if series == 'v2' else 'figS2'
    sources = [directory / f for f in ['summary.csv','manifest.json','comparisons.json']]
    scope = ('Four fixed core conditions equally weighted within training seed; mean and descriptive 95% '
             'Student-t interval across five seeds, conditional on the common test pool. Left dots also '
             'show seed means. These are not the crossed seed/topology bootstrap intervals reported in text. ')
    scope += ('Violation: fraction of fact slots with child marginal > parent marginal + 1e-6.' if series == 'v2'
              else 'Violation: adjacent-candidate shadow union success probability > min(two member probabilities) + 1e-4.')
    save(fig, name, sources, description, scope)


def main():
    OUT.mkdir(parents=True, exist_ok=True); REVIEW.mkdir(parents=True, exist_ok=True)
    style()
    rows = load_csv(REAL / 'summary.csv'); curves = load_csv(REAL / 'curves.csv'); cal = load_csv(REAL / 'calibration.csv')
    manifest = load_json(REAL / 'manifest.json'); comp = load_json(REAL / 'comparisons.json')
    gpu = load_json(REAL / 'gpu_audit.json'); cal_audit = load_json(REAL / 'calibration_audit.json')
    audit(manifest, comp, REAL)
    expected = set(itertools.product(SEEDS, METHODS, CONDITIONS, CAPTURES))
    actual = {(int(r['seed']),r['method'],r['condition'],int(r['scenario'])) for r in rows}
    if len(rows) != 975 or actual != expected:
        raise ValueError('Incomplete or duplicate real-data result rows')
    for key, filename in [('calibration_sha256','calibration.csv'),('calibration_audit_sha256','calibration_audit.json'),('gpu_audit_sha256','gpu_audit.json')]:
        if comp[key] != sha(REAL / filename):
            raise ValueError('Changed audited file: ' + filename)
    if not gpu['passed'] or gpu['manifest_sha256'] != sha(REAL / 'manifest.json') or gpu['summary_sha256'] != sha(REAL / 'summary.csv'):
        raise ValueError('Incomplete or stale common-session GPU audit')
    for relative, expected_hash in gpu['source_hashes'].items():
        if sha(ROOT / relative) != expected_hash:
            raise ValueError('Changed GPU-audit source: ' + relative)
        record(ROOT / relative)
    for kind in ['manifest','summary','curves','calibration']:
        ext = '.json' if kind == 'manifest' else '.csv'
        path = ROOT / 'results/real' / (kind + ext)
        if manifest['base_' + kind + '_sha256'] != sha(path):
            raise ValueError('Changed imported baseline source: ' + kind)
        record(path)
    clean = subset(rows, condition='clean')
    sources = [REAL / f for f in ['summary.csv','manifest.json','comparisons.json']]
    seed_scope = ('Equal capture means within each training seed, then five-seed means and descriptive 95% '
                  'Student-t intervals. Three fixed captures; previously inspected test data; exploratory follow-up.')

    # Figure 1: retained baselines and the two prespecified exploratory contrasts.
    fig, axs = canvas(3.45, gridspec_kw={'width_ratios':[1.2,1]})
    order = METHODS
    values = [estimate(subset(clean, method=m), 'ap') for m in order]
    axs[0].barh(range(13), [v for v,e in values], xerr=[e for v,e in values],
                color=[COLORS[m] for m in order], capsize=2, height=.67)
    axs[0].set_yticks(range(13), [SHORT[m] for m in order]); axs[0].invert_yaxis()
    axs[0].set_xlim(0,1.055); axs[0].set_xticks([0,.25,.5,.75,1])
    panel(axs[0], 'a  All 13 methods', 'Clean capture-macro AP')
    for i, m in enumerate(['EDGE_GMAE','CONS_GMAE']):
        p = comp['primary'][m]; lo,hi = p['bonferroni_97_5_ci']; inner = p['percentile_95_ci']
        delta = paired(clean,m,'GMAE')
        if not np.isclose(delta.mean(),p['estimate'],atol=1e-10):
            raise ValueError('Contrast/CSV mean mismatch')
        if not np.allclose(delta,[p['seed_effects'][str(s)] for s in SEEDS],atol=1e-10):
            raise ValueError('Contrast/CSV seed mismatch')
        axs[1].hlines(i, lo,hi,color=COLORS[m],lw=1.5)
        axs[1].hlines(i,inner[0],inner[1],color=COLORS[m],lw=5,alpha=.55)
        axs[1].vlines([lo,hi],i-.06,i+.06,color=COLORS[m],lw=1)
        axs[1].plot(p['estimate'],i,'o',color=COLORS[m],ms=6)
        axs[1].scatter(delta,i+np.linspace(-.13,.13,5),color=COLORS[m],s=15,alpha=.6)
    axs[1].set_yticks([0,1],['EDGE\n− GMAE','CONS\n− GMAE']); axs[1].set_ylim(1.55,-.55)
    axs[1].axvline(0,color='#555',ls='--',lw=.9); axs[1].xaxis.set_major_locator(MaxNLocator(4))
    legend = [Line2D([0],[0],color='#555',lw=5,alpha=.55,label='95% interval'),
              Line2D([0],[0],color='#555',lw=1.5,label='97.5% interval'),
              Line2D([0],[0],color='#555',marker='o',ls='',ms=3,label='Seed effect')]
    axs[1].legend(handles=legend,loc='lower right',frameon=False,handlelength=1.7)
    panel(axs[1],'b  Paired primary contrasts','Clean AP difference')
    save(fig,'fig1',sources,'Clean performance and primary paired contrasts',seed_scope +
         ' Panel b: recorded shared paired seed/time-block percentile intervals, 95% thick and97.5% thin, '
         'with seed points; fixed equal capture weights. The wider interval is nominally adjusted for two comparisons.')

    # Figure 2: fixed-capture differences and support-weighted calibration.
    fig, axs = canvas(3.2, gridspec_kw={'width_ratios':[1.03,1]})
    matrix = np.array([[estimate(subset(clean,method=m,scenario=float(c)),'ap')[0] for c in CAPTURES] for m in PR_FOCUS])
    im = axs[0].imshow(matrix,cmap='Blues',vmin=.65,vmax=1,aspect='auto'); axs[0].grid(False)
    for i in range(6):
        for j in range(3):
            axs[0].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=8,
                        color='white' if matrix[i,j]>.86 else 'black')
    prevalence = []
    for c in CAPTURES:
        rr = subset(clean,scenario=float(c))
        pairs = {(int(r['n']),int(r['positives'])) for r in rr}
        if len(pairs) != 1:
            raise ValueError('Inconsistent capture labels/counts')
        n,p = next(iter(pairs)); prevalence.append(p/n)
    axs[0].set_xticks(range(3),[f'{c}\nπ={p:.3f}' for c,p in zip(CAPTURES,prevalence)])
    axs[0].set_yticks(range(6),[SHORT[m] for m in PR_FOCUS])
    cb = fig.colorbar(im,ax=axs[0],orientation='horizontal',pad=.11,aspect=28,fraction=.045)
    cb.set_ticks([.7,.8,.9,1]); cb.ax.tick_params(labelsize=7.5)
    panel(axs[0],'a  AP by held-out capture')
    for m in PR_FOCUS:
        chosen = subset(cal,method=m,condition='clean'); xx=[]; yy=[]
        for b in range(10):
            br = [r for r in subset(chosen,bin=float(b)) if r['count']>0]
            count = sum(r['count'] for r in br)
            if count:
                xx.append(sum(r['confidence']*r['count'] for r in br)/count)
                yy.append(sum(r['accuracy']*r['count'] for r in br)/count)
        axs[1].plot(xx,yy,color=COLORS[m],marker=MARKERS[m],ms=3,label=SHORT[m])
    axs[1].plot([0,1],[0,1],ls='--',color='#777',lw=.8)
    axs[1].set(xlim=(0,1),ylim=(0,1)); axs[1].legend(ncol=2,loc='lower right',frameon=False,columnspacing=.9)
    panel(axs[1],'b  Pooled reliability','Predicted attack probability','Recorded attack frequency')
    save(fig,'fig2',sources+[REAL/'calibration.csv',REAL/'calibration_audit.json'],
         'Capture-specific AP and pooled clean reliability',
         'Panel a: mean AP across five seeds for each fixed capture; π is attack prevalence from identical per-capture n/positives. '
         'Color scale .65–1.00. Panel b: confidence/accuracy bin means pooled by recorded flow count over captures and seeds. '
         'Repeated predictions on the same flows are not independent observations; no uncertainty bars are asserted.')

    # Figure 3: artificial corruption and matched mechanisms.
    fig, axs = canvas(3.15)
    for m in FOCUS:
        stats = [estimate(subset(rows,method=m,condition=c),'ap') for c in CONDITIONS]
        axs[0].errorbar(range(5),[v for v,e in stats],yerr=[e for v,e in stats],color=COLORS[m],
                       marker=MARKERS[m],ms=3,capsize=2,ls='--' if m in ['EDGE_DENSE','CONS_PRE'] else '-',label=SHORT[m])
    axs[0].legend(ncol=1,loc='lower left',frameon=False,handlelength=1.3)
    contrasts = [('EDGE_GMAE','EDGE_DENSE','EDGE − EDGE-0'),('CONS_GMAE','CONS_PRE','CONS − CONS-P'),('CONS_PRE','GMAE','CONS-P − GMAE')]
    for i,(a,b,label) in enumerate(contrasts):
        effects = [paired(subset(rows,condition=c),a,b) for c in CONDITIONS]
        axs[1].errorbar(np.arange(5)+(i-1)*.055,[v.mean() for v in effects],
            yerr=[2.776445*v.std(ddof=1)/np.sqrt(5) for v in effects],color=COLORS[a],
            marker=['o','s','^'][i],ms=3,capsize=2,label=label)
    axs[1].axhline(0,color='#666',ls='--',lw=.8)
    axs[1].legend(loc='best',frameon=False,handlelength=1.5)
    for ax in axs:
        ax.set_xticks(range(5),COND_LABEL); ax.tick_params(axis='x',labelsize=7.5)
    panel(axs[0],'a  Artificial corruption stress',ylabel='Capture-macro attack AP')
    panel(axs[1],'b  Matched control contrasts',ylabel='AP difference')
    save(fig,'fig3',sources,'Artificial stress and matched controls',seed_scope+
         ' Paired differences use the same five seeds. CONS-P minus GMAE changes both view count and latent loss; '
         'it is a joint intervention. Missing/replaced features are constructed corruptions, not observed deception.')

    # Figure 4: measured effort and a common-session inference comparison.
    fig, axs = canvas(3.05)
    methods = ['DENSE','GMAE']+NEW; records=manifest['checkpoints']; pre=[]; sup=[]; err=[]
    for m in methods:
        rr=[r for r in records if r['method']==m]
        if len(rr)!=5:
            raise ValueError('Incomplete training cost records')
        pre.append(np.mean([r['pretrain_seconds'] for r in rr])); sup.append(np.mean([r['supervised_seconds'] for r in rr]))
        total=np.array([r['pretrain_seconds']+r['supervised_seconds'] for r in rr])
        err.append(2.776445*total.std(ddof=1)/np.sqrt(5))
    axs[0].bar(range(6),sup,color=[COLORS[m] for m in methods],label='Supervised')
    axs[0].bar(range(6),pre,bottom=sup,color=[COLORS[m] for m in methods],alpha=.6,hatch='///',label='Pretraining')
    axs[0].errorbar(range(6),np.array(pre)+sup,yerr=err,fmt='none',ecolor='#333',capsize=2)
    axs[0].set_xticks(range(6),[SHORT[m] for m in methods],rotation=35,ha='right')
    axs[0].legend(frameon=False,loc='upper left')
    timing={r['method']:r for r in gpu['timing_rows']}
    for m in ['NB','DAE']+FOCUS:
        t=timing[m];y,ye=estimate(subset(clean,method=m),'ap')
        axs[1].errorbar(t['wall_median_ms'],y,
            xerr=np.array([[t['wall_median_ms']-t['wall_p10_ms']],[t['wall_p95_ms']-t['wall_median_ms']]]),
            yerr=ye,fmt=MARKERS[m],color=COLORS[m],ms=4,label=SHORT[m],capsize=2)
    axs[1].legend(ncol=2,loc='lower right',frameon=False,columnspacing=.7,handlelength=1.2)
    panel(axs[0],'a  Actual training effort',ylabel='Pretraining + supervision (s)')
    panel(axs[1],'b  Common-session wall cost','Median wall time (ms); p10–p95','Clean capture-macro AP')
    save(fig,'fig4',sources+[REAL/'gpu_audit.json'],'Training and measured common-session inference cost',seed_scope+
         ' Training: actual pretraining and supervision seconds, mean and total seed-t interval; baseline training '
         'times were imported from their original session. Inference: all models remeasured together on a prepared '
         '32-flow host-to-GPU-to-host window, horizontal p10–p95 repeat quantiles, not confidence intervals; '
         'vertical descriptive seed-t intervals. Data loading from disk is excluded.')

    synthetic_figure('v2',['JH','JD','IH','ID','JH_ERM','JH_product'],'logical_violation',
                     'Child-above-parent fraction','Synthetic matched methods and fact-marginal consistency')
    synthetic_figure('v3',['SCR','CAND_CTX','CAND_LOCAL','DAE','LOGIT','ID','JH'],'shadow_union_violation',
                     'Shadow-union violation fraction','Synthetic candidate ranking and union consistency')

    # Figure S3: comparable losses, not the full regularized CONS training objective.
    fig, axs = canvas(3.0)
    methods=['GMAE']+NEW
    for ax,split,letter in zip(axs,['train','validation'],'ab'):
        for m in methods:
            chosen=subset(curves,method=m,split='train_base' if split=='train' and m in NEW else split)
            epochs=sorted({r['epoch'] for r in chosen})
            if not epochs:
                raise ValueError('Missing common base-loss history')
            vals=[estimate(subset(chosen,epoch=e),'loss')[0] for e in epochs]
            ax.plot(np.array(epochs)+1,vals,color=COLORS[m],label=SHORT[m],
                    ls='--' if m in ['EDGE_DENSE','CONS_PRE'] else '-')
        panel(ax,letter+'  '+('Training' if split=='train' else 'Validation')+' base loss','Epoch (1-based)','BCE + Brier')
        ax.xaxis.set_major_locator(MaxNLocator(5,integer=True))
    axs[0].legend(ncol=2,loc='upper right',frameon=False,columnspacing=.8,handlelength=1.7)
    save(fig,'figS3',[REAL/'curves.csv',REAL/'manifest.json'],'Comparable optimization histories',
         'Available-seed means at each epoch, no uncertainty intervals. New methods use train_base; imported '
         'GMAE uses train. All plotted training/validation losses are the common BCE+Brier objective; '
         'CONS supervised consistency penalty is excluded. Different stopping times change seed support.')

    # Figure S4: fixed-capture paired effects without pooling away heterogeneity.
    fig, axs = canvas(2.8,sharey=True)
    matrices=[np.array([paired(subset(clean,scenario=float(c)),m,'GMAE') for c in CAPTURES])
              for m in ['EDGE_GMAE','CONS_GMAE']]
    lim=max(np.abs(a).max() for a in matrices)
    for ax,m,matrix,letter in zip(axs,['EDGE_GMAE','CONS_GMAE'],matrices,'ab'):
        im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto'); ax.grid(False)
        for i in range(3):
            for j in range(5):
                ax.text(j,i,f'{matrix[i,j]:+.4f}',ha='center',va='center',fontsize=7.5,
                        color='white' if abs(matrix[i,j])>.5*lim else 'black')
        ax.set_xticks(range(5),SEEDS); ax.set_yticks(range(3),[f'Capture {c}' for c in CAPTURES])
        panel(ax,letter+'  '+SHORT[m]+' − GMAE','Training seed')
    cb=fig.colorbar(im,ax=axs,orientation='horizontal',pad=.19,fraction=.055,aspect=35)
    cb.set_label('Paired clean AP difference (positive favors variant)',fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    save(fig,'figS4',sources,'Paired seed effects for each fixed capture',
         'Each cell is variant-minus-GMAE AP for one training seed and one fixed clean capture. '
         'Shared symmetric color scale across panels; no independent-capture or uncertainty inference.')

    provenance={'script':'scripts/make_anonymous_figures.py','script_sha256':sha(__file__),
        'inputs':INPUTS,'figures':PROVENANCE,'pdf_metadata':'Anonymous; author/subject/keywords/dates omitted; creator/producer empty.',
        'source_integrity':'All comparison manifest/summary hashes, real calibration/audit hashes, common-session GPU audit hashes, '
            'imported baseline hashes, and recorded scientific-source hashes verified before plotting.',
        'rendering':'Fixed 6.53-inch PDF width; no bounding-box crop; all visible figure text at least7.5pt. '
            'PNG previews are ignored work artifacts. No training, inference, GPU selection, or raw result changes.'}
    (ROOT/'manuscript/figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Created8publication-size figures; source checks passed; minimum visible font7.5pt.')


if __name__=='__main__':
    main()
