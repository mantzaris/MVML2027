#!/usr/bin/env python3
"""Four empirical GraphMAE follow-up figures and two supplemental diagnostics."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'work/matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_real_gmae import METHODS,NEW,VARIANTS,LABELS,SEEDS,SCENARIOS,CONDITIONS,raw_path,ranked

DATA=ROOT/'results/real_gmae';OUT=ROOT/'figures/real_gmae';SUPP=ROOT/'supplemental/real_gmae'
FOCUS=['GMAE','DENSE','EDGE_GMAE','EDGE_DENSE','CONS_GMAE','CONS_PRE']
PR_FOCUS=['GMAE','DENSE','HDC','DAE','EDGE_GMAE','CONS_GMAE']
SHORT={'NB':'NB','LOGIT':'LOGIT','DAE':'DAE','DENSE':'DENSE','GMAE':'GMAE','HDC':'HDC','SCR_FLOW':'SCR','CTX_FLOW':'CTX','LOCAL_FLOW':'LOCAL',
'EDGE_GMAE':'EDGE','EDGE_DENSE':'EDGE-0','CONS_GMAE':'CONS','CONS_PRE':'CONS-P'}
COLORS={'NB':'#7b8794','LOGIT':'#967149','DAE':'#9365b8','DENSE':'#254d66','GMAE':'#0072b2','HDC':'#a09549',
'SCR_FLOW':'#a45c7d','CTX_FLOW':'#777b5d','LOCAL_FLOW':'#b3a992','EDGE_GMAE':'#d55e00','EDGE_DENSE':'#edaa69','CONS_GMAE':'#009e73','CONS_PRE':'#80bcaa'}
MARKERS={m:('o' if m in VARIANTS else '^' if m in NEW else 's' if m in ['GMAE','DENSE'] else 'D' if m=='DAE' else 'x') for m in METHODS}
COND_LABEL={'clean':'Recorded','missing20':'20% missing','missing50':'50% missing','spoof20':'20% replaced','spoof40':'40% replaced'}
MAIN=['gmae_fig01_quality','gmae_fig02_capture_calibration','gmae_fig03_stress_ablations','gmae_fig04_resource_cost']
EXTRA=['gmae_figS1_training','gmae_figS2_seed_effects']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    with (DATA/name).open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            try:row[key]=float(value)
            except (ValueError,TypeError):pass
    return rows


def subset(rows,**criteria):return [r for r in rows if all(r.get(k)==v for k,v in criteria.items())]


def seed_values(rows,metric):
    out={}
    for s in SEEDS:
        values=[r[metric] for r in rows if int(r['seed'])==s and np.isfinite(r.get(metric,float('nan')))]
        if values:out[s]=float(np.mean(values))
    return out


def estimate(rows,metric):
    x=np.array(list(seed_values(rows,metric).values()))
    if not len(x):return np.nan,np.nan
    return x.mean(),2.776445*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.


def paired(rows,first,second,metric='ap'):
    a=seed_values(subset(rows,method=first),metric);b=seed_values(subset(rows,method=second),metric)
    return np.array([a[s]-b[s] for s in SEEDS])


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.8,'axes.titleweight':'bold',
    'axes.labelsize':9,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':.15,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})


def panel(ax,title,xlabel=None,ylabel=None):
    ax.set_title(title,loc='left',pad=10)
    if xlabel:ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)


def bars(ax,rows,metric,methods=METHODS):
    values=[estimate(subset(rows,method=m),metric) for m in methods]
    ax.bar(range(len(methods)),[x[0] for x in values],yerr=[x[1] for x in values],color=[COLORS[m] for m in methods],capsize=2)
    ax.set_xticks(range(len(methods)),[SHORT[m] for m in methods],rotation=48,ha='right',fontsize=7.5)


def save(fig,stem,inputs,title,supplement=False):
    directory=SUPP if supplement else OUT;directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(directory/f'{stem}.png',dpi=280,bbox_inches='tight',pil_kwargs={'optimize':True})
    unique=list(dict.fromkeys(Path(p) for p in inputs+[ROOT/'scripts/analyze_real_gmae.py',ROOT/'scripts/analyze_real.py']))
    record=dict(title=title,script='scripts/make_real_gmae_results.py',script_sha256=sha(__file__),
        dependency_sha256={'scripts/analyze_real_gmae.py':sha(ROOT/'scripts/analyze_real_gmae.py'),'scripts/analyze_real.py':sha(ROOT/'scripts/analyze_real.py')},
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in unique},seeds=SEEDS,test_captures=SCENARIOS,
        aggregation='Equal capture means within training seed, then five-seed mean and descriptive 95% Student-t interval unless marked paired block bootstrap. Two primary contrasts show 95% and 97.5% marginal percentile intervals on shared draws. PR curves illustrate seed 37 and at most700 display points. Calibration pools bins by support. Timing quantiles describe common-session repeats, not confidence intervals.',
        scope='Exploratory follow-up on previously inspected CTU-13 test captures. Captured telemetry with controlled botnet activity; artificial feature corruption is not observed deception. No operational COA outcomes.',
        baseline_reuse='Nine earlier baseline predictions/checkpoints and training times are imported unchanged. GPUaudit remeasures all 13 inference architectures in one follow-up session.',
        output_sha256={ext:sha(directory/f'{stem}.{ext}') for ext in ['png','pdf']})
    (directory/f'{stem}.provenance.json').write_text(json.dumps(record,indent=2)+'\n');plt.close(fig)


def precision_recall(y,q):
    order,ends=ranked(y,q);tp=np.cumsum(np.asarray(y,dtype=float)[order])[ends]
    return np.r_[0,tp/np.asarray(y).sum()],np.r_[1,tp/(ends+1)]


def heatmap(ax,matrix,labels_x,labels_y,title,color_label,fig):
    lim=max(float(np.abs(matrix).max()),.005)
    im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>.5*lim else 'black')
    ax.set_xticks(range(len(labels_x)),labels_x,rotation=25,ha='right');ax.set_yticks(range(len(labels_y)),labels_y)
    fig.colorbar(im,ax=ax,label=color_label,shrink=.8);panel(ax,title)


def gallery():
    cards=[(MAIN[0],'Clean quality and two primary contrasts','A/B show all 13 method clean capture-macro AP/Brier with descriptive seed-t intervals. C separates captures. D distinguishes95%and97.5%paired seed/time-block intervals for the two planned variant-versus-GMAE contrasts. Wider intervals target a nominal two-comparison adjustment; reused test data remains exploratory.'),
    (MAIN[1],'Captured-traffic discrimination and calibration','A-C are illustrative seed 37 precision-recall curves, with each capture\'s prevalence reference. D pools recorded clean probability bins by flow support. Repeated predictions on shared flows are not independent samples.'),
    (MAIN[2],'Artificial stress and matched controls','A shows accuracy under constructed masking/replacement. B gives macro variant-minus-GMAE effects. C isolates the matched controls where possible; CONS-P versus GMAE changes both view count and latent loss. D shows clean operating points at frozen validation thresholds.'),
    (MAIN[3],'Training and common-session inference costs','Training seconds include actual pretraining/supervision; imported baseline training times come from the prior run. Inference/VRAM are remeasured for all 13 architectures in one follow-up GPUaudit on the same 32-flow window. Wall timing includes decoded transfers, scaling, graph and prediction return; p10-p95 repeat quantiles are not confidence intervals. Epoch selections do not establish convergence.')]
    content=[]
    for stem,title,caption in cards:
        content.append(f'<section><h2>{html.escape(title)}</h2><img src="{stem}.png" alt="{html.escape(title)}"><p>{html.escape(caption)}</p><a href="{stem}.pdf">Vector PDF</a> · <a href="{stem}.provenance.json">Provenance</a></section>')
    for stem,title in zip(EXTRA,['S1: Optimization histories','S2: Paired seed effects by capture']):
        content.append(f'<section><h2>{title}</h2><img src="../../supplemental/real_gmae/{stem}.png" alt="{title}"><a href="../../supplemental/real_gmae/{stem}.pdf">Vector PDF</a></section>')
    key=''.join(f'<tr><td><code>{SHORT[m]}</code></td><td><code>{m}</code></td><td>{html.escape(LABELS[m])}</td></tr>' for m in METHODS)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GraphMAE variants on captured telemetry</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:30px;color:#182b42;background:#f4f8fb}section{background:white;padding:24px;margin:24px 0;border:1px solid #d6e2ec;border-radius:12px}img{width:100%;height:auto}a{color:#256b8a}td{padding:4px 20px 4px 0}</style><h1>Two exploratory GraphMAE variants</h1><p>Four empirical main figures and two supplementary diagnostics. Test captures were inspected in the earlier study; these results are an exploratory follow-up. Baseline predictions are imported unchanged. All new model training and inference ran on the RTX 5000 Ada GPU.</p><p><a href="../../docs/RESULTS_REAL_GMAE.md">Results and limits</a> · <a href="../../supplemental/real_gmae/Supplement_real_gmae.pdf">Three-page supplement</a> · <a href="../../docs/REAL_GMAE_VARIANTS.md">Methods and equations</a></p>'''+''.join(content)+'<section><h2>Method key</h2><table>'+key+'</table></section></html>'
    (OUT/'index.html').write_text(page)


def main():
    style();rows=load('summary.csv');curves=load('curves.csv');cal=load('calibration.csv')
    manifest=json.loads((DATA/'manifest.json').read_text());comparison=json.loads((DATA/'comparisons.json').read_text())
    gpu=json.loads((DATA/'gpu_audit.json').read_text())
    if len(rows)!=975 or set(manifest['methods'])!=set(METHODS):raise ValueError('Expected complete13method follow-up')
    if comparison['manifest_sha256']!=sha(DATA/'manifest.json') or comparison['summary_sha256']!=sha(DATA/'summary.csv'):raise ValueError('Stale comparison provenance')
    if not gpu['passed'] or gpu['manifest_sha256']!=sha(DATA/'manifest.json'):raise ValueError('Completed common-session GPUaudit required')
    if comparison['calibration_sha256']!=sha(DATA/'calibration.csv'):raise ValueError('Changed calibration source')
    clean=subset(rows,condition='clean');base=[DATA/'summary.csv',DATA/'manifest.json',DATA/'comparisons.json']
    fig,axs=plt.subplots(2,2,figsize=(11.4,8),layout='constrained')
    bars(axs[0,0],clean,'ap');panel(axs[0,0],'A  Clean capture-macro discrimination',None,'Attack average precision')
    bars(axs[0,1],clean,'brier');panel(axs[0,1],'B  Clean probability accuracy',None,'Brier score (lower is better)')
    shown=['GMAE','DENSE']+VARIANTS;width=.18
    for i,m in enumerate(shown):
        stats=[estimate(subset(clean,method=m,scenario=float(c)),'ap') for c in SCENARIOS]
        axs[1,0].bar(np.arange(3)+(i-1.5)*width,[x[0] for x in stats],yerr=[x[1] for x in stats],width=width,color=COLORS[m],capsize=2,label=SHORT[m])
    axs[1,0].set_xticks(range(3),['10: Rbot','11: Rbot','12: NSIS.ay']);axs[1,0].legend(ncol=2,loc='lower left')
    panel(axs[1,0],'C  Effects differ across fixed captures','Held-out capture','Attack average precision')
    for i,m in enumerate(VARIANTS):
        p=comparison['primary'][m];lo,hi=p['bonferroni_97_5_ci'];inner=p['percentile_95_ci']
        axs[1,1].hlines(i,lo,hi,color=COLORS[m],lw=1.5,label='97.5% interval' if i==0 else None)
        axs[1,1].hlines(i,inner[0],inner[1],color=COLORS[m],lw=5,alpha=.5,label='95% interval' if i==0 else None)
        axs[1,1].vlines([lo,hi],i-.07,i+.07,color=COLORS[m]);axs[1,1].plot(p['estimate'],i,'o',color=COLORS[m],ms=7)
        axs[1,1].scatter(list(p['seed_effects'].values()),i+np.linspace(-.1,.1,5),s=18,color=COLORS[m],alpha=.55)
    axs[1,1].set_yticks(range(2),['EDGE - GMAE','CONS - GMAE']);axs[1,1].invert_yaxis();axs[1,1].axvline(0,color='#555',ls='--');axs[1,1].legend(loc='center right')
    panel(axs[1,1],'D  Exploratory paired block intervals','Clean AP difference (positive favors variant)')
    save(fig,MAIN[0],base,'Clean quality and two exploratory GraphMAE contrasts')

    fig,axs=plt.subplots(2,2,figsize=(11.1,8),layout='constrained');raw_inputs=[]
    for ax,c,letter in zip(axs.flat[:3],SCENARIOS,'ABC'):
        for m in PR_FOCUS:
            path=raw_path(37,m,'clean',c);raw_inputs.append(path)
            with np.load(path,allow_pickle=False) as a:x,y=precision_recall(a['labels'],a['prob']);prevalence=a['labels'].mean()
            ids=np.unique(np.linspace(0,len(x)-1,min(700,len(x))).astype(int));ax.plot(x[ids],y[ids],color=COLORS[m],lw=1.25,label=SHORT[m])
        ax.axhline(prevalence,color='#555',ls='--',label=f'Prevalence {prevalence:.3f}');ax.set(xlim=(0,1),ylim=(0,1.03));ax.legend(ncol=2,loc='lower left')
        panel(ax,f'{letter}  Capture {c}, seed 37','Attack recall','Attack precision')
    for m in PR_FOCUS:
        chosen=subset(cal,method=m,condition='clean');xx=[];yy=[]
        for b in range(10):
            br=[r for r in subset(chosen,bin=float(b)) if r['count']>0];n=sum(r['count'] for r in br)
            if n:xx.append(sum(r['confidence']*r['count'] for r in br)/n);yy.append(sum(r['accuracy']*r['count'] for r in br)/n)
        axs[1,1].plot(xx,yy,color=COLORS[m],marker=MARKERS[m],label=SHORT[m],ms=4)
    axs[1,1].plot([0,1],[0,1],color='#555',ls='--');axs[1,1].set(xlim=(0,1),ylim=(0,1));axs[1,1].legend(ncol=2)
    panel(axs[1,1],'D  Pooled clean reliability','Predicted attack probability','Recorded attack frequency')
    save(fig,MAIN[1],base+[DATA/'calibration.csv',DATA/'calibration_audit.json']+raw_inputs,'Capture-specific PR curves and reliability')

    fig,axs=plt.subplots(2,2,figsize=(11.4,8),layout='constrained')
    for m in FOCUS:
        stats=[estimate(subset(rows,method=m,condition=c),'ap') for c in CONDITIONS]
        axs[0,0].errorbar(range(5),[x[0] for x in stats],yerr=[x[1] for x in stats],color=COLORS[m],marker=MARKERS[m],ls='--' if m in ['EDGE_DENSE','CONS_PRE'] else '-',label=SHORT[m],capsize=2)
    axs[0,0].set_xticks(range(5),[COND_LABEL[c] for c in CONDITIONS],rotation=25,ha='right');axs[0,0].legend(ncol=3,loc='best')
    panel(axs[0,0],'A  Artificial feature-corruption stress',None,'Capture-macro attack AP')
    matrix=np.array([[paired(subset(rows,condition=c),m,'GMAE').mean() for c in CONDITIONS] for m in VARIANTS])
    heatmap(axs[0,1],matrix,[COND_LABEL[c] for c in CONDITIONS],[SHORT[m]+' - GMAE' for m in VARIANTS],'B  Macro variant effects','Positive favors variant',fig)
    pairs=[('EDGE_GMAE','EDGE_DENSE','EDGE pretraining'),('CONS_GMAE','CONS_PRE','CONS supervised term'),('CONS_PRE','GMAE','CONS two-view pretrain')]
    for i,(a,b,label) in enumerate(pairs):
        means=[];errors=[]
        for c in CONDITIONS:
            v=paired(subset(rows,condition=c),a,b);means.append(v.mean());errors.append(2.776445*v.std(ddof=1)/np.sqrt(5))
        axs[1,0].errorbar(np.arange(5)+(i-1)*.05,means,yerr=errors,color=COLORS[a],marker=['o','s','^'][i],capsize=2,label=label)
    axs[1,0].axhline(0,color='#555',ls='--');axs[1,0].set_xticks(range(5),[COND_LABEL[c] for c in CONDITIONS],rotation=25,ha='right');axs[1,0].legend(loc='best')
    panel(axs[1,0],'C  Matched controls (descriptive seed t)',None,'AP difference for added component')
    for m in ['NB','DAE']+FOCUS:
        x,xe=estimate(subset(clean,method=m),'fpr');y,ye=estimate(subset(clean,method=m),'recall')
        axs[1,1].errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
    axs[1,1].legend(ncol=3,loc='best');panel(axs[1,1],'D  Validation-selected operating points','Clean false-positive rate','Clean attack recall')
    save(fig,MAIN[2],base,'Artificial stress, controls and frozen-threshold operating points')

    timing={r['method']:r for r in gpu['timing_rows']}
    if set(timing)!=set(METHODS):raise ValueError('Common-session timing requires all13models')
    records=manifest['checkpoints'];fig,axs=plt.subplots(2,2,figsize=(11.4,8),layout='constrained')
    methods=['DENSE','GMAE']+NEW;pre=[];sup=[];err=[]
    for m in methods:
        chosen=[r for r in records if r['method']==m]
        pre.append(np.mean([r['pretrain_seconds'] for r in chosen]));sup.append(np.mean([r['supervised_seconds'] for r in chosen]));tot=np.array([r['pretrain_seconds']+r['supervised_seconds'] for r in chosen]);err.append(2.776445*tot.std(ddof=1)/np.sqrt(5))
    axs[0,0].bar(range(len(methods)),sup,color=[COLORS[m] for m in methods],label='Supervised')
    axs[0,0].bar(range(len(methods)),pre,bottom=sup,color=[COLORS[m] for m in methods],hatch='///',alpha=.6,label='Pretraining')
    axs[0,0].errorbar(range(len(methods)),np.array(pre)+sup,yerr=err,fmt='none',ecolor='#333',capsize=3)
    axs[0,0].set_xticks(range(len(methods)),[SHORT[m] for m in methods],rotation=30,ha='right');axs[0,0].legend()
    panel(axs[0,0],'A  Actual training effort','Same graph parameter count; unequal forward passes','Pretraining + supervision seconds')
    for m in ['NB','DAE']+FOCUS:
        t=timing[m];y,ye=estimate(subset(clean,method=m),'ap')
        axs[0,1].errorbar(t['wall_median_ms'],y,xerr=np.array([[t['wall_median_ms']-t['wall_p10_ms']],[t['wall_p95_ms']-t['wall_median_ms']]]),yerr=ye,fmt=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
    axs[0,1].legend(ncol=2);panel(axs[0,1],'B  Common-session wall cost and quality','Prepared host-to-GPU-to-host ms (p10-p95)','Clean macro attack AP')
    for m in ['NB','DAE']+FOCUS:
        t=timing[m];axs[1,0].scatter(t['cuda_median_ms'],t['peak_vram_mb'],marker=MARKERS[m],color=COLORS[m],label=SHORT[m],s=45)
    axs[1,0].legend(ncol=2);panel(axs[1,0],'C  Warm GPU inference allocation','Common-session CUDA median ms','Incremental peak VRAM MiB')
    selected=np.array([[next(r['best_epoch'] for r in records if r['method']==m and int(r['seed'])==s)+1 for s in SEEDS] for m in NEW])
    im=axs[1,1].imshow(selected,cmap='Blues',vmin=1,vmax=40,aspect='auto');axs[1,1].grid(False)
    for i in range(len(NEW)):
        for j in range(5):axs[1,1].text(j,i,str(selected[i,j]),ha='center',va='center',color='white' if selected[i,j]>25 else 'black',fontsize=8)
    axs[1,1].set_xticks(range(5),SEEDS);axs[1,1].set_yticks(range(4),[SHORT[m] for m in NEW]);fig.colorbar(im,ax=axs[1,1],label='Selected epoch (1-based)',shrink=.8)
    panel(axs[1,1],'D  New-model validation checkpoints','Training seed')
    save(fig,MAIN[3],base+[DATA/'gpu_audit.json'],'Measured training effort and common-session inference resources')

    fig,axs=plt.subplots(1,2,figsize=(8.9,3.2),layout='constrained')
    for ax,split,letter in zip(axs,['train','validation'],'AB'):
        for m in FOCUS:
            chosen=subset(curves,method=m,split='train_base' if split=='train' and m in NEW else split);epochs=sorted({r['epoch'] for r in chosen})
            values=[estimate(subset(chosen,epoch=e),'loss')[0] for e in epochs]
            ax.plot(np.array(epochs)+1,values,color=COLORS[m],label=SHORT[m],ls='--' if m in ['EDGE_DENSE','CONS_PRE'] else '-',lw=1.2)
        panel(ax,f'{letter}  {split.capitalize()} histories','Epoch (1-based)','Common training BCE + Brier' if split=='train' else 'Common validation BCE + Brier')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.06),ncol=6,fontsize=7)
    fig.suptitle('Available-seed means; common base loss excludes the CONS regularizer',fontsize=8)
    save(fig,EXTRA[0],[DATA/'curves.csv',DATA/'manifest.json'],'Comparable base-loss optimization histories',True)
    fig,axs=plt.subplots(1,2,figsize=(8.9,3.3),layout='constrained')
    for ax,m,letter in zip(axs,VARIANTS,'AB'):
        matrix=np.array([paired(subset(clean,scenario=float(c)),m,'GMAE') for c in SCENARIOS])
        heatmap(ax,matrix,[str(s) for s in SEEDS],[f'Capture {c}' for c in SCENARIOS],f'{letter}  {SHORT[m]} - GMAE clean AP','Positive favors variant',fig)
        ax.set_xlabel('Training seed')
    save(fig,EXTRA[1],base,'Per-seed paired effects on fixed captures',True)
    gallery();print('Created four main GraphMAE follow-up figures and two extra diagnostics.')


if __name__=='__main__':main()
