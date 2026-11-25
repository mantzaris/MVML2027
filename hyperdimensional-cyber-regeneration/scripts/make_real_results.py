#!/usr/bin/env python3
"""Four empirical CTU-13 figures plus two supplemental diagnostics."""
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
from analyze_real import METHODS, LABELS, CONDITIONS, SCENARIOS, SEEDS, raw_path, ranked

DATA=ROOT/'results/real';OUT=ROOT/'figures/real';SUPP=ROOT/'supplemental/real'
NEW=[m for m in METHODS if m!='NB']
FOCUS=['LOGIT','DAE','GMAE','HDC','CTX_FLOW','SCR_FLOW']
COLORS=dict(zip(METHODS,['#7b8794','#967149','#9365b8','#5896ac','#0072b2','#009e73','#cc3366','#d55e00','#d9a114']))
MARKERS=dict(zip(METHODS,['x','P','D','s','s','o','o','^','v']))
COND_LABEL={'clean':'Recorded','missing20':'20% missing','missing50':'50% missing','spoof20':'20% replaced','spoof40':'40% replaced'}
SHORT={'NB':'NB','LOGIT':'LOGIT','DAE':'DAE','DENSE':'DENSE','GMAE':'GMAE','HDC':'HDC','SCR_FLOW':'SCR','CTX_FLOW':'CTX','LOCAL_FLOW':'LOCAL'}
MAIN=['real_fig01_external_quality','real_fig02_capture_calibration','real_fig03_corruption_transfer','real_fig04_resource_cost']
EXTRA=['real_figS1_training','real_figS2_seed_operating_points']


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
    for seed in SEEDS:
        values=[r[metric] for r in rows if int(r['seed'])==seed and np.isfinite(r.get(metric,float('nan')))]
        if values:out[seed]=float(np.mean(values))
    return out


def estimate(rows,metric):
    values=np.array(list(seed_values(rows,metric).values()))
    if not len(values):return np.nan,np.nan
    return values.mean(),(2.776445*values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else 0.)


def paired(rows,first,second,metric='ap'):
    a=seed_values(subset(rows,method=first),metric);b=seed_values(subset(rows,method=second),metric)
    return np.array([a[s]-b[s] for s in SEEDS])


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.titleweight':'bold',
        'axes.labelsize':9,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.15,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})


def panel(ax,title,xlabel=None,ylabel=None):
    ax.set_title(title,loc='left',pad=10)
    if xlabel:ax.set_xlabel(xlabel)
    if ylabel:ax.set_ylabel(ylabel)


def bars(ax,rows,metric,methods=METHODS):
    stats=[estimate(subset(rows,method=m),metric) for m in methods]
    ax.bar(np.arange(len(methods)),[x[0] for x in stats],yerr=[x[1] for x in stats],color=[COLORS[m] for m in methods],capsize=3)
    ax.set_xticks(np.arange(len(methods)),[SHORT[m] for m in methods],rotation=32,ha='right',fontsize=8)


def save(fig,stem,inputs,title,supplement=False):
    directory=SUPP if supplement else OUT;directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/f'{stem}.pdf',bbox_inches='tight')
    fig.savefig(directory/f'{stem}.png',dpi=280,bbox_inches='tight',pil_kwargs={'optimize':True})
    metadata={'title':title,'script':'scripts/make_real_results.py','script_sha256':sha(__file__),
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'inputs':{str(Path(p).relative_to(ROOT)):sha(p) for p in inputs},'seeds':SEEDS,'test_captures':SCENARIOS,
        'aggregation':'Equal capture means within each training seed; mean and Student-t 95% interval across five seeds unless explicitly marked primary block-bootstrap. Raw PR curves illustrate seed37, not independent-seed averaging; at most700 score-threshold points are displayed. Calibration pools recorded flows by bin support.',
        'scope':'Real captured CTU-13 traffic with controlled botnet executions. Artificial telemetry missingness/replacement stress is not observed deception. No real course-of-action outcomes are supplied.',
        'uncertainty_limit':'Three test captures are fixed. Seed intervals condition on their recorded flow pool; the primary crossed temporal-block interval is recorded separately.',
        'output_sha256':{ext:sha(directory/f'{stem}.{ext}') for ext in ['pdf','png']}}
    (directory/f'{stem}.provenance.json').write_text(json.dumps(metadata,indent=2)+'\n');plt.close(fig)


def precision_recall(labels,prob):
    order,ends=ranked(labels,prob);truth=np.asarray(labels,dtype=float)
    tp=np.cumsum(truth[order])[ends]
    return np.r_[0.,tp/truth.sum()],np.r_[1.,tp/(ends+1)]


def gallery():
    cards=[
        (MAIN[0],'External classification quality','Clean recorded telemetry only. A/B give equal-weight capture-macro scores; C separates the three future captures. D distinguishes the primary seed/time-block interval from descriptive seed-t contrasts. IDs map to the method table below.'),
        (MAIN[1],'Capture-specific precision and calibration','A-C show seed37 precision-recall curves on the three held-out captures, each with its own attack-prevalence reference. These curves are illustrative, not a confidence interval. D pools clean calibration bins across fixed captures and training seeds; repeated labels are not independent samples.'),
        (MAIN[2],'Artificial telemetry stress and transfer','Missingness and benign-centroid feature replacement are deliberately applied stresses on real captured records. A/B use attack AP. C shows validation-threshold recall/FPR, without selecting thresholds on the test captures. D relates clean accuracy to warmed batch-one-window CUDA inference time.'),
        (MAIN[3],'Measured resource and optimization costs','Actual new-model trainable parameter counts, training wall time, incremental GPU allocation and checkpoint selections. Retained pretraining decoders contribute to checkpoint counts but are inactive during inference. Epoch choices do not prove convergence.'),
    ]
    content=[]
    for stem,title,caption in cards:
        content.append(f'<section><h2>{html.escape(title)}</h2><img src="{stem}.png" alt="{html.escape(title)}"><p>{html.escape(caption)}</p><a href="{stem}.pdf">Vector PDF</a> · <a href="{stem}.provenance.json">Provenance</a></section>')
    for stem,title in zip(EXTRA,['S1: Training and validation objectives','S2: Seed effects and threshold choices']):
        content.append(f'<section><h2>{title}</h2><img src="../../supplemental/real/{stem}.png" alt="{title}"><a href="../../supplemental/real/{stem}.pdf">Vector PDF</a></section>')
    legend=''.join(f'<tr><td><code>{m}</code></td><td>{html.escape(LABELS[m])}</td></tr>' for m in METHODS)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CTU-13 external study</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:auto;padding:30px;color:#182b42;background:#f4f8fb}section{background:white;padding:24px;margin:24px 0;border:1px solid #d6e2ec;border-radius:12px}img{width:100%;height:auto}a{color:#256b8a}td{padding:4px 20px 4px 0}</style><h1>External application to captured CTU-13 traffic</h1><p>Four empirical main figures and two supplementary diagnostics. Five training seeds share the same three fixed held-out captures. Error bars are conditional seed-t intervals except the explicitly marked primary block-bootstrap interval.</p><p><a href="../../docs/RESULTS_REAL.md">Results and limits</a> · <a href="../../supplemental/real/Supplement_real.pdf">Three-page supplement</a> · <a href="https://www.stratosphereips.org/datasets-ctu13">Original dataset</a></p>'''+''.join(content)+'<section><h2>Method key</h2><table>'+legend+'</table></section></html>'
    (OUT/'index.html').write_text(page)


def main():
    style();rows=load('summary.csv');curves=load('curves.csv');cal=load('calibration.csv')
    manifest=json.loads((DATA/'manifest.json').read_text());comparison=json.loads((DATA/'comparisons.json').read_text())
    if len(rows)!=675 or set(manifest['methods'])!=set(METHODS):raise ValueError('Complete real-data study required')
    if comparison['manifest_sha256']!=sha(DATA/'manifest.json') or comparison['summary_sha256']!=sha(DATA/'summary.csv'):raise ValueError('Stale comparison metadata')
    clean=subset(rows,condition='clean');base=[DATA/'summary.csv',DATA/'manifest.json']
    fig,axs=plt.subplots(2,2,figsize=(10.8,7.8),layout='constrained')
    bars(axs[0,0],clean,'ap');panel(axs[0,0],'A  Clean capture-macro discrimination','Adapted method','Attack average precision')
    bars(axs[0,1],clean,'brier');panel(axs[0,1],'B  Clean probability accuracy','Adapted method','Brier score (lower is better)')
    width=.13
    for i,m in enumerate(FOCUS):
        stats=[estimate(subset(clean,method=m,scenario=float(s)),'ap') for s in SCENARIOS]
        axs[1,0].bar(np.arange(3)+(i-2.5)*width,[v for v,e in stats],yerr=[e for v,e in stats],width=width,color=COLORS[m],label=SHORT[m],capsize=2)
    axs[1,0].set_xticks(range(3),['10: Rbot','11: Rbot','12: NSIS.ay']);axs[1,0].legend(ncol=3,loc='lower left')
    panel(axs[1,0],'C  Future captures and malware shift','Fixed held-out capture','Attack average precision')
    comps=['GMAE','CTX_FLOW','LOCAL_FLOW','DAE','LOGIT']
    for i,m in enumerate(comps):
        values=paired(clean,'SCR_FLOW',m);point=values.mean()
        if m=='GMAE':lo,hi=comparison['primary']['percentile_95_ci'];color='#162d43'
        else:
            error=2.776445*values.std(ddof=1)/np.sqrt(5);lo,hi=point-error,point+error;color=COLORS[m]
        axs[1,1].hlines(i,lo,hi,color=color);axs[1,1].vlines([lo,hi],i-.06,i+.06,color=color);axs[1,1].plot(point,i,'o',color=color)
        axs[1,1].scatter(values,i+np.linspace(-.09,.09,5),s=13,color=color,alpha=.55)
    axs[1,1].axvline(0,color='#666',ls='--');axs[1,1].set_yticks(range(5),['SCR - GMAE (block CI)','SCR - CTX (seed t)','SCR - LOCAL (seed t)','SCR - DAE (seed t)','SCR - LOGIT (seed t)']);axs[1,1].invert_yaxis()
    panel(axs[1,1],'D  Primary and descriptive contrasts','AP difference (positive favors SCR)')
    save(fig,MAIN[0],base+[DATA/'comparisons.json'],'External discrimination, calibration loss and paired model contrasts')

    fig,axs=plt.subplots(2,2,figsize=(10.8,7.8),layout='constrained');raw_inputs=[]
    for ax,scenario,letter in zip(axs.flat[:3],SCENARIOS,'ABC'):
        prevalence=None
        for m in FOCUS:
            path=raw_path(37,m,'clean',scenario);raw_inputs.append(path)
            with np.load(path,allow_pickle=False) as a:
                xx,yy=precision_recall(a['labels'],a['prob']);prevalence=a['labels'].mean()
            # Keep endpoints and a uniform index sample only for graphical size.
            ids=np.unique(np.linspace(0,len(xx)-1,min(700,len(xx))).astype(int))
            ax.plot(xx[ids],yy[ids],color=COLORS[m],label=SHORT[m],lw=1.3)
        ax.axhline(prevalence,color='#555',ls='--',label=f'Prevalence {prevalence:.3f}')
        ax.set(xlim=(0,1),ylim=(0,1.03));ax.legend(ncol=2,loc='lower left')
        panel(ax,f'{letter}  Capture {scenario}, seed 37','Attack recall','Attack precision')
    for m in FOCUS:
        mr=subset(cal,method=m,condition='clean');xx=[];yy=[]
        for bin_id in sorted({r['bin'] for r in mr}):
            br=[r for r in subset(mr,bin=bin_id) if r['count']>0];count=sum(r['count'] for r in br)
            if count:
                xx.append(sum(r['confidence']*r['count'] for r in br)/count);yy.append(sum(r['accuracy']*r['count'] for r in br)/count)
        axs[1,1].plot(xx,yy,marker=MARKERS[m],color=COLORS[m],label=SHORT[m],ms=4)
    axs[1,1].plot([0,1],[0,1],color='#666',ls='--');axs[1,1].set(xlim=(0,1),ylim=(0,1));axs[1,1].legend(ncol=2)
    panel(axs[1,1],'D  Pooled clean reliability','Predicted attack probability','Recorded attack frequency')
    save(fig,MAIN[1],base+[DATA/'calibration.csv']+raw_inputs,'Per-capture precision-recall and pooled reliability')

    fig,axs=plt.subplots(2,2,figsize=(10.8,7.8),layout='constrained')
    for m in FOCUS:
        stats=[estimate(subset(rows,method=m,condition=c),'ap') for c in CONDITIONS]
        axs[0,0].errorbar(range(5),[v for v,e in stats],yerr=[e for v,e in stats],marker=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
    axs[0,0].set_xticks(range(5),[COND_LABEL[c] for c in CONDITIONS],rotation=25,ha='right');axs[0,0].legend(ncol=3,loc='lower left')
    panel(axs[0,0],'A  Artificial stress on real observations',None,'Capture-macro attack AP')
    matrix=np.array([[paired(subset(rows,condition=c,scenario=float(s)),'SCR_FLOW','GMAE').mean() for c in CONDITIONS] for s in SCENARIOS]);lim=max(abs(matrix).max(),.005)
    im=axs[0,1].imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');axs[0,1].grid(False)
    for i in range(3):
        for j in range(5):axs[0,1].text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>.5*lim else 'black')
    axs[0,1].set_xticks(range(5),[COND_LABEL[c] for c in CONDITIONS],rotation=25,ha='right');axs[0,1].set_yticks(range(3),[f'Capture {s}' for s in SCENARIOS]);fig.colorbar(im,ax=axs[0,1],label='Positive favors SCR',shrink=.8)
    panel(axs[0,1],'B  Capture-specific SCR - GMAE AP')
    for m in METHODS:
        x,xe=estimate(subset(clean,method=m),'fpr');y,ye=estimate(subset(clean,method=m),'recall')
        axs[1,0].errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
        x,xe=estimate(subset(clean,method=m),'latency_ms');y,ye=estimate(subset(clean,method=m),'ap')
        axs[1,1].errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
    axs[1,0].legend(ncol=3,loc='best');axs[1,1].legend(ncol=3,loc='best')
    panel(axs[1,0],'C  Validation-selected operating points','Clean false-positive rate','Clean attack recall')
    panel(axs[1,1],'D  Accuracy and warm GPU latency','One 32-flow window (ms)','Clean capture-macro attack AP')
    save(fig,MAIN[2],base,'Artificial telemetry stresses and conditional operating-cost tradeoffs')

    fig,axs=plt.subplots(2,2,figsize=(10.8,7.8),layout='constrained');records=manifest['checkpoints']
    for m in NEW:
        selected=[r for r in records if r['method']==m]
        if len(selected)!=5:raise ValueError(f'Expected five new checkpoints for {m}')
        parameters=np.mean([r['parameters'] for r in selected]);times=np.array([r['train_seconds'] for r in selected]);error=2.776445*times.std(ddof=1)/np.sqrt(5)
        axs[0,0].errorbar(parameters,times.mean(),yerr=error,fmt=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
        score,err=estimate(subset(clean,method=m),'ap');axs[0,1].errorbar(parameters,score,yerr=err,fmt=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
    axs[0,0].set_xscale('log');axs[0,1].set_xscale('log');axs[0,0].legend(ncol=2);axs[0,1].legend(ncol=2)
    panel(axs[0,0],'A  Actual new-model training cost','Trainable parameters (pretraining decoder included)','Pretraining + supervision wall time (s)')
    panel(axs[0,1],'B  Capacity and external discrimination','Trainable parameters (pretraining decoder included)','Clean capture-macro attack AP')
    for m in METHODS:
        x,xe=estimate(subset(clean,method=m),'latency_ms');y,ye=estimate(subset(clean,method=m),'peak_vram_mb')
        axs[1,0].errorbar(x,y,xerr=xe,yerr=ye,fmt=MARKERS[m],color=COLORS[m],label=SHORT[m],capsize=2)
    axs[1,0].legend(ncol=3);panel(axs[1,0],'C  Warm inference allocation','One-window CUDA latency (ms)','Incremental GPU allocation (MiB)')
    epochs=np.array([[next(r for r in records if r['method']==m and int(r['seed'])==s).get('best_epoch',next(r for r in records if r['method']==m and int(r['seed'])==s).get('epoch'))+1 for s in SEEDS] for m in NEW])
    im=axs[1,1].imshow(epochs,cmap='Blues',vmin=1,vmax=40,aspect='auto');axs[1,1].grid(False)
    for i in range(len(NEW)):
        for j in range(5):axs[1,1].text(j,i,str(epochs[i,j]),ha='center',va='center',fontsize=8,color='white' if epochs[i,j]>25 else 'black')
    axs[1,1].set_xticks(range(5),SEEDS);axs[1,1].set_yticks(range(len(NEW)),[SHORT[m] for m in NEW]);fig.colorbar(im,ax=axs[1,1],label='Selected epoch (1-based)',shrink=.8)
    panel(axs[1,1],'D  Validation checkpoint selections','Training seed')
    save(fig,MAIN[3],base,'Measured capacity, optimization and GPU-resource comparisons')

    fig,axs=plt.subplots(1,2,figsize=(8.7,3.2),layout='constrained')
    for ax,split,letter in zip(axs,['train','validation'],'AB'):
        for m in NEW:
            chosen=subset(curves,method=m,split=split);epochs=sorted({r['epoch'] for r in chosen})
            values=[estimate(subset(chosen,epoch=e),'loss')[0] for e in epochs]
            ax.plot(np.array(epochs)+1,values,color=COLORS[m],label=SHORT[m],lw=1.2)
        panel(ax,f'{letter}  {split.capitalize()} supervised objective','Epoch (1-based)','BCE + Brier');ax.legend(ncol=3,fontsize=7)
    fig.suptitle('Means over available seeds; checkpoint records specify stopping and selection',fontsize=8)
    save(fig,EXTRA[0],[DATA/'curves.csv',DATA/'manifest.json'],'Recorded training and validation objectives',True)
    fig,axs=plt.subplots(1,2,figsize=(8.7,3.3),layout='constrained')
    effects=np.array([paired(subset(clean,scenario=float(c)),'SCR_FLOW','GMAE') for c in SCENARIOS]);lim=max(abs(effects).max(),.005)
    im=axs[0].imshow(effects,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');axs[0].grid(False)
    for i in range(3):
        for j in range(5):axs[0].text(j,i,f'{effects[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(effects[i,j])>.5*lim else 'black')
    axs[0].set_xticks(range(5),SEEDS);axs[0].set_yticks(range(3),[f'Capture {s}' for s in SCENARIOS]);fig.colorbar(im,ax=axs[0],label='Positive favors SCR',shrink=.8)
    panel(axs[0],'A  SCR - GMAE clean AP','Training seed')
    for i,m in enumerate(METHODS):
        values=[subset(clean,method=m,seed=float(s))[0]['threshold'] for s in SEEDS]
        axs[1].scatter(np.full(5,i)+np.linspace(-.12,.12,5),values,color=COLORS[m],s=22)
    axs[1].set_xticks(range(9),[SHORT[m] for m in METHODS],rotation=35,ha='right',fontsize=7)
    panel(axs[1],'B  Frozen validation thresholds',None,'Attack probability threshold')
    save(fig,EXTRA[1],base,'Paired seed/capture effects and validation threshold choices',True)
    gallery();print('Created four real-data main figures, two extras and gallery.')


if __name__=='__main__':main()
