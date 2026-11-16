"""Operator-readable static figures from the frozen identification results."""
from __future__ import annotations
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from underwater_reasoning.io import atomic_json, canonical_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/figures'
LABELS = ['Background', 'Cargo vessel', 'Passenger vessel', 'Tanker', 'Tug / tow']
NAMES = {'anchor':'Single acoustic model', 'ensemble':'Three-model ensemble', 'acoustic_refit':'Acoustic-only refit',
         'fusion':'Proposed evidence fusion', 'fusion_maxprob':'Fusion: confidence only', 'physical_only':'Measured features only',
         'no_physical':'Without measured features', 'no_prototypes':'Without exemplar evidence', 'prototype_vote':'Training-source analogies'}
COLORS = {'anchor':'#758797', 'ensemble':'#2c5a88', 'acoustic_refit':'#ab7d20', 'fusion':'#00857b'}


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=300, facecolor='white', pil_kwargs={'optimize':True})
    fig.savefig(OUT/(name+'.pdf'), facecolor='white', metadata={'Author':'Anonymous','Title':name.replace('_',' ')})
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42})
    r = json.loads((ROOT/'results/tables/identification_results.json').read_text())
    clean = r['conditions']['clean']['methods']
    comparator = r['primary_comparator']
    p, b = clean['fusion'], clean[comparator]
    n = r['fresh_clean_clips']
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={'width_ratios':[1.05, 1.25]})
    fig.subplots_adjust(left=.16, right=.98, top=.79, bottom=.24, wspace=.48)
    fig.suptitle('Identifying vessel categories in previously unused recordings', x=.5, y=.97, fontsize=16, fontweight='bold')
    fig.text(.5,.89,f"{n} five-second recordings · {r['source_groups']} new recorded source groups · dataset category labels", ha='center', color='#4c5965')
    y = np.arange(5)
    for method, offset, color in [(comparator,-.24,COLORS[comparator]),('acoustic_refit',0,COLORS['acoustic_refit']),('fusion',.24,COLORS['fusion'])]:
        values = np.array(clean[method]['per_class_recall'])*100
        bars = left.barh(y+offset, values, height=.21, color=color, label=NAMES[method])
        for bar, value in zip(bars,values):
            left.text(value+1.2,bar.get_y()+bar.get_height()/2,f'{value:.0f}%',va='center',fontsize=9)
    left.set_yticks(y, [f'{label} (n={count})' for label,count in zip(LABELS,r['class_counts'])])
    left.invert_yaxis();left.set_xlim(0,112);left.set_xticks([0,25,50,75,100]);left.set_xlabel('Correct category within each group (%)')
    left.set_title('A. Which categories are recognized?', loc='left', fontsize=11, pad=16)
    left.grid(axis='x',alpha=.15);left.set_axisbelow(True)
    left.legend(loc='upper left',bbox_to_anchor=(-.3,-.15),frameon=False,fontsize=8)
    methods = ['anchor','ensemble','acoustic_refit','fusion','fusion_maxprob']
    yy = np.arange(len(methods))
    offset = np.zeros(len(methods))
    for key,color,label in [('correct_per_100','#00857b','Correct label'),('wrong_per_100','#c95b48','Wrong label'),('referred_per_100','#dfe4e8','Referred for review')]:
        values = np.array([clean[m][key] for m in methods])
        bars = right.barh(yy,values,left=offset,color=color,height=.62,label=label)
        for bar,value,start in zip(bars,values,offset):
            if value>=7:
                right.text(start+value/2,bar.get_y()+bar.get_height()/2,f'{value:.1f}',ha='center',va='center',fontsize=9,
                           color='white' if key!='referred_per_100' else '#26333c')
        offset += values
    right.set_yticks(yy,[NAMES[m] for m in methods],fontsize=9);right.invert_yaxis()
    right.set_xlim(0,100);right.set_xlabel('Outcomes per 100 requested recordings')
    right.set_title('B. Correct labels, errors and review requests', loc='left', fontsize=11, pad=16)
    right.legend(loc='upper left',bbox_to_anchor=(-.2,-.18),frameon=False,fontsize=9,ncol=2)
    difference = r['conditions']['clean']['primary_contrast']['macro_f1']
    fig.text(.04,.045,f"All-recording accuracy: ensemble {100*clean['ensemble']['accuracy']:.1f}% · proposed fusion {100*p['accuracy']:.1f}%.  "
             f"Fusion − locked comparator macro-F1: {100*difference['difference']:+.1f} points "
             f"(95% interval {100*difference['ci95'][0]:+.1f} to {100*difference['ci95'][1]:+.1f}).",fontsize=9)
    fig.text(.04,.012,'Panel B uses separately frozen calibration thresholds; actual reporting rates differ. Review counts do not measure human time saved.',fontsize=8,color='#4c5965')
    save(fig,'identification_operator_results')

    # Native publication-width version keeps category labels readable in the paper.
    fig,ax=plt.subplots(figsize=(7.1,3.8))
    fig.subplots_adjust(left=.27,right=.92,top=.87,bottom=.25)
    for method,offset,color in [(comparator,-.24,COLORS[comparator]),('acoustic_refit',0,COLORS['acoustic_refit']),('fusion',.24,COLORS['fusion'])]:
        values=np.array(clean[method]['per_class_recall'])*100
        bars=ax.barh(y+offset,values,height=.21,color=color,label=NAMES[method])
        for bar,value in zip(bars,values):
            ax.text(value+1.3,bar.get_y()+bar.get_height()/2,f'{value:.0f}%',va='center',fontsize=8)
    ax.set_yticks(y,[f'{label} (n={count})' for label,count in zip(LABELS,r['class_counts'])],fontsize=9)
    ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Correct category within each group (%)',fontsize=10)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.012),ncol=2,frameon=False,fontsize=8)
    fig.suptitle('Which vessel categories are identified correctly?',y=.98,fontsize=12,fontweight='bold')
    save(fig,'identification_class_accuracy')

    conditions = ['clean','noise_20','noise_10','noise_0','lowpass_2000','gain_025','clipping']
    ticks = ['Original','Noise\n20 dB SNR','Noise\n10 dB SNR','Noise\n0 dB SNR','Low-pass\n2 kHz','Gain\n×0.25','Digital\nclipping']
    methods = ['anchor','ensemble','acoustic_refit','physical_only','fusion']
    values = np.array([[100*r['conditions'][c]['methods'][m]['accuracy'] for c in conditions] for m in methods])
    fig, ax = plt.subplots(figsize=(7.1,3.1))
    fig.subplots_adjust(left=.27,right=.93,top=.78,bottom=.25)
    im = ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    for i in range(len(methods)):
        for j in range(len(conditions)):
            ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]>60 else '#152c3e',fontsize=8)
    ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods],fontsize=8)
    ax.set_xticks(range(len(conditions)),ticks,fontsize=8)
    ax.tick_params(length=0)
    fig.suptitle('Category accuracy when the recording changes',y=.98,fontsize=12,fontweight='bold')
    fig.text(.5,.87,f'All {n} recordings scored in every condition; no rejection or retraining',ha='center',fontsize=9)
    bar = fig.colorbar(im,ax=ax,fraction=.035,pad=.035);bar.set_label('Correct category (%)',fontsize=8);bar.ax.tick_params(labelsize=8)
    fig.text(.03,.035,'Artificial changes; nominal category labels retained. No claim about real-world prevalence or human recognition.',fontsize=8)
    save(fig,'identification_robustness')

    # Two outcome-balanced, hash-ordered illustrations. These are explicitly post-hoc examples.
    rows = json.loads((ROOT/'data/private/identification_manifest.json').read_text())['records']
    raw = np.load(ROOT/'results/raw/identification/evaluation_predictions.npz')
    positions = {uid:i for i,uid in enumerate(raw['ids'])}
    examples = []
    for proposed_correct, comparator_correct in [(True,False),(False,True)]:
        eligible = [row for row in rows if (raw['fusion'][positions[row['id']]].argmax()==row['label_index'])==proposed_correct
                    and (raw[comparator][positions[row['id']]].argmax()==row['label_index'])==comparator_correct]
        if eligible:
            examples.append(sorted(eligible,key=lambda row:canonical_hash(['identification_example',row['id']]))[0])
    if len(examples)==2:
        fig,axes = plt.subplots(1,2,figsize=(7.1,3.5))
        fig.subplots_adjust(left=.1,right=.98,top=.64,bottom=.25,wspace=.3)
        metadata = []
        for ax,row in zip(axes,examples):
            wave,fs=sf.read(ROOT/row['audio_path'],dtype='float32')
            f,t,z=signal.stft(wave,fs,nperseg=4096,noverlap=3072,boundary=None)
            selected=(f>=20)&(f<=2000)
            power=20*np.log10(np.maximum(abs(z[selected]),1e-12));power-=power.max()
            ax.pcolormesh(t,f[selected],power,vmin=-70,vmax=0,cmap='magma',shading='auto',rasterized=True)
            ax.set_ylim(20,2000);ax.set_xlim(0,5);ax.set_xlabel('Time (s)');ax.set_ylabel('Frequency (Hz)')
            i=positions[row['id']];pp=int(raw['fusion'][i].argmax());bp=int(raw[comparator][i].argmax())
            ax.set_title(f"Dataset: {LABELS[row['label_index']]}\nFusion: {LABELS[pp]}\nEnsemble: {LABELS[bp]}",fontsize=9,pad=10)
            metadata.append({'anonymous_example_id':canonical_hash(row['id'])[:10],
                             'dataset_category':LABELS[row['label_index']], 'fusion':LABELS[pp], 'comparator':LABELS[bp],
                             'audio_sha256':file_hash(ROOT/row['audio_path'])})
        fig.suptitle('An identification gained and an identification lost',y=.98,fontsize=12,fontweight='bold')
        fig.text(.03,.07,'Hash-ordered illustrative cases. Intensity is relative to each panel’s peak (−70 to 0 dB).',fontsize=8)
        fig.text(.03,.025,'Post-hoc examples; spectral patterns alone do not prove a vessel category.',fontsize=8)
        save(fig,'identification_examples')
        atomic_json(ROOT/'results/tables/identification_examples.json',{'selection':'post-hoc opposing outcomes; fixed hash ordering within each group','examples':metadata})
    print('Identification figures written')


if __name__ == '__main__':
    main()
