"""Publication figures from measured result tables; no synthetic result stand-ins."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
OUT=ROOT/'results/figures';OUT.mkdir(parents=True,exist_ok=True)
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
NAMES={'P':'Sequential risk','B3_fixed':'Fixed pair','B9_entropy':'Entropy heuristic','B7_class_only':'Class-only policy','A1_one_shot':'One-shot policy','B5_all':'All four tools','B8_deterministic':'Deterministic report','B6_planner':'Prompt planner'}

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=300);plt.close(fig)

def pipeline():
    fig,ax=plt.subplots(figsize=(7.3,2.3));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    boxes=[(.1,1.6,1.7,.9,'5 s audio\nSwin classifier\nClass prior'),(2.25,1.6,2,.9,'Observed state\nRisk ensemble\n(3 small MLPs)'),(4.7,1.6,1.9,.9,'Choose / stop\nAt most 2 tools'),(7.1,1.6,2.7,.9,'Frozen Omni\nOne terminal call\n5 slots + interpretation'),
        (2.5,.1,3.8,.8,'A Spectrum · B Modulation\nC Energy · D Retrieval'),(7.1,.1,2.7,.8,'Score against separate\nnumeric proxies and\nblinded AI assessments')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',facecolor='#F1F6FA',edgecolor='#40576B',linewidth=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
    for start,end in [((1.85,2.05),(2.2,2.05)),((4.3,2.05),(4.65,2.05)),((6.65,2.05),(7.05,2.05)),((5.65,1.55),(5.65,.95)),((3.25,.95),(3.25,1.55)),((8.45,1.55),(8.45,.95))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,color='#40576B'))
    ax.text(5,.01,'Numerical correctness and citation fidelity are checked separately. No human reference labels.',ha='center',va='top',fontsize=8)
    ax.plot([.95,.95,8.45],[2.55,2.85,2.85],color='#40576B',lw=.8)
    ax.add_patch(FancyArrowPatch((8.45,2.85),(8.45,2.55),arrowstyle='-|>',mutation_scale=8,color='#40576B'))
    ax.text(4.7,2.87,'Audio + prior',ha='center',va='bottom',fontsize=7.5)
    save(fig,'pipeline')

def main():
    result=json.loads((ROOT/'results/tables/main_results.json').read_text());curves=json.loads((ROOT/'results/tables/risk_coverage_curves.json').read_text());base=result['locked_comparator'];methods=['P',base,'B7_class_only','A1_one_shot','B8_deterministic','B5_all']
    pipeline();fig,axes=plt.subplots(1,2,figsize=(7.3,3.0))
    for color,method in zip(COLORS,methods):
        c=curves[method];axes[0].plot([v['coverage'] for v in c],[v['supported_coverage'] for v in c],label=NAMES.get(method,method),color=color,lw=1.6)
        axes[1].plot([v['coverage'] for v in c],[np.nan if v['unsupported_rate'] is None else v['unsupported_rate'] for v in c],color=color,lw=1.6)
        point=result['methods'][method]
        if point['supported_coverage'] is not None:axes[0].scatter(point['report_coverage'],point['supported_coverage'],color=color,s=22,zorder=5)
        if point['unsupported_rate'] is not None:axes[1].scatter(point['report_coverage'],point['unsupported_rate'],color=color,s=22,zorder=5)
        for ax,metric in zip(axes,('supported_coverage','unsupported_rate')):
            if point.get(metric+'_ci95') is not None:ax.vlines(point['report_coverage'],*point[metric+'_ci95'],color=color,lw=1,zorder=4)
    for ax in axes:ax.set(xlim=(.09,1.01),ylim=(-.02,1.02),xlabel='Fraction of reports retained');ax.grid(alpha=.2)
    axes[0].set_ylabel('Proxy supported physical coverage');axes[1].set_ylabel('Wrong / unsupported assertion rate')
    axes[0].legend(fontsize=7,loc='best');fig.tight_layout();save(fig,'risk_coverage')
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.9));budgets=result['tool_budget_sensitivity']
    for ax,xfield,yfield in [(axes[0],'mean_tools_all','supported_coverage'),(axes[1],'mean_evidence_tokens_all','whole_test_useful_coverage')]:
        values=[budgets[str(b)] for b in (1,2,3)]
        ax.plot([v[xfield] for v in values],[np.nan if v[yfield] is None else v[yfield] for v in values],marker='o',color=COLORS[0],label='Sequential; budgets 1, 2, 3')
        locations={}
        for budget,value in zip((1,2,3),values):
            if value[yfield] is not None:locations.setdefault((value[xfield],value[yfield]),[]).append(budget)
        for point,labels in locations.items():
            ax.annotate(', '.join(map(str,labels)),point,xytext=(-5,-9) if labels==[1] else (6,7),textcoords='offset points',fontsize=8)
        for value in values:
            if value.get(yfield+'_ci95') is not None:ax.vlines(value[xfield],*value[yfield+'_ci95'],color=COLORS[0],lw=1)
        for color,method in zip([COLORS[1],COLORS[5],COLORS[4]],(base,'B5_all','B8_deterministic')):
            value=result['methods'][method]
            if value[yfield] is None:continue
            ax.scatter(value[xfield],value[yfield],marker='s',color=color,label=NAMES[method],s=25)
            if value.get(yfield+'_ci95') is not None:ax.vlines(value[xfield],*value[yfield+'_ci95'],color=color,lw=1)
        ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
    axes[0].set(xlabel='Mean tool calls',ylabel='Supported coverage\n(retained reports)')
    axes[1].set(xlabel='Mean prompt evidence tokens',ylabel='Useful coverage\n(all requested slots)')
    axes[0].legend(fontsize=6.7,loc='best');fig.tight_layout();save(fig,'evidence_cost')
    fig,ax=plt.subplots(figsize=(7.3,2.6));names=[base,'B7_class_only','A1_one_shot'];d=result['primary_contrasts'];y=np.arange(3)
    for i,name in enumerate(names):
        if d[name]['difference'] is None:
            ax.text(0,i,'Not estimable: no retained reports',va='center');continue
        point=d[name]['difference']*100;lo,hi=np.array(d[name]['ci95'])*100
        ax.plot([lo,hi],[i,i],color=COLORS[i]);ax.scatter([point],[i],color=COLORS[i],s=24)
        ax.text(max(hi,point)+.6,i,'Holm p={:.3f}'.format(d[name]['p_holm']),va='center',fontsize=8,zorder=5,bbox={'facecolor':'white','edgecolor':'none','pad':1})
    ax.axvline(0,color='.4',lw=1);ax.set(yticks=y,yticklabels=['P − '+NAMES[n] for n in names],xlabel='Supported-coverage difference (percentage points)');ax.invert_yaxis();ax.margins(x=.35);ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'primary_contrasts')
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8));matrix=np.asarray(result['classifier_all_test']['17']['confusion_matrix']);normal=matrix/matrix.sum(1,keepdims=True)
    im=axes[0].imshow(normal,vmin=0,vmax=1,cmap='Blues')
    for i in range(5):
        for j in range(5):axes[0].text(j,i,f'{normal[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if normal[i,j]>.6 else 'black')
    short=['Bkg','Cargo','Pass','Tank','Tug'];axes[0].set(xticks=range(5),yticks=range(5),xticklabels=short,yticklabels=short,xlabel='Predicted class',ylabel='Dataset class',title='Swin seed 17; all eligible test clips')
    raw=json.loads((ROOT/'results/raw/scored_test_records.json').read_text())['P'];counts={mask:sum(r['mask']==mask for r in raw) for mask in range(16)};present=[m for m,n in counts.items() if n]
    axes[1].bar(range(len(present)),[counts[m] for m in present],color=COLORS[0]);axes[1].set(xticks=range(len(present)),xticklabels=[''.join(f for j,f in enumerate('ABCD') if m&(1<<j)) or 'None' for m in present],ylabel='Test reports',xlabel='Acquired tool set',title='Sequential policy decisions');fig.tight_layout();save(fig,'classifier_and_tools')
    robustness=ROOT/'results/tables/robustness_results.json'
    if robustness.exists():
        stress=json.loads(robustness.read_text());conditions=['noise_20','noise_10','noise_0','lowpass_2000','gain_025','clipping'];methods=['P',base,'B5_all','B8_deterministic']
        assert set(conditions)==set(stress['conditions']), 'Stress figure labels must match condition keys'
        fig,axes=plt.subplots(1,2,figsize=(7.3,3.0));x=np.arange(len(conditions));width=.19
        for j,method in enumerate(methods):
            color=[COLORS[0],COLORS[1],COLORS[5],COLORS[4]][j]
            values=[stress['conditions'][c]['methods'][method]['supported_coverage'] for c in conditions]
            values=[np.nan if v is None else v for v in values]
            axes[0].bar(x+(j-1.5)*width,values,width,color=color,label=NAMES[method])
            axes[1].plot(x,[stress['conditions'][c]['methods'][method]['report_coverage'] for c in conditions],marker='o',ms=3,color=color)
        for ax in axes:ax.set(xticks=x,xticklabels=['Noise\n20 dB','Noise\n10 dB','Noise\n0 dB','LP\n2 kHz','Gain\n0.25','Clip'],xlim=(-.6,len(conditions)-.4),ylim=(0,1.02));ax.grid(axis='y',alpha=.2)
        for i,condition in enumerate(conditions):
            if all(stress['conditions'][condition]['methods'][m]['supported_coverage'] is None for m in methods):
                axes[0].text(i,.22,'No reports\nretained\n(support NA)',ha='center',va='center',fontsize=6.5)
        axes[0].set_ylabel('Proxy supported coverage');axes[1].set_ylabel('Realized report coverage');axes[0].legend(fontsize=7,ncol=2,loc='upper center');fig.tight_layout();save(fig,'robustness')
    print('Figures saved from measured tables',flush=True)

if __name__=='__main__':main()
