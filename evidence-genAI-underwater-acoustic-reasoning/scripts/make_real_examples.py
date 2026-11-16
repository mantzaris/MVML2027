"""Two traceable illustrations from actual test recordings, selected by fixed rules.

Examples are illustrative, not a representative performance estimate. The
selection rules are written before test unlock; no example changes the study.
"""
import json,os,textwrap
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})

def main():
    if not json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise ValueError('Test locked')
    records=json.loads((ROOT/'results/raw/scored_test_records.json').read_text())['P']
    ordered=sorted(records,key=lambda r:canonical_hash([17,'illustration',r['id']]))
    success=[r for r in ordered if r['score']['supported']>=1 and r['score']['unsupported']==0 and any(c['appropriate_uncertainty'] for c in r['score']['physical'])]
    errors=[r for r in ordered if r['score']['unsupported']>0]
    first=success[0] if success else ordered[0]
    second=next((r for r in errors if r['id']!=first['id']),next(r for r in ordered if r['id']!=first['id']))
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());lookup={r['id']:r for r in manifest['records']}
    fig=plt.figure(figsize=(7.3,4.2));grid=fig.add_gridspec(2,2,height_ratios=[1.35,1],hspace=.8,wspace=.25);examples=[]
    for column,record in enumerate((first,second)):
        uid=record['id'];row=lookup[uid];mask=record['mask'];path=ROOT/f'results/raw/terminal/main/test/{uid}_{mask}.json'
        raw=json.loads(path.read_text());report=raw['result']['parsed'];reference=json.loads((ROOT/f'data/derived/references/{uid}.json').read_text())['reference']
        x,fs=sf.read(ROOT/row['audio_path'],dtype='float32')
        f,t,z=signal.stft(x,fs,nperseg=fs,noverlap=3*fs//4,boundary=None);power=np.abs(z)**2
        band=(f>=20)&(f<=2000);db=10*np.log10(np.maximum(power[band],1e-20)/max(float(power[band].max()),1e-20))
        ax=fig.add_subplot(grid[0,column]);mesh=ax.pcolormesh(t,f[band],db,vmin=-75,vmax=0,cmap='magma',shading='auto',rasterized=True)
        ax.set(xlim=(0,5),ylim=(20,2000),yscale='log',xlabel='Time (s)',ylabel='Frequency (Hz)',yticks=[20,100,500,2000])
        ax.set_yticklabels(['20','100','500','2000']);ax.set_title(('A. Supported claims with uncertainty' if success else 'A. Hash-selected example') if column==0 else ('B. Wrong / unsupported assertion' if second in errors else 'B. Hash-selected example'),fontsize=9)
        name='E'+canonical_hash(uid)[:8];tools=''.join(family for i,family in enumerate('ABCD') if mask&(1<<i)) or 'none'
        lines=[f"{name}; dataset class: {row['label']}; tools: {tools}",f"Retained: {record['retained']}; supported: {record['score']['supported']}/4; wrong/unsupported: {record['score']['unsupported']}"]
        for claim in record['score']['physical']:
            slot=claim['slot'];value=report[slot];detail=value['status'] if value['status']=='uncertain' else str(value['value'])
            if value['status']=='asserted' and slot in ('tone','modulation'):detail+=' '+str(value['frequency_hz' if slot=='tone' else 'rate_hz'])+' Hz'
            verdict='supported' if claim['supported'] else 'wrong/unsupported' if claim['asserted'] else 'ambiguous proxy' if not claim['identifiable'] else 'identifiable proxy withheld'
            lines.append(f'{slot}: {detail} ({verdict})')
        text_ax=fig.add_subplot(grid[1,column]);text_ax.axis('off');text_ax.text(0,1,'\n'.join('\n'.join(textwrap.wrap(line,62)) for line in lines),ha='left',va='top',fontsize=7.5,linespacing=1.35)
        examples.append({'example_id':name,'selection':'first seed-17 hash-ordered qualifying example','source_clip_id':uid,'waveform_sha256':raw['key']['waveform_sha256'],
            'prediction_path':str(path.relative_to(ROOT)),'prediction_sha256':file_hash(path),'selected_mask':mask,'selected_actions':record['actions'],
            'dataset_class':row['label'],'retained':record['retained'],'report':report,'score':record['score'],'independent_reference':reference})
    colorbar=fig.colorbar(mesh,cax=fig.add_axes([.41,.415,.22,.014]),orientation='horizontal',ticks=[-75,-50,-25,0])
    colorbar.ax.tick_params(labelsize=7,pad=1);colorbar.set_label('Power relative to each peak (dB)',fontsize=7,labelpad=1)
    out=ROOT/'results/figures';out.mkdir(exist_ok=True,parents=True)
    fig.savefig(out/'real_examples.pdf',bbox_inches='tight');fig.savefig(out/'real_examples.png',dpi=300,bbox_inches='tight');plt.close(fig)
    atomic_json(ROOT/'results/tables/real_examples.json',{'illustrative_not_representative':True,'positive_eligible':len(success),'error_eligible':len(errors),
        'spectrogram':'Native-rate 1 s Hann STFT, 0.25 s hop, 20-2000 Hz, per-example relative power -75 to 0 dB; not calibrated sound pressure.',
        'rules':'First hash-ordered P report with supported>=1, unsupported=0 and appropriate uncertainty; first distinct report with unsupported>0. Fall back to distinct hash-ordered records if a category is empty.',
        'examples':examples})
    print('Two real-recording illustrations written; source paths and hashes retained in compact trace table')

if __name__=='__main__':main()
