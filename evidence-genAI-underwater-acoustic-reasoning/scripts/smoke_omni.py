"""Development-only text-output Omni benchmark, before prompt freeze."""
import argparse,json,time
from pathlib import Path
import soundfile as sf
from underwater_reasoning.omni import Omni
from underwater_reasoning.reports import parse_report
from underwater_reasoning.io import atomic_json,file_hash
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='omni_smoke_v2');parser.add_argument('--repeat',type=int,default=1);args=parser.parse_args()
    start=time.monotonic();runner=Omni(ROOT);system=(ROOT/'prompts/terminal_v1.txt').read_text();items=[]
    for wav in sorted((ROOT/'data/raw/release_probe').glob('sample_*.wav')):
        x,fs=sf.read(wav,dtype='float32');items.append({'audio':x,'sample_rate':fs,'id':wav.name,
            'text':system+'\nAvailable evidence: W0 raw audio only. Produce the required JSON report.'})
    items*=args.repeat;results=runner.generate(items)
    for item,result in zip(items,results):
        parsed,errors=parse_report(result['raw']);result.update({'id':item['id'],'parsed':parsed,'errors':errors})
    atomic_json(ROOT/f'results/raw/{args.name}.json',{'reports':results,'total_seconds':time.monotonic()-start,'prompt_sha256':file_hash(ROOT/'prompts/terminal_v1.txt')})
    print(json.dumps({'batch_seconds':results[0]['batch_seconds'],'batch_size':len(items),'peak_vram_gb':results[0]['peak_vram_bytes']/1e9,'errors':[r['errors'] for r in results]}),flush=True)
