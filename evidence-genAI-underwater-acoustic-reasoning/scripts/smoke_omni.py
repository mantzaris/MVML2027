"""Development-only text-output Omni benchmark, before prompt freeze."""
import argparse,json,time
from pathlib import Path
import soundfile as sf
from underwater_reasoning.omni import Omni
from underwater_reasoning.reports import parse_report
from underwater_reasoning.io import atomic_json,file_hash
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='omni_smoke_v2');parser.add_argument('--repeat',type=int,default=1);parser.add_argument('--report-schema',action='store_true');args=parser.parse_args()
    start=time.monotonic();runner=Omni(ROOT);system=(ROOT/'prompts/terminal_v1.txt').read_text();items=[]
    for wav in sorted((ROOT/'data/raw/release_probe').glob('sample_*.wav')):
        x,fs=sf.read(wav,dtype='float32');items.append({'audio':x,'sample_rate':fs,'id':wav.name,
            'text':system+'\nAvailable evidence: W0 raw audio only. Produce the required JSON report.'})
    items*=args.repeat;results=runner.generate(items,report_schema=args.report_schema)
    for item,result in zip(items,results):
        parsed,errors=parse_report(result['raw']);result.update({'id':item['id'],'parsed':parsed,'errors':errors})
    atomic_json(ROOT/f'results/raw/{args.name}.json',{'reports':results,'total_seconds':time.monotonic()-start,'prompt_sha256':file_hash(ROOT/'prompts/terminal_v1.txt')})
    structured={r['id']:[v['parsed'] for v in results if v['id']==r['id']] for r in results}
    repeat_exact=all(all(v==values[0] for v in values) for values in structured.values())
    atomic_json(ROOT/f'results/manifests/{args.name}.json',{'reports':len(results),'constrained_schema':args.report_schema,'repeat':args.repeat,'structured_repeat_exact':repeat_exact,'schema_valid':sum(not [e for e in r['errors'] if e!='interpretation_word_count'] for r in results),'format_valid':sum(not r['errors'] for r in results),'peak_vram_bytes':results[0]['peak_vram_bytes'],'batch_seconds':results[0]['batch_seconds']})
    print(json.dumps({'batch_seconds':results[0]['batch_seconds'],'batch_size':len(items),'peak_vram_gb':results[0]['peak_vram_bytes']/1e9,'errors':[r['errors'] for r in results]}),flush=True)
