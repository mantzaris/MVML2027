"""Development-only text-output Omni benchmark, before prompt freeze."""
import json,time,os
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
from underwater_reasoning.reports import parse_report
from underwater_reasoning.io import atomic_json,file_hash
from underwater_reasoning.runtime import budget_check
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)
if __name__=='__main__':
    budget_check();path=(ROOT/'.cache/omni_path.txt').read_text().strip();started=time.monotonic()
    processor=Qwen2_5OmniProcessor.from_pretrained(path);processor.tokenizer.padding_side='left'
    model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(path,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    system=(ROOT/'prompts/terminal_v1.txt').read_text()
    xs=[];texts=[];ids=[]
    for wav in sorted((ROOT/'data/raw/release_probe').glob('sample_*.wav')):
        x,fs=sf.read(wav,dtype='float32');xs.append(resample_poly(x,1,2))
        messages=[{'role':'system','content':[{'type':'text','text':system}]},{'role':'user','content':[{'type':'audio','audio_url':str(wav)},{'type':'text','text':'Available evidence: W0 raw audio only. Produce the required JSON report.'}]}]
        texts.append(processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False));ids.append(wav.name)
    inputs=processor(text=texts,audio=xs,return_tensors='pt',padding=True,sampling_rate=16000).to(model.device)
    # Float audio features must match the BF16 audio tower; IDs remain integers.
    inputs={k:v.to(dtype=model.dtype) if v.is_floating_point() else v for k,v in inputs.items()}
    torch.cuda.synchronize();t=time.monotonic()
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=768,do_sample=False)
    torch.cuda.synchronize();elapsed=time.monotonic()-t
    raw=processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    results=[]
    for uid,text in zip(ids,raw):
        parsed,errors=parse_report(text);results.append({'id':uid,'raw':text,'parsed':parsed,'errors':errors})
    atomic_json(ROOT/'results/raw/omni_smoke.json',{'reports':results,'seconds':elapsed,'load_and_infer_seconds':time.monotonic()-started,
        'batch_size':len(xs),'peak_vram_bytes':torch.cuda.max_memory_allocated(),'prompt_sha256':file_hash(ROOT/'prompts/terminal_v1.txt')})
    print(json.dumps({'seconds':elapsed,'peak_vram_gb':torch.cuda.max_memory_allocated()/1e9,'errors':[r['errors'] for r in results]}),flush=True)
