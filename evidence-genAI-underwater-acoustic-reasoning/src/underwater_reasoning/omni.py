"""Pinned text-only Omni inference with explicit end-of-turn stopping."""
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniProcessor
from .runtime import budget_check

DEFAULT_SYSTEM='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
REVISION='ae9e1690543ffd5c0221dc27f79834d0294cba00'

class Omni:
    def __init__(self,root):
        torch.set_num_threads(8);self.root=Path(root);path=(self.root/'.cache/omni_path.txt').read_text().strip()
        self.processor=Qwen2_5OmniProcessor.from_pretrained(path,use_fast=False);self.processor.tokenizer.padding_side='left'
        self.model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        self.eos=self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if self.eos!=151645:raise ValueError('Unexpected end-of-turn token')

    def tokens(self,text):return len(self.processor.tokenizer.encode(text,add_special_tokens=False))

    def generate(self,items,max_new_tokens=768):
        budget_check();texts=[];waves=[];text_counts=[]
        for item in items:
            x,fs=item['audio'],item['sample_rate']
            if fs!=16000:
                from math import gcd
                g=gcd(fs,16000);x=resample_poly(x,16000//g,fs//g)
            waves.append(np.asarray(x,dtype='float32'));text_counts.append(self.tokens(item['text']))
            if text_counts[-1]>4096:raise ValueError('Text prompt budget exceeded')
            conversation=[{'role':'system','content':[{'type':'text','text':DEFAULT_SYSTEM}]},
                {'role':'user','content':[{'type':'audio','audio_url':'anonymous.wav'},{'type':'text','text':item['text']}]}]
            texts.append(self.processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False))
        inputs=self.processor(text=texts,audio=waves,return_tensors='pt',padding=True,sampling_rate=16000).to(self.model.device)
        inputs={k:v.to(dtype=self.model.dtype) if v.is_floating_point() else v for k,v in inputs.items()}
        torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,eos_token_id=self.eos,
                pad_token_id=self.processor.tokenizer.pad_token_id,use_cache=True)
        torch.cuda.synchronize();elapsed=time.monotonic()-start;ids=output[:,inputs['input_ids'].shape[1]:]
        raw=self.processor.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False);results=[]
        for i,text in enumerate(raw):
            tokens=ids[i].tolist();count=tokens.index(self.eos)+1 if self.eos in tokens else len(tokens)
            results.append({'raw':text,'input_text_tokens':text_counts[i],'input_tokens_with_audio':int(inputs['attention_mask'][i].sum()),
                'output_tokens':count,'hit_output_limit':self.eos not in tokens,'batch_seconds':elapsed,'batch_size':len(items),
                'amortized_seconds':elapsed/len(items),'peak_vram_bytes':torch.cuda.max_memory_allocated()})
        return results
