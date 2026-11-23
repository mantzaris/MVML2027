"""Batched model-driven tool selection with finite legal-JSON token constraints.

The model greedily chooses among token prefixes of all currently legal action /
candidate / enumerated-reason combinations. No action ranking or retry policy is
inserted. JSON syntax and available-tool validity are guaranteed by the grammar,
not evidence of unconstrained model competence or acoustic correctness.
"""
from __future__ import annotations
import json
import time
from uac.agent import validate_state,parse_action,SYSTEM_PROMPT,MODEL_ID,REASONS,_reject
from uac.runtime import fingerprint

CONSTRAINT_VERSION='finite_legal_json_trie_v1'


def legal_action_texts(state):
    """Use the independent strict parser as the legality oracle for enumeration."""
    state=validate_state(state);texts=[]
    for tool in state['available_tools']:
        candidates=[None] if tool=='stop' else [c['candidate_id'] for c in state['candidates']]
        for candidate_id in candidates:
            for reason in REASONS:
                text=json.dumps({'action':tool,'candidate_id':candidate_id,'reason_code':reason},separators=(',',':'),ensure_ascii=True)
                if parse_action(text,state)['valid']:texts.append(text)
    if not texts:raise ValueError('No legal stop completion in validated state')
    return tuple(texts)


class ActionTokenTrie:
    """Finite exact token sequences, each ending in one explicit generation EOS."""
    TERMINAL=-1
    def __init__(self,texts,tokenizer,*,max_output_tokens):
        self.eos=tokenizer.eos_token_id;self.pad=tokenizer.pad_token_id
        if not isinstance(self.eos,int) or not isinstance(self.pad,int):raise ValueError('Explicit integer EOS and pad tokens are required')
        self.root={};self.texts=tuple(texts);self.max_sequence_tokens=0
        special=set(tokenizer.all_special_ids)
        for text in self.texts:
            tokens=list(tokenizer.encode(text,add_special_tokens=False))
            if not tokens or any(token in special for token in tokens):raise ValueError('Action JSON contains an empty encoding or a special token')
            decoded=tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if decoded!=text:raise ValueError('Action JSON tokenization is not an exact round trip')
            sequence=tokens+[self.eos]
            if len(sequence)>max_output_tokens:raise ValueError('A legal JSON completion plus EOS exceeds the output token budget')
            self.max_sequence_tokens=max(self.max_sequence_tokens,len(sequence));node=self.root
            for token in sequence:node=node.setdefault(token,{})
            node[self.TERMINAL]=True
    def allowed(self,prefix):
        node=self.root
        for offset,token in enumerate(prefix):
            if token not in node:raise ValueError('Generated prefix is outside its legal action trie')
            node=node[token]
            if token==self.eos:
                if not node.get(self.TERMINAL):raise ValueError('EOS reached outside a complete legal JSON action')
                if any(item not in (self.pad,self.eos) for item in prefix[offset+1:]):
                    raise ValueError('Non-padding content follows the terminal EOS')
                # HF still invokes the processor for finished batch rows before
                # replacing their next token with pad. This path must be nonempty.
                return [self.pad]
        allowed=sorted(token for token in node if token!=self.TERMINAL)
        if not allowed:raise ValueError('An unfinished constrained prefix has no legal next token')
        return allowed
    def complete(self,tokens):
        if self.eos not in tokens:return False
        try:self.allowed(tokens)
        except ValueError:return False
        return True


class BatchedActionConstraints:
    def __init__(self,states,tokenizer,*,prompt_width,max_output_tokens):
        if prompt_width<1:raise ValueError('A positive shared padded prompt width is required')
        self.prompt_width=prompt_width
        self.tries=[ActionTokenTrie(legal_action_texts(state),tokenizer,max_output_tokens=max_output_tokens) for state in states]
    def __call__(self,batch_id,input_ids):
        if not 0<=batch_id<len(self.tries):raise ValueError('Unexpected generation batch ID; beam count must be one')
        if len(input_ids)<self.prompt_width:raise ValueError('Generation prefix is shorter than padded prompt width')
        suffix=input_ids[self.prompt_width:]
        tokens=suffix.tolist() if hasattr(suffix,'tolist') else list(suffix)
        return self.tries[batch_id].allowed(tokens)


def select_actions(controller,states):
    torch=controller.torch
    cleaned=[validate_state(state) for state in states]
    if not cleaned or any(s['remaining_actions']<=0 for s in cleaned):raise ValueError('Only nonempty active states may enter batched generation')
    started=time.perf_counter();texts=[]
    for state in cleaned:
        messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':json.dumps(state,sort_keys=True,separators=(',',':'))}]
        texts.append(controller.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    controller.tokenizer.padding_side='left'
    if controller.tokenizer.pad_token_id is None:controller.tokenizer.pad_token=controller.tokenizer.eos_token
    encoded=controller.tokenizer(texts,padding=True,return_tensors='pt',add_special_tokens=False)
    lengths=encoded['attention_mask'].sum(-1).tolist();prompt_width=encoded['input_ids'].shape[1]
    if prompt_width+controller.max_output_tokens>controller.max_context_tokens:raise ValueError('Batched context budget exceeded; no truncation')
    constraints=BatchedActionConstraints(cleaned,controller.tokenizer,prompt_width=prompt_width,max_output_tokens=controller.max_output_tokens)
    encoded={key:value.to('cuda') for key,value in encoded.items()}
    torch.cuda.synchronize();event_start=torch.cuda.Event(enable_timing=True);event_end=torch.cuda.Event(enable_timing=True);event_start.record()
    with torch.inference_mode():
        output=controller.model.generate(**encoded,max_new_tokens=controller.max_output_tokens,
          do_sample=False,use_cache=True,num_beams=1,num_return_sequences=1,
          eos_token_id=controller.tokenizer.eos_token_id,pad_token_id=controller.tokenizer.pad_token_id,
          prefix_allowed_tokens_fn=constraints,return_dict_in_generate=True,output_scores=True)
    event_end.record();torch.cuda.synchronize();gpu_ms=event_start.elapsed_time(event_end)
    generated=output.sequences;result=[];token_log_probabilities=[]
    # These are conditional probabilities under each legal-token mask. They are
    # not probabilities of detection, correctness, or unconstrained compliance.
    with torch.inference_mode():
        for step,scores in enumerate(output.scores):
            selected=generated[:,prompt_width+step]
            token_log_probabilities.append(scores.gather(1,selected[:,None]).squeeze(1)-torch.logsumexp(scores,dim=1))
        log_probabilities=torch.stack(token_log_probabilities,dim=1).cpu().tolist()
    for i,state in enumerate(cleaned):
        tokens=generated[i,prompt_width:].tolist();trie=constraints.tries[i]
        eos_present=trie.eos in tokens;count=tokens.index(trie.eos)+1 if eos_present else len(tokens)
        raw=controller.tokenizer.decode(tokens[:count],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        action=parse_action(raw,state)
        if not trie.complete(tokens):action=_reject('constrained_generation_missing_eos_or_invalid_prefix')
        result.append({**action,'raw_text':raw,'model':MODEL_ID,'revision':controller.revision,
          'prompt_sha256':controller.prompt_sha256,'state_sha256':fingerprint(state),
          'input_tokens':int(lengths[i]),'output_tokens':count,'batch_size':len(states),
          'constrained_decoding':CONSTRAINT_VERSION,'schema_validity_by_construction':True,
          'legal_completion_count':len(trie.texts),'largest_legal_completion_tokens_including_eos':trie.max_sequence_tokens,
          'terminal_eos_observed':eos_present,'constrained_token_log_probability_sum':sum(log_probabilities[i][:count]),
          'log_probability_scope':'conditional token probabilities after legal-prefix masking; not detector or acoustic confidence',
          'measurement_mode':'live_batched_model_call_with_finite_json_constraints',
          'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    torch.cuda.synchronize();wall=time.perf_counter()-started
    for record in result:
        record.update({'batch_wall_seconds':wall,'batch_cuda_ms':float(gpu_ms),'amortized_seconds':wall/len(states),
          'latency_seconds':wall,'batch_cuda_scope':'model generation including prefix constraints; wall additionally includes tokenization, grammar construction, parsing and score logging'})
    return result
