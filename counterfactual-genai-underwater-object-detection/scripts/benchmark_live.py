#!/usr/bin/env python3
"""Isolated, restartable live inference benchmark on a prospectively chosen calibration batch.

All waveform I/O, native checksum/decoding, CUDA resampling and representations,
actual requested generation, actual bounded Qwen calls, detector scoring and frozen
calibration are timed. Model loading and durable receipt writes are separately
reported. No acoustic evidence is read from the feature/intervention caches.

The cohort is one hash-selected complete production batch of 32 calibration rows.
This preserves both production controller batching and canonical flow RNG layout.
A queried verification executes its complete canonical 8-clip microbatch, charging
unqueried companion overhead. All primary transforms are initially computed and
recomputed for acquired tools: this implementation claims no transform savings.
"""
from __future__ import annotations
import argparse, gc, json, os, statistics, sys, time
from contextlib import contextmanager
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import torch
from uac.agent import BoundedQwenController
from uac.agent_batch import select_actions
from uac.dsp import PRIMARY_GROUPS, extract_features
from uac.verification import (DetectorBank, load_flow, generate_intervention_features,
    intervention_components, verification_score, SUBSET_NAMES)
from uac.runtime import (RunLock, StopRequested, EventLedger, atomic_json, fingerprint,
    sha256_file, source_fingerprint, project_path, utc_now)
from extract_real_features import _decode_to_cuda, _safe_gpu
from evaluate_robustness import calibrate_cases

CONFIG_PATH='configs/runtime_benchmark.json'
CODE_PATHS=['scripts/benchmark_live.py','scripts/extract_real_features.py',
 'scripts/evaluate_robustness.py','scripts/generate_interventions.py',
 'scripts/run_policies.py','scripts/analyze_real.py','src/uac/data.py','src/uac/dsp.py',
 'src/uac/verification.py','src/uac/models.py','src/uac/agent.py',
 'src/uac/agent_batch.py','src/uac/metrics.py','src/uac/runtime.py']
RESAMPLE={'method':'sinc_interp_kaiser','lowpass_filter_width':64,
 'rolloff':.9475937167399596,'beta':14.769656459379492,'device':'cuda'}
METHODS=['main','fixed_verification','heuristic_verification','proposed_agent','agent_no_generation']


def planned_config():
    path=PROJECT/'data/manifests/planned_real.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows=[r for r in rows if r['split']=='calibration']
    blocks=[(start,rows[start:start+32]) for start in range(0,len(rows),32) if len(rows[start:start+32])==32]
    if not blocks:raise ValueError('No complete planned 32-clip calibration batch')
    start,cohort=min(blocks,key=lambda item:fingerprint({'namespace':'live_runtime_batch_v1','seed':17,'ids':[r['id'] for r in item[1]]}))
    return {'schema_version':2,'seed':17,'split':'calibration','batch_size':32,
      'scope_amendment':'Add prespecified heuristic_verification cost after real development results existed, before runtime or test outcomes; preserve four-method v1 configuration',
      'heuristic_rule':'abs(sigmoid(raw_STFT_logit)-0.5)<0.3; query canonical verification blocks only for uncertain clips; main base if uncertain, STFT base otherwise; no Qwen',
      'flow_batch_size':8,'repetitions':3,'warmup_passes_per_method_per_process':1,
      'methods':METHODS,'cohort_start':start,'cohort_ids':[r['id'] for r in cohort],
      'cohort_selection':'Hash-select one complete production batch32 from canonical calibration manifest order; no outcome selection',
      'planned_manifest_sha256':sha256_file(path),'sample_rate':16000,'resample':RESAMPLE,
      'groups':list(PRIMARY_GROUPS),'initial_view':'stft','max_actions':3,'max_verifications':1,
      'probe_seconds':[.75,2.25],'taper_samples':160,'counterfactual_samples':4,'flow_steps':8,
      'flow_seed_namespace':7100000,'flow_rng':'Preserve complete canonical8 microbatches and seed namespace+canonical_start',
      'acquisition_work':'Compute fullprimary152 initially and recompute fullprimary152 for clips requesting a new view; only acquired views affect predictions',
      'verification_work':'Generate all8 canonical companions on first verify in their block; cache only within this live batch; charge companion overhead',
      'timing_scope':'Wall time including verified raw-WAV I/O/decode, H2D transfer, CUDA resample/transforms/generation, tokenization, actual Qwen, scoring, calibration and final D2H output; excludes model loading and receipt commits',
      'storage_scope':'Warm OS file cache allowed; no feature/intervention/policy cache serves timed evidence; no cold-disk claim',
      'peak_memory_scope':'Maximum allocated CUDA memory including method-resident detector bank and required flow/controller; all loaded detector heads retained and charged',
      'reporting_scope':'32 calibration clips, seed17, batch-throughput milliseconds per clip; not single-request latency or test-set accuracy'}


def write_config():
    config=planned_config();path=PROJECT/CONFIG_PATH
    if path.exists() and json.loads(path.read_text())!=config:raise RuntimeError('Existing prospective benchmark config differs; do not overwrite')
    if not path.exists():atomic_json(path,config)
    return config


def verified_cohort(config):
    path=PROJECT/'data/manifests/real_calibration.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    planned=[json.loads(line) for line in (PROJECT/'data/manifests/planned_real.jsonl').read_text().splitlines() if line.strip()]
    planned=[r for r in planned if r['split']=='calibration']
    if [r['id'] for r in rows]!=[r['id'] for r in planned]:raise RuntimeError('Verified calibration order differs from planned canonical order')
    start=config['cohort_start'];cohort=rows[start:start+config['batch_size']]
    if [r['id'] for r in cohort]!=config['cohort_ids']:raise RuntimeError('Benchmark cohort identity mismatch')
    if any(r['split']!='calibration' or not r.get('header_verified') or not r.get('sha256') for r in cohort):raise RuntimeError('Only verified calibration WAVs may enter benchmark')
    return cohort


def controller_file_hashes(directory=None):
    """Pin the actual local controller bytes, not only its declared revision.

    Hugging Face's mutable download bookkeeping is excluded. Every regular model,
    configuration and tokenizer file at the snapshot root is bound to this run;
    the safetensors index must resolve to existing pinned weight shards.
    """
    if directory is None:
        lock=json.loads((PROJECT/'manifests/model_lock.json').read_text())['Qwen/Qwen3-4B-Instruct-2507']
        if not lock.get('download_complete'):raise RuntimeError('Controller download is incomplete')
        directory=project_path(lock['path'])
    else:directory=project_path(directory)
    required=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']
    for name in required:
        if not (directory/name).is_file():raise RuntimeError('Missing controller file: '+name)
    index=json.loads((directory/'model.safetensors.index.json').read_text())
    shards=set(index.get('weight_map',{}).values())
    if not shards:raise RuntimeError('Controller weight index is empty')
    for name in shards:
        path=project_path(directory/name)
        if not path.is_relative_to(directory) or not path.is_file():raise RuntimeError('Missing or external controller weight shard: '+name)
    files=sorted(path for path in directory.rglob('*') if path.is_file() and not any(part.startswith('.') for part in path.relative_to(directory).parts))
    return {str(path.relative_to(PROJECT)):sha256_file(path) for path in files}


def verify_production_geometry(config):
    research=json.loads((PROJECT/'configs/research.json').read_text())
    if research['policy_batch_size']!=config['batch_size'] or config['batch_size']!=32:
        raise RuntimeError('Production controller batch must match prospective batch32')
    if research['counterfactual_samples']!=config['counterfactual_samples'] or config['counterfactual_samples']!=4:
        raise RuntimeError('Production counterfactual sample count must remain4')
    if research['flow_sampling_steps']!=config['flow_steps'] or config['flow_steps']!=8:
        raise RuntimeError('Production flow sampling steps must remain8')
    path=PROJECT/'data/interventions/calibration/index.json'
    index=json.loads(path.read_text());settings=index['config']
    if index.get('status')!='complete':raise RuntimeError('Calibration intervention index is incomplete')
    if (settings.get('split')!='calibration' or settings.get('probe')!='central'
        or settings.get('probe_seconds')!=config['probe_seconds']
        or settings.get('batch_size')!=config['flow_batch_size'] or config['flow_batch_size']!=8
        or settings.get('shard_size',0)%8!=0
        or settings.get('k')!=config['counterfactual_samples'] or settings.get('steps')!=config['flow_steps']):
        raise RuntimeError('Canonical calibration flow batch/RNG/probe geometry differs')
    if settings.get('generator_hash')!=sha256_file(PROJECT/'checkpoints/background_flow/final.pt'):
        raise RuntimeError('Calibration interventions used a different generator')
    if settings.get('data_fingerprint')!=config['planned_manifest_sha256']:
        raise RuntimeError('Calibration intervention cohort differs')
    if settings.get('feature_index_hash')!=sha256_file(PROJECT/'data/features/real/calibration/index.json'):
        raise RuntimeError('Calibration intervention feature order/index changed')
    return {'calibration_intervention_index_sha256':sha256_file(path),
      'canonical_flow_microbatch_size':8,'counterfactual_samples':4,'flow_steps':8}


class Timings:
    def __init__(self):self.seconds={}
    @contextmanager
    def stage(self,name):
        torch.cuda.synchronize();start=time.perf_counter()
        try:yield
        finally:
            torch.cuda.synchronize();self.seconds[name]=self.seconds.get(name,0.)+time.perf_counter()-start


class LiveEvidence:
    def __init__(self,audio,features,bank,flow,config,timer):
        self.audio=audio;self.features=features;self.bank=bank;self.flow=flow;self.config=config;self.timer=timer
        self.components={};self.generated_blocks=[];self.feature_recomputed_clips=0
    def acquire(self,indices):
        if not indices:return
        with self.timer.stage('acquired_view_full_feature_recomputation'):
            index=torch.tensor(indices,device='cuda')
            self.features[index]=extract_features(self.audio[index],groups=PRIMARY_GROUPS)['features']
            self.feature_recomputed_clips+=len(indices)
    def verify(self,indices):
        for begin in sorted({(i//8)*8 for i in indices}):
            if begin in self.components:continue
            with self.timer.stage('generator_sham_transforms_and_verification_heads'):
                raw=generate_intervention_features(self.flow,self.audio[begin:begin+8],
                  seed=self.config['flow_seed_namespace']+self.config['cohort_start']+begin,
                  k=self.config['counterfactual_samples'],steps=self.config['flow_steps'],probe='central')
                self.components[begin]=intervention_components(self.bank,raw,{})
                self.generated_blocks.append(begin)
    def component(self,index):
        begin=(index//8)*8;offset=index-begin
        return {key:value[offset] for key,value in self.components[begin].items()}


def run_live_policy(controller,evidence,ids,*,generation):
    n=len(ids);bank=evidence.bank;features=evidence.features;timer=evidence.timer
    with timer.stage('detector_state_and_final_record_construction'):
        view_logits=bank.view_logits(features)
        states=[];traces=[[] for _ in ids];views=[['stft'] for _ in ids];verified=[False]*n;active=list(range(n))
        for i in range(n):
            probability=float(torch.sigmoid(view_logits[i,1]))
            states.append({'candidates':[{'candidate_id':'candidate0','interval_seconds':[.75,2.25],
              'base_probability':probability,'uncertainty':1-2*abs(probability-.5)}],
              'observations':{'candidate0':{'stft':{'detector_logit':float(view_logits[i,1]),'presence_probability':probability}}},
              'history':[],'available_tools':['fourier','wavelet','stop']+(['verify'] if generation else []),
              'remaining_actions':3,'remaining_verifications':int(generation)})
    for _ in range(3):
        if not active:break
        with timer.stage('controller_tokenization_generation_parse'):
            actions=select_actions(controller,[states[i] for i in active])
        acquired=[i for i,a in zip(active,actions) if a['valid'] and a['action'] in ('fourier','wavelet')]
        queries=[i for i,a in zip(active,actions) if a['valid'] and a['action']=='verify']
        evidence.acquire(acquired)
        evidence.verify(queries)
        with timer.stage('detector_state_and_final_record_construction'):
            # Recomputed only for acquired tools; other tool logits are never exposed.
            if acquired:
                selected=torch.tensor(acquired,device='cuda')
                view_logits[selected]=bank.view_logits(features[selected])
            next_active=[]
            for i,action in zip(active,actions):
                traces[i].append(action);state=states[i];tool=action['action']
                if tool=='stop' or not action['valid']:continue
                state['history'].append({'action':tool,'candidate_id':'candidate0','valid':True})
                state['remaining_actions']-=1;state['available_tools'].remove(tool)
                if tool=='verify':
                    verified[i]=True;state['remaining_verifications']=0;comp=evidence.component(i)
                    observed={v:float(comp['delta'][:,PRIMARY_GROUPS.index(v)].median()) for v in views[i]}
                    state['observations']['candidate0']['verify']={'valid':bool(comp['valid']),'view_delta_medians':observed,'background_penalty':float(comp['penalty'])}
                else:
                    views[i].append(tool);position=PRIMARY_GROUPS.index(tool)
                    state['observations']['candidate0'][tool]={'detector_logit':float(view_logits[i,position]),'presence_probability':float(torch.sigmoid(view_logits[i,position]))}
                    if verified[i]:state['observations']['candidate0']['verify']['view_delta_medians'][tool]=float(evidence.component(i)['delta'][:,position].median())
                current=float(torch.sigmoid(bank.subset_logit(features[i:i+1],views[i])[0]))
                state['candidates'][0]['base_probability']=current;state['candidates'][0]['uncertainty']=1-2*abs(current-.5)
                if state['remaining_actions']>0:next_active.append(i)
            active=next_active
    with timer.stage('detector_state_and_final_record_construction'):
        records=[]
        for i,identity in enumerate(ids):
            comp=evidence.component(i) if verified[i] else None
            selected=views[i];positions=[PRIMARY_GROUPS.index(v) for v in selected]
            records.append({'id':identity,'seed':17,'mode':'agent' if generation else 'agent_no_generation',
              'views':selected,'subset_method':SUBSET_NAMES[frozenset(selected)],'verified':verified[i],
              'intervention_valid':bool(comp['valid']) if comp is not None else False,
              'base_logit':float(bank.subset_logit(features[i:i+1],selected)[0]),
              'delta':comp['delta'][:,positions].flatten().cpu().tolist() if comp is not None else [],
              'penalty':float(comp['penalty']) if comp is not None else 0.,'trace':traces[i],
              'amortized_llm_seconds':sum(t['amortized_seconds'] for t in traces[i])})
    return records


def heuristic_query_mask(stft_logit):
    """Frozen analyze_real rule, evaluated before any generative evidence."""
    return (torch.sigmoid(stft_logit)-.5).abs()<.3


def heuristic_case(main,stft,uncertain,components,calibration):
    """Match analyze_real's base/fallback and keep companion evidence unused."""
    value=torch.zeros_like(main);valid=torch.zeros_like(uncertain)
    for begin,component in components.items():
        end=begin+len(component['valid'])
        value[begin:end]=verification_score(component,PRIMARY_GROUPS,calibration['dispersion'],calibration['penalty'])
        valid[begin:end]=component['valid']
    return {'base':torch.where(uncertain,main,stft),'v':torch.where(uncertain,value,torch.zeros_like(value)),
      'verify':uncertain&valid,'base_names':['main' if bool(flag) else 'stft' for flag in uncertain]}


@torch.inference_mode()
def timed_pass(method,rows,bank,flow,controller,calibration,config):
    timer=Timings();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with timer.stage('raw_checksum_decode_transfer_cuda_resample'):
        audio=_decode_to_cuda(rows,16000,config['resample'])
    with timer.stage('initial_full_primary_features'):
        features=extract_features(audio,groups=PRIMARY_GROUPS)['features']
    evidence=LiveEvidence(audio,features,bank,flow,config,timer);n=len(rows);ids=[r['id'] for r in rows]
    policy=[];queries=[];uncertain=None;stft=None
    if method in ('proposed_agent','agent_no_generation'):
        policy=run_live_policy(controller,evidence,ids,generation=method=='proposed_agent')
    elif method=='fixed_verification':
        queries=list(range(n));evidence.verify(queries)
    elif method=='heuristic_verification':
        with timer.stage('heuristic_query_detector_and_rule'):
            stft=bank.logits('stft',features);uncertain=heuristic_query_mask(stft)
            queries=uncertain.nonzero().flatten().cpu().tolist()
        evidence.verify(queries)
    with timer.stage('final_detection_frozen_calibration_output_transfer'):
        if policy:
            v=[]
            for row in policy:
                values=torch.tensor(row['delta'],device='cuda');median=values.median() if len(values) else torch.tensor(0.,device='cuda')
                score=median-calibration['dispersion']*(values-median).abs().median()-calibration['penalty']*row['penalty'] if len(values) else median
                v.append(float(score))
            case={'base':torch.tensor([r['base_logit'] for r in policy],device='cuda'),'v':torch.tensor(v,device='cuda'),
              'verify':torch.tensor([r['verified'] and r['intervention_valid'] for r in policy],device='cuda'),
              'base_names':[r['subset_method'] for r in policy]}
        else:
            case={'base':bank.logits('main',features),'v':torch.zeros(n,device='cuda'),
              'verify':torch.zeros(n,dtype=torch.bool,device='cuda'),'base_names':['main']*n}
            if method=='fixed_verification':
                comp={key:torch.cat([evidence.components[b][key] for b in sorted(evidence.components)]) for key in next(iter(evidence.components.values()))}
                case['v']=verification_score(comp,PRIMARY_GROUPS,calibration['dispersion'],calibration['penalty']);case['verify']=comp['valid']
            elif method=='heuristic_verification':
                case=heuristic_case(case['base'],stft,uncertain,evidence.components,calibration)
        fit=calibration['seeds']['17'];prob=calibrate_cases({method:case},fit)[method]
        scores=prob.cpu().tolist()
        # Keep the frozen float64 nextafter boundary: an FP32 comparison can
        # round a just-above-score threshold back onto that score (even at1).
        decisions=(prob.to(torch.float64)>=torch.tensor(fit['thresholds'][method]['threshold'],dtype=torch.float64,device=prob.device)).cpu().tolist()
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    queried=sum(r['verified'] for r in policy) if policy else len(queries)
    return {'method':method,'ids':ids,'probabilities':scores,'decisions':decisions,'policy_records':policy,
      'seconds':elapsed,'milliseconds_per_clip':elapsed*1000/n,'stage_seconds':timer.seconds,
      'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,
      'peak_reserved_memory_mib':torch.cuda.max_memory_reserved()/1024**2,
      'verification_query_ids':[r['id'] for r in policy if r['verified']] if policy else [ids[i] for i in queries],
      'generation_queried_clips':queried,'generation_executed_clips':len(evidence.generated_blocks)*8,
      'canonical_generated_block_starts':[config['cohort_start']+b for b in evidence.generated_blocks],
      'canonical_unqueried_block_starts':[config['cohort_start']+b for b in range(0,n,8) if b not in evidence.generated_blocks],
      'generation_unqueried_companion_clips':len(evidence.generated_blocks)*8-queried,
      'acquired_view_full_feature_recomputed_clips':evidence.feature_recomputed_clips,
      'mode':'live','completed_utc':utc_now()}


def valid_receipt(path,expected,ledger):
    sidecar=path.with_suffix('.receipt.json')
    if not path.exists() and not sidecar.exists():return None
    try:
        receipt=json.loads(sidecar.read_text())
        if receipt.get('config_hash')!=expected:raise RuntimeError('Benchmark config/source/model mismatch; preserve existing run and use a new directory')
        if path.exists() and sha256_file(path)==receipt['sha256'] and path.stat().st_size==receipt['bytes']:
            return json.loads(path.read_text())
    except (OSError,ValueError,KeyError):pass
    suffix='.corrupt-'+str(time.time_ns())
    for item in (path,sidecar):
        if item.exists():item.rename(item.with_name(item.name+suffix))
    ledger.append('runtime_benchmark_corruption_replay',path=str(path.relative_to(PROJECT)))
    return None


def commit(path,payload,config_hash):
    atomic_json(path,payload)
    atomic_json(path.with_suffix('.receipt.json'),{'config_hash':config_hash,'sha256':sha256_file(path),'bytes':path.stat().st_size,'completed_utc':utc_now()})


def equivalence_observation(method,result,config):
    """Untimed action/evidence/final-probability parity against archived calibration.

    Absolute tolerances: base logits and paired deltas1e-4, penalty1e-6,
    calibrated probability1e-5. Actions, validity and threshold decisions must
    agree exactly. Different traces remain reported, never silently rewritten.
    """
    tolerances={'base_logit_absolute':1e-4,'verification_delta_absolute':1e-4,
      'penalty_absolute':1e-6,'calibrated_probability_absolute':1e-5}
    references={};different=[];validity_differences=[];max_error=0.;max_delta_error=0.;max_penalty_error=0.
    if method in ('proposed_agent','agent_no_generation'):
        mode='agent' if method=='proposed_agent' else method
        path=PROJECT/'results/raw/policies/calibration'/f'{mode}_seed17'/'all.json'
        index_path=path.parent/'index.json'
        index=json.loads(index_path.read_text())
        if index.get('status')!='complete' or index.get('sha256')!=sha256_file(path):raise RuntimeError('Archived policy parity reference is incomplete or corrupt')
        expected=json.loads(path.read_text())['records'][config['cohort_start']:config['cohort_start']+32]
        if [r['id'] for r in expected]!=result['ids']:raise RuntimeError('Canonical reference cohort ordering differs')
        references[str(path.relative_to(PROJECT))]=sha256_file(path)
        for reference,observed in zip(expected,result['policy_records']):
            if ([(x['action'],x['valid']) for x in reference['trace']]!=[(x['action'],x['valid']) for x in observed['trace']]
                or reference['views']!=observed['views'] or reference['verified']!=observed['verified']):different.append(reference['id'])
            max_error=max(max_error,abs(reference['base_logit']-observed['base_logit']))
            if reference['verified'] and observed['verified']:
                if reference['intervention_valid']!=observed['intervention_valid']:validity_differences.append(reference['id'])
                max_penalty_error=max(max_penalty_error,abs(reference['penalty']-observed['penalty']))
                if len(reference['delta'])==len(observed['delta']):
                    max_delta_error=max(max_delta_error,max((abs(a-b) for a,b in zip(reference['delta'],observed['delta'])),default=0.))
                elif reference['id'] not in different:different.append(reference['id'])
    path=PROJECT/'results/raw/real_predictions_calibration_seed17.pt'
    reference=torch.load(path,map_location='cpu',weights_only=False)
    start=config['cohort_start'];stop=start+32
    if reference['ids'][start:stop]!=result['ids']:raise RuntimeError('Archived calibrated prediction identity differs')
    expected_probability=reference['probabilities'][method][start:stop].tolist()
    if len(expected_probability)!=len(result['probabilities']):raise RuntimeError('Archived probability count differs')
    max_probability_error=max(abs(a-b) for a,b in zip(expected_probability,result['probabilities']))
    threshold=json.loads((PROJECT/'manifests/calibration.json').read_text())['seeds']['17']['thresholds'][method]['threshold']
    decisions=[identity for identity,probability,observed in zip(result['ids'],expected_probability,result['decisions']) if (probability>=threshold)!=observed]
    references[str(path.relative_to(PROJECT))]=sha256_file(path)
    matched=(not different and not validity_differences and not decisions
      and max_error<=tolerances['base_logit_absolute'] and max_delta_error<=tolerances['verification_delta_absolute']
      and max_penalty_error<=tolerances['penalty_absolute'] and max_probability_error<=tolerances['calibrated_probability_absolute'])
    return {'status':'matched' if matched else 'numerical_or_policy_difference','absolute_tolerances':tolerances,
      'action_difference_ids':different,'validity_difference_ids':validity_differences,'decision_difference_ids':decisions,
      'max_delta_abs_error':max_delta_error,'max_penalty_abs_error':max_penalty_error,
      'max_base_logit_abs_error':max_error,'max_calibrated_probability_abs_error':max_probability_error,
      'reference_sha256':references,'action_evidence_audit_applicable':method in ('proposed_agent','agent_no_generation'),
      'interpretation':'Untimed comparison of actual live outputs with archived calibration; differences remain visible and do not alter measured timing'}


def summarize(directory,config,settings,setup):
    records=[];receipts=[]
    for method in METHODS:
        values=[]
        for repeat in range(config['repetitions']):
            path=directory/f'{method}-repeat{repeat}.json'
            if not path.exists() or not path.with_suffix('.receipt.json').exists():continue
            receipt=json.loads(path.with_suffix('.receipt.json').read_text())
            if receipt['config_hash']!=fingerprint(settings) or receipt['sha256']!=sha256_file(path):raise RuntimeError('Runtime summary receipt mismatch')
            values.append(json.loads(path.read_text()));receipts.append({'path':str(path.relative_to(PROJECT)),**receipt})
        if not values:continue
        latencies=[v['milliseconds_per_clip'] for v in values]
        records.append({'method':method,'seed':17,'milliseconds_per_clip':statistics.mean(latencies),
          'milliseconds_per_clip_sd':statistics.stdev(latencies) if len(values)>1 else None,
          'milliseconds_per_clip_min':min(latencies),'milliseconds_per_clip_max':max(latencies),
          'peak_memory_mib':max(v['peak_memory_mib'] for v in values),'mode':'live','repetitions':len(values),
          'clips_per_repetition':32,'measurement_kind':'warm_batch32_throughput',
          'stage_seconds_mean':{key:statistics.mean(v['stage_seconds'].get(key,0.) for v in values) for key in sorted({k for v in values for k in v['stage_seconds']})},
          'generation_queried_clips':[v['generation_queried_clips'] for v in values],
          'generation_executed_clips':[v['generation_executed_clips'] for v in values],
          'canonical_policy_parity':[v.get('parity_audit') for v in values]})
    complete=len(receipts)==len(METHODS)*config['repetitions']
    output={'status':'complete' if complete else 'partial','config':config,'settings':settings,
      'records':records,'receipts':receipts,'setup_observations':[json.loads(p.read_text()) for p in sorted(directory.glob('*-setup-*.json'))],
      'limitations':['Warm OS file cache, no cold-storage latency claim.','32 calibration clips and one detector seed; no population runtime interval.',
        'Batch-throughput cost, not individual-request response time.','All primary transforms computed and charged; no adaptive transform-savings claim.',
        'Unqueried canonical flow companions are generated to preserve random draws, with their full overhead charged.',
        'CUDA allocated memory includes resident per-method detector bank; reserved allocator memory separately saved per pass.',
        'Every timed pass recomputes waveform evidence and runs the controller; no cached evidence is timed as live execution.']}
    atomic_json('results/tables/runtime_benchmark.json',output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-config',action='store_true')
    parser.add_argument('--max-units',type=int);parser.add_argument('--output-dir',default='results/raw/runtime_benchmark')
    args=parser.parse_args()
    if args.write_config:
        print(json.dumps(write_config(),indent=2));return 0
    config=json.loads((PROJECT/CONFIG_PATH).read_text())
    if config!=planned_config():raise RuntimeError('Frozen prospective benchmark configuration changed')
    production_geometry=verify_production_geometry(config)
    directory=project_path(args.output_dir);directory.mkdir(parents=True,exist_ok=True)
    with RunLock('runs/training.lock'),StopRequested() as stop:
        gpu=_safe_gpu();rows=verified_cohort(config);ledger=EventLedger();calibration=json.loads((PROJECT/'manifests/calibration.json').read_text())
        provenance_started=time.perf_counter();controller_bytes=controller_file_hashes()
        settings={'config':config,'production_geometry':production_geometry,'source_fingerprint':source_fingerprint(CODE_PATHS),
          'model_sha256':{p:sha256_file(p) for p in ['manifests/detector_index.json','manifests/calibration.json','manifests/model_lock.json','checkpoints/background_flow/final.pt']},
          'controller_file_sha256':controller_bytes,
          'parity_reference_sha256':{p:sha256_file(p) for p in ['results/raw/real_predictions_calibration_seed17.pt',
            'results/raw/policies/calibration/agent_seed17/all.json','results/raw/policies/calibration/agent_seed17/index.json',
            'results/raw/policies/calibration/agent_no_generation_seed17/all.json','results/raw/policies/calibration/agent_no_generation_seed17/index.json']},
          'cohort_audio_sha256':{r['id']:r['sha256'] for r in rows},'gpu':gpu,
          'torch_version':torch.__version__,'cuda_version':torch.version.cuda}
        config_hash=fingerprint(settings);setup=[];units=0
        ledger.append('live_runtime_untimed_provenance_verified',controller_files=len(controller_bytes),seconds_excluded=time.perf_counter()-provenance_started)
        # Inspect every unit before partial summary reads so a power cut on a later
        # method cannot strand recovery while earlier methods are being replayed.
        for method in METHODS:
            for repeat in range(config['repetitions']):
                valid_receipt(directory/f'{method}-repeat{repeat}.json',config_hash,ledger)
        for method in METHODS:
            pending=[]
            for repeat in range(config['repetitions']):
                if valid_receipt(directory/f'{method}-repeat{repeat}.json',config_hash,ledger) is None:pending.append(repeat)
            if not pending:continue
            if stop.requested:break
            torch.cuda.empty_cache();started=time.perf_counter();bank=DetectorBank(17)
            flow=load_flow() if method in ('fixed_verification','heuristic_verification','proposed_agent') else None
            controller=BoundedQwenController() if method in ('proposed_agent','agent_no_generation') else None
            torch.cuda.synchronize();loaded=time.perf_counter()-started
            warmup=timed_pass(method,rows,bank,flow,controller,calibration,config)
            observation={'method':method,'config_hash':config_hash,'model_loading_seconds_excluded':loaded,'warmup_seconds_excluded':warmup['seconds'],
              'resident_allocated_mib_after_warmup':torch.cuda.memory_allocated()/1024**2,'completed_utc':utc_now()}
            setup.append(observation);atomic_json(directory/f'{method}-setup-{time.time_ns()}.json',observation)
            del warmup
            for repeat in pending:
                if stop.requested or (args.max_units is not None and units>=args.max_units):break
                result=timed_pass(method,rows,bank,flow,controller,calibration,config)
                result['repeat']=repeat;result['parity_audit']=equivalence_observation(method,result,config)
                commit(directory/f'{method}-repeat{repeat}.json',result,config_hash);units+=1
                ledger.append('live_runtime_batch_complete',method=method,repeat=repeat,seconds=result['seconds'])
                print(json.dumps({'method':method,'repeat':repeat,'milliseconds_per_clip':result['milliseconds_per_clip'],'peak_memory_mib':result['peak_memory_mib']}),flush=True)
                summarize(directory,config,settings,setup)
            if controller is not None:controller.close()
            del bank,flow,controller;gc.collect();torch.cuda.empty_cache()
            if stop.requested or (args.max_units is not None and units>=args.max_units):break
        report=summarize(directory,config,settings,setup)
        ledger.append('live_runtime_benchmark_status',status=report['status'],completed_receipts=len(report['receipts']))
        return 75 if stop.requested else 0


if __name__=='__main__':raise SystemExit(main())
