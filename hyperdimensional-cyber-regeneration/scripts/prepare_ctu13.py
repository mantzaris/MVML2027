#!/usr/bin/env python3
"""Acquire only whitelisted CTU-13 flow tables; CPU parsing/I/O, never model compute.

The source manifest declares files and capture splits before held-out features
or model outcomes are inspected. Only .binetflow CSV files are downloaded; the
provider's executables, packet payloads and full archives are not requested.
"""
import argparse
from array import array
import csv
from collections import Counter
from datetime import datetime, timezone
import math
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/ctu13_sources.json'
OUT=ROOT/'data/external/ctu13'
RECORD=ROOT/'data/ctu13_downloads.json'
MANIFEST=ROOT/'results/real/ctu13_data_manifest.json'
FEATURES=['log1p_duration','log1p_packets','log1p_bytes','log1p_source_bytes',
          'source_byte_fraction','log1p_bytes_per_packet','source_port_fraction','destination_port_fraction',
          'is_tcp','is_udp','is_icmp','state_has_S','state_has_F','state_has_R']


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def acquire(source,previous):
    url=source['url'];parsed=urlparse(url)
    if (parsed.scheme!='https' or parsed.netloc!='mcfp.felk.cvut.cz'
            or not parsed.path.endswith('.binetflow')
            or '/detailed-bidirectional-flow-labels/' not in parsed.path):
        raise ValueError('Only declared official bidirectional text tables are permitted')
    path=OUT/source['local_filename']
    if path.is_file() and previous and previous.get('url')==url and path.stat().st_size==previous['bytes'] and sha(path)==previous['sha256']:
        print(json.dumps(dict(scenario=source['scenario'],cached=True,bytes=previous['bytes'])),flush=True)
        return previous
    partial=path.with_suffix('.binetflow.part')
    headers={'User-Agent':'MVML2027-public-research/1.0'}
    request=urllib.request.Request(url,headers=headers)
    started=time.monotonic()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=120) as response, partial.open('wb') as dest:
                expected=response.headers.get('Content-Length')
                digest=hashlib.sha256();size=0;first=True
                for chunk in iter(lambda:response.read(1024*1024),b''):
                    if first:
                        if not chunk.startswith(b'StartTime,'):
                            raise ValueError('Unexpected flow-table header; refusing non-CSV content')
                        first=False
                    dest.write(chunk);digest.update(chunk);size+=len(chunk)
                if expected is not None and size!=int(expected):
                    raise IOError(f'Truncated download {size}/{expected}')
                record=dict(scenario=source['scenario'],url=url,path=str(path.relative_to(ROOT)),bytes=size,
                    sha256=digest.hexdigest(),last_modified=response.headers.get('Last-Modified'),
                    etag=response.headers.get('ETag'),retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    download_seconds=time.monotonic()-started)
            partial.replace(path)
            print(json.dumps(dict(scenario=source['scenario'],downloaded=True,bytes=size)),flush=True)
            return record
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))
    raise AssertionError('Unreachable')


def download(workers=3):
    sources=json.loads(SOURCE.read_text());OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads(RECORD.read_text()).get('files',[]) if RECORD.is_file() else []
    cache={r['scenario']:r for r in previous};records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        jobs=[executor.submit(acquire,source,cache.get(source['scenario'])) for source in sources['sources']]
        for future in concurrent.futures.as_completed(jobs):
            records.append(future.result())
            data=dict(dataset=sources['dataset'],source_manifest_sha256=sha(SOURCE),
                      files=sorted(records,key=lambda r:r['scenario']),complete=len(records)==len(jobs))
            RECORD.write_text(json.dumps(data,indent=2)+'\n')
    return records



def nonnegative(value):
    try:
        parsed=float(value)
        return (parsed,True) if math.isfinite(parsed) and parsed>=0 else (0.,False)
    except (ValueError,TypeError):
        return 0.,False


def port(value):
    try:
        text=value.strip()
        parsed=int(text,16 if text.lower().startswith('0x') else 10)
        return (parsed/65535.,True) if 0<=parsed<=65535 else (0.,False)
    except (ValueError,TypeError):
        return 0.,False


def observed_features(row):
    # Parsing transformations only: no fitted statistic or ML is computed here.
    dur,md=nonnegative(row[1]);pkts,mp=nonnegative(row[11])
    total,mt=nonnegative(row[12]);source,ms=nonnegative(row[13])
    sport,msp=port(row[4]);dport,mdp=port(row[7])
    proto=row[2].strip().lower();mproto=proto not in ('','?')
    state=row[8].strip().upper();mstate=state not in ('','?')
    fractions=source/max(total,1.) if ms and mt else 0.
    per_packet=math.log1p(total/max(pkts,1.)) if mt and mp else 0.
    values=[math.log1p(dur),math.log1p(pkts),math.log1p(total),math.log1p(source),
            fractions,per_packet,sport,dport,float(proto=='tcp'),float(proto=='udp'),
            float(proto=='icmp'),float('S' in state),float('F' in state),float('R' in state)]
    observed=[md,mp,mt,ms,ms and mt,mt and mp,msp,mdp,mproto,mproto,mproto,mstate,mstate,mstate]
    # Reject overflow when exporting float32 rather than inventing finite values.
    for i,value in enumerate(values):
        if not math.isfinite(value) or abs(value)>3.4028234663852886e38:
            values[i]=0.;observed[i]=False
    return values,observed


def identity_hash(value):
    if value.strip() in ('','?'):return 0
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest()[:8],'little')


def prepare_one(source,download_record):
    import numpy as np
    path=ROOT/download_record['path']
    if path.stat().st_size!=download_record['bytes'] or sha(path)!=download_record['sha256']:
        raise ValueError(f'Raw file changed: {path}')
    columns=json.loads(SOURCE.read_text())['columns']
    features=array('f');observed=array('b');labels=array('b');timestamps=array('d')
    row_hashes=set();duplicates=0
    rowids=array('q');src_ids=array('Q');dst_ids=array('Q')
    counts=Counter();retained_counts=Counter();excluded=Counter();total_rows=0
    min_all=None;max_all=None;max_completion=None
    missing_source=0;missing_destination=0;unknown_duration=0
    with path.open('r',encoding='utf-8',newline='') as stream:
        reader=csv.reader(stream)
        header=next(reader)
        if header!=columns:raise ValueError(f'Unexpected schema for scenario {source["scenario"]}: {header}')
        for rowid,row in enumerate(reader):
            total_rows+=1
            if len(row)!=len(columns):raise ValueError(f'Malformed CSV record {rowid} of {path.name}')
            label=row[14].strip();counts[label]+=1
            bot='From-Botnet' in label;normal='From-Normal' in label
            if bot==normal:
                excluded['ambiguous_both' if bot else 'unknown_label']+=1
                continue
            row_digest=hashlib.sha256('\x1f'.join(row).encode('utf-8')).digest()
            if row_digest in row_hashes:duplicates+=1
            row_hashes.add(row_digest)
            # Naive capture-local clock interpreted with a fixed UTC epoch anchor.
            # It is metadata for ordering/splits, never a predictive feature or a
            # claim that the provider's capture clock was actually UTC.
            dt=datetime.fromisoformat(row[0].replace('/','-'))
            if dt.tzinfo is not None:raise ValueError('Unexpected timezone-bearing timestamp')
            timestamp=dt.replace(tzinfo=timezone.utc).timestamp()
            values,mask=observed_features(row)
            features.extend(values);observed.extend(mask);labels.append(bot)
            timestamps.append(timestamp);rowids.append(rowid)
            src_id=identity_hash(row[3]);dst_id=identity_hash(row[6])
            src_ids.append(src_id);dst_ids.append(dst_id)
            missing_source+=src_id==0;missing_destination+=dst_id==0
            duration,duration_known=nonnegative(row[1])
            completion=timestamp+duration
            if duration_known and math.isfinite(completion):
                max_completion=completion if max_completion is None else max(max_completion,completion)
            else:unknown_duration+=1
            retained_counts[label]+=1
            if min_all is None or dt<min_all:min_all=dt
            if max_all is None or dt>max_all:max_all=dt
    count=len(labels)
    if not count:raise ValueError(f'No supervised records in scenario {source["scenario"]}')
    time_array=np.asarray(timestamps,dtype=np.float64)
    row_array=np.asarray(rowids,dtype=np.int64)
    order=np.lexsort((row_array,time_array))
    arrays=dict(features=np.asarray(features,dtype=np.float32).reshape(count,14)[order],
                observed=np.asarray(observed,dtype=np.bool_).reshape(count,14)[order],
                label=np.asarray(labels,dtype=np.bool_)[order],timestamp=time_array[order],
                row_id=row_array[order],src=np.asarray(src_ids,dtype=np.uint64)[order],
                dst=np.asarray(dst_ids,dtype=np.uint64)[order])
    output=OUT/'parsed'/f'scenario_{source["scenario"]:02d}.npz'
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    record=dict(scenario=source['scenario'],capture_id=source['capture_id'],family=source['family'],
        split=source['split'],raw_path=str(path.relative_to(ROOT)),raw_bytes=path.stat().st_size,
        raw_sha256=download_record['sha256'],prepared_path=str(output.relative_to(ROOT)),
        prepared_sha256=sha(output),prepared_bytes=output.stat().st_size,rows_total=total_rows,
        retained_exact_duplicate_rows=duplicates,duplicate_policy='Counted but retained; exact original CSV row equality includes timestamp and addresses.',
        numpy_version=np.__version__,rows_retained=count,rows_attack=int(arrays['label'].sum()),
        rows_normal=int((~arrays['label']).sum()),rows_excluded=total_rows-count,
        excluded_reasons=dict(sorted(excluded.items())),labels_all=dict(sorted(counts.items())),
        labels_retained=dict(sorted(retained_counts.items())),schema=header,
        retained_time_min=min_all.isoformat(),retained_time_max=max_all.isoformat(),
        retained_completion_max=datetime.fromtimestamp(max_completion,tz=timezone.utc).replace(tzinfo=None).isoformat() if max_completion is not None else None,
        missing_source_endpoints=missing_source,missing_destination_endpoints=missing_destination,unknown_durations=unknown_duration,
        unobserved_features={name:int((~arrays['observed'][:,i]).sum()) for i,name in enumerate(FEATURES)},
        sorted_by=['timestamp','row_id'],source=download_record)
    print(json.dumps(dict(scenario=source['scenario'],parsed=True,rows_total=total_rows,rows_retained=count,
                          attacks=record['rows_attack'],normal=record['rows_normal'])),flush=True)
    return record


def prepare():
    source=json.loads(SOURCE.read_text());downloads=json.loads(RECORD.read_text())
    if not downloads.get('complete') or len(downloads['files'])!=13:
        raise ValueError('Finish all thirteen downloads before preparing the frozen data split')
    by_scenario={r['scenario']:r for r in downloads['files']};records=[]
    MANIFEST.parent.mkdir(parents=True,exist_ok=True)
    for item in source['sources']:
        records.append(prepare_one(item,by_scenario[item['scenario']]))
        manifest=dict(dataset=source['dataset'],source_manifest_sha256=sha(SOURCE),download_manifest_sha256=sha(RECORD),
            preparation_script_sha256=sha(Path(__file__)),features=FEATURES,feature_count=14,
            timestamp_semantics='Naive provider capture-local timestamp mapped to a fixed UTC epoch for ordering only; not an assertion of true UTC time. Excluded from model features.',
            identity_semantics='Metadata only: first64 bits of SHA256(address UTF8), little endian uint64. Hashing is not a claim of strong anonymization. Raw identities never enter model features or committed tables.',
            target='label=True only for From-Botnet; False only for From-Normal; other or ambiguous labels excluded.',
            splits=source['splits'],files=records,complete=False,
            preprocessing='CPU parsing, feature transformations, metadata sorting and I/O only. No fitting, normalization or model inference. GPU scaling is fitted later from training observations only.')
        MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    # Assert the intended temporal separation using the actual capture clocks.
    latest_train_start=max(r['retained_time_max'] for r in records if r['split']=='train')
    latest_train=max(r['retained_completion_max'] for r in records if r['split']=='train')
    earliest_val=min(r['retained_time_min'] for r in records if r['split']=='validation')
    latest_val_start=max(r['retained_time_max'] for r in records if r['split']=='validation')
    latest_val=max(r['retained_completion_max'] for r in records if r['split']=='validation')
    earliest_test=min(r['retained_time_min'] for r in records if r['split']=='test')
    if any(r['unknown_durations'] for r in records):
        raise ValueError('Unknown flow durations prevent a full chronological completion check')
    if not latest_train<earliest_val or not latest_val<earliest_test:
        raise ValueError('Provider timestamps do not support the prespecified chronological split')
    manifest['temporal_split_check']=dict(passed=True,latest_train_start=latest_train_start,latest_train_completion=latest_train,earliest_validation_start=earliest_val,latest_validation_start=latest_val_start,latest_validation_completion=latest_val,earliest_test_start=earliest_test)
    manifest['totals']={split:{key:sum(r[key] for r in records if r['split']==split) for key in ['rows_total','rows_retained','rows_attack','rows_normal','rows_excluded']} for split in ['train','validation','test']}
    manifest['complete']=True
    MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['download','prepare','all'])
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers not in range(1,5):raise ValueError('Use one to four concurrent downloads')
    if args.action in ['download','all']:download(args.workers)
    if args.action in ['prepare','all']:prepare()


if __name__=='__main__':main()
