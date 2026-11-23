#!/usr/bin/env python3
"""Resume verified PORTIA metadata/split/download work entirely inside this project.

Usage: .venv/bin/python scripts/prepare_data.py --stage split
       .venv/bin/python scripts/prepare_data.py --stage download --workers 4
No learned model is run. Failed downloads never become successful manifest rows.
"""
from __future__ import annotations
import argparse, concurrent.futures, json, os, sys, time, traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uac.data import (RemoteSplitZip,fetch_portia_metadata,portia_records,design_split,
                      acquire_member,audit_rows,atomic_bytes,save_json,digest,object_hash,load_rows)


def jsonl(rows):return ''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows).encode()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['metadata','split','download','audit'],required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--limit',type=int,default=0,help='Positive value performs only a bounded acquisition pilot; never finalizes a partial cohort')
    args=ap.parse_args();out=ROOT/'data/manifests';out.mkdir(parents=True,exist_ok=True)
    if args.stage=='audit':
        rows=load_rows(ROOT,verify=True);audit=audit_rows(rows)
        if len({r['sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate complete audio file')
        if len({r['decoded_channel_sha256'] for r in rows})!=len(rows):raise ValueError('Duplicate decoded analysis-channel audio')
        manifest_hash=digest((out/'real.jsonl').read_bytes())
        previous=json.loads((out/'audio_audit.json').read_text()) if (out/'audio_audit.json').exists() else {}
        completed_utc=previous.get('completed_utc') if previous.get('manifest_sha256')==manifest_hash else None
        audit.update(manifest_sha256=manifest_hash,audio_bytes=sum(r['bytes'] for r in rows),
            all_headers_verified=True,all_member_crcs_verified=True,decoded_channel_duplicates=0,complete=True,
            completed_utc=completed_utc or datetime.now(timezone.utc).isoformat(),
            archive_integrity_scope='selected ZIP members only; full archive MD5 not claimed',release_url='https://zenodo.org/records/21409381')
        save_json(out/'audio_audit.json',audit);print(json.dumps(audit,indent=2));return
    parts=fetch_portia_metadata(ROOT)
    if args.stage=='metadata':
        print(json.dumps({'metadata_ready':True,'parts':len(parts),'archive_bytes':sum(p['size'] for p in parts)},indent=2));return
    planned=out/'planned_real.jsonl'
    if args.stage=='split':
        rows,exclusions,audit=design_split(portia_records(ROOT),seed=17)
        data=jsonl(rows)
        if planned.exists() and planned.read_bytes()!=data:raise ValueError('Refusing to mutate a frozen cohort; use a documented new cohort version')
        atomic_bytes(planned,data);save_json(out/'split_exclusions.json',exclusions)
        conflict=out/'cross_task_conflicts.json'
        audit.update(planned_manifest_sha256=digest(data),cross_task_conflicts_sha256=digest(conflict.read_bytes()),
                     release_metadata_sha256={p.name:digest(p.read_bytes()) for p in sorted((out/'portia_release').glob('*.csv'))},
                     locked_before_acoustic_model_outcomes=True)
        save_json(out/'split_audit.json',audit)
        print(json.dumps(audit,indent=2));return
    if not planned.exists():raise ValueError('Run --stage split and inspect the audit before download')
    rows=[json.loads(s) for s in planned.read_text().splitlines()]
    archive=RemoteSplitZip(parts,out/'portia_release/ranges')
    idx=out/'portia_release/audio_index.json'
    entries=json.loads(idx.read_text()) if idx.exists() else archive.entries()
    if not idx.exists():save_json(idx,entries)
    index={Path(e['name']).name:e for e in entries if e['name'].endswith('.wav')}
    if len(index)!=sum(e['name'].endswith('.wav') for e in entries):raise ValueError('Nonunique audio archive basenames')
    if any(r['window_id'] not in index for r in rows):raise ValueError('Planned audio absent from public archive')
    receipts=out/'download_records';receipts.mkdir(exist_ok=True)
    todo=[];completed=[]
    for row in rows:
        p=receipts/(row['id']+'.json')
        if p.exists():
            item=json.loads(p.read_text());audio=ROOT/item['path']
            if audio.exists() and digest(audio.read_bytes())==item['sha256']:
                completed.append(item);continue
        todo.append(row)
    # Group access by disk/offset, improving provider locality; training starts only after completion.
    rank={'train':0,'development':1,'calibration':2,'test':3}
    todo.sort(key=lambda r:(rank[r['split']],index[r['window_id']]['disk'],index[r['window_id']]['offset']))
    if args.limit:todo=todo[:args.limit]
    print(json.dumps({'already_verified':len(completed),'to_acquire':len(todo),'total':len(rows),'partial_pilot':bool(args.limit)}),flush=True)
    errors=[];started=time.monotonic();published=set()
    def publish_completed_splits():
        for split in ('train','development','calibration','test'):
            current=sorted((r for r in completed if r['split']==split),key=lambda r:r['id'])
            expected=sum(r['split']==split for r in rows)
            if split in published or len(current)!=expected:continue
            data=jsonl(current);path=out/('real_'+split+'.jsonl')
            if path.exists() and path.read_bytes()!=data:raise ValueError('Immutable completed split manifest mismatch')
            if len({r['decoded_channel_sha256'] for r in current})!=len(current):raise ValueError('Duplicate audio within completed split')
            atomic_bytes(path,data);published.add(split)
            print(json.dumps({'split_complete':split,'rows':expected,'manifest_sha256':digest(data)}),flush=True)
    publish_completed_splits()
    def acquire(row):
        result=acquire_member(ROOT,archive,index[row['window_id']],row)
        save_json(receipts/(row['id']+'.json'),result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(acquire,r):r for r in todo}
        for future in concurrent.futures.as_completed(futures):
            row=futures[future]
            try:
                completed.append(future.result())
                publish_completed_splits()
            except Exception as exc:
                errors.append({'id':row['id'],'error':str(exc),'type':type(exc).__name__})
                print('FAILED',row['id'],str(exc)[:200],flush=True)
            if (len(completed)+len(errors))%25==0:
                status={'verified':len(completed),'expected':len(rows),'errors':len(errors),'elapsed_seconds':round(time.monotonic()-started,1)}
                save_json(out/'download_progress.json',status);print(json.dumps(status),flush=True)
    save_json(out/'download_errors.json',errors)
    if errors:raise RuntimeError(f'{len(errors)} downloads failed; completed files retained; rerun same command')
    if len(completed)!=len(rows):
        print('Pilot complete; full real.jsonl not published until every planned file validates.',flush=True);return
    completed.sort(key=lambda r:r['id'])
    if len({r['decoded_channel_sha256'] for r in completed})!=len(completed):
        dup=Counter(r['decoded_channel_sha256'] for r in completed)
        save_json(out/'duplicate_audio.json',[r for r in completed if dup[r['decoded_channel_sha256']]>1])
        raise ValueError('Duplicate decoded audio: resolve with a documented exclusion before fitting')
    atomic_bytes(out/'real.jsonl',jsonl(completed))
    audit=audit_rows(completed);audit.update(manifest_sha256=digest((out/'real.jsonl').read_bytes()),audio_bytes=sum(r['bytes'] for r in completed),
         all_headers_verified=True,all_member_crcs_verified=True,decoded_channel_duplicates=0,complete=True,completed_utc=datetime.now(timezone.utc).isoformat(),
         archive_integrity_scope='selected ZIP members only; full archive MD5 not claimed',release_url='https://zenodo.org/records/21409381')
    save_json(out/'audio_audit.json',audit);save_json(out/'download_progress.json',{'verified':len(completed),'expected':len(rows),'errors':0,'complete':True})
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
