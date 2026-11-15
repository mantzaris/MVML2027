"""Acquire only selected public WAV members with exact ranges and CRC checks.

No 18 GB archive copy is needed. Original PCM WAV bytes and decoded float32
hashes are retained outside Git. Interrupted members are retried, never trusted.
"""
from __future__ import annotations
import concurrent.futures as cf
import hashlib, io, json, struct, threading, time, zlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
from underwater_reasoning.io import atomic_json, file_hash, canonical_hash
from probe_release import PARTS

ROOT=Path(__file__).resolve().parents[1]
SIZES=[10737418240,7059560361]
LOCAL=threading.local()

def read(disk,offset,size):
    if offset+size>SIZES[disk]:
        first=SIZES[disk]-offset
        return read(disk,offset,first)+read(disk+1,0,size-first)
    if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
    end=offset+size-1
    with LOCAL.session.get('https://drive.usercontent.google.com/download',
        params={'id':PARTS[disk],'export':'download','confirm':'t'},
        headers={'Range':f'bytes={offset}-{end}'},timeout=90) as response:
        response.raise_for_status()
        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{SIZES[disk]}':
            raise ValueError('Byte-range mismatch')
        data=response.content
    if len(data)!=size:raise ValueError('Range length mismatch')
    return data

def acquire(item):
    row,entry=item;out=ROOT/row['audio_path'];out.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(6):
        try:
            if out.exists():data=out.read_bytes()
            else:
                # This release's local extra fields fit within 1 KiB; validate below.
                data=read(entry['disk'],entry['offset'],entry['packed']+1024)
                header=struct.unpack_from('<4s5H3I2H',data)
                if header[0]!=b'PK\x03\x04':raise ValueError('Bad local ZIP header')
                start=30+header[-2]+header[-1]
                if start>1024:raise ValueError('Local header exceeds audited bound')
                data=data[start:start+entry['packed']]
                if entry['compression']==8:data=zlib.decompress(data,-15)
                elif entry['compression']!=0:raise ValueError('Unknown compression')
            if len(data)!=entry['unpacked'] or zlib.crc32(data)!=entry['crc32']:raise ValueError('CRC mismatch')
            x,fs=sf.read(io.BytesIO(data),dtype='float32')
            if fs!=32000 or x.shape!=(160000,) or not np.isfinite(x).all():raise ValueError('Audio contract mismatch')
            if not out.exists():
                tmp=out.with_suffix('.part');tmp.write_bytes(data);tmp.replace(out)
            return {'id':row['id'],'file_sha256':hashlib.sha256(data).hexdigest(),
                'waveform_sha256':hashlib.sha256(str(fs).encode()+x.astype('<f4').tobytes()).hexdigest(),
                'bytes':len(data),'partition':row['partition']}
        except Exception:
            if attempt==5:raise
            time.sleep(min(2**attempt,20))

def main():
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text())
    index={e['name']:e for e in json.loads((ROOT/'.cache/release_index.json').read_text())}
    rows=manifest['records'];items=[(r,index[r['archive_name']]) for r in rows]
    # Complete training and development first so GPU preparation can overlap.
    order={'acoustic_fit':0,'development':1,'policy_fit':2,'calibration':3,'test':4}
    items.sort(key=lambda t:(order[t[0]['partition']],t[0]['id']))
    results=[];started=time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(acquire,items):
            results.append(result)
            if len(results)%100==0:
                atomic_json(ROOT/'data/private/download_progress.json',{'complete':len(results),'total':len(rows),'elapsed_seconds':time.monotonic()-started})
                print(len(results),'/',len(rows),'seconds',round(time.monotonic()-started),flush=True)
    atomic_json(ROOT/'data/private/waveform_hashes.json',results)
    by_hash=defaultdict(list)
    for r in results:by_hash[r['waveform_sha256']].append(r)
    collisions=[v for v in by_hash.values() if len({r['partition'] for r in v})>1]
    atomic_json(ROOT/'results/manifests/audio_download_audit.json',{
        'clips':len(results),'bytes':sum(r['bytes'] for r in results),'manifest_hash':canonical_hash(manifest),
        'decoded_hash_manifest_sha256':file_hash(ROOT/'data/private/waveform_hashes.json'),
        'cross_partition_hash_collisions':len(collisions),'duplicate_hash_groups':sum(len(v)>1 for v in by_hash.values()),
        'all_crc_checked':True,'native_sample_rate':32000,'channels':1,'duration_seconds':5,
        'elapsed_seconds':round(time.monotonic()-started,2)})
    if collisions:raise ValueError('Cross-partition decoded duplicates; resolve before training')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
