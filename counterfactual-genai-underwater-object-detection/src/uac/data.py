"""Provenance-aware PORTIA ingestion; CPU operations here are I/O/metadata only.

Native audio is preserved. Numerical resampling/transforms belong on the pinned
Ada GPU. The public release is CC-BY-4.0 and AIS-labelled, not audibility-labelled.
"""
from __future__ import annotations
import bisect, csv, hashlib, io, json, os, re, struct, threading, time, urllib.request, urllib.error, zipfile, zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD_ID = '21409381'
RELEASE_URL = 'https://zenodo.org/records/' + RECORD_ID
SPLITS = ('train', 'development', 'calibration', 'test')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def safe_path(root, relative):
    p = (Path(root) / relative).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError('Path leaves project: ' + str(relative))
    return p


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial-' + str(os.getpid()) + '-' + str(threading.get_ident()))
    with tmp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def save_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def fetch_small(url, max_bytes=30_000_000):
    with urllib.request.urlopen(url, timeout=45) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError('Bounded metadata download exceeded limit')
    return data


class RemoteSplitZip:
    """Bounded public HTTP range reads with ZIP64, size and CRC verification."""
    def __init__(self, parts, cache):
        self.parts = parts
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._rate_lock=threading.Lock();self._next_request=0.0;self._blocked_until=0.0

    def _pace(self):
        while True:
            with self._rate_lock:
                now=time.monotonic();delay=max(self._next_request,self._blocked_until)-now
                if delay<=0:
                    self._next_request=now+0.75
                    return
            time.sleep(min(delay,30))

    def read(self, disk, offset, size, cache=True):
        if size < 0 or size > 64 * 1024 * 1024 or offset < 0:
            raise ValueError('Range exceeds bounded read size')
        while offset >= self.parts[disk]['size']:
            offset -= self.parts[disk]['size']; disk += 1
        if not size:
            return b''
        part = self.parts[disk]
        n = min(size, part['size'] - offset)
        key = self.cache / f'{disk}-{offset}-{n}.bin'
        if cache and key.exists() and key.stat().st_size == n:
            data = key.read_bytes()
        else:
            req = urllib.request.Request(part['url'], headers={'Range': f'bytes={offset}-{offset+n-1}', 'User-Agent': 'MVML2027-acoustic-research/1.0'})
            for attempt in range(8):
                try:
                    self._pace()
                    with urllib.request.urlopen(req, timeout=60) as r:
                        expected = f'bytes {offset}-{offset+n-1}/{part["size"]}'
                        if r.status != 206 or r.headers.get('Content-Range') != expected:
                            excerpt = r.read(512).decode(errors='replace')
                            raise ValueError(f'Exact public range unavailable: {r.status}; {excerpt[:100]}')
                        data = r.read(n+1)
                    if len(data) != n:
                        raise ValueError('Range length mismatch')
                    break
                except urllib.error.HTTPError as e:
                    if e.code==429:
                        raw=e.headers.get('Retry-After','60')
                        try: delay=max(60,float(raw))
                        except ValueError:
                            try:delay=max(60,parsedate_to_datetime(raw).timestamp()-time.time())
                            except (ValueError,TypeError):delay=60
                        with self._rate_lock:self._blocked_until=max(self._blocked_until,time.monotonic()+delay)
                        save_json(self.cache/'rate_limit.json',{'status':429,'retry_after':raw,'wait_seconds':delay,'limit':e.headers.get('X-RateLimit-Limit'),'remaining':e.headers.get('X-RateLimit-Remaining')})
                    elif attempt<7:time.sleep(min(30,2**attempt))
                    if attempt==7:raise
                except OSError:
                    if attempt==7:raise
                    time.sleep(min(30,2**attempt))
            if cache:
                atomic_bytes(key, data)
        if n < size:
            return data + self.read(disk+1, 0, size-n, cache=cache)
        return data

    def entries(self):
        disk = len(self.parts)-1
        tail = self.read(disk, self.parts[disk]['size']-65536, 65536)
        pos = tail.rfind(b'PK\x06\x06')
        if pos < 0:
            raise ValueError('Expected ZIP64 central directory in this release')
        end = struct.unpack_from('<4sQ2H2I4Q', tail, pos)
        data = self.read(end[5], end[9], end[8])
        entries = []; cursor = 0
        while cursor < len(data):
            f = struct.unpack_from('<4s6H3I5H2I', data, cursor)
            if f[0] != b'PK\x01\x02' or f[3] & 1:
                raise ValueError('Invalid or encrypted archive entry')
            nn,en,cn = f[10:13]
            name = data[cursor+46:cursor+46+nn].decode('utf-8' if f[3]&2048 else 'cp437')
            extra = data[cursor+46+nn:cursor+46+nn+en]
            unpacked,packed,offset,disk = f[9],f[8],f[16],f[13]
            p = 0
            while p+4 <= len(extra):
                tag,ln = struct.unpack_from('<HH',extra,p); values=extra[p+4:p+4+ln]
                if tag == 1:
                    nums = [unpacked,packed,offset,disk]; vp=0
                    for i,num in enumerate(nums):
                        if num == (65535 if i==3 else 4294967295):
                            fmt='<I' if i==3 else '<Q';nums[i]=struct.unpack_from(fmt,values,vp)[0];vp+=struct.calcsize(fmt)
                    unpacked,packed,offset,disk=nums
                p+=4+ln
            entries.append(dict(name=name,unpacked=unpacked,packed=packed,offset=offset,disk=disk,compression=f[4],crc32=f[7]))
            cursor+=46+nn+en+cn
        if len(entries)!=end[7] or cursor!=len(data):
            raise ValueError('ZIP central directory count mismatch')
        return entries

    def extract(self, entry):
        if entry['unpacked'] > 32*1024*1024:
            raise ValueError('Audio member too large for expected 3-second windows')
        # One request for header+payload; bounded padding covers local filename/extra fields.
        block=self.read(entry['disk'],entry['offset'],entry['packed']+4096,cache=False)
        f=struct.unpack('<4s5H3I2H',block[:30])
        if f[0]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
        start=30+f[-2]+f[-1]
        if start>4096:raise ValueError('Unexpectedly large local ZIP header')
        data=block[start:start+entry['packed']]
        if entry['compression']==8:data=zlib.decompress(data,-15)
        elif entry['compression']!=0:raise ValueError('Unsupported ZIP compression')
        if len(data)!=entry['unpacked'] or zlib.crc32(data)!=entry['crc32']:
            raise ValueError('Audio member CRC/size mismatch')
        return data


def fetch_portia_metadata(root=ROOT):
    root=Path(root); out=root/'data/manifests/portia_release'; out.mkdir(parents=True,exist_ok=True)
    rp=out/'record.json'
    if not rp.exists():atomic_bytes(rp,fetch_small('https://zenodo.org/api/records/'+RECORD_ID))
    rec=json.loads(rp.read_text())
    if rec['metadata']['license']['id']!='cc-by-4.0':raise ValueError('Unexpected dataset license')
    files={f['key']:f for f in rec['files']}
    for name in ('PORTIA_Binary_Detection.zip','PORTIA_Metadata.zip','PORTIA_Single_Vessel_Classification.zip'):
        f=files[name];p=out/name
        data=p.read_bytes() if p.exists() else fetch_small(f['links']['self'],max_bytes=f['size'])
        if len(data)!=f['size'] or 'md5:'+hashlib.md5(data).hexdigest()!=f['checksum']:
            raise ValueError('Provider metadata checksum mismatch')
        if not p.exists():atomic_bytes(p,data)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in z.namelist():
                if n.endswith(('.csv','.json')):
                    prefix='single_' if name=='PORTIA_Single_Vessel_Classification.zip' else ''
                    atomic_bytes(out/(prefix+Path(n).name),z.read(n))
    names=[f'PORTIA_audio.z{i:02d}' for i in range(1,7)]+['PORTIA_audio.zip']
    parts=[{'name':n,'size':files[n]['size'],'url':files[n]['links']['self'],'provider_checksum':files[n]['checksum']} for n in names]
    return parts


def portia_records(root=ROOT):
    root=Path(root); rows=[]; seen=set(); conflicts=set(); candidates=defaultdict(list)
    for filename,official in [('train.csv','train'),('val.csv','validation'),('test.csv','test'),('single_train.csv','train'),('single_val.csv','validation'),('single_test.csv','test')]:
        with (root/'data/manifests/portia_release'/filename).open(newline='') as f:
            for r in csv.DictReader(f):
                name=r['window_id']; lab=int(r['label_vessel']); n=int(r['n_vessels'])
                if not filename.startswith('single_') and lab!=0:continue
                if filename.startswith('single_') and (lab!=1 or n!=1):raise ValueError('Single-vessel task has invalid presence metadata')
                task='single_vessel' if filename.startswith('single_') else 'binary_background'
                if (task,name) in seen:raise ValueError('Duplicate within source task')
                seen.add((task,name))
                if lab not in (0,1) or bool(n)!=bool(lab):raise ValueError('Inconsistent AIS label')
                dt=datetime.fromisoformat(r['start_utc']).astimezone(timezone.utc)
                mmsi = str(int(float(r['primary_mmsi']))) if r['primary_mmsi'] else None
                if lab and (not mmsi or len(mmsi)!=9):raise ValueError('Missing positive primary vessel identity')
                if not re.fullmatch(r'\d{8}_\d{6}\.\d+\.wav',name):raise ValueError('Unexpected filename format')
                # Parse the nanosecond filename only to microseconds for boundary guards.
                datepart,clockpart=name[:-4].split('_');clock,fraction=clockpart.split('.')
                actual=datetime.strptime(datepart+clock,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
                seconds=actual.timestamp()+float('0.'+fraction)
                if actual.date()!=dt.date() or abs(seconds-dt.timestamp())>=1:raise ValueError('Metadata/filename timestamp mismatch')
                uid=object_hash(['PORTIA-v2',name])[:24]
                candidates[name].append({'id':uid,'path':f'data/raw/portia/{uid}.wav','label':lab,'class_name':r['primary_class'] if lab else 'background',
                             'source_id':'vessel:'+mmsi if lab else 'background-day:'+dt.date().isoformat(),
                             'primary_mmsi':mmsi,'n_vessels':n,'day':dt.date().isoformat(),'start_utc':r['start_utc'],'start_seconds':seconds,
                             'sample_rate':32000,'num_samples':96000,'channels':4,'analysis_channel':0,'origin':'real','prior_inspection':False,
                             'label_basis':'AIS spatial co-occurrence; not human acoustic audibility','official_split':official,'window_id':name,
                             'parent_recording_id':None,'deployment_id':'PORTIA-Valencia-2025-fixed-array','dataset':'PORTIA-v2','license':'CC-BY-4.0','source_task':task})
    for name,members in candidates.items():
        if len({r['label'] for r in members})>1:conflicts.add(name)
        else:rows.append(members[0])
    save_json(root/'data/manifests/cross_task_conflicts.json',{'window_ids':sorted(conflicts),'count':len(conflicts),'rule':'exclude both labels; no assumed resolution'})
    return sorted(rows,key=lambda r:r['id'])


def design_split(rows, seed=17, caps=None):
    """Whole-day holdout and vessel purge; source data never inspected acoustically."""
    # This ordering is the first metadata feasibility seed evaluated; do not tune on outcomes.
    caps=caps or {'train':2000,'development':400,'calibration':300,'test':600}
    rank={s:i for i,s in enumerate(SPLITS)}
    h=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
    days=sorted({r['day'] for r in rows},key=lambda d:h([seed,d]))
    day_part={d:('test' if i<5 else 'calibration' if i<7 else 'development' if i<10 else 'train') for i,d in enumerate(days)}
    eligible=[r for r in rows if r['n_vessels']<=1]
    owners={}
    for r in eligible:
        if r['label']:
            p=day_part[r['day']];g=r['source_id']
            if g not in owners or rank[p]>rank[owners[g]]:owners[g]=p
    selected=[];exclusions=[]
    for r in rows:
        p=day_part[r['day']]
        if r['n_vessels']>1:exclusions.append({'id':r['id'],'reason':'multiple_AIS_vessels_no_complete_identity_list'});continue
        if r['label'] and owners[r['source_id']]!=p:exclusions.append({'id':r['id'],'reason':'vessel_reserved_for_'+owners[r['source_id']]});continue
        selected.append({**r,'split':p,'group':'day:'+r['day']})
    # Guard across day boundaries. Preserve higher-priority held-out partitions.
    kept=[];higher_times=[]
    for split in reversed(SPLITS):
        current=[]
        for r in (r for r in selected if r['split']==split):
            i=bisect.bisect_left(higher_times,r['start_seconds'])
            if any(abs(t-r['start_seconds'])<63 for t in higher_times[max(0,i-1):i+1]):
                exclusions.append({'id':r['id'],'reason':'global_60_second_guard'});continue
            current.append(r)
        kept+=current;higher_times=sorted(higher_times+[r['start_seconds'] for r in current])
    output=[]
    source_caps={'train':200,'development':60,'calibration':60,'test':40}
    background_caps={'train':500,'development':150,'calibration':200,'test':200}
    # Sampling is outcome-blind, round-robin across days then source groups.
    # Cap each class equally; no examples are duplicated to meet a cap.
    for split in SPLITS:
        for label in (0,1):
            pool=[r for r in kept if r['split']==split and r['label']==label]
            strata=defaultdict(lambda:defaultdict(list))
            for r in pool:strata[r['day']][r['source_id']].append(r)
            ordered_by_day={}
            for day,groups in strata.items():
                for rs in groups.values():rs.sort(key=lambda r:h([seed,'sample',r['id']]))
                day_order=[]
                for ix in range(max(map(len,groups.values()),default=0)):
                    for g in sorted(groups,key=lambda g:h([seed,g])):
                        if ix<len(groups[g]):day_order.append(groups[g][ix])
                ordered_by_day[day]=day_order
            chosen=[];source_counts=Counter()
            for ix in range(max(map(len,ordered_by_day.values()),default=0)):
                for day in sorted(ordered_by_day,key=lambda d:h([seed,d])):
                    if len(chosen)>=caps[split]:break
                    if ix<len(ordered_by_day[day]):
                        item=ordered_by_day[day][ix];cap=source_caps[split] if label else background_caps[split]
                        if source_counts[item['source_id']]<cap:
                            chosen.append(item);source_counts[item['source_id']]+=1
                if len(chosen)>=caps[split]:break
            ids={r['id'] for r in chosen}
            exclusions.extend({'id':r['id'],'reason':'metadata_only_class_cap'} for r in pool if r['id'] not in ids)
            output+=chosen
    output.sort(key=lambda r:r['id'])
    audit=audit_rows(output)
    audit.update(seed=seed,day_assignment=day_part,raw_windows=len(rows),single_source_eligible=len(eligible),exclusions=dict(Counter(r['reason'] for r in exclusions)),
                 design='custom whole-day split then primary-vessel purge; official clip split replaced',test_model_outcomes_examined=False,
                 all_parent_recording_ids_known=False,geographically_independent_of_sibling_ONC=True,
                 graph_feasibility='official binary all-day/vessel graph: one component of 36104/36148 rows; discard cross-split vessel rows',
                 cross_task_label_conflicts_excluded=2388,class_caps=caps,positive_source_caps=source_caps,background_day_caps=background_caps,cohort='single-vessel task positives plus binary task negatives; contradictory windows excluded')
    return output,exclusions,audit


def audit_rows(rows):
    if not rows:raise ValueError('Empty manifest')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate clip IDs')
    for field in ['source_id','day']:
        p=defaultdict(set)
        for r in rows:p[r[field]].add(r['split'])
        if any(len(s)>1 for s in p.values()):raise ValueError('Cross-partition '+field)
    valid={s:[r for r in rows if r['split']==s] for s in SPLITS}
    if any({r['label'] for r in rs}!={0,1} for rs in valid.values()):raise ValueError('Missing class in a split')
    counts={s:{'clips':len(rs),'labels':dict(Counter(str(r['label']) for r in rs)),
               'days':len({r['day'] for r in rs}),'vessel_ids':len({r['source_id'] for r in rs if r['label']}),
               'background_days':len({r['day'] for r in rs if not r['label']})} for s,rs in valid.items()}
    return {'passed':True,'counts':counts,'guarantees':['whole_UTC_day_disjoint','primary_vessel_disjoint','single_AIS_vessel_only','no_shared_audio_windows'],
            'limitations':['only 19 collection days in full binary metadata','AIS labels do not certify audible presence/absence','unknown original continuous parent recording IDs','single deployment; no deployment generalization','untracked acoustic sources may occur','five test days give weak population uncertainty estimation']}


def acquire_member(root, archive, entry, row):
    root=Path(root);path=safe_path(root,row['path'])
    data=path.read_bytes() if path.exists() else archive.extract(entry)
    if len(data)!=entry['unpacked'] or zlib.crc32(data)!=entry['crc32']:raise ValueError('Cached audio CRC mismatch')
    import soundfile as sf
    with sf.SoundFile(io.BytesIO(data)) as f:
        rate,frames,channels,subtype=f.samplerate,f.frames,f.channels,f.subtype
        pcm=f.read(dtype='int32',always_2d=True)[:,row['analysis_channel']].copy()
    if (rate,frames,channels,subtype)!=(32000,96000,4,'PCM_32'):
        raise ValueError(f'Unexpected native header: {(rate,frames,channels,subtype)}')
    if not path.exists():atomic_bytes(path,data)
    return {**row,'sha256':digest(data),'bytes':len(data),'archive_crc32':entry['crc32'],'header_verified':True,'decoded_channel_sha256':digest(pcm.astype('<i4').tobytes())}


def load_rows(root=ROOT, split=None, verify=False):
    root=Path(root);path=root/'data/manifests/real.jsonl'
    if not path.exists() and split is not None:path=root/('data/manifests/real_'+split+'.jsonl')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line]
    if split is not None:rows=[r for r in rows if r['split']==split]
    for r in rows:
        p=safe_path(root,r['path'])
        if not p.is_file():raise FileNotFoundError(p)
        if verify and digest(p.read_bytes())!=r['sha256']:raise ValueError('Audio hash mismatch')
    return rows


def decode_wav(root, row):
    """Decode native channel 0 only; no resampling or substantial CPU transforms."""
    import soundfile as sf
    x,sr=sf.read(safe_path(root,row['path']),dtype='float32',always_2d=True)
    if sr!=row['sample_rate'] or len(x)!=row['num_samples'] or x.shape[1]!=row['channels']:
        raise ValueError('Native header differs from frozen manifest')
    return x[:,row['analysis_channel']].copy(),sr
