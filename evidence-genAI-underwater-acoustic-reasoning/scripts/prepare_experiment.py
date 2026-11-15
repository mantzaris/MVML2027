"""Prospective source/day-separated amendment using actual release metadata.

No fictional parent/deployment IDs are created. All unknown parent IDs remain
unknown; the actual guarantees are vessel, whole-UTC-day, temporal, and hash
separation. Missing parent provenance remains an explicit study limitation.
"""
from __future__ import annotations
import csv
import json
import posixpath
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from underwater_reasoning.io import atomic_json, canonical_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {"background":"background", "cargo":"cargo", "passengership":"passenger", "tanker":"tanker", "tug":"tug_tow"}
PARTS = ["acoustic_fit", "policy_fit", "development", "calibration", "test"]
LABELS = list(LABEL_MAP.values())


def load_records():
    rows, exclusions = [], []
    for split in ("train", "validation", "test"):
        csv_path = ROOT / "data/raw/release_probe" / f"metadata_5s_{split}.csv"
        for r in csv.DictReader(csv_path.open()):
            if r["label"] not in LABEL_MAP: continue
            name = posixpath.normpath("allclass_5hours/fold_1/" + r["path"])
            basename = Path(r["path"]).stem
            start, end = [datetime.strptime(x, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc) for x in basename.split("_")]
            if start.strftime("%Y%m%d") != r["date"]:
                exclusions.append({"archive_name":name,"reason":"filename_metadata_date_disagreement"});continue
            if int(r["sample_rate"]) != 32000: raise ValueError("Unexpected sample rate")
            if abs((end-start).total_seconds()-5) > .001 or abs(float(r["duration_sec"])-5) > .001:
                exclusions.append({"archive_name":name,"reason":"not_five_seconds"});continue
            label = LABEL_MAP[r["label"]]
            mmsi = Decimal(r["MMSI"])
            if mmsi != mmsi.to_integral_value(): raise ValueError("Noninteger vessel identity")
            mmsi = str(int(mmsi))
            group = "day:" + r["date"] if label == "background" else "vessel:" + mmsi
            if label != "background" and len(mmsi) != 9: raise ValueError("Invalid vessel identity")
            uid = canonical_hash(["UniqueShip-v1.0", name])[:24]
            rows.append({"id":uid,"archive_name":name,"audio_path":"data/raw/experiment_audio/"+uid+".wav",
                         "label":label,"label_index":LABELS.index(label),"official_split":split,"source_group":group,
                         "date":r["date"],"start_utc":start.isoformat(),"start_seconds":start.timestamp(),
                         "hydrophone":r["hydrophone"] or None,"parent_recording_id":None,"deployment_id":None})
    return rows, exclusions


def assign(rows):
    strata = defaultdict(set)
    for r in rows: strata[(r["official_split"],r["label"])].add(r["source_group"])
    assignments = {}
    for (split,label),groups in sorted(strata.items()):
        ordered=sorted(groups,key=lambda g:canonical_hash([17,split,label,g]))
        if split=="test": assignments.update({g:"test" for g in ordered});continue
        cutoff=max(1,min(len(ordered)-1,round(len(ordered)*(.75 if split=="train" else .5))))
        names=("acoustic_fit","policy_fit") if split=="train" else ("development","calibration")
        assignments.update({g:names[i>=cutoff] for i,g in enumerate(ordered)})
    for r in rows:r["partition"]=assignments[r["source_group"]]
    # Highest-priority partition owns the whole day across ALL hydrophones/classes.
    owners={}
    for r in rows:
        if r["date"] not in owners or PARTS.index(r["partition"])>PARTS.index(owners[r["date"]]):
            owners[r["date"]]=r["partition"]
    retained,excluded=[],[]
    for r in rows:
        if owners[r["date"]]!=r["partition"]:
            excluded.append({"id":r["id"],"reason":"whole_utc_day_reserved_for_"+owners[r["date"]]})
        else:retained.append(r)
    # Audit midnight boundaries globally, without relying on absent device IDs.
    kept=[]
    for r in sorted(retained,key=lambda x:(-PARTS.index(x["partition"]),x["id"])):
        if any(x["partition"]!=r["partition"] and abs(x["start_seconds"]-r["start_seconds"])<65 for x in kept):
            excluded.append({"id":r["id"],"reason":"global_60_second_guard"})
        else:kept.append(r)
    # Cap long sources only outside official test.
    per_group=defaultdict(list)
    for r in kept:per_group[r["source_group"]].append(r)
    final=[]
    for group,members in per_group.items():
        ordered=sorted(members,key=lambda r:canonical_hash([17,r["id"]]))
        for i,r in enumerate(ordered):
            if r["partition"]!="test" and i>=100:excluded.append({"id":r["id"],"reason":"source_cap100"})
            else:final.append(r)
    return sorted(final,key=lambda r:r["id"]),excluded


def sample(rows,partition,per_class):
    selected=[];details={}
    for label in LABELS:
        groups=defaultdict(list)
        for r in rows:
            if r["partition"]==partition and r["label"]==label:groups[r["source_group"]].append(r)
        ordered=sorted(groups,key=lambda g:canonical_hash([17,"assess",partition,label,g]))
        chosen=[];times=defaultdict(list)
        for round_index in range(3):
            for g in ordered:
                candidates=sorted(groups[g],key=lambda r:canonical_hash([17,"assess",r["id"]]))
                for r in candidates:
                    if all(abs(r["start_seconds"]-t)>=600 for t in times[g]):
                        chosen.append(r);times[g].append(r["start_seconds"]);break
        target=min(per_class,len(chosen))
        selected.extend(chosen[:target]);details[label]={"requested":per_class,"selected":target,"capacity":len(chosen),"groups":len({r['source_group'] for r in chosen[:target]})}
    return selected,details


def main():
    rows,excluded=load_records();rows,purge=assign(rows);excluded+=purge
    for field in ("source_group","date"):
        seen=defaultdict(set)
        for r in rows:seen[r[field]].add(r["partition"])
        assert all(len(v)==1 for v in seen.values()),field
    ordered=sorted(rows,key=lambda r:r["start_seconds"])
    for a,b in zip(ordered,ordered[1:]):
        assert a['partition']==b['partition'] or b['start_seconds']-a['start_seconds']>=65
    counts={p:{"clips":sum(r['partition']==p for r in rows),"classes":dict(Counter(r['label'] for r in rows if r['partition']==p)),
               "groups":len({r['source_group'] for r in rows if r['partition']==p})} for p in PARTS}
    assessments={};sampling={}
    # Prospective budget-limited matrix: 100 policy, 40 development, 40 calibration,
    # and up to 100 test clips. Class shortfalls are retained, never oversampled.
    for partition,target in [('policy_fit',20),('development',8),('calibration',8),('test',20)]:
        selected,info=sample(rows,partition,target)
        assessments[partition]=[r['id'] for r in selected];sampling[partition]=info
    manifest={"schema_version":1,"labels":LABELS,"records":rows,"assessments":assessments,
              "data_protocol":"vessel_and_whole_utc_day_separated_v1",
              "missing_parent_provenance":True,"split_seed":17}
    atomic_json(ROOT/'data/private/experiment_manifest.json',manifest)
    atomic_json(ROOT/'data/private/experiment_exclusions.json',excluded)
    summary={"protocol":manifest['data_protocol'],"manifest_sha256":canonical_hash(manifest),"counts":counts,"assessment_sampling":sampling,
             "excluded_records":len(excluded),"exclusion_reasons":dict(Counter(r['reason'] for r in excluded)),
             "passed_checks":["vessel_disjoint","all_class_all_hydrophone_utc_day_disjoint","global_60_second_guard"],
             "pending_checks":["decoded_waveform_hashes_after_download"],
             "unverifiable_original_fields":["parent_recording_id","deployment_id"],"test_model_outcomes_examined":False}
    atomic_json(ROOT/'results/manifests/experiment_split_audit.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
