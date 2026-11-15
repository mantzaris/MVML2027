"""Read selected public ZIP members by HTTP ranges, checking size and CRC.

Only metadata and a small training-audio sample are acquired. The two original
multipart files are never rewritten. Full-release acquisition remains a later
stage after metadata independence and storage gates pass.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import posixpath
import struct
import zlib
from collections import Counter
from pathlib import Path

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PARTS = ["1IaDNuCOtg9cl7TfCXyEfj-uizIwCTX34", "1gBOsHTfSKmAhXKODFK2twz5AGtQT3Od1"]


class RemoteSplitZip:
    def __init__(self):
        self.cache = ROOT / ".cache" / "ranges"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sizes = [self._part_size(i) for i in range(len(PARTS))]

    def _get(self, disk: int, span: str):
        return requests.get("https://drive.usercontent.google.com/download",
                            params={"id": PARTS[disk], "export": "download", "confirm": "t"},
                            headers={"Range": "bytes=" + span}, timeout=60, stream=True)

    def _part_size(self, disk: int) -> int:
        with self._get(disk, "0-0") as r:
            if r.status_code != 206:
                raise ValueError(f"Public byte range unavailable: HTTP {r.status_code}")
            return int(r.headers["Content-Range"].split("/")[-1])

    def read(self, disk: int, offset: int, size: int) -> bytes:
        if size < 0 or size > 50 * 1024 * 1024 or offset < 0:
            raise ValueError("Probe member/range outside bounded limits")
        if offset + size > self.sizes[disk]:
            first = self.sizes[disk] - offset
            return self.read(disk, offset, first) + self.read(disk + 1, 0, size - first)
        if not size:
            return b""
        key = self.cache / f"{disk}-{offset}-{size}.bin"
        if key.exists():
            data = key.read_bytes()
            if len(data) == size:
                return data
        with self._get(disk, f"{offset}-{offset + size - 1}") as r:
            expected = f"bytes {offset}-{offset + size - 1}/{self.sizes[disk]}"
            if r.status_code != 206 or r.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor exact requested byte range")
            data = r.raw.read(size + 1)
            if len(data) != size:
                raise ValueError("Truncated or oversized range")
        key.write_bytes(data)
        return data

    def entries(self) -> list[dict]:
        tail = self.read(1, self.sizes[1] - 65536, 65536)
        pos = tail.rfind(b"PK\x06\x06")
        if pos < 0:
            raise ValueError("Expected the verified release's ZIP64 directory")
        end = struct.unpack_from("<4sQ2H2I4Q", tail, pos)
        data = self.read(end[5], end[9], end[8])
        output, cursor = [], 0
        while cursor < len(data):
            fields = struct.unpack_from("<4s6H3I5H2I", data, cursor)
            if fields[0] != b"PK\x01\x02":
                raise ValueError("Invalid central directory")
            name_n, extra_n, comment_n = fields[10:13]
            name = data[cursor + 46:cursor + 46 + name_n].decode("utf-8" if fields[3] & 2048 else "cp437")
            extra = data[cursor + 46 + name_n:cursor + 46 + name_n + extra_n]
            unpacked, packed, offset, disk = fields[9], fields[8], fields[16], fields[13]
            epos = 0
            while epos + 4 <= len(extra):
                tag, length = struct.unpack_from("<HH", extra, epos)
                values = extra[epos + 4:epos + 4 + length]
                if tag == 1:
                    vpos = 0
                    numbers = [unpacked, packed, offset, disk]
                    for i, number in enumerate(numbers):
                        sentinel = 65535 if i == 3 else 4294967295
                        if number == sentinel:
                            fmt = "<I" if i == 3 else "<Q"
                            numbers[i] = struct.unpack_from(fmt, values, vpos)[0]
                            vpos += struct.calcsize(fmt)
                    unpacked, packed, offset, disk = numbers
                epos += 4 + length
            if fields[3] & 1:
                raise ValueError("Encrypted archive is not supported")
            output.append(dict(name=name, unpacked=unpacked, packed=packed,
                               offset=offset, disk=disk, compression=fields[4], crc32=fields[7]))
            cursor += 46 + name_n + extra_n + comment_n
        if len(output) != end[7]:
            raise ValueError("Central directory entry count mismatch")
        return output

    def extract_bytes(self, entry: dict) -> bytes:
        if entry["unpacked"] > 50 * 1024 * 1024:
            raise ValueError("Probe limit exceeded")
        local = self.read(entry["disk"], entry["offset"], 30)
        header = struct.unpack("<4s5H3I2H", local)
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid local header")
        data = self.read(entry["disk"], entry["offset"] + 30 + header[-2] + header[-1], entry["packed"])
        if entry["compression"] == 8:
            data = zlib.decompress(data, -15)
        elif entry["compression"] != 0:
            raise ValueError("Unsupported compression")
        if len(data) != entry["unpacked"] or zlib.crc32(data) != entry["crc32"]:
            raise ValueError("Member size/CRC mismatch")
        return data


def main():
    archive = RemoteSplitZip()
    entries = archive.entries()
    index = {e["name"]: e for e in entries}
    (ROOT / ".cache" / "release_index.json").write_text(json.dumps(entries))
    print("Verified archive directory:", len(entries), "entries; bytes:", archive.sizes, flush=True)
    out = ROOT / "data" / "raw" / "release_probe"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"kind": "partial_release_audit_not_full_download", "archive_sizes": archive.sizes,
               "archive_entries": len(entries), "metadata": [], "audio_samples": []}
    training = []
    for entry in entries:
        if "fold_1/" not in entry["name"] or not entry["name"].endswith(".csv"):
            continue
        data = archive.extract_bytes(entry)
        (out / Path(entry["name"]).name).write_bytes(data)
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        rows = list(reader)
        nulls = {key: sum(str(row[key]).strip().lower() in {"", "nan", "none", "null"} for row in rows)
                 for key in reader.fieldnames}
        item = {"archive_path": entry["name"], "sha256": hashlib.sha256(data).hexdigest(),
                "rows": len(rows), "columns": reader.fieldnames, "missing_by_column": nulls,
                "labels": dict(Counter(r.get("label") for r in rows)),
                "sample_rates": dict(Counter(r.get("sample_rate") for r in rows))}
        summary["metadata"].append(item)
        print(json.dumps(item), flush=True)
        if "_train.csv" in entry["name"]:
            training = [(entry, r) for r in rows]
    for label in ("background", "cargo", "passengership", "tanker", "tug"):
        match = next(((entry, row) for entry, row in training if row.get("label") == label), None)
        if match is None:
            continue
        entry, row = match
        name = posixpath.normpath(posixpath.join(posixpath.dirname(entry["name"]), row["path"]))
        member = index.get(name)
        if member is None:
            raise ValueError("CSV audio path does not resolve in archive")
        data = archive.extract_bytes(member)
        local = out / ("sample_" + label + ".wav")
        local.write_bytes(data)
        info = sf.info(local)
        sample = {"label": label, "sample_rate": info.samplerate, "channels": info.channels,
                  "frames": info.frames, "subtype": info.subtype, "duration": info.duration,
                  "sha256": hashlib.sha256(data).hexdigest()}
        summary["audio_samples"].append(sample)
        print(sample, flush=True)
    (ROOT / "sources" / "release_metadata_audit.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
