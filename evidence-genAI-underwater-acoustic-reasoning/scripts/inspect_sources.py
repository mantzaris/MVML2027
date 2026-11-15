"""Fetch public source metadata and template; never fetch audio or model weights."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "mvml_home": "https://mvml.org/",
    "mvml_guidelines": "https://mvml.org/author-guidelines/",
    "mvml_dates": "https://mvml.org/important-dates/",
    "mvml_template": "https://mvml.org/documents/MVML27_Paper_Template.docx",
    "uniqueship_release": "https://uniqueshipdata.org/",
    "onc_policy": "https://www.oceannetworks.ca/data/data-policy/",
    "uniqueship_drive": "https://drive.google.com/drive/folders/1HxV8AGh8ckZS3JehAU_gY2fuCnscX-eV",
    "autagent": "https://proceedings.mlr.press/v306/tong26l.html",
    "autagent_project": "https://tongsiqian.github.io/AuTAgent/",
    "signals1": "https://arxiv.org/abs/2610.22064",
    "signals2": "https://arxiv.org/abs/2610.31346",
    "audio_mind": "https://arxiv.org/abs/2605.28480",
    "sear": "https://arxiv.org/abs/2609.39847",
    "ace_cap": "https://arxiv.org/abs/2608.16162",
    "ase_clap": "https://doi.org/10.1109/ICASSP55912.2026.11461966",
    "uniqueship": "https://arxiv.org/abs/2609.13659",
    "ua_bench": "https://arxiv.org/abs/2610.16942",
    "wow_bench": "https://aclanthology.org/2026.findings-acl.1562/",
    "self_rag": "https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html",
    "alce": "https://aclanthology.org/2023.emnlp-main.398/",
    "conformal_factuality": "https://proceedings.mlr.press/v235/mohri24a.html",
    "cofact": "https://proceedings.iclr.cc/paper_files/paper/2026/hash/bd074b828bbd782d4bece89306caff84-Abstract-Conference.html",
    "aha": "https://arxiv.org/abs/2512.24052",
    "hallu_audio": "https://aclanthology.org/2026.acl-long.1797/",
    "underwater_embeddings": "https://doi.org/10.1016/j.apacoust.2025.111103",
    "passive_contrastive": "https://doi.org/10.1371/journal.pcbi.1014005",
    "omni_model": "https://huggingface.co/api/models/Qwen/Qwen2.5-Omni-7B",
    "planner_model": "https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct",
}


def fetch(item):
    name, url = item
    record = {"id": name, "url": url, "accessed_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=40, headers={"User-Agent": "MVML-research-source-audit/0.1"})
        record.update(http_status=response.status_code, resolved_url=response.url,
                      bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        if not response.content:
            raise ValueError("Empty response: publication contents were not retrieved")
        suffix = ".docx" if name == "mvml_template" else ".html"
        (ROOT / ".cache" / "sources" / (name + suffix)).write_bytes(response.content)
        if suffix == ".html":
            soup = BeautifulSoup(response.content, "html.parser")
            record["title"] = soup.title.get_text(" ", strip=True) if soup.title else None
            record["citation_metadata"] = {}
            for tag in soup.find_all("meta"):
                key = tag.get("name", "")
                if key.startswith("citation_") and key not in {"citation_reference", "citation_abstract", "citation_author_institution"}:
                    record["citation_metadata"].setdefault(key, []).append(tag.get("content", ""))
            if name == "uniqueship_release":
                record["drive_links"] = [a.get("href") for a in soup.find_all("a", href=True)
                                         if "drive.google.com" in a["href"]]
            if name.endswith("_model"):
                model = response.json()
                record["model_revision"] = model.get("sha")
                record["license"] = model.get("cardData", {}).get("license")
        record["status"] = "access_incomplete" if record.get("title") == "Redirecting" else "retrieved_requires_content_review"
    except Exception as exc:
        record.update(status="access_failed", error=str(exc))
    return record


if __name__ == "__main__":
    (ROOT / ".cache" / "sources").mkdir(parents=True, exist_ok=True)
    (ROOT / "sources").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    (ROOT / "sources" / "access_audit.json").write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(record["id"], record["status"], record.get("http_status"), record.get("bytes"))
