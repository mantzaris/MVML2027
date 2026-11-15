"""Generate source-linked BibTeX only from retrieved publisher/arXiv metadata."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
records = json.loads((ROOT / "sources/access_audit.json").read_text())
bib, bibliography = [], []
for record in records:
    m = record.get("citation_metadata", {})
    if "citation_title" not in m or "citation_author" not in m:
        continue
    get = lambda k: m.get(k, [None])[0]
    date = get("citation_publication_date") or get("citation_date")
    year = re.search(r"\b(?:19|20)\d{2}\b", date or "")
    if not year:
        continue
    arxiv = get("citation_arxiv_id")
    conference = get("citation_conference_title")
    journal = get("citation_journal_title")
    if journal == "International Conference on Learning Representations":
        conference, journal = journal, None
    fields = {"title": get("citation_title"), "author": " and ".join(a.strip() for a in m["citation_author"]),
              "year": year.group(), "url": record["url"]}
    if arxiv:
        kind = "misc"
        fields.update(eprint=arxiv, archivePrefix="arXiv", note="Preprint; bibliographic metadata verified 2026-11-15")
        if record["id"] == "uniqueship":
            fields["note"] = "Author manuscript; arXiv records acceptance at OCEANS 2026; final proceedings pages unverified"
    elif conference:
        kind = "inproceedings"; fields["booktitle"] = conference
    else:
        kind = "article"; fields["journal"] = journal
    if get("citation_doi"): fields["doi"] = get("citation_doi")
    if get("citation_firstpage") and get("citation_lastpage"):
        fields["pages"] = get("citation_firstpage") + "--" + get("citation_lastpage")
    bib.append("@" + kind + "{" + record["id"] + ",\n" +
               ",\n".join("  " + k + " = {" + str(v) + "}" for k, v in fields.items() if v) + "\n}\n")
    bibliography.append({"key": record["id"], "type": kind, "fields": fields,
                         "verification": "bibliographic_metadata_only_unless_discussed_in_nearest_work"})
for name in ("ase_clap", "underwater_embeddings"):
    path = ROOT / "sources" / (name + "_crossref.json")
    if not path.exists(): continue
    m = json.loads(path.read_text())
    kind = "inproceedings" if m["type"] == "proceedings-article" else "article"
    fields = {"title": m["title"][0], "author": " and ".join(a["family"] + ", " + a["given"] for a in m["author"]),
              "year": str(m["published"]["date-parts"][0][0]), "doi": m["DOI"], "url": m["URL"],
              "booktitle" if kind == "inproceedings" else "journal": m["container-title"][0],
              "note": "Crossref metadata verified; full method not independently reviewed in this preparation stage"}
    bib.append("@" + kind + "{" + name + ",\n" + ",\n".join("  " + k + " = {" + str(v) + "}" for k, v in fields.items()) + "\n}\n")
    bibliography.append({"key": name, "type": kind, "fields": fields, "verification": "crossref_metadata"})
(ROOT / "manuscript").mkdir(exist_ok=True)
(ROOT / "manuscript/references.bib").write_text("% Verified metadata, not a claim of full-text review of every entry.\n\n" + "\n".join(bib))
(ROOT / "sources/bibliography.json").write_text(json.dumps(bibliography, indent=2, ensure_ascii=False) + "\n")
print("Created", len(bibliography), "source-linked bibliography entries")
