"""Build a 100-compound document-discovery scope from existing source audits.

No network calls and no approval of candidate text. The manifest records where
description text was previously obtained; only the reviewed CSV feeds RAG.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import compound_data
from external_documents import load_reviewed_documents

BASE = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BASE / "data" / "document_discovery_manifest.csv"


def build_manifest(limit: int = 100) -> list[dict[str, str]]:
    compounds = {r["ID"]: r for r in compound_data.load_compounds() if r["_split"] == "train"}
    with (BASE / "research/provenance/pubchem_source_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        candidates = {
            r["AqSolDB_ID"]: r for r in csv.DictReader(stream)
            if r["AqSolDB_ID"] in compounds
            and r["InChIKey"] == compounds[r["AqSolDB_ID"]]["InChIKey"]
            and r["annotation_text"].strip() and r["PubChem_CID"].strip()
        }
    approved = {}
    reviewed_text = {}
    for doc in load_reviewed_documents():
        meta = doc["metadata"]
        if meta["compound_id"] not in approved:
            approved[meta["compound_id"]] = meta
            reviewed_text[meta["compound_id"]] = doc["page_content"]
    selected = sorted(approved)
    groups = defaultdict(list)
    for cid in sorted(candidates):
        if cid not in approved:
            groups[cid.split("-", 1)[0]].append(cid)
    while len(selected) < limit:
        added = False
        for code in sorted(groups):
            if groups[code] and len(selected) < limit:
                selected.append(groups[code].pop(0))
                added = True
        if not added:
            raise ValueError("Not enough train records with previously retrieved description text")
    records = []
    for cid in selected[:limit]:
        row = compounds[cid]
        source = approved.get(cid)
        candidate = candidates.get(cid, {})
        text = reviewed_text[cid] if source else candidate["annotation_text"]
        records.append({
            "AqSolDB_ID": cid, "Name": row["Name"], "InChIKey": row["InChIKey"],
            "PubChem_CID": str(source["pubchem_cid"] if source else candidate["PubChem_CID"]),
            "document_url": source["source_url"] if source else candidate["PubChem_URL"],
            "document_heading": source["document_title"] if source else candidate["heading"],
            "discovery_evidence": "approved_external_sources.csv" if source else "pubchem_source_audit.csv: annotation_text present; exact InChIKey mapping",
            "review_status": "reviewed_excerpt_available" if source else "description_obtained_terms_and_relevance_pending",
            "description_character_count": str(len(text)),
            "description_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return records


def main() -> None:
    records = build_manifest()
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    approved = sum(r["review_status"] == "reviewed_excerpt_available" for r in records)
    print(f"Document discovery scope: {len(records)} train compounds; reviewed {approved}; pending {len(records)-approved}")
    print("Candidate description text is not approved or indexed by this command.")


if __name__ == "__main__":
    main()
