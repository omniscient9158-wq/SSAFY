"""Load only reviewed external excerpts for compounds with approved evidence."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import compound_data

PROJECT_DIR = Path(__file__).resolve().parent.parent
APPROVED_SOURCES = PROJECT_DIR / "data" / "approved_external_sources.csv"
OUTPUT_JSONL = compound_data.PROJECT_DIR / "data" / "curated_external_documents.jsonl"
DEMO_COMPOUND_IDS = ("A-454", "A-5538", "A-4518", "B-524", "C-533", "E-1274")


def load_reviewed_documents() -> list[dict[str, Any]]:
    """Validate approved excerpts against local AqSolDB identity records."""
    compounds = {row["ID"]: row for row in compound_data.load_compounds()}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    with APPROVED_SOURCES.open("r", encoding="utf-8-sig", newline="") as stream:
        for source in csv.DictReader(stream):
            compound_id = source["AqSolDB_ID"].strip()
            if source.get("index_eligible", "").lower() != "yes":
                raise ValueError(f"External document is not approved for indexing: {source['document_id']}")
            row = compounds.get(compound_id)
            if row is None:
                raise ValueError(f"Selected AqSolDB compound not found: {compound_id}")
            if row["_split"] != "train":
                raise ValueError(f"Test compound cannot enter reviewed search evidence: {compound_id}")
            if source["InChIKey"].strip() != row["InChIKey"].strip():
                raise ValueError(f"InChIKey mismatch for {compound_id}; refusing to include the document")
            if not source["approved_excerpt"].strip() or not source["source_url"].startswith("https://"):
                raise ValueError(f"Incomplete reviewed evidence record: {source['document_id']}")

            content = "\n".join((
                f"화합물: {row['Name']} ({compound_id})",
                f"문서 항목: {source['heading']}",
                f"검토된 발췌: {source['approved_excerpt']}",
                f"문서 제공자: {source['source_provider']}",
                f"원출처: {source['original_attribution']}",
            ))
            if source["document_id"] in selected_ids:
                raise ValueError(f"Duplicate selected evidence record: {source['document_id']}")
            selected_ids.add(source["document_id"])
            selected.append({
                "page_content": content,
                "metadata": {
                    "chunk_id": source["document_id"],
                    "source_type": "reviewed_external_evidence",
                    "compound_id": compound_id,
                    "compound_name": row["Name"],
                    "inchikey": source["InChIKey"],
                    "pubchem_cid": source["PubChem_CID"],
                    "document_title": source["heading"],
                    "source_provider": source["source_provider"],
                    "original_attribution": source["original_attribution"],
                    "source_url": source["source_url"],
                    "source_terms_status": source["source_terms_status"],
                    "terms_basis_url": source["terms_basis_url"],
                    "reviewed_on": source["reviewed_on"],
                    "scope_note": source["scope_note"],
                    "index_eligible": True,
                },
            })

    missing = sorted(set(DEMO_COMPOUND_IDS) - {item["metadata"]["compound_id"] for item in selected})
    if missing:
        raise ValueError(f"Selected demo compounds lack reviewed evidence: {', '.join(missing)}")
    demo_order = {compound_id: order for order, compound_id in enumerate(DEMO_COMPOUND_IDS)}
    return sorted(selected, key=lambda item: (demo_order.get(item["metadata"]["compound_id"], len(demo_order)), item["metadata"]["compound_id"], item["metadata"]["chunk_id"]))


def reviewed_scope(documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Derive UI counts from approved evidence and the preserved discovery scope."""
    documents = load_reviewed_documents() if documents is None else documents
    compound_ids = tuple(dict.fromkeys(doc["metadata"]["compound_id"] for doc in documents))
    with (PROJECT_DIR / "data/document_discovery_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest_ids = {row["AqSolDB_ID"] for row in csv.DictReader(stream)}
    return {"compound_ids": compound_ids, "compound_count": len(compound_ids),
            "excerpt_count": len(documents), "target_count": len(manifest_ids),
            "pending_count": len(manifest_ids - set(compound_ids))}
