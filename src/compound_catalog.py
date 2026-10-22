"""Browsable local Train records with explicit document availability."""
from __future__ import annotations

import csv
import io

import compound_data
from external_documents import load_reviewed_documents

READY = "문헌 발췌 준비됨"
PENDING = "문헌 검토 대기"
RECORD_ONLY = "실험 기록만"


def load_catalog() -> list[dict]:
    compound_data.refresh_name_links()
    reviewed_ids = {doc["metadata"]["compound_id"] for doc in load_reviewed_documents()}
    manifest_path = compound_data.DATA_DIR / "document_discovery_manifest.csv"
    discovered_ids = set()
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            discovered_ids = {row["AqSolDB_ID"] for row in csv.DictReader(stream)}
    items = []
    for row in compound_data.load_compounds():
        if row["_split"] != "train":
            continue
        status = READY if row["ID"] in reviewed_ids else PENDING if row["ID"] in discovered_ids else RECORD_ONLY
        linked = compound_data.wikidata_link(row)
        items.append({
            "화합물": compound_data.display_name(row),
            "데이터셋 이름": row["Name"],
            "별칭": ", ".join(compound_data.searchable_aliases(row)),
            "AqSolDB ID": row["ID"],
            "실험 LogS": float(row["Y"]),
            "분자량 (g/mol)": float(row["MolWt"]),
            "문헌 근거": status,
            "InChIKey": row["InChIKey"],
            "영문 위키 후보": linked.get("en_wikipedia", ""),
            "한글 위키 후보": linked.get("ko_wikipedia", ""),
        })
    return sorted(items, key=lambda item: (item["화합물"].casefold(), item["AqSolDB ID"]))


def filter_catalog(items: list[dict], query: str = "", status: str = "전체", *, document_only: bool = False) -> list[dict]:
    needle = query.strip().casefold()
    return [
        item for item in items
        if (status == "전체" or item["문헌 근거"] == status)
        and (not document_only or item["영문 위키 후보"] or item["한글 위키 후보"])
        and (not needle or any(needle in str(item[field]).casefold() for field in (
            "화합물", "데이터셋 이름", "별칭", "AqSolDB ID", "InChIKey",
        )))
    ]


def catalog_csv(items: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    fields = ("화합물", "데이터셋 이름", "별칭", "AqSolDB ID", "실험 LogS", "분자량 (g/mol)", "문헌 근거", "InChIKey", "영문 위키 후보", "한글 위키 후보")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(items)
    return stream.getvalue().encode("utf-8-sig")
