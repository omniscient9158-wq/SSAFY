"""Retrieve compound metadata and reviewed evidence locally, without API calls.

Usage from the project root:
    python src/retrieve_local_evidence.py A-4518 "물에 대한 용해도"

This deterministic lexical retriever is a data-flow prototype, not vector search
or a generated RAG answer. It keeps metadata and external evidence as separate
result groups and only considers evidence for the selected AqSolDB ID.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Any

import compound_data
import external_documents

ALIASES = {
    "용해도": ("water", "solubility"),
    "물에 녹": ("water", "solubility"),
    "녹는": ("water", "solubility"),
    "외관": ("description", "liquid", "solid"),
    "생김새": ("description", "liquid", "solid"),
    "색": ("yellow", "white", "colorless"),
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _query_terms(query: str) -> set[str]:
    normalized = _normalize(query)
    expanded = [normalized]
    for phrase, replacements in ALIASES.items():
        if phrase in normalized:
            expanded.extend(replacements)
    return {
        term
        for value in expanded
        for term in re.findall(r"[a-z0-9]+|[가-힣]+", value)
        if len(term) > 1
    }


def retrieve(compound_id: str, query: str, limit: int = 3) -> dict[str, Any]:
    """Return one exact metadata record and up to `limit` same-compound sources."""
    if not query.strip():
        raise ValueError("질문을 입력하세요.")
    if limit < 1 or limit > 10:
        raise ValueError("외부 근거 결과 수는 1~10 사이여야 합니다.")

    row = next((r for r in compound_data.load_compounds() if r["ID"] == compound_id), None)
    if row is None:
        raise ValueError(f"AqSolDB 화합물 ID를 찾을 수 없습니다: {compound_id}")

    source = compound_data.source_info(row)
    metadata_evidence = {
        "citation_id": "근거 1",
        "source_type": "AqSolDB compound metadata",
        "compound_id": row["ID"],
        "name": row["Name"],
        "inchikey": row["InChIKey"],
        "smiles": row["SMILES"],
        "experimental_logS": row.get("Y", ""),
        "split": row.get("_split", ""),
        "source_group_code": compound_data._source_code(row),
        "upstream_source_dataset": source.get("Original_source_dataset", "unmapped"),
        "paper_reference_number": source.get("Reference_number_in_Sorkun_2019", ""),
        "dataset_doi": compound_data.DATASET_DOI,
        "dataset_license": compound_data.DATASET_LICENSE,
    }

    terms = _query_terms(query)
    ranked: list[tuple[int, dict[str, Any]]] = []
    for record in external_documents.load_reviewed_documents():
        metadata = record["metadata"]
        if metadata["compound_id"] != compound_id:
            continue
        haystack = _normalize(record["page_content"])
        matched_terms = sorted(term for term in terms if term in haystack)
        if not matched_terms:
            continue
        ranked.append((len(matched_terms), {
            "citation_id": "",
            "source_type": metadata["source_type"],
            "compound_id": metadata["compound_id"],
            "compound_name": metadata["compound_name"],
            "document_title": metadata["document_title"],
            "excerpt": metadata["document_title"] + ": " + next(
                line.removeprefix("검토된 발췌: ")
                for line in record["page_content"].splitlines()
                if line.startswith("검토된 발췌: ")
            ),
            "matched_query_terms": matched_terms,
            "source_provider": metadata["source_provider"],
            "original_attribution": metadata["original_attribution"],
            "source_url": metadata["source_url"],
            "source_terms_status": metadata["source_terms_status"],
            "terms_basis_url": metadata["terms_basis_url"],
            "reviewed_on": metadata["reviewed_on"],
            "scope_note": metadata["scope_note"],
        }))

    ranked.sort(key=lambda item: (-item[0], item[1]["source_url"]))
    external = [item for _, item in ranked[:limit]]
    for i, item in enumerate(external, start=2):
        item["citation_id"] = f"근거 {i}"

    return {
        "query": query,
        "retrieval_method": "local lexical match; exact AqSolDB ID filter",
        "compound_metadata": metadata_evidence,
        "external_evidence": external,
        "external_evidence_status": "matched" if external else "no query-matched external excerpt",
        "answer_policy": "이 결과는 검색 근거다. 답변 생성은 구현하지 않았으며, 외부 근거가 없으면 외부 문서 기반 주장을 하지 않는다.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local compound metadata and reviewed evidence.")
    parser.add_argument("compound_id", help="Exact AqSolDB ID, for example A-4518")
    parser.add_argument("query", help="Question or search terms")
    parser.add_argument("--limit", type=int, default=3, help="Maximum external evidence records (1-10)")
    args = parser.parse_args()
    try:
        result = retrieve(args.compound_id, args.query, args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
