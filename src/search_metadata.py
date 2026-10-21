"""Search local AqSolDB metadata without embeddings or external API calls.

This is a deterministic metadata lookup, not the final vector/RAG retriever.
Run from the project root:
    python src/search_metadata.py "aspirin"
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from typing import Any

import compound_data


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _terms(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[가-힣]+", _normalize(query))


def _score(query: str, row: dict[str, str]) -> float:
    normalized_query = _normalize(query)
    item_id = _normalize(row.get("ID", ""))
    inchikey = _normalize(row.get("InChIKey", ""))
    name = _normalize(row.get("Name", ""))
    smiles = _normalize(row.get("SMILES", ""))

    if normalized_query == item_id or normalized_query == inchikey:
        return 100.0
    if normalized_query and normalized_query in name:
        return 80.0 + min(len(normalized_query), 20) / 100
    if normalized_query and normalized_query in smiles:
        return 70.0

    terms = _terms(query)
    if not terms:
        return 0.0
    searchable = _normalize(compound_data.to_text(row))
    matches = sum(term in searchable for term in terms)
    return matches / len(terms) * 50 if matches else 0.0


def search_compounds(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return ranked local records with dataset and upstream-source attribution."""
    if not query.strip():
        raise ValueError("검색어를 입력하세요.")
    if limit < 1 or limit > 100:
        raise ValueError("결과 수는 1~100 사이여야 합니다.")

    ranked = []
    for row in compound_data.load_compounds():
        score = _score(query, row)
        if score <= 0:
            continue
        metadata = compound_data.to_metadata(row)
        source_info = compound_data.source_info(row)
        ranked.append({
            "score": score,
            "id": row.get("ID", ""),
            "name": row.get("Name", ""),
            "inchikey": row.get("InChIKey", ""),
            "smiles": row.get("SMILES", ""),
            "experimental_logS": row.get("Y", ""),
            "split": row.get("_split", ""),
            "source_code": metadata["source_code"],
            "upstream_source_group": source_info.get("Original_source_dataset", "미매핑"),
            "paper_reference_number": source_info.get("Reference_number_in_Sorkun_2019", ""),
            "source_paper_url": "https://www.nature.com/articles/s41597-019-0151-1",
            "dataset_doi": compound_data.DATASET_DOI,
            "dataset_license": compound_data.DATASET_LICENSE,
            "evidence_text": compound_data.to_text(row),
        })
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return ranked[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local AqSolDB metadata (no API calls).")
    parser.add_argument("query", help="화합물 이름, AqSolDB ID, InChIKey 또는 descriptor 값")
    parser.add_argument("--limit", type=int, default=10, help="최대 결과 수 (기본값: 10)")
    args = parser.parse_args()

    results = search_compounds(args.query, args.limit)
    print("검색 유형: 로컬 화합물 메타데이터 조회 (벡터 검색/RAG 아님)")
    print("출처 주의: A-I 원자료 코드는 논문 표의 데이터 묶음 추적용이며, 개별 행의 독립 라이선스를 뜻하지 않습니다.")
    print(f"검색 결과: {len(results)}건")
    for index, item in enumerate(results, start=1):
        print(f"\n[{index}] {item['id']} - {item['name']}")
        print(f"  LogS(Y): {item['experimental_logS']} | split: {item['split']}")
        print(f"  InChIKey: {item['inchikey']}")
        print(f"  원자료 코드: {item['source_code']} / {item['upstream_source_group']} {item['paper_reference_number']}")
        print(f"  원자료 묶음 근거: Sorkun et al. (2019), Table 1 - {item['source_paper_url']}")
        print(f"  데이터 출처: AqSolDB v1, {item['dataset_license']}")
        print(f"  DOI: {item['dataset_doi']}")
        print(f"  SMILES: {item['smiles']}")


if __name__ == "__main__":
    main()
