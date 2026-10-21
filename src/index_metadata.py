"""F2001-F2002: prepare and optionally index AqSolDB metadata.

Default: prepare/check a deterministic 100-record sample from train.csv without API calls.
To incur embedding API usage, explicitly run:
    python src/index_metadata.py --index --limit 100

Index re-runs skip existing IDs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from langchain_core.documents import Document

import compound_data
import config
import vector_store


def select_training_rows(limit: int) -> list[dict[str, str]]:
    """Choose a deterministic, source-balanced subset without using test.csv."""
    if limit < 1:
        raise ValueError("--limit must be a positive integer")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in compound_data.load_compounds():
        if row["_split"] == "train":
            groups[compound_data._source_code(row)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["ID"])

    # Keep the three externally documented pilot compounds and the aspirin /
    # caffeine demo records searchable, then fill the remaining sample evenly.
    featured_ids = {"A-454", "A-5538", "A-4518", "E-1274", "C-533"}
    selected = [
        row for rows in groups.values() for row in rows
        if row["ID"] in featured_ids
    ]
    selected.sort(key=lambda row: row["ID"])
    if len(selected) >= limit:
        return selected[:limit]
    for rows in groups.values():
        rows[:] = [row for row in rows if row["ID"] not in featured_ids]

    source_codes = sorted(groups)
    while len(selected) < limit:
        added = False
        for code in source_codes:
            if groups[code] and len(selected) < limit:
                selected.append(groups[code].pop(0))
                added = True
        if not added:
            break
    return selected


def build_documents(limit: int = 100) -> list[Document]:
    """Create one searchable document per source-balanced train record."""
    return [
        Document(
            page_content=compound_data.to_text(row),
            metadata=compound_data.to_metadata(row),
        )
        for row in select_training_rows(limit)
    ]


def index_documents(documents: list[Document]) -> None:
    store = vector_store.get_store(config.META_COLLECTION)
    ids = [str(doc.metadata["chunk_id"]) for doc in documents]
    already = vector_store.existing_ids(store, ids)
    pending = [doc for doc in documents if doc.metadata["chunk_id"] not in already]

    print(f"[기존 색인] {len(already):,}건")
    if not pending:
        print(f"[완료] 추가할 항목 없음 (총 {vector_store.count(store):,}건)")
        return

    print(f"[임베딩/색인] 신규 {len(pending):,}건 - 외부 임베딩 API를 호출합니다")
    for start in range(0, len(pending), config.BATCH_SIZE):
        batch = pending[start:start + config.BATCH_SIZE]
        batch_ids = [str(doc.metadata["chunk_id"]) for doc in batch]
        store.add_documents(batch, ids=batch_ids)
        done = min(start + len(batch), len(pending))
        print(f"  진행: {done:,}/{len(pending):,}")

    actual = vector_store.count(store)
    print(f"[완료] 로드 {len(documents):,}건 / 색인 {actual:,}청크")
    if actual != len(documents):
        print("[확인 필요] 로드 항목 수와 색인 청크 수가 다릅니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or index AqSolDB metadata.")
    parser.add_argument("--index", action="store_true", help="Call the configured embedding API and write to Chroma.")
    parser.add_argument("--limit", type=int, default=100, help="Source-balanced train records to prepare (default: 100).")
    args = parser.parse_args()
    documents = build_documents(limit=args.limit)
    print(f"[준비 완료] train 출처 균형 샘플 {len(documents):,}건 / 요청 {args.limit:,}건")
    if args.index:
        index_documents(documents)
    else:
        print("[API 미호출] 색인 비용을 확인하고 명시적으로 --index를 지정하세요.")


if __name__ == "__main__":
    main()
