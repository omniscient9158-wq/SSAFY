"""Prepare curated evidence locally, or index it only with explicit --index.

Default: write data/curated_external_documents.jsonl without network/API calls.
Indexing: python src/index_external_documents.py --index (calls embeddings API).
"""
from __future__ import annotations

import argparse
import json

import external_documents


def write_jsonl(records: list[dict]) -> None:
    external_documents.OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with external_documents.OUTPUT_JSONL.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def index_records(
    records: list[dict],
    *,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    # Imports are intentionally delayed: the offline preparation path must not
    # load environment secrets or initialize any API-backed client.
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    import config
    import vector_store

    effective_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    effective_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    if effective_size < 1 or effective_overlap < 0 or effective_overlap >= effective_size:
        raise ValueError("청크 크기는 양수여야 하며 중복 길이는 0 이상, 청크 크기 미만이어야 합니다.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_size,
        chunk_overlap=effective_overlap,
        add_start_index=True,
    )
    documents: list[Document] = []
    for record in records:
        source_metadata = record["metadata"]
        chunks = splitter.create_documents([record["page_content"]])
        for sequence, chunk in enumerate(chunks, start=1):
            metadata = dict(source_metadata)
            metadata["document_id"] = source_metadata["chunk_id"]
            metadata["chunk_id"] = f"{source_metadata['compound_id']}:{source_metadata['chunk_id']}:{sequence:03d}"
            metadata["chunk_sequence"] = sequence
            metadata["source_start_index"] = chunk.metadata.get("start_index", 0)
            documents.append(Document(page_content=chunk.page_content, metadata=metadata))
    target_collection = collection_name or config.DOC_COLLECTION
    store = vector_store.get_store(target_collection)
    ids = [str(doc.metadata["chunk_id"]) for doc in documents]
    existing = vector_store.existing_ids(store, ids)
    pending = [doc for doc in documents if doc.metadata["chunk_id"] not in existing]
    if pending:
        store.add_documents(pending, ids=[str(doc.metadata["chunk_id"]) for doc in pending])
    print(
        f"[외부 문서 색인] 컬렉션 {target_collection} · "
        f"청크 {effective_size}/{effective_overlap} · 기존 {len(existing)}건 / "
        f"추가 {len(pending)}건 / 총 {vector_store.count(store)}건"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reviewed external evidence for the demo compounds.")
    parser.add_argument("--index", action="store_true", help="Call the configured embedding API and add documents to Chroma.")
    parser.add_argument("--collection", help="Target Chroma collection (defaults to DOC_COLLECTION)")
    parser.add_argument("--chunk-size", type=int, help="Chunk size in characters (defaults to CHUNK_SIZE)")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap in characters (defaults to CHUNK_OVERLAP)")
    args = parser.parse_args()

    records = external_documents.load_reviewed_documents()
    write_jsonl(records)
    print(f"[로컬 준비 완료] 검토 근거 {len(records)}건 -> {external_documents.OUTPUT_JSONL}")
    if args.index:
        index_records(
            records,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        print("[API 미호출] 벡터 색인이 필요하면 --index 옵션을 명시하세요.")


if __name__ == "__main__":
    main()
