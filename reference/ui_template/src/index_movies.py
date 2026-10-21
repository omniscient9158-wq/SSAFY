"""RAG 1 — 영화 메타데이터를 ChromaDB에 색인한다. (최초 1회 실행)

실행:
    python src/index_movies.py
    python src/index_movies.py --rebuild   # 기존 컬렉션을 지우고 다시 색인
"""

from __future__ import annotations

import argparse
import shutil

from langchain_core.documents import Document

import config
import movies as movie_data
import vector_store


def build_documents() -> list[Document]:
    """영화 1편 = 청크 1개. (overview 최대 930자로 분할이 불필요하다)"""
    documents = []
    for movie in movie_data.load_movies():
        documents.append(
            Document(
                page_content=movie_data.to_text(movie),
                metadata=movie_data.to_metadata(movie),
            )
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="기존 색인을 지우고 다시 만든다")
    args = parser.parse_args()

    persist_dir = config.INDEX_DIR / config.MOVIE_COLLECTION
    if args.rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"[초기화] {persist_dir} 삭제")

    documents = build_documents()
    print(f"[로드] 영화 {len(documents)}편")

    store = vector_store.get_store(config.MOVIE_COLLECTION)
    ids = [doc.metadata["chunk_id"] for doc in documents]

    already = vector_store.existing_ids(store, ids)
    new_documents = [doc for doc in documents if doc.metadata["chunk_id"] not in already]

    if not new_documents:
        print(f"[완료] 이미 색인되어 있습니다. (총 {vector_store.count(store)}건)")
        return

    print(f"[색인] {len(new_documents)}건 임베딩 중...")
    batch = 100
    for start in range(0, len(new_documents), batch):
        part = new_documents[start:start + batch]
        store.add_documents(part, ids=[doc.metadata["chunk_id"] for doc in part])
        print(f"      {min(start + batch, len(new_documents))}/{len(new_documents)}")

    print(f"[완료] {persist_dir} (총 {vector_store.count(store)}건)")


if __name__ == "__main__":
    main()
