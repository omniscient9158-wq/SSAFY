"""ChromaDB 컬렉션 열기와 공통 조회."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma

import config
from llm import get_embeddings


def get_store(collection_name: str) -> Chroma:
    """컬렉션을 연다. 없으면 새로 만들어진다."""
    persist_dir = config.INDEX_DIR / collection_name
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def count(store: Chroma) -> int:
    """저장된 청크 수."""
    try:
        return store._collection.count()
    except Exception:
        return len(store.get().get("ids", []))


def existing_ids(store: Chroma, ids: list[str]) -> set[str]:
    """이미 색인된 chunk_id를 돌려준다."""
    if not ids:
        return set()
    found: set[str] = set()
    # 한 번에 너무 많은 id를 넘기지 않는다.
    for start in range(0, len(ids), 500):
        part = ids[start:start + 500]
        found.update(store.get(ids=part).get("ids", []))
    return found


def get_by_movie(store: Chroma, movie_id: str) -> list[dict[str, Any]]:
    """특정 영화의 청크를 메타데이터·본문과 함께 가져온다."""
    result = store.get(where={"movie_id": movie_id}, include=["documents", "metadatas"])
    return [
        {"page_content": doc, "metadata": meta}
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]
