"""Persistent Chroma collections for metadata and external-document chunks."""
from __future__ import annotations

from langchain_chroma import Chroma

import config
from llm import get_embeddings


def get_store(collection_name: str) -> Chroma:
    path = config.INDEX_DIR / collection_name
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(path),
    )


def count(store: Chroma) -> int:
    return int(store._collection.count())


def count_persisted(collection_name: str) -> int:
    """Read an existing collection count without initializing embeddings."""
    import chromadb

    path = config.INDEX_DIR / collection_name
    if not path.is_dir():
        return 0
    client = chromadb.PersistentClient(path=str(path))
    try:
        return int(client.get_collection(collection_name).count())
    except Exception:
        # A missing collection is an empty index; no embedding call is made.
        return 0


def existing_ids(store: Chroma, ids: list[str]) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(ids), 500):
        found.update(store.get(ids=ids[start:start + 500]).get("ids", []))
    return found
