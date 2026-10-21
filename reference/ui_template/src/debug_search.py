"""검색 결과 확인용 도구. 어떤 청크가 근거로 뽑혔는지 눈으로 본다.

실행:
    python src/debug_search.py <movie_id> "질문"
    python src/debug_search.py 12477 "이순신이 배 13척으로 싸운 게 사실인가?"
    python src/debug_search.py 12477 "..." --grep 13척    # 색인 전체에서 단어 찾기
"""

from __future__ import annotations

import argparse

import config
import movies as movie_data
import vector_store


def show_index_status(movie_id: str) -> None:
    wiki_store = vector_store.get_store(config.WIKI_COLLECTION)
    result = wiki_store.get(where={"movie_id": movie_id}, include=["metadatas"])
    metadatas = result.get("metadatas", [])

    titles: dict[str, int] = {}
    for meta in metadatas:
        titles[meta.get("title", "?")] = titles.get(meta.get("title", "?"), 0) + 1

    print(f"[색인 상태] 이 영화의 위키 청크 {len(metadatas)}건")
    for title, count in sorted(titles.items(), key=lambda x: -x[1]):
        print(f"  - {title}: {count}청크")
    print()


def grep_chunks(movie_id: str, word: str) -> None:
    """색인된 청크 중 특정 단어가 들어간 것을 찾는다. (검색이 아니라 단순 문자열 확인)"""
    wiki_store = vector_store.get_store(config.WIKI_COLLECTION)
    result = wiki_store.get(where={"movie_id": movie_id}, include=["documents", "metadatas"])

    hits = [
        (meta.get("title", "?"), doc)
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
        if word in doc
    ]

    print(f"[단어 확인] '{word}'가 포함된 청크 {len(hits)}건")
    for title, doc in hits[:5]:
        index = doc.find(word)
        start, end = max(0, index - 120), index + 120
        print(f"  - {title}: ...{doc[start:end]}...")
    print()


def show_search(movie_id: str, question: str) -> None:
    movie_store = vector_store.get_store(config.MOVIE_COLLECTION)
    wiki_store = vector_store.get_store(config.WIKI_COLLECTION)

    print(f"[질문] {question}\n")

    print("── 영화 메타데이터 ──")
    for item in vector_store.get_by_movie(movie_store, movie_id):
        print(f"  · {item['metadata'].get('title')}: {item['page_content'][:100]}...")

    print("\n── 위키백과 검색 결과 (실제 근거로 들어가는 청크) ──")
    results = wiki_store.similarity_search_with_score(
        question, k=config.TOP_K_WIKI, filter={"movie_id": movie_id}
    )
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"  {rank}. [{score:.3f}] {doc.metadata.get('title')} "
              f"({doc.metadata.get('chunk_id')})")
        print(f"     {doc.page_content[:150].replace(chr(10), ' ')}...")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("movie_id")
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--grep", help="색인된 청크에서 찾을 단어")
    args = parser.parse_args()

    movie = movie_data.find_movie(args.movie_id)
    if movie:
        print(f"[영화] {movie_data.label(movie)} / 감독 {', '.join(movie['director'])}\n")

    show_index_status(args.movie_id)

    if args.grep:
        grep_chunks(args.movie_id, args.grep)

    if args.question:
        show_search(args.movie_id, args.question)


if __name__ == "__main__":
    main()
