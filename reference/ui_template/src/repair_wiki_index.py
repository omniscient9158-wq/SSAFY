"""chunk_id 충돌로 유실된 위키 청크를 복구한다. (한 번만 실행하면 되는 도구)

이전 chunk_id는 `{page_id}:{순번}`이라 영화 구분이 없었다. 시리즈물처럼 여러 영화가
같은 문서(감독·원작·세계관)를 수집하면 두 번째 영화의 청크가 "이미 색인됨"으로
걸러져 사라졌고, 검색은 movie_id로 필터하므로 먼저 수집된 영화만 그 문서를 볼 수 있었다.

    python src/repair_wiki_index.py             # 무엇을 할지 보기만 한다
    python src/repair_wiki_index.py --apply     # 실제로 고친다
    python src/repair_wiki_index.py --compact   # 삭제 후 남은 빈 슬롯을 정리한다

두 단계로 진행한다.
    1단계  기존 청크의 id를 `{movie_id}:{page_id}:{순번}`으로 옮긴다.
           저장된 임베딩을 그대로 재사용하므로 비용이 들지 않는다.
    2단계  manifest에는 있는데 색인에는 없는 (영화, 문서) 쌍을 다시 넣는다.
           원문이 resources/rag/wikipedia에 남아 있어 위키 API를 호출하지 않는다.
           이 단계에서만 임베딩 비용이 발생한다.

--compact는 위 복구와 별개다. HNSW는 삭제를 표시만 하고 슬롯을 비워 두며 그 자리를
재사용하지 않는다. 1단계처럼 id를 바꾸는 작업을 하면 지운 만큼 파일이 부풀어 있으므로,
컬렉션을 새로 만들어 살아 있는 것만 옮긴다. 임베딩은 그대로 옮기므로 비용이 없다.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import collect_wiki
import config
import vector_store

BATCH = 500


def new_id(movie_id: str, page_id: str, order: str) -> str:
    return f"{movie_id}:{page_id}:{order}"


def is_old_format(chunk_id: str) -> bool:
    """옛 id는 `{page_id}:{순번}` 두 토막, 새 id는 세 토막이다."""
    return chunk_id.count(":") == 1


# ── 1단계 : id 옮기기 (임베딩 재사용) ──────────────────────────────────
def renumber(collection, apply: bool) -> int:
    """저장된 임베딩을 그대로 두고 id만 새 형식으로 바꾼다.

    지우면서 offset으로 순회하면 컬렉션이 줄어들어 뒤쪽을 건너뛴다.
    대상 id를 먼저 전부 모은 뒤 id로 집어서 처리한다.
    """
    stale = [i for i in collection.get(include=[])["ids"] if is_old_format(i)]
    if not apply or not stale:
        return len(stale)

    for start in range(0, len(stale), BATCH):
        page = collection.get(
            ids=stale[start:start + BATCH],
            include=["documents", "metadatas", "embeddings"],
        )
        ids, documents, metadatas, embeddings = [], [], [], []
        for index, chunk_id in enumerate(page["ids"]):
            metadata = dict(page["metadatas"][index])
            fresh = new_id(metadata["movie_id"], metadata["page_id"], chunk_id.split(":")[1])
            metadata["chunk_id"] = fresh

            ids.append(fresh)
            documents.append(page["documents"][index])
            metadatas.append(metadata)
            embeddings.append(page["embeddings"][index])

        # upsert라 중간에 멈췄다 다시 돌려도 안전하다.
        collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        collection.delete(ids=page["ids"])
        print(f"      {min(start + BATCH, len(stale)):,}/{len(stale):,}")

    return len(stale)


# ── 2단계 : 유실된 (영화, 문서) 쌍 다시 색인 ───────────────────────────
def indexed_pairs(collection) -> set[tuple[str, str]]:
    """색인에 실제로 들어 있는 (영화, 문서) 쌍."""
    found: set[tuple[str, str]] = set()
    total = collection.count()
    for offset in range(0, total, BATCH):
        page = collection.get(limit=BATCH, offset=offset, include=["metadatas"])
        for metadata in page["metadatas"]:
            found.add((str(metadata["movie_id"]), str(metadata["page_id"])))
    return found


def claimed_pairs() -> set[tuple[str, str]]:
    """manifest가 '수집했다'고 기록해 둔 (영화, 문서) 쌍."""
    manifest = collect_wiki.load_manifest()
    return {
        (str(movie_id), str(page_id))
        for movie_id, record in manifest.items()
        for page_id in record.get("page_ids", [])
    }


def local_pages() -> dict[tuple[str, str], dict]:
    """수집 당시 저장해 둔 원문. 위키 API를 다시 부르지 않기 위한 것."""
    pages: dict[tuple[str, str], dict] = {}
    for path in glob.glob(str(config.WIKI_DIR / "*.json")):
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        pages[(str(payload["movie_id"]), str(payload["page_id"]))] = payload
    return pages


def rebuild(store, missing: list[dict], apply: bool) -> int:
    """원문을 다시 청킹해 색인한다. (임베딩 비용은 여기서 발생)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.WIKI_CHUNK_SIZE,
        chunk_overlap=config.WIKI_CHUNK_OVERLAP,
    )

    documents: list[Document] = []
    for payload in missing:
        movie_id = str(payload["movie_id"])
        page_id = str(payload["page_id"])
        for order, chunk in enumerate(splitter.split_text(payload["text"])):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chunk_id": new_id(movie_id, page_id, f"{order:03d}"),
                        "source_type": "wikipedia",
                        "title": payload["title"],
                        "source_url": payload["url"],
                        "page_id": page_id,
                        "movie_id": movie_id,
                        "related_movie": payload["related_movie"],
                        "search_query": payload.get("search_query", ""),
                        "collected_at": payload.get("collected_at", ""),
                    },
                )
            )

    if apply:
        for start in range(0, len(documents), 100):
            part = documents[start:start + 100]
            store.add_documents(part, ids=[d.metadata["chunk_id"] for d in part])
            print(f"      {min(start + 100, len(documents))}/{len(documents)}")

    return len(documents)


# ── 3단계 : manifest를 색인과 맞추기 ───────────────────────────────────
def reconcile(collection) -> None:
    """색인을 기준으로 manifest의 문서 목록과 청크 수를 다시 쓴다.

    `--refill`은 재수집을 위해 manifest 기록을 지우는데, 그때 이미 색인된 문서는
    다시 들어가지 않으므로 기록만 사라진다. 그 차이를 메운다.
    """
    pages: dict[str, dict[str, str]] = {}   # movie_id → {page_id: title}
    counts: dict[str, int] = {}             # movie_id → 청크 수
    total = collection.count()
    for offset in range(0, total, BATCH):
        batch = collection.get(limit=BATCH, offset=offset, include=["metadatas"])
        for metadata in batch["metadatas"]:
            movie_id = str(metadata["movie_id"])
            pages.setdefault(movie_id, {})[str(metadata["page_id"])] = metadata["title"]
            counts[movie_id] = counts.get(movie_id, 0) + 1

    manifest = collect_wiki.load_manifest()
    for movie_id, record in manifest.items():
        titles = pages.get(movie_id, {})
        record["page_ids"] = sorted(titles)
        # 기존 순서(중요도 순)를 지키고 빠진 것만 뒤에 붙인다.
        kept = [t for t in record.get("titles", []) if t in titles.values()]
        record["titles"] = kept + [t for t in titles.values() if t not in kept]
        record["chunks"] = counts.get(movie_id, 0)
    collect_wiki.save_manifest(manifest)


LIVE_DIR = lambda: config.INDEX_DIR / config.WIKI_COLLECTION
STAGING_DIR = lambda: config.INDEX_DIR / f"{config.WIKI_COLLECTION}__new"
RETIRED_DIR = lambda: config.INDEX_DIR / f"{config.WIKI_COLLECTION}__old"


def slot_count(persist_dir: Path, dimensions: int = 1536, neighbors: int = 16) -> int:
    """HNSW가 잡아 둔 슬롯 수. 삭제해도 줄지 않으므로 실제 건수보다 클 수 있다."""
    per_element = neighbors * 2 * 4 + 4 + dimensions * 4 + 8
    files = list(persist_dir.glob("*/data_level0.bin"))
    return files[0].stat().st_size // per_element if files else 0


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def live_count() -> int:
    """sqlite를 직접 읽어 건수만 센다. Chroma를 열면 파일이 잠기므로 열지 않는다."""
    path = LIVE_DIR() / "chroma.sqlite3"
    if not path.exists():
        return 0
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    finally:
        connection.close()


def copy_collection() -> None:
    """살아 있는 청크만 새 컬렉션으로 옮긴다. (임베딩 재사용, 비용 없음)

    반드시 별도 프로세스에서 돈다. Chroma가 sqlite 파일을 프로세스가 끝날 때까지
    붙들고 있어서, 같은 프로세스 안에서는 디렉터리를 바꿔치기할 수 없다.
    """
    import chromadb

    live, staging = LIVE_DIR(), STAGING_DIR()
    if staging.exists():
        shutil.rmtree(staging)

    source = chromadb.PersistentClient(path=str(live)).get_collection(config.WIKI_COLLECTION)
    total = source.count()

    # 거리 계산 방식과 HNSW 파라미터가 바뀌면 검색 결과가 달라진다. 그대로 가져간다.
    settings = source.configuration_json.get("hnsw") or {}
    target = chromadb.PersistentClient(path=str(staging)).create_collection(
        config.WIKI_COLLECTION, configuration={"hnsw": settings}
    )
    print(f"새 컬렉션 생성 (hnsw: {settings})")

    moved = 0
    for offset in range(0, total, BATCH):
        page = source.get(
            limit=BATCH,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        if not page["ids"]:
            break
        target.add(
            ids=page["ids"],
            embeddings=page["embeddings"],
            documents=page["documents"],
            metadatas=page["metadatas"],
        )
        moved += len(page["ids"])
        print(f"      {moved:,}/{total:,}")

    if moved != total:
        raise SystemExit(f"옮긴 수가 맞지 않습니다 ({moved:,} != {total:,}). 원본은 그대로 둡니다.")


def compact(apply: bool) -> None:
    """빈 슬롯을 털어낸다. 복사는 자식 프로세스가, 교체는 이 프로세스가 한다."""
    live, staging, retired = LIVE_DIR(), STAGING_DIR(), RETIRED_DIR()

    total = live_count()
    slots = slot_count(live)
    print(f"현재: 청크 {total:,} · 슬롯 {slots:,} · {directory_size(live) / 1048576:.1f} MB")
    print(f"      빈 슬롯 {max(slots - total, 0):,}개")
    if not apply:
        print()
        print("계획만 출력했습니다. 실제로 정리하려면 --apply 를 붙이세요.")
        return

    print()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--copy-only"], text=True
    )
    if result.returncode != 0 or not staging.exists():
        raise SystemExit("복사에 실패했습니다. 원본은 그대로입니다.")

    before = directory_size(live)
    if retired.exists():
        shutil.rmtree(retired)
    live.rename(retired)          # 지우지 않고 옆으로 치운 뒤 바꿔 넣는다
    staging.rename(live)
    saved = before - directory_size(live)
    shutil.rmtree(retired)

    print()
    print(
        f"[완료] 청크 {live_count():,} · 슬롯 {slot_count(live):,}"
        f" · {directory_size(live) / 1048576:.1f} MB (회수 {saved / 1048576:.1f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="위키 색인 복구·정리")
    parser.add_argument("--apply", action="store_true", help="실제로 고친다 (없으면 계획만 출력)")
    parser.add_argument("--compact", action="store_true",
                        help="삭제 후 남은 빈 슬롯을 정리한다 (복구와는 별개)")
    parser.add_argument("--copy-only", action="store_true",
                        help=argparse.SUPPRESS)   # --compact가 자식 프로세스로 호출한다
    args = parser.parse_args()

    if args.copy_only:
        copy_collection()
        return

    if args.compact:
        compact(args.apply)
        return

    store = vector_store.get_store(config.WIKI_COLLECTION)
    collection = store._collection
    print(f"현재 색인: {collection.count():,}청크\n")

    print("[1단계] chunk_id를 영화별로 분리")
    moved = renumber(collection, args.apply)
    print(f"    옮길 청크 {moved:,}개 (임베딩 재사용, 비용 없음)\n")

    print("[2단계] 유실된 문서 다시 색인")
    have = indexed_pairs(collection)
    want = claimed_pairs()
    pages = local_pages()

    lost = sorted(want - have)
    missing = [pages[pair] for pair in lost if pair in pages]
    orphan = [pair for pair in lost if pair not in pages]

    print(f"    manifest 기록 {len(want):,}쌍 · 색인 확인 {len(have):,}쌍")
    print(f"    유실 {len(lost)}쌍 · 원문 보유 {len(missing)}쌍 · 원문 없음 {len(orphan)}쌍")
    if orphan:
        print("    (원문이 없는 쌍은 prebuild_wiki.py --refill 로 다시 수집해야 한다)")

    if missing:
        by_movie: dict[str, int] = {}
        for payload in missing:
            by_movie[payload["related_movie"]] = by_movie.get(payload["related_movie"], 0) + 1
        top = sorted(by_movie.items(), key=lambda kv: -kv[1])[:8]
        print("    많이 유실된 영화: " + ", ".join(f"{t}({n})" for t, n in top))

    chunks = rebuild(store, missing, args.apply)
    print(f"    다시 넣을 청크 {chunks:,}개 (임베딩 발생)\n")

    print("[3단계] manifest를 색인과 맞추기")
    stray = sorted(have - want)
    print(f"    색인에는 있는데 manifest에 없는 쌍 {len(stray)}건")
    print("    (--refill이 기록을 지우고 재수집하면서 생긴다. 기록만 복원한다.)")
    if args.apply:
        reconcile(collection)
        print("    manifest의 page_ids·titles·chunks를 색인 기준으로 다시 씀")
    print()

    if not args.apply:
        print("계획만 출력했습니다. 실제로 고치려면 --apply 를 붙여 다시 실행하세요.")
        return

    print(f"[완료] 색인 {collection.count():,}청크")
    remaining = claimed_pairs() - indexed_pairs(collection)
    if remaining:
        print(f"    아직 {len(remaining)}쌍이 남았습니다. (원문 없음)")
    else:
        print("    manifest의 모든 (영화, 문서) 쌍이 검색 가능합니다.")


if __name__ == "__main__":
    main()
