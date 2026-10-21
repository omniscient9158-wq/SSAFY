"""위키데이터 수집·색인 (F2016 — 여러 종류의 문서 출처 결합).

위키백과(서술형 문서)와 수집·청킹 방식이 다르다.
    - 수집 : 검색어 추출·후보 선별이 필요 없다. TMDB ID로 해당 영화 항목 하나를 바로 찾는다.
    - 청킹 : "속성: 값" 한 줄이 사실 하나다. 글자 수로 자르면 속성명과 값이 갈라지므로
             줄 단위로 묶고, 청크마다 영화 제목 머리줄을 붙인다. 중복 길이는 두지 않는다.
    - 색인 : 별도 컬렉션(movie_wikidata). 영화당 1~3청크라 검색 없이 전부 근거에 넣는다.

실행:
    python src/collect_wikidata.py --movies 282631,157336   # 지정한 영화만
    python src/collect_wikidata.py --all                    # 964편 전부 (중단해도 이어서 진행)
    python src/collect_wikidata.py --reindex                # 저장된 원문 JSON으로 색인만 다시 (API 미호출)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.documents import Document

import config
import movies as movie_data
import vector_store
import wikidata_api


# ── 수집 이력 ─────────────────────────────────────────────────────────────
def load_manifest() -> dict[str, Any]:
    if config.WIKIDATA_MANIFEST_PATH.exists():
        with config.WIKIDATA_MANIFEST_PATH.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    config.WIKIDATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.WIKIDATA_MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def raw_path(movie_id: str) -> Path:
    return config.WIKIDATA_DIR / f"{movie_id}.json"


# ── 문서 만들기 ───────────────────────────────────────────────────────────
def fetch(movie: dict[str, Any]) -> dict[str, Any] | None:
    """위키데이터에서 영화 항목을 받아 원문 JSON 형태로 만든다. 항목이 없으면 None."""
    qid = wikidata_api.find_qid(movie["movie_id"])
    if not qid:
        return None
    entity = wikidata_api.get_entity(qid)
    labels = wikidata_api.get_labels(wikidata_api.referenced_ids(entity))
    lines = wikidata_api.fact_lines(entity, labels, config.WIKIDATA_MAX_VALUES)
    return {
        "movie_id": movie["movie_id"],
        "related_movie": movie["title"],
        "qid": qid,
        "title": f"{wikidata_api.label_of(entity) or movie['title']} ({qid})",
        "url": wikidata_api.ENTITY_URL.format(qid=qid),
        "lines": lines,
        "source": "위키데이터",
        "license": "CC0 1.0",
        "search_query": f"haswbstatement:P4947={movie['movie_id']}",
        "collected_at": date.today().isoformat(),
    }


def chunk_lines(header: str, lines: list[str], size: int) -> list[str]:
    """줄을 자르지 않고 묶는다. 청크마다 머리줄을 붙여 어느 영화의 사실인지 잃지 않게 한다."""
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and len(header) + sum(len(l) + 1 for l in current) + len(line) > size:
            chunks.append("\n".join([header, *current]))
            current = []
        current.append(line)
    if current:
        chunks.append("\n".join([header, *current]))
    return chunks


def to_documents(payload: dict[str, Any]) -> list[Document]:
    header = f"[위키데이터] {payload['related_movie']} 영화 항목의 사실 기록"
    documents = []
    for index, text in enumerate(chunk_lines(header, payload["lines"], config.WIKIDATA_CHUNK_SIZE)):
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "chunk_id": f"{payload['movie_id']}:{payload['qid']}:{index:03d}",
                    "source_type": "wikidata",
                    "title": payload["title"],
                    "source_url": payload["url"],
                    "page_id": payload["qid"],
                    "movie_id": payload["movie_id"],
                    "related_movie": payload["related_movie"],
                    "search_query": payload["search_query"],
                    "collected_at": payload["collected_at"],
                },
            )
        )
    return documents


def index(payload: dict[str, Any]) -> int:
    """그 영화의 위키데이터 청크를 새로 쓴다. 다시 받으면 줄 수가 달라질 수 있어 지우고 넣는다."""
    store = vector_store.get_store(config.WIKIDATA_COLLECTION)
    old = store.get(where={"movie_id": payload["movie_id"]}).get("ids", [])
    if old:
        store.delete(ids=old)
    documents = to_documents(payload)
    if documents:
        store.add_documents(documents, ids=[d.metadata["chunk_id"] for d in documents])
    return len(documents)


def indexed_count(movie_id: str) -> int:
    store = vector_store.get_store(config.WIKIDATA_COLLECTION)
    return len(store.get(where={"movie_id": movie_id}).get("ids", []))


# ── 진입점 ───────────────────────────────────────────────────────────────
def collect_for_movie(movie: dict[str, Any], refresh: bool = False) -> dict[str, Any]:
    """영화 한 편의 위키데이터 문서를 준비한다.

    수집 여부는 manifest가 아니라 색인을 보고 판단한다. manifest만 믿으면 색인이 비어도
    "이미 수집함"으로 건너뛰어 근거가 빈다. 색인에 없고 원문 JSON이 있으면 API 없이 다시 색인한다.
    """
    movie_id = movie["movie_id"]
    manifest = load_manifest()
    record = manifest.get(movie_id)

    if record and not refresh:
        if not record.get("qid"):
            return {"status": "not_found", "chunks": 0}
        count = indexed_count(movie_id)
        if count:
            return {"status": "cached", "qid": record["qid"], "chunks": count}
        if raw_path(movie_id).exists():
            payload = json.loads(raw_path(movie_id).read_text(encoding="utf-8"))
            return {"status": "reindexed", "qid": payload["qid"], "chunks": index(payload)}

    payload = fetch(movie)
    if payload is None:
        manifest[movie_id] = {"qid": "", "collected_at": date.today().isoformat()}
        save_manifest(manifest)
        return {"status": "not_found", "chunks": 0}

    config.WIKIDATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path(movie_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks = index(payload)
    manifest[movie_id] = {
        "qid": payload["qid"],
        "title": payload["title"],
        "facts": len(payload["lines"]),
        "chunks": chunks,
        "collected_at": payload["collected_at"],
    }
    save_manifest(manifest)
    return {"status": "collected", "qid": payload["qid"], "chunks": chunks}


def reindex_all() -> None:
    """저장된 원문 JSON으로 색인만 다시 만든다. 배포본을 받은 PC에서 쓴다."""
    paths = sorted(config.WIKIDATA_DIR.glob("*.json"))
    total = 0
    for order, path in enumerate(paths, start=1):
        total += index(json.loads(path.read_text(encoding="utf-8")))
        if order % 100 == 0:
            print(f"  {order}/{len(paths)}")
    print(f"[완료] 원문 {len(paths)}건 → 청크 {total}개")


def main() -> None:
    parser = argparse.ArgumentParser(description="위키데이터 수집·색인 (F2016)")
    parser.add_argument("--movies", default="", help="쉼표로 구분한 영화 ID")
    parser.add_argument("--all", action="store_true", help="전체 영화")
    parser.add_argument("--refresh", action="store_true", help="이미 받은 영화도 다시 받는다")
    parser.add_argument("--reindex", action="store_true", help="원문 JSON으로 색인만 다시 (API 미호출)")
    args = parser.parse_args()

    if args.reindex:
        reindex_all()
        return

    if args.all:
        targets = list(movie_data.load_movies())
    else:
        targets = [m for m in (movie_data.find_movie(i.strip()) for i in args.movies.split(",") if i.strip()) if m]
    if not targets:
        parser.print_help()
        return

    tally: dict[str, int] = {}
    for order, movie in enumerate(targets, start=1):
        try:
            result = collect_for_movie(movie, refresh=args.refresh)
        except Exception as error:
            result = {"status": "error", "chunks": 0}
            print(f"  [실패] {movie['title']}: {error}")
        tally[result["status"]] = tally.get(result["status"], 0) + 1
        if result["status"] == "collected":
            time.sleep(0.5)   # 위키미디어 호출 예절
        if order % 50 == 0 or len(targets) <= 10:
            print(f"  {order}/{len(targets)} {movie['title']}: {result}")
    print(f"[완료] {tally}")


if __name__ == "__main__":
    main()
