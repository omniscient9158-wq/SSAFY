"""data/korean_seed.json의 한국 영화를 TMDB에서 찾아 data/movies.json에 합친다.

시드 목록은 웹 검색(흥행 순위·천만 영화·연도별 개봉작)으로 만든 대중적인 한국 영화다.
TMDB의 vote_count 기준 수집만으로는 한국 영화가 적게 잡히기 때문에 제목으로 직접 조회한다.

실행:
    python src/collect_korean_movies.py
    python src/collect_korean_movies.py --dry-run   # 매칭 결과만 확인
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from collect_movies import (
    OUTPUT_PATH,
    SLEEP,
    fetch_movie,
    get_key,
    make_session,
    request_tmdb,
)

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "korean_seed.json"


def load_seed() -> list[dict[str, Any]]:
    with SEED_PATH.open(encoding="utf-8") as file:
        return json.load(file)["movies"]


def load_existing() -> list[dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return []
    with OUTPUT_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def search_movie(session, title: str, year: int) -> dict[str, Any] | None:
    """제목으로 TMDB를 검색해 가장 그럴듯한 한국 영화를 고른다."""
    data = request_tmdb(
        session,
        "/search/movie",
        query=title,
        language="ko-KR",
        include_adult="false",
    )
    results = data.get("results", [])
    if not results:
        return None

    def score(item: dict[str, Any]) -> tuple[int, int, float]:
        item_year = (item.get("release_date") or "")[:4]
        is_korean = item.get("original_language") == "ko"
        year_match = item_year == str(year)
        near_year = item_year.isdigit() and abs(int(item_year) - year) <= 1
        return (
            int(is_korean) * 4 + int(year_match) * 3 + int(near_year),
            int(item.get("vote_count") or 0),
            float(item.get("popularity") or 0),
        )

    best = max(results, key=score)

    # 한국 영화도 아니고 연도도 안 맞으면 잘못 매칭된 것으로 본다.
    best_year = (best.get("release_date") or "")[:4]
    if best.get("original_language") != "ko" and best_year != str(year):
        return None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 매칭만 확인")
    args = parser.parse_args()

    session = make_session(get_key())
    seed = load_seed()
    existing = load_existing()
    known = {movie["movie_id"] for movie in existing}

    print(f"[시작] 시드 {len(seed)}편 / 기존 데이터 {len(existing)}편")

    added: list[dict[str, Any]] = []
    duplicated: list[str] = []
    not_found: list[str] = []

    for index, item in enumerate(seed, start=1):
        title, year = item["title"], item["year"]

        try:
            found = search_movie(session, title, year)
        except Exception as error:
            print(f"[검색 실패] {title}: {error}")
            not_found.append(f"{title} ({year})")
            continue

        if not found:
            not_found.append(f"{title} ({year})")
            time.sleep(SLEEP)
            continue

        movie_id = str(found["id"])
        if movie_id in known:
            duplicated.append(title)
            time.sleep(SLEEP)
            continue

        time.sleep(SLEEP)
        movie = fetch_movie(session, found["id"])
        if not movie or not movie["overview"]:
            not_found.append(f"{title} ({year}) - 줄거리 없음")
            continue

        known.add(movie_id)
        added.append(movie)
        print(f"  [{index:3d}/{len(seed)}] + {movie['title']} ({movie.get('year')})")
        time.sleep(SLEEP)

    print(f"\n[결과] 신규 {len(added)}편 / 기존 중복 {len(duplicated)}편 / 미발견 {len(not_found)}편")
    if not_found:
        print("  미발견:", ", ".join(not_found))

    if args.dry_run:
        print("[dry-run] 저장하지 않았습니다.")
        return

    merged = existing + added
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False, indent=2)

    korean = sum(1 for m in merged if m["original_language"] == "ko")
    print(f"[저장] {OUTPUT_PATH} (총 {len(merged)}편, 한국 영화 {korean}편)")


if __name__ == "__main__":
    main()
