"""TMDB에서 유명 영화 메타데이터를 수집해 data/movies.json으로 저장한다.

실행:
    python src/collect_movies.py                # 기본 1000편
    python src/collect_movies.py --target 300   # 편수 조정

src/.env 의 TMDB_KEY 를 사용한다. (v3 API Key / v4 Read Access Token 모두 지원)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

BASE_URL = "https://api.themoviedb.org/3"
ENV_PATH = Path(__file__).resolve().parent / ".env"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "movies.json"

# 유명한 영화 = 평점 참여자 수가 많은 영화
GLOBAL_RATIO = 0.8          # 전체 인기작 비율 (나머지는 한국 영화)
MIN_VOTE_COUNT = 500
SLEEP = 0.05                # TMDB 호출 간격


def get_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.environ.get("TMDB_KEY")
    if not key:
        raise RuntimeError(f"TMDB_KEY가 없습니다. {ENV_PATH} 를 확인하세요.")
    return key


def make_session(key: str) -> requests.Session:
    session = requests.Session()
    # v4 토큰(JWT)은 Authorization 헤더, v3 키는 쿼리 파라미터로 보낸다.
    if key.startswith("eyJ"):
        session.headers.update({"Authorization": f"Bearer {key}"})
        session.params = {}
    else:
        session.params = {"api_key": key}
    return session


def request_tmdb(session: requests.Session, path: str, **params: Any) -> dict[str, Any]:
    """429를 처리하면서 TMDB API를 호출한다."""
    for attempt in range(1, 6):
        response = session.get(f"{BASE_URL}{path}", params=params, timeout=20)

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            wait = float(response.headers.get("Retry-After", 2)) + 0.5
            print(f"[429] {wait:.1f}초 후 재시도 ({attempt}/5)")
            time.sleep(wait)
            continue

        response.raise_for_status()

    raise RuntimeError(f"TMDB 요청 실패: {path}")


def discover_ids(
    session: requests.Session,
    target: int,
    original_language: str | None = None,
) -> list[int]:
    """평점 참여자 수 기준 상위 영화 ID를 모은다."""
    ids: list[int] = []
    page = 1

    while len(ids) < target and page <= 500:
        params = {
            "language": "ko-KR",
            "sort_by": "vote_count.desc",
            "vote_count.gte": MIN_VOTE_COUNT,
            "include_adult": "false",
            "page": page,
        }
        if original_language:
            params["with_original_language"] = original_language

        data = request_tmdb(session, "/discover/movie", **params)
        results = data.get("results", [])
        if not results:
            break

        ids.extend(item["id"] for item in results)
        page += 1
        time.sleep(SLEEP)

    return ids[:target]


def to_movie(detail: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """TMDB 응답을 프로젝트에서 쓰는 형태로 정리한다."""
    credits = detail.get("credits", {})

    directors = [
        member["name"]
        for member in credits.get("crew", [])
        if member.get("job") == "Director"
    ]
    cast = [member["name"] for member in credits.get("cast", [])[:5]]

    overview = (detail.get("overview") or "").strip()
    if not overview and fallback:
        overview = (fallback.get("overview") or "").strip()

    release_date = detail.get("release_date") or ""

    return {
        "movie_id": str(detail["id"]),
        "title": detail.get("title") or detail.get("original_title", ""),
        "original_title": detail.get("original_title", ""),
        "release_date": release_date,
        "year": int(release_date[:4]) if release_date[:4].isdigit() else None,
        "genres": [g["name"] for g in detail.get("genres", [])],
        "director": directors,
        "cast": cast,
        "overview": overview,
        "runtime": detail.get("runtime"),
        "original_language": detail.get("original_language", ""),
        "production_countries": [
            c["name"] for c in detail.get("production_countries", [])
        ],
        "vote_average": detail.get("vote_average"),
        "vote_count": detail.get("vote_count"),
        "tmdb_url": f"https://www.themoviedb.org/movie/{detail['id']}",
    }


def fetch_movie(session: requests.Session, movie_id: int) -> dict[str, Any] | None:
    """영화 상세와 출연·제작진을 한 번에 가져온다."""
    try:
        detail = request_tmdb(
            session,
            f"/movie/{movie_id}",
            language="ko-KR",
            append_to_response="credits",
        )
    except Exception as error:
        print(f"[건너뜀] {movie_id}: {error}")
        return None

    fallback = None
    if not (detail.get("overview") or "").strip():
        # 한국어 줄거리가 없으면 영어 줄거리로 대체한다.
        time.sleep(SLEEP)
        try:
            fallback = request_tmdb(session, f"/movie/{movie_id}", language="en-US")
        except Exception:
            fallback = None

    return to_movie(detail, fallback)


def collect(target: int) -> list[dict[str, Any]]:
    session = make_session(get_key())

    global_target = int(target * GLOBAL_RATIO)
    korean_target = target - global_target

    print(f"[1/3] 영화 목록 조회 (전체 {global_target}편 + 한국 {korean_target}편)")
    ids = discover_ids(session, global_target)
    korean_ids = discover_ids(session, korean_target, original_language="ko")

    # 중복 제거 후 순서 유지
    seen: set[int] = set()
    merged: list[int] = []
    for movie_id in ids + korean_ids:
        if movie_id not in seen:
            seen.add(movie_id)
            merged.append(movie_id)

    print(f"[2/3] 상세 정보 수집: {len(merged)}편")
    movies: list[dict[str, Any]] = []
    for index, movie_id in enumerate(merged, start=1):
        movie = fetch_movie(session, movie_id)
        if movie and movie["overview"]:
            movies.append(movie)
        if index % 100 == 0:
            print(f"      {index}/{len(merged)} 완료")
        time.sleep(SLEEP)

    return movies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000, help="수집할 영화 편수")
    args = parser.parse_args()

    movies = collect(args.target)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(movies, file, ensure_ascii=False, indent=2)

    korean = sum(1 for m in movies if m["original_language"] == "ko")
    print(f"[3/3] 저장 완료: {OUTPUT_PATH} ({len(movies)}편, 한국 영화 {korean}편)")


if __name__ == "__main__":
    main()
