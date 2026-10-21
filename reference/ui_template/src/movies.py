"""영화 메타데이터 로드와 색인용 텍스트 구성."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import config


@lru_cache(maxsize=1)
def load_movies() -> tuple[dict[str, Any], ...]:
    """data/movies.json을 읽어 영화 목록을 반환한다."""
    if not config.MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"영화 데이터가 없습니다: {config.MOVIES_PATH}\n"
            "먼저 python src/collect_movies.py 를 실행하세요."
        )

    with config.MOVIES_PATH.open(encoding="utf-8") as file:
        return tuple(json.load(file))


def label(movie: dict[str, Any]) -> str:
    """화면 표시용 이름. 동명 작품 구분을 위해 연도를 붙인다."""
    year = movie.get("year")
    return f"{movie['title']} ({year})" if year else movie["title"]


def find_movie(movie_id: str) -> dict[str, Any] | None:
    for movie in load_movies():
        if movie["movie_id"] == movie_id:
            return movie
    return None


def to_text(movie: dict[str, Any]) -> str:
    """색인·프롬프트에 사용할 영화 소개 텍스트.

    제목·감독·배우로 검색될 확률이 높으므로 앞쪽에 배치한다.
    """
    title = movie["title"]
    original = movie.get("original_title", "")
    head = f"제목: {title}" + (f" ({original})" if original and original != title else "")

    lines = [
        head,
        f"개봉: {movie.get('release_date', '')} ({movie.get('year', '')}년)",
        f"장르: {', '.join(movie.get('genres', []))}",
        f"감독: {', '.join(movie.get('director', []))}",
        f"출연: {', '.join(movie.get('cast', []))}",
        f"제작 국가: {', '.join(movie.get('production_countries', []))}"
        f" | 상영시간: {movie.get('runtime', '')}분"
        f" | 평점: {movie.get('vote_average', '')} (참여 {movie.get('vote_count', '')}명)",
        f"줄거리: {movie.get('overview', '')}",
    ]
    return "\n".join(lines)


def to_metadata(movie: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB 메타데이터. 값은 문자열·숫자만 넣는다(리스트 불가)."""
    return {
        "chunk_id": f"movies.json:{movie['movie_id']}",
        "source_type": "movie_meta",
        "movie_id": movie["movie_id"],
        "title": movie["title"],
        "original_title": movie.get("original_title", ""),
        "year": movie.get("year") or 0,
        "genres": ", ".join(movie.get("genres", [])),
        "director": ", ".join(movie.get("director", [])),
        "tmdb_url": movie.get("tmdb_url", ""),
    }
