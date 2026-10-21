"""사전 구축 대상 영화 300편을 고르고 목록을 저장한다.

    python src/select_movies.py

TMDB의 vote_count는 영어권 기준이라 한국 영화가 구조적으로 밀린다.
(명량 872위, 극한직업 870위) 그래서 해외 상위와 한국 상위를 나눠 뽑는다.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import movies as movie_data

GLOBAL_COUNT = 200         # 전체에서 뽑을 수
KOREAN_COUNT = 100         # 한국 영화에서 추가로 뽑을 수
OUTPUT_PATH = config.BASE_DIR / "data" / "prebuilt_movies.json"


def is_korean(movie: dict) -> bool:
    countries = " ".join(movie.get("production_countries", []))
    return movie.get("original_language") == "ko" or "Korea" in countries


def popularity(movie: dict) -> int:
    return movie.get("vote_count") or 0


def select() -> list[dict]:
    """해외 상위 + 한국 상위를 합쳐 중복 없이 목록을 만든다."""
    everything = sorted(movie_data.load_movies(), key=popularity, reverse=True)

    picked: list[dict] = []
    seen: set[str] = set()

    def add(movie: dict, reason: str) -> None:
        if movie["movie_id"] in seen:
            return
        seen.add(movie["movie_id"])
        picked.append({
            "movie_id": movie["movie_id"],
            "title": movie["title"],
            "year": movie.get("year"),
            "vote_count": popularity(movie),
            "korean": is_korean(movie),
            "reason": reason,
        })

    for movie in everything[:GLOBAL_COUNT]:
        add(movie, "global")
    for movie in [m for m in everything if is_korean(m)][:KOREAN_COUNT]:
        add(movie, "korean")

    return picked


def main() -> None:
    picked = select()
    korean = sum(1 for m in picked if m["korean"])

    payload = {
        "생성일": date.today().isoformat(),
        "기준": f"vote_count 상위 {GLOBAL_COUNT}편 + 한국 영화 상위 {KOREAN_COUNT}편",
        "총_영화_수": len(picked),
        "한국_영화_수": korean,
        "movies": picked,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"선정 {len(picked)}편 (한국 {korean}편) → {OUTPUT_PATH}")
    print("\n한국 영화 상위 15편:")
    for movie in [m for m in picked if m["korean"]][:15]:
        print(f"  {movie['title']} ({movie['year']})  vote {movie['vote_count']}")


if __name__ == "__main__":
    main()
