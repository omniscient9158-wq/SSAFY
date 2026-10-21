"""영화 추천 — RAG 1(movie_meta) 검색으로 후보를 넓게 뽑고 LLM이 골라낸다.

벡터 검색은 "가까운 순서로 k개"를 돌려줄 뿐 얼마나 가까워야 하는지는 모른다.
그래서 검색은 후보를 넓게 던지는 역할만 하고, 요청에 맞는지는 LLM이 판정한다.
질의응답과 달리 위키 문서(RAG 2)는 쓰지 않는다.
"""

from __future__ import annotations

import json
import re
from typing import Any

import config
import movies as movie_data
import vector_store
from llm import get_llm

# ── 프롬프트 4 : 후보 선별 + 추천 이유 ──────────────────────────────────
SELECT_PROMPT = """넌 영화 큐레이터야.
아래 [후보 영화] 중에서 [사용자 요청]에 맞는 영화만 골라줘.

1. 후보는 검색으로 넓게 뽑은 목록이라 요청과 무관한 영화가 섞여 있다.
   **맞지 않는 영화는 반드시 빼라.**
2. 개수를 채우려고 억지로 고르지 않는다. 맞는 영화가 1편뿐이면 1편만, 없으면 빈 배열([])을 낸다.
3. 잘 맞는 순서로 최대 {limit}편까지 고른다.
4. 판단은 후보 영화의 정보(장르·줄거리 등)에 있는 내용만으로 한다. 없는 내용을 지어내지 않는다.
5. 이유는 요청과 어떻게 연결되는지를 한 문장(40자 내외)으로 쓴다.
6. 다른 설명 없이 JSON 배열만 출력한다. 번호는 후보 번호다.
   예: [{{"번호": 3, "이유": "웜홀을 지나는 우주 탐사를 다룹니다"}}]

[사용자 요청]
{query}

[후보 영화]
{candidates}
"""


def _parse_json_array(text: str) -> list[Any]:
    """LLM 응답에서 JSON 배열만 뽑아낸다."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def search(query: str, k: int, exclude_movie_id: str = "") -> list[dict[str, Any]]:
    """movie_meta 컬렉션에서 요청과 가까운 영화를 찾는다."""
    store = vector_store.get_store(config.MOVIE_COLLECTION)
    # 제외·중복으로 빠질 수 있으니 넉넉히 뽑는다.
    docs = store.similarity_search(query, k=k + 3)

    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in docs:
        movie_id = doc.metadata.get("movie_id", "")
        if not movie_id or movie_id in seen or movie_id == exclude_movie_id:
            continue
        movie = movie_data.find_movie(movie_id)
        if movie:
            seen.add(movie_id)
            found.append(movie)
    return found[:k]


def select(query: str, found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """후보 중 요청에 맞는 영화만 고르고 이유를 붙인다."""
    candidates = "\n\n".join(
        f"{number}. {movie_data.to_text(movie)}"
        for number, movie in enumerate(found, start=1)
    )
    prompt = SELECT_PROMPT.format(
        limit=config.RECOMMEND_COUNT, query=query, candidates=candidates
    )
    try:
        response = get_llm().invoke(prompt)
        values = _parse_json_array(response.content)
    except Exception:
        values = []

    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            number = int(value.get("번호", 0))
        except (TypeError, ValueError):
            continue
        # 없는 번호나 중복은 버린다.
        if not (1 <= number <= len(found)) or number in used:
            continue
        used.add(number)
        picked.append({
            "movie": found[number - 1],
            "reason": str(value.get("이유", "")).strip(),
        })
        if len(picked) >= config.RECOMMEND_COUNT:
            break
    return picked


def recommend(query: str, exclude_movie_id: str = "") -> list[dict[str, Any]]:
    """사용자 요청으로 영화를 찾는다. 검색은 넓게, 선별은 LLM이 한다."""
    found = search(query, config.RECOMMEND_FETCH, exclude_movie_id)
    if not found:
        return []
    return select(query, found)


def similar(movie_id: str, k: int = 0) -> list[dict[str, Any]]:
    """선택한 영화와 비슷한 작품.

    영화를 바꿀 때마다 부르므로 선별(LLM 호출) 없이 검색만 한다.
    """
    movie = movie_data.find_movie(movie_id)
    if not movie:
        return []
    return search(movie_data.to_text(movie), k or config.SIMILAR_COUNT, movie_id)
