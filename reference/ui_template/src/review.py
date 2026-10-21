"""리뷰 생성 — 선택한 영화의 메타데이터(RAG 1)와 수집된 위키 문서(RAG 2)로 관람 가이드를 쓴다.

질문에 답하는 게 아니라 영화 한 편을 통째로 소개하므로,
질문 대신 영화 정보 자체를 검색어로 써서 위키 문서를 고르게 훑는다.
"""

from __future__ import annotations

from typing import Any

import config
import movies as movie_data
import rag_graph
import vector_store
from llm import get_llm

# ── 프롬프트 5 : 관람 가이드 ────────────────────────────────────────────
REVIEW_PROMPT = """넌 영화 큐레이터야. 아래 [문서]에 있는 내용만 사용해서 이 영화의 관람 가이드를 써.

문서 읽는 법
- "영화 메타데이터" 출처는 영화의 줄거리·소개다. 영화가 어떻게 묘사했는지를 알려줄 뿐 실제 사실이 아니다.
- "위키백과" 출처는 실제 사실 확인에 사용한다.
- "위키데이터" 출처는 공개일·출연과 배역·원작·수상·흥행 수입 같은 구조화된 사실 기록이다.

규칙
1. **문서에 없는 장면·연기·연출·촬영·음악은 쓰지 않는다.** 보지 않은 영화를 본 것처럼 묘사하지 않는다.
2. 좋다/나쁘다는 평가 대신, 어떤 영화인지와 무엇을 알고 보면 좋은지를 알려준다.
3. 줄거리는 이야기의 출발점까지만 쓰고 결말은 밝히지 않는다.
4. 주요 내용 뒤에는 [근거 N] 형태로 번호를 붙인다. 근거가 여러 개면 [근거 2, 5]처럼 한 괄호 안에 쉼표로 쓴다.
5. 아래 형식과 제목을 그대로 지킨다. 문서에 근거가 없는 항목은 "문서에서 확인되지 않습니다"라고 적는다.

## 어떤 영화인가
(2~3문장. 장르·소재·이야기의 출발점)

## 이런 분에게
- (2~3항목. 어떤 취향의 관객에게 맞는지)

## 알고 보면 좋은 배경
- (위키백과 문서에서 확인된 실제 사건·인물·원작 등)

## 볼 때 참고할 점
- (상영시간, 관객 평점, 영화와 실제 기록의 차이 등 문서로 확인되는 것만)

[영화]
{title}

[문서]
{context}
"""


def collect(movie: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """리뷰에 쓸 근거를 모은다. 영화 메타데이터 전부 + 그 영화의 위키 청크."""
    movie_id = movie["movie_id"]
    movie_store = vector_store.get_store(config.MOVIE_COLLECTION)
    wiki_store = vector_store.get_store(config.WIKI_COLLECTION)

    collected: list[dict[str, Any]] = []
    for item in vector_store.get_by_movie(movie_store, movie_id):
        collected.append(rag_graph.to_source(item["page_content"], item["metadata"]))

    # 특정 질문이 없으므로 영화 정보 자체를 검색어로 쓰고,
    # 한 문서에 쏠리지 않도록 MMR로 다양하게 고른다.
    wiki_docs = wiki_store.max_marginal_relevance_search(
        movie_data.to_text(movie),
        k=config.REVIEW_TOP_K_WIKI,
        fetch_k=config.FETCH_K,
        filter={"movie_id": movie_id},
    )
    for doc in wiki_docs:
        collected.append(rag_graph.to_source(doc.page_content, doc.metadata))

    # 위키데이터(F2016) 사실 목록은 영화당 1~3청크라 전부 넣는다.
    wikidata_store = vector_store.get_store(config.WIKIDATA_COLLECTION)
    for item in vector_store.get_by_movie(wikidata_store, movie_id):
        collected.append(rag_graph.to_source(item["page_content"], item["metadata"]))

    return rag_graph.pack_sources(collected)


def write(movie_id: str) -> dict[str, Any]:
    """관람 가이드 리뷰와 실제로 인용한 근거를 돌려준다."""
    movie = movie_data.find_movie(movie_id)
    if not movie:
        return {"review": "", "sources": [], "cited": False}

    sources, context = collect(movie)
    response = get_llm().invoke(
        REVIEW_PROMPT.format(
            title=movie_data.to_text(movie),
            context=context or "(검색된 문서가 없습니다)",
        )
    )
    review = response.content
    # 질의응답과 같은 근거 표기 방식을 쓴다. 인용이 없으면 빈 목록과 cited=False. (F2008)
    cited = rag_graph.cited_sources(review, sources)
    return {"review": review, "sources": cited, "cited": bool(cited)}
