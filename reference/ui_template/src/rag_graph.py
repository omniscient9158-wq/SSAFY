"""LangGraph로 구성한 RAG 파이프라인. retrieve → generate 두 노드."""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any

from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import config
import vector_store
from llm import get_llm

# ── 프롬프트 3 : 근거 기반 답변 (요구사항 2.4 — 검색된 문서만 사용) ────────────────────────────────────────
ANSWER_PROMPT = """너는 영화 큐레이터야. 아래 [문서]를 근거로 [질문]에 답해.

문서 읽는 법
- "영화 메타데이터" 출처는 영화의 줄거리·소개다. 영화가 어떻게 묘사했는지를 알려줄 뿐 실제 사실이 아니다.
- "위키백과" 출처는 실제 사실 확인에 사용한다.
- "위키데이터" 출처는 공개일·출연과 배역·원작·배경 장소·수상·흥행 수입 같은 구조화된 사실 기록이다.
  실제 사실 확인에 사용하며, 숫자·연도·이름은 이 출처의 값을 우선한다.

공통 규칙
1. [문서]에 있는 내용만 말한다. 실제 사실(역사·인물·과학·제작 정보)뿐 아니라
   영화 속 장면·설정·줄거리도 마찬가지다. 자신의 지식으로 채우지 않는다.
   문서에 없으면 "문서에서 확인되지 않는다"고 밝힌다.
2. 사용자가 질문에서 영화의 묘사를 직접 제시했으면("영화처럼 ~했다는데") 그 묘사는
   질문의 전제로 받아 실제 기록과 대조한다. 질문과 문서에 없는 장면을 덧붙이지 않는다.
3. 문서에서 온 주장 뒤에는 [근거 N] 형태로 번호를 붙인다.
   근거가 여러 개면 [근거 2, 5]처럼 한 괄호 안에 쉼표로 쓴다. 범위(2~5)나 다른 괄호는 쓰지 않는다.
4. 문서에서 확인한 내용은 추측이 아니라 사실로 단정해 말한다.
5. 질문에 맞는 만큼만 쓴다. 묻지 않은 항목이나 빈 제목은 넣지 않는다.

답변 형식은 질문 유형에 맞춰 고른다.

(A) 사실 여부를 묻는 질문일 때
   ("실제로 그랬어?", "이거 맞아?", "고증이 정확해?"처럼 참·거짓 판단을 요구하는 경우)
   영화의 묘사와 실제 기록을 나누어 대조한 뒤, 맨 끝에 아래 한 줄을 붙인다.
   (판정을 먼저 쓰면 설명과 어긋나기 쉽다. 설명을 다 쓰고 그 결론대로 판정한다.)

   판정: (사실 / 대체로 사실 / 각색됨 / 문서로 확인 불가 중 하나)

   - 영화의 묘사: 영화 메타데이터 근거와, 사용자가 질문에서 제시한 묘사로만 설명한다.
     둘 다에 없는 장면이면 "영화 정보에서 확인되지 않는다"고 쓴다.
   - 실제 기록: 위키백과·위키데이터 근거로만 설명한다. 숫자·인물·연도는 문서에서 값을 찾아 비교한다.
   - 판정은 실제 기록 쪽 근거로만 내린다. 기준은 다음과 같다.
     · 사실: 질문의 핵심 주장이 위키백과·위키데이터 근거와 일치한다.
     · 대체로 사실: 핵심 배경·사건은 근거로 뒷받침되지만 세부(의도·정확한 수치·대사·장면)는 근거에 없다.
     · 각색됨: 근거의 기록과 영화 묘사가 어긋나거나, 근거에 없는 인물·사건·장치가 극적으로 추가됐다.
     · 문서로 확인 불가: 질문과 관련된 실제 기록 자체가 근거에 없을 때만 쓴다.
   - 근거가 질문 문장과 똑같이 쓰여 있지 않아도 된다. 근거의 사실들로부터 판단할 수 있으면 판단한다.
   - "과학적으로 가능한가"류 질문은 근거에 있는 원리로 판단한다.

(B) 그 외의 질문일 때
   ("줄거리 알려줘", "감독이 누구야?", "비슷한 영화 추천", "이 사건 배경 설명해줘" 등)
   판정 줄을 쓰지 않는다. 질문에 맞는 자연스러운 형태로 답한다.
   - 사실을 묻는 짧은 질문이면 한두 문장으로 바로 답한다.
   - 설명·요약이면 2~5문장 문단으로, 항목이 여러 개면 짧은 목록으로 답한다.

[문서]
{context}

[질문]
{question}
"""


# 근거 출처 종류 (F2016). 화면과 프롬프트가 같은 이름을 쓰도록 한곳에 둔다.
SOURCE_LABELS = {
    "movie_meta": "영화 메타데이터",
    "wikipedia": "위키백과",
    "wikidata": "위키데이터",
}


def source_label(source_type: str) -> str:
    return SOURCE_LABELS.get(source_type, "외부 문서")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    movie_id: str
    context: str
    sources: list[dict[str, Any]]


def to_source(document: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": metadata.get("chunk_id", ""),
        "source_type": metadata.get("source_type", ""),
        "title": metadata.get("title", ""),
        "url": metadata.get("source_url") or metadata.get("tmdb_url", ""),
        "text": document,
    }


def pack_sources(
    collected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """중복 청크를 지우고 [근거 N] 번호를 붙여 프롬프트에 넣을 문맥을 만든다.

    질의응답과 리뷰가 같은 근거 번호 체계를 쓰도록 한곳에 모아둔다.
    """
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in collected:
        if source["chunk_id"] in seen:
            continue
        seen.add(source["chunk_id"])
        sources.append(source)

    # 답변의 [근거 N]과 맞추기 위해 번호를 함께 저장한다.
    for index, source in enumerate(sources, start=1):
        source["index"] = index

    context = "\n\n".join(
        f"[근거 {source['index']}] 출처: {source_label(source['source_type'])}"
        f" · {source['title']}\n{source['text']}"
        for source in sources
    )
    return sources, context


def retrieve(state: State) -> dict[str, Any]:
    """영화 메타데이터·위키백과·위키데이터를 검색해 병합하고 중복 청크를 제거한다."""
    question = state["messages"][-1].content
    movie_id = state.get("movie_id", "")

    movie_store = vector_store.get_store(config.MOVIE_COLLECTION)
    wiki_store = vector_store.get_store(config.WIKI_COLLECTION)

    collected: list[dict[str, Any]] = []

    # 사용자가 고른 영화의 청크는 항상 근거에 포함한다.
    if movie_id:
        for item in vector_store.get_by_movie(movie_store, movie_id):
            collected.append(to_source(item["page_content"], item["metadata"]))

    # 다른 영화까지 유사도로 검색하면 질문과 무관한 작품이 근거에 섞인다.
    # 영화를 고르지 않았거나 비교 질문을 다루고 싶을 때만 켠다.
    if not movie_id or config.INCLUDE_SIMILAR_MOVIES:
        for doc in movie_store.similarity_search(question, k=config.TOP_K_MOVIE):
            collected.append(to_source(doc.page_content, doc.metadata))

    wiki_filter = {"movie_id": movie_id} if movie_id else None
    if config.USE_MMR:
        # 비슷한 청크가 몰리면 정작 필요한 문단이 밀려난다. 넉넉히 뽑아 다양성으로 고른다.
        wiki_docs = wiki_store.max_marginal_relevance_search(
            question, k=config.TOP_K_WIKI, fetch_k=config.FETCH_K, filter=wiki_filter
        )
    else:
        wiki_docs = wiki_store.similarity_search(
            question, k=config.TOP_K_WIKI, filter=wiki_filter
        )
    for doc in wiki_docs:
        collected.append(to_source(doc.page_content, doc.metadata))

    # 위키데이터(F2016)는 영화당 1~3청크의 사실 목록이라 검색 없이 전부 넣는다.
    # 유사도로 고르면 "수상", "흥행"처럼 한 줄짜리 사실이 질문과 멀다고 빠진다.
    if movie_id:
        wikidata_store = vector_store.get_store(config.WIKIDATA_COLLECTION)
        for item in vector_store.get_by_movie(wikidata_store, movie_id):
            collected.append(to_source(item["page_content"], item["metadata"]))

    sources, context = pack_sources(collected)
    return {"context": context, "sources": sources}


# 답변 본문의 근거 표기를 찾는다. 프롬프트는 [근거 3] · [근거 3, 5]로 쓰게 하지만, LLM이 형식을
# 어겨도 받아 준다. ([근거3] · [근거 2~4] · [근거 2, 근거 5] · ［근거 3］ · (근거 3))
# 하나라도 놓치면 실제로 인용한 답변이 '인용 없음'으로 보이므로 넓게 잡는다.
CITE_PATTERN = re.compile(r"[\[［(（]\s*근거\s*([^\]］)）]*)[\]］)）]")
CITE_NUMBER = re.compile(r"(\d+)(?:\s*[~～\-–]\s*(\d+))?")


def cited_numbers(answer: str) -> set[int]:
    """답변 본문에서 인용한 근거 번호를 모은다."""
    used: set[int] = set()
    for group in CITE_PATTERN.findall(answer):
        for start, end in CITE_NUMBER.findall(group):
            first = int(start)
            last = int(end) if end else first
            if first <= last <= first + 50:   # 범위 표기 [근거 2~4]
                used.update(range(first, last + 1))
            else:
                used.add(first)
    return used


def cited_sources(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """답변에서 실제로 인용한 근거만 골라낸다. (F2008)

    인용이 하나도 없으면 빈 목록을 돌려준다. 검색된 근거 전체로 대신 채우면 문서를 쓰지 않은
    답변도 근거가 있는 것처럼 보인다. 화면은 빈 목록이면 '인용된 근거가 없다'고 안내한다.
    """
    used = cited_numbers(answer)
    return [source for source in sources if source["index"] in used]


def generate(state: State) -> dict[str, Any]:
    """검색된 근거만으로 답변을 생성한다."""
    question = state["messages"][-1].content
    context = state["context"] or "(검색된 문서가 없습니다)"

    response = get_llm().invoke([
        SystemMessage(
            content="너는 검색된 문서만 근거로 답하는 영화 큐레이터다. "
            "영화 속 장면이라도 문서에 없는 내용은 지어내지 않는다."
        ),
        HumanMessage(content=ANSWER_PROMPT.format(context=context, question=question)),
    ])

    return {"messages": [response]}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(question: str, movie_id: str = "") -> dict[str, Any]:
    """질문 1건을 처리하고 답변과 근거 목록을 돌려준다."""
    agent = build_graph()
    result = agent.invoke({
        "messages": [HumanMessage(content=question)],
        "movie_id": movie_id,
        "context": "",
        "sources": [],
    })
    answer = result["messages"][-1].content
    sources = cited_sources(answer, result.get("sources", []))
    # cited: 답변이 근거를 하나라도 인용했는지. False면 화면이 '인용된 근거 없음'을 안내한다. (F2008)
    return {"answer": answer, "sources": sources, "cited": bool(sources)}
