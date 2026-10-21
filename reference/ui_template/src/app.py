"""Movie Lab — 영화 큐레이터 (필수 기능)

실행: streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import collect_wiki
import collect_wikidata
import config
import movies as movie_data
import rag_graph
import recommend
import review
import vector_store

st.set_page_config(page_title="Movie Lab", page_icon="🎬")

# 좁은 사이드바에서 입력창의 "Press Enter to apply" 안내가 입력 글자와 겹친다.
st.markdown(
    "<style>[data-testid='InputInstructions'] {display: none;}</style>",
    unsafe_allow_html=True,
)


# 영화 선택 selectbox의 위젯 키. 선택된 영화는 이 위젯 값 하나로만 관리한다.
MOVIE_KEY = "movie_select"


@st.cache_data(show_spinner=False)
def movie_options() -> dict[str, str]:
    """{movie_id: 표시 이름}. 표시 이름 순으로 정렬한다."""
    items = sorted(
        ((movie_data.label(m), m["movie_id"]) for m in movie_data.load_movies()),
        key=lambda x: x[0],
    )
    return {movie_id: name for name, movie_id in items}


@st.cache_data(show_spinner=False)
def similar_movies(movie_id: str) -> list[dict]:
    """사이드바용. 영화를 바꿀 때마다 부르므로 결과를 캐시한다."""
    return recommend.similar(movie_id)


def pick_movie(movie_id: str) -> None:
    """추천 카드·비슷한 작품 버튼에서 영화를 고른다. 버튼의 on_click 콜백으로만 부른다.

    selectbox 값을 직접 바꿔 사이드바 선택과 같은 상태를 쓴다. 콜백은 다음 실행이
    시작되기 전에 돌아서 위젯 값을 바꿀 수 있다. (본문에서 바꾸면 이미 그려진 위젯이라
    StreamlitAPIException이 난다.)
    """
    st.session_state[MOVIE_KEY] = movie_id
    st.session_state.pop("recommendations", None)


def get_history(movie_id: str) -> list[dict]:
    """영화별 대화 기록. 영화를 바꾸면 그 영화의 기록만 보인다."""
    histories = st.session_state.setdefault("histories", {})
    return histories.setdefault(movie_id, [])


def sidebar() -> str:
    """영화 선택 + 추천 + 색인 상태. 선택된 movie_id를 반환한다."""
    with st.sidebar:
        st.header("영화 선택")

        options = movie_options()
        # 선택값은 key로 유지한다. index를 직전 선택값으로 계산하면 실행마다 인자가 달라져
        # Streamlit이 다른 위젯으로 보고 사용자가 방금 고른 값을 버린다. (영화 변경이 한 번 무시됨)
        # 추천 카드·비슷한 작품에서 고른 영화는 pick_movie 콜백이 이 위젯 값을 바꿔 반영한다.
        movie_id = st.selectbox(
            "질문할 영화",
            list(options),
            format_func=lambda movie_id: options.get(movie_id, movie_id),
            index=None,
            key=MOVIE_KEY,
            placeholder="영화 제목을 입력하세요",
        ) or ""
        if movie_id != st.session_state.get("movie_id"):
            # 영화를 바꾸면 추천 결과 화면을 닫고 그 영화의 대화로 넘어간다.
            st.session_state.pop("recommendations", None)
        st.session_state.movie_id = movie_id

        # 진입점 — 제목을 몰라도 영화를 찾을 수 있게 한다.
        with st.expander("🔎 이런 영화 찾아줘", expanded=not movie_id):
            # 폼으로 묶어야 엔터키로도 검색된다.
            with st.form("movie_search"):
                query = st.text_input(
                    "어떤 영화를 찾으세요?",
                    placeholder="우주 배경에 가족이 나오는 감동적인 영화",
                )
                submitted = st.form_submit_button("영화 찾기", use_container_width=True)

            if submitted and query.strip():
                with st.spinner("영화를 찾는 중..."):
                    try:
                        results = recommend.recommend(query.strip())
                    except Exception:
                        results = []
                st.session_state.recommendations = results
                st.session_state.recommend_query = query.strip()

        st.divider()
        try:
            store = vector_store.get_store(config.MOVIE_COLLECTION)
            movie_count = vector_store.count(store)
        except Exception:
            movie_count = 0

        st.caption(f"영화 메타데이터 색인: {movie_count}건")
        if movie_count == 0:
            st.warning("먼저 `python src/index_movies.py`를 실행하세요.")

        if movie_id:
            movie = movie_data.find_movie(movie_id)
            if movie:
                st.caption(f"감독: {', '.join(movie['director'])}")
                st.caption(f"출연: {', '.join(movie['cast'][:3])}")

            titles = collect_wiki.collected_titles(movie_id)
            if titles:
                st.caption(f"수집된 위키 문서 {len(titles)}건")
                st.caption(" · ".join(titles))

            # 리뷰는 대화 흐름 안에 assistant 메시지로 들어간다.
            if st.button("📝 리뷰 생성", use_container_width=True):
                st.session_state.review_request = movie_id
                st.rerun()

            # 대화 중 자연스럽게 다음 영화로 넘어가는 통로.
            try:
                others = similar_movies(movie_id)
            except Exception:
                others = []
            if others:
                st.divider()
                st.caption("이 영화와 비슷한 작품")
                for other in others:
                    st.button(
                        movie_data.label(other),
                        key=f"similar_{other['movie_id']}",
                        use_container_width=True,
                        on_click=pick_movie,
                        args=(other["movie_id"],),
                    )

        # 대화 중인 영화 목록 — 영화마다 기록이 따로 쌓인다는 걸 보여준다.
        histories = st.session_state.get("histories", {})
        talking = [mid for mid, items in histories.items() if items]
        if talking:
            st.divider()
            st.caption("대화 중인 영화")
            for mid in talking:
                movie = movie_data.find_movie(mid)
                name = movie_data.label(movie) if movie else mid
                mark = "▶ " if mid == movie_id else "· "
                asked = sum(1 for item in histories[mid] if item["role"] == "user")
                st.caption(f"{mark}{name} ({asked}문답)")

            if movie_id and histories.get(movie_id):
                if st.button("이 영화의 대화 지우기", use_container_width=True):
                    histories[movie_id] = []
                    st.rerun()

        return movie_id


def prepare_wiki(movie_id: str, question: str) -> None:
    """질문에 맞춰 위키 문서를 수집·색인한다.

    이미 받은 키워드는 건너뛰므로, 질문이 바뀌어 새 키워드가 나오면 그만큼만 추가된다.
    """
    movie = movie_data.find_movie(movie_id)
    if not movie:
        return

    with st.status("관련 위키백과 문서를 확인하고 있습니다...", expanded=False) as status:
        result = collect_wiki.collect_for_movie(movie, question)
        added = result.get("added_titles", [])
        total = result.get("titles", [])

        if added:
            label = f"위키 문서 {len(added)}건 추가 (총 {len(total)}건)"
        else:
            label = f"이미 수집된 위키 문서 {len(total)}건 사용"
        # 두 번째 출처(F2016). 실패해도 위키백과만으로 답할 수 있으니 흐름을 끊지 않는다.
        try:
            wikidata = collect_wikidata.collect_for_movie(movie)
        except Exception:
            wikidata = {"status": "error", "chunks": 0}
        if wikidata["chunks"]:
            label += f" · 위키데이터 {wikidata['qid']}"
        status.update(label=label, state="complete")
        st.caption("수집 문서: " + (", ".join(total) or "없음"))


def merge_sources(sources: list[dict]) -> list[dict]:
    """같은 문서에서 나온 근거는 한 줄로 합친다.

    위키 문서 하나가 여러 청크로 쪼개져 있어 같은 제목이 여러 번 뜨기 때문이다.
    번호는 모두 남겨 답변 본문의 [근거 N]을 그대로 찾을 수 있게 한다.
    """
    merged: dict[tuple, dict] = {}
    for position, source in enumerate(sources, start=1):
        # 답변 본문의 [근거 N]과 번호를 맞춘다(인용되지 않은 번호는 빠져 있다).
        index = source.get("index", position)
        key = (source["title"], source["source_type"], source["url"])
        if key in merged:
            merged[key]["indexes"].append(index)
        else:
            merged[key] = {
                "indexes": [index],
                "title": source["title"],
                "source_type": source["source_type"],
                "url": source["url"],
            }
    return list(merged.values())


def render_sources(sources: list[dict], cited: bool = True) -> None:
    """답변에 실제로 인용한 근거만 보여준다. (F2008)

    인용이 하나도 없으면 그 사실을 안내한다. 검색됐지만 인용되지 않은 문서를
    '답변에 사용한 근거'로 보여주면 문서에 기대지 않은 답변도 근거가 있는 것처럼 보인다.
    """
    if not cited or not sources:
        st.caption("ⓘ 답변에 인용된 근거가 없습니다.")
        return
    documents = merge_sources(sources)
    with st.expander(f"답변에 사용한 근거 {len(documents)}건"):
        for document in documents:
            numbers = ", ".join(str(index) for index in sorted(document["indexes"]))
            origin = rag_graph.source_label(document["source_type"])
            line = f"**[근거 {numbers}]** {document['title']} · {origin}"
            if document["url"]:
                line += f" · [원문]({document['url']})"
            st.markdown(line)


def render_recommendations() -> bool:
    """추천 결과 화면. 화면을 그렸으면 True를 돌려준다."""
    results = st.session_state.get("recommendations")
    if results is None:
        return False

    # 돌아가는 길을 맨 위에 둔다. 결과가 길면 아래쪽 버튼은 눈에 띄지 않는다.
    if st.button("← 대화 화면으로 돌아가기"):
        st.session_state.pop("recommendations", None)
        st.rerun()

    st.subheader(f"\"{st.session_state.get('recommend_query', '')}\" 추천 결과")
    st.caption(
        f"영화 메타데이터 색인에서 후보 {config.RECOMMEND_FETCH}편을 찾고"
        " 요청에 맞는 것만 골랐습니다. 고르면 그 영화의 대화가 시작됩니다."
    )

    if not results:
        st.info("요청에 맞는 영화를 찾지 못했습니다. 다른 표현으로 다시 찾아보세요.")

    for item in results:
        movie = item["movie"]
        with st.container(border=True):
            st.markdown(f"**🎬 {movie_data.label(movie)}**")
            st.caption(
                f"{', '.join(movie.get('genres', []))}"
                f" · 감독 {', '.join(movie.get('director', []))}"
            )
            if item["reason"]:
                st.write(item["reason"])
            st.button(
                "이 영화에 대해 질문하기",
                key=f"pick_{movie['movie_id']}",
                use_container_width=True,
                on_click=pick_movie,
                args=(movie["movie_id"],),
            )

    return True


def main() -> None:
    st.title("🎬 Movie Lab")
    st.caption("영화 메타데이터와 위키백과 문서에서 근거를 찾아 답변하는 영화 큐레이터")

    movie_id = sidebar()

    if render_recommendations():
        return

    if not movie_id:
        st.info("왼쪽에서 질문할 영화를 선택하거나, `🔎 이런 영화 찾아줘`로 영화를 찾아보세요.\n\n영화마다 대화가 따로 저장됩니다.")
        return

    movie = movie_data.find_movie(movie_id)
    st.subheader(movie_data.label(movie) if movie else movie_id)

    history = get_history(movie_id)
    for item in history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                render_sources(item.get("sources", []), item.get("cited", True))

    if st.session_state.pop("review_request", None) == movie_id:
        with st.chat_message("assistant"):
            with st.spinner("리뷰를 작성하는 중..."):
                try:
                    result = review.write(movie_id)
                except Exception:
                    result = None
            if result and result["review"]:
                st.markdown(result["review"])
                render_sources(result["sources"], result["cited"])
                history.append({
                    "role": "assistant",
                    "content": result["review"],
                    "sources": result["sources"],
                    "cited": result["cited"],
                })
            else:
                st.error("리뷰를 만들지 못했습니다. 색인 상태와 API 키 설정을 확인해 주세요.")

    question = st.chat_input("이 영화에 대해 궁금한 점을 물어보세요")
    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            prepare_wiki(movie_id, question)
            with st.spinner("문서를 검색하고 답변을 만드는 중..."):
                result = rag_graph.ask(question, movie_id)
        except Exception:
            st.error("답변을 만들지 못했습니다. 색인 상태와 API 키 설정을 확인해 주세요.")
            history.pop()  # 답변이 없는 질문은 기록에 남기지 않는다.
            return

        st.markdown(result["answer"])
        render_sources(result["sources"], result["cited"])

    history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "cited": result["cited"],
    })


if __name__ == "__main__":
    main()
