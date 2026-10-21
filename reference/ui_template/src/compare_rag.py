"""RAG 사용 전후 답변 비교 (F2015).

같은 질문을 같은 모델에 두 번 묻는다.
    - RAG 없음 : 영화 제목과 질문만 준다. 모델이 알고 있는 지식으로 답한다.
    - RAG 사용 : rag_graph.ask() — 영화 메타데이터·위키백과·위키데이터를 근거로 답한다.

두 답을 나란히 저장해 사실 정확성·근거 제시·모름 인정 세 기준으로 비교한다.
판정은 사람이 한다. 결과 파일의 "정답 메모"는 비교 기준으로 삼은 사실과 그 출처다.

실행:
    python src/compare_rag.py                  # 대표 질문 전부
    python src/compare_rag.py --only 1,3       # 일부만
결과:
    outputs/rag_compare/compare_<날짜시각>.json · .md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain.messages import HumanMessage, SystemMessage

import collect_wiki
import collect_wikidata
import config
import llm
import movies as movie_data
import rag_graph

OUTPUT_DIR = config.BASE_DIR / "outputs" / "rag_compare"

# 근거 없이 답하면 틀리기 쉬운 질문. 수치·실존 인물·원작·수상처럼 기억에 기대면 흔들리는 것,
# 그리고 어디에도 기록이 없어 "모른다"가 정답인 질문을 섞었다.
# (영화 ID, 질문, 정답 메모 — 비교 기준으로 삼은 사실과 출처)
QUESTIONS = [
    ("282631", "영화 명량의 국내 누적 관객 수는 몇 명이야?",
     "1,761만 명 (한국 영화 역대 1위). 출처: 위키백과 '명량 (영화)'"),
    ("282631", "명량해전 당시 일본 수군의 배는 몇 척이었어?",
     "난중일기 기준 적선 133척(130여 척). 흔히 알려진 330척은 난중일기 간행 때의 오독으로 설명된다. "
     "출처: 위키백과 '명량 해전', '이순신'"),
    ("437068", "택시운전사에서 송강호가 연기한 김만섭의 실제 모델은 누구야? 실제 이름도 김만섭이야?",
     "실존 인물 김사복이 모티브, 영화 속 이름 김만섭은 각색. 출처: 위키백과 '택시운전사 (영화)'"),
    ("437081", "영화 남한산성의 원작은 무엇이고 작가는 누구야?",
     "김훈의 장편소설 『남한산성』(2007). 출처: 위키백과 '남한산성 (영화)'"),
    ("157336", "인터스텔라는 아카데미 시상식에서 어떤 상을 받았어?",
     "제87회 아카데미 시각효과상 수상 (음악·음향편집·음향효과·미술 후보). 출처: 위키데이터 Q13417189"),
    ("919207", "서울의 봄에서 황정민이 연기한 전두광과 정우성이 연기한 이태신은 각각 어떤 실존 인물을 모티브로 했어?",
     "전두광 = 전두환, 이태신 = 장태완(수도경비사령관). 출처: 위키백과 '서울의 봄 (영화)'"),
    ("157336", "인터스텔라 촬영 첫날 크리스토퍼 놀란 감독이 점심으로 먹은 음식은 뭐야?",
     "공개된 기록이 없다. '모른다/확인되지 않는다'가 정답. (모름 인정 확인용)"),
]

NO_RAG_SYSTEM = "너는 영화 큐레이터다."
NO_RAG_PROMPT = "[영화]\n{movie}\n\n[질문]\n{question}"


def answer_without_rag(movie: dict, question: str) -> str:
    response = llm.get_llm().invoke([
        SystemMessage(content=NO_RAG_SYSTEM),
        HumanMessage(content=NO_RAG_PROMPT.format(movie=movie_data.label(movie), question=question)),
    ])
    return response.content


def answer_with_rag(movie: dict, question: str) -> tuple[str, list[dict]]:
    # 앱과 같은 순서: 질문에 맞춰 문서를 준비한 뒤 답한다.
    collect_wiki.collect_for_movie(movie, question)
    try:
        collect_wikidata.collect_for_movie(movie)
    except Exception as error:
        print(f"  [위키데이터 실패] {error}")
    result = rag_graph.ask(question, movie["movie_id"])
    sources = [
        {"index": s["index"], "source": rag_graph.source_label(s["source_type"]), "title": s["title"]}
        for s in result["sources"]
    ]
    return result["answer"], sources


def to_markdown(rows: list[dict], meta: dict) -> str:
    lines = [
        f"# RAG 사용 전후 답변 비교 ({meta['run_at']})",
        "",
        f"- 모델: `{meta['model']}` (두 방식 모두 같은 모델)",
        f"- RAG 없음: 영화 제목과 질문만 제공 · RAG 사용: `rag_graph.ask()`",
        "",
    ]
    for number, row in enumerate(rows, start=1):
        lines += [
            f"## {number}. [{row['movie']}] {row['question']}",
            "",
            f"**정답 메모** — {row['reference']}",
            "",
            f"### RAG 없음 ({row['seconds_no_rag']}초)",
            "",
            row["no_rag"].strip(),
            "",
            f"### RAG 사용 ({row['seconds_rag']}초)",
            "",
            row["rag"].strip(),
            "",
            "근거: " + (", ".join(f"[{s['index']}] {s['source']} · {s['title']}" for s in row["sources"]) or "없음"),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 사용 전후 답변 비교 (F2015)")
    parser.add_argument("--only", default="", help="질문 번호 (예: 1,3)")
    args = parser.parse_args()

    picked = {int(n) for n in args.only.split(",") if n.strip()} if args.only else None
    rows = []
    for number, (movie_id, question, reference) in enumerate(QUESTIONS, start=1):
        if picked and number not in picked:
            continue
        movie = movie_data.find_movie(movie_id)
        print(f"[{number}] {movie['title']} — {question}")

        started = time.perf_counter()
        no_rag = answer_without_rag(movie, question)
        seconds_no_rag = round(time.perf_counter() - started, 1)

        started = time.perf_counter()
        rag, sources = answer_with_rag(movie, question)
        seconds_rag = round(time.perf_counter() - started, 1)

        rows.append({
            "number": number,
            "movie_id": movie_id,
            "movie": movie_data.label(movie),
            "question": question,
            "reference": reference,
            "no_rag": no_rag,
            "rag": rag,
            "sources": sources,
            "seconds_no_rag": seconds_no_rag,
            "seconds_rag": seconds_rag,
        })

    meta = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": llm.CHAT_MODEL}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (OUTPUT_DIR / f"compare_{stamp}.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / f"compare_{stamp}.md").write_text(to_markdown(rows, meta), encoding="utf-8")
    print(f"[완료] {OUTPUT_DIR / f'compare_{stamp}.md'}")


if __name__ == "__main__":
    main()
