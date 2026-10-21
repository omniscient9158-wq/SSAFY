"""Movie Lab 웹 API 서버 (F2013 — 프론트엔드 직접 구현).

화면(src/web/: HTML·CSS·JavaScript)과 서버(이 파일)를 분리한다. 서버는 JSON API만 제공하고
RAG 파이프라인은 Streamlit 앱과 같은 모듈(rag_graph·collect_wiki·recommend·review)을 그대로 쓴다.

실행:
    python src/server.py                 # http://localhost:8000
    python src/server.py --port 8080

요청·응답 형식은 README "웹 API" 절에 정리했다.
Streamlit 앱과 같은 색인을 쓰므로 둘을 동시에 띄우지 않는다. (chromadb 다중 프로세스 쓰기 금지)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import collect_wiki
import collect_wikidata
import config
import llm
import movies as movie_data
import rag_graph
import recommend
import review
import vector_store

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Movie Lab API", docs_url="/api/docs", openapi_url="/api/openapi.json")

# FastAPI는 일반 함수 엔드포인트를 스레드 풀에서 돌린다. chromadb 클라이언트를 여러 스레드가
# 동시에 열면 "Could not connect to tenant" 오류가 나고 색인이 손상될 수 있어 한 번에 하나만 처리한다.
_index_lock = threading.Lock()


# ── 요청 본문 ─────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    movie_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=500)


class CollectRequest(BaseModel):
    question: str = Field("", max_length=500)


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)


# ── 공통 ───────────────────────────────────────────────────────────────────
def _movie_or_404(movie_id: str) -> dict[str, Any]:
    movie = movie_data.find_movie(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="영화를 찾을 수 없습니다.")
    return movie


def _movie_brief(movie: dict[str, Any]) -> dict[str, Any]:
    return {
        "movie_id": movie["movie_id"],
        "label": movie_data.label(movie),
        "title": movie["title"],
        "year": movie.get("year"),
        "genres": movie.get("genres", []),
        "director": movie.get("director", []),
    }


def _sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """본문(text)은 빼고 화면에 필요한 것만 보낸다."""
    return [
        {
            "index": s.get("index"),
            "title": s["title"],
            "source_type": s["source_type"],
            "source_label": rag_graph.source_label(s["source_type"]),
            "url": s["url"],
        }
        for s in sources
    ]


def _count(collection: str) -> int:
    try:
        return vector_store.count(vector_store.get_store(collection))
    except Exception:
        return 0


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    """입력 형식 오류. 화면에 그대로 보여줄 수 있는 한 문장으로 돌려준다."""
    return JSONResponse(
        status_code=422,
        content={"detail": "입력값을 확인해 주세요. (질문은 1~500자, 검색 조건은 1~200자)"},
    )


@app.exception_handler(Exception)
async def hide_internal_errors(request: Request, error: Exception) -> JSONResponse:
    """오류 추적 정보·키·경로를 화면에 보내지 않는다. (NF2004) 서버 로그에만 남긴다."""
    print(f"[오류] {request.method} {request.url.path}: {type(error).__name__}: {error}")
    return JSONResponse(
        status_code=500,
        content={"detail": "요청을 처리하지 못했습니다. 색인 상태와 API 키 설정을 확인해 주세요."},
    )


# ── API ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status() -> dict[str, Any]:
    """색인 건수와 API 키 설정 여부. 화면이 첫 안내를 정하는 데 쓴다. (F2012)"""
    with _index_lock:
        counts = {
            "movie_meta": _count(config.MOVIE_COLLECTION),
            "wikipedia": _count(config.WIKI_COLLECTION),
            "wikidata": _count(config.WIKIDATA_COLLECTION),
        }
    return {
        "indexes": counts,
        "api_key": bool(os.environ.get("GMS_KEY")),
        "chat_model": llm.CHAT_MODEL,
        "movies": len(movie_data.load_movies()),
    }


@app.get("/api/movies")
def list_movies() -> list[dict[str, Any]]:
    items = [
        {"movie_id": m["movie_id"], "label": movie_data.label(m)}
        for m in movie_data.load_movies()
    ]
    return sorted(items, key=lambda item: item["label"])


@app.get("/api/movies/{movie_id}")
def movie_detail(movie_id: str) -> dict[str, Any]:
    movie = _movie_or_404(movie_id)
    with _index_lock:
        try:
            similar = [_movie_brief(m) for m in recommend.similar(movie_id)]
        except Exception:
            similar = []
    return {
        **_movie_brief(movie),
        "cast": movie.get("cast", [])[:5],
        "overview": movie.get("overview", ""),
        "wiki_titles": collect_wiki.collected_titles(movie_id),
        "wikidata": collect_wikidata.load_manifest().get(movie_id, {}).get("title", ""),
        "similar": similar,
    }


@app.post("/api/movies/{movie_id}/collect")
def collect(movie_id: str, body: CollectRequest) -> dict[str, Any]:
    """질문에 맞춰 위키백과·위키데이터 문서를 수집·색인한다. 답변 전에 따로 불러 진행 상태를 나눠 보여준다."""
    movie = _movie_or_404(movie_id)
    with _index_lock:
        wiki = collect_wiki.collect_for_movie(movie, body.question)
        try:
            wikidata = collect_wikidata.collect_for_movie(movie)
        except Exception:
            wikidata = {"status": "error", "chunks": 0}
    return {
        "added_titles": wiki.get("added_titles", []),
        "titles": wiki.get("titles", []),
        "wikidata": {"status": wikidata["status"], "qid": wikidata.get("qid", ""), "chunks": wikidata["chunks"]},
    }


@app.post("/api/ask")
def ask(body: AskRequest) -> dict[str, Any]:
    _movie_or_404(body.movie_id)
    with _index_lock:
        result = rag_graph.ask(body.question.strip(), body.movie_id)
    # sources에는 답변이 실제로 인용한 근거만 들어간다. 인용이 없으면 빈 목록, cited=False. (F2008)
    return {"answer": result["answer"], "sources": _sources(result["sources"]), "cited": result["cited"]}


@app.post("/api/movies/{movie_id}/review")
def write_review(movie_id: str) -> dict[str, Any]:
    _movie_or_404(movie_id)
    with _index_lock:
        result = review.write(movie_id)
    if not result["review"]:
        raise HTTPException(status_code=500, detail="리뷰를 만들지 못했습니다.")
    return {"review": result["review"], "sources": _sources(result["sources"]), "cited": result["cited"]}


@app.post("/api/recommend")
def recommend_movies(body: RecommendRequest) -> dict[str, Any]:
    with _index_lock:
        results = recommend.recommend(body.query.strip())
    return {
        "query": body.query.strip(),
        "results": [{**_movie_brief(item["movie"]), "reason": item["reason"]} for item in results],
    }


# 정적 파일은 API 경로 뒤에 붙인다. 먼저 붙이면 "/" 마운트가 /api 요청을 가로챈다.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Movie Lab 웹 서버 (F2013)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
