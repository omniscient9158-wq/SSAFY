"""위키백과 API 부하 · GMS 크레딧 소모 시뮬레이션.

실제 수집 경로(collect_wiki)를 그대로 쓰되, 저장 위치만 outputs/load_sim 으로 격리한다.
프로젝트의 wiki_manifest.json · resources/rag · movie_wiki 인덱스는 건드리지 않는다.

    python src/simulate_load.py --limit 10        # 파일럿 (기본)
    python src/simulate_load.py --limit 300       # 전체
    python src/simulate_load.py --limit 300 --restart   # 처음부터 다시

중단해도(Ctrl+C) 지금까지의 결과로 보고서를 남기며, 다시 실행하면 남은 영화부터 이어간다.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from langchain_core.embeddings import Embeddings

import config
import llm as llm_module
import movies as movie_data
import vector_store

KEY_INFO_URL = "https://gms.ssafy.io/gmsapi/key-info"
EXTRAPOLATE_TO = 300          # 파일럿 결과를 몇 편 기준으로 환산할지

# ── 측정 기록 ──────────────────────────────────────────────────────────
WIKI_LOG: list[dict[str, Any]] = []
LLM_LOG: list[dict[str, Any]] = []
EMBED_LOG = {"batches": 0, "texts": 0, "chars": 0}
STOPPED = {"flag": False}


def _install_probes() -> None:
    """실제 호출 경로에 계측을 끼워 넣는다."""
    import collect_wiki
    import wiki_api

    # 1) 위키백과 HTTP 요청 — 재시도까지 모두 남는다.
    original_get = wiki_api.session.get

    def logged_get(url, **kwargs):
        params = kwargs.get("params") or {}
        kind = "search" if params.get("list") == "search" else "fetch_page"
        started = time.perf_counter()
        error = ""
        response = None
        try:
            response = original_get(url, **kwargs)
            status = response.status_code
        except Exception as exc:
            status = -1
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        WIKI_LOG.append({
            "at": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "query": params.get("srsearch") or params.get("titles") or "",
            "status": status,
            "seconds": round(elapsed, 3),
            "retry_after": (response.headers.get("Retry-After") if response is not None else None),
            "bytes": (len(response.content) if response is not None else 0),
            "error": error,
        })

        # 제한을 건 주체가 위키백과인지 중간 프록시인지 확인하려면 본문과 헤더가 필요하다.
        if response is not None and response.status_code != 200:
            WIKI_LOG[-1]["headers"] = dict(response.headers)
            WIKI_LOG[-1]["body"] = response.text[:1000]
        if response is None:
            raise RuntimeError(error)
        return response

    wiki_api.session.get = logged_get

    # 2) LLM 키워드 추출 — 호출 수와 토큰 사용량
    class CountingLLM:
        def __init__(self, inner):
            self.inner = inner

        def invoke(self, *args, **kwargs):
            started = time.perf_counter()
            response = self.inner.invoke(*args, **kwargs)
            usage = getattr(response, "usage_metadata", None) or {}
            LLM_LOG.append({
                "seconds": round(time.perf_counter() - started, 3),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
            return response

    collect_wiki.get_llm = lambda *a, **k: CountingLLM(llm_module.get_llm(*a, **k))

    # 3) 임베딩 — 크레딧의 대부분이 여기서 나간다.
    class CountingEmbeddings(Embeddings):
        def __init__(self, inner):
            self.inner = inner

        def embed_documents(self, texts):
            EMBED_LOG["batches"] += 1
            EMBED_LOG["texts"] += len(texts)
            EMBED_LOG["chars"] += sum(len(t) for t in texts)
            return self.inner.embed_documents(texts)

        def embed_query(self, text):
            EMBED_LOG["chars"] += len(text)
            return self.inner.embed_query(text)

    vector_store.get_embeddings = lambda: CountingEmbeddings(llm_module.get_embeddings())


def _isolate_storage(sim_dir: Path) -> None:
    """수집 결과를 시뮬레이션 전용 공간으로 돌린다."""
    sim_dir.mkdir(parents=True, exist_ok=True)
    config.WIKI_DIR = sim_dir / "wikipedia"
    config.MANIFEST_PATH = sim_dir / "manifest.json"
    config.INDEX_DIR = sim_dir / "indexes"
    config.WIKI_COLLECTION = "sim_movie_wiki"


# ── GMS 크레딧 ─────────────────────────────────────────────────────────
def key_info() -> dict[str, Any]:
    key = os.environ.get("GMS_KEY", "")
    if not key:
        return {"error": "GMS_KEY가 없습니다."}
    try:
        response = requests.get(
            KEY_INFO_URL, headers={"authorization": f"Bearer {key}"}, timeout=20
        )
        return {"status": response.status_code, "body": response.json()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _flatten(value: Any, prefix: str = "") -> dict[str, float]:
    """응답 형식을 모르므로 숫자 필드를 전부 펼쳐 비교한다."""
    found: dict[str, float] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            found.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        found[prefix] = float(value)
    return found


def credit_delta(before: dict, after: dict) -> dict[str, dict[str, float]]:
    left = _flatten(before.get("body", {}))
    right = _flatten(after.get("body", {}))
    delta = {}
    for field in sorted(set(left) | set(right)):
        b, a = left.get(field), right.get(field)
        if b is None or a is None or b == a:
            continue
        delta[field] = {"before": b, "after": a, "change": round(a - b, 6)}
    return delta


# ── 요약 ───────────────────────────────────────────────────────────────
def summarize(done: int, seconds: float) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for entry in WIKI_LOG:
        statuses[str(entry["status"])] = statuses.get(str(entry["status"]), 0) + 1

    latencies = sorted(e["seconds"] for e in WIKI_LOG if e["status"] == 200)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        return round(latencies[min(int(len(latencies) * p), len(latencies) - 1)], 3)

    wiki_calls = len(WIKI_LOG)
    per_movie = (wiki_calls / done) if done else 0
    return {
        "영화_처리_수": done,
        "총_소요_초": round(seconds, 1),
        "영화당_평균_초": round(seconds / done, 1) if done else 0,
        "위키_요청_수": wiki_calls,
        "위키_영화당_요청": round(per_movie, 1),
        "위키_검색_요청": sum(1 for e in WIKI_LOG if e["kind"] == "search"),
        "위키_본문_요청": sum(1 for e in WIKI_LOG if e["kind"] == "fetch_page"),
        "위키_상태코드": statuses,
        "위키_429_횟수": statuses.get("429", 0),
        "위키_초당_요청": round(wiki_calls / seconds, 3) if seconds else 0,
        "위키_지연_p50": pct(0.50),
        "위키_지연_p95": pct(0.95),
        "위키_지연_최대": round(latencies[-1], 3) if latencies else 0,
        "LLM_호출_수": len(LLM_LOG),
        "LLM_입력_토큰": sum(e["input_tokens"] for e in LLM_LOG),
        "LLM_출력_토큰": sum(e["output_tokens"] for e in LLM_LOG),
        "임베딩_배치": EMBED_LOG["batches"],
        "임베딩_청크": EMBED_LOG["texts"],
        "임베딩_문자수": EMBED_LOG["chars"],
    }


def extrapolate(summary: dict[str, Any], target: int) -> dict[str, Any]:
    done = summary["영화_처리_수"]
    if not done:
        return {}
    scale = target / done
    hours = summary["총_소요_초"] * scale / 3600
    return {
        "기준_영화_수": target,
        "예상_위키_요청": round(summary["위키_요청_수"] * scale),
        "예상_소요_시간": f"{hours:.1f}시간",
        "예상_LLM_토큰": round(
            (summary["LLM_입력_토큰"] + summary["LLM_출력_토큰"]) * scale
        ),
        "예상_임베딩_청크": round(summary["임베딩_청크"] * scale),
        "예상_임베딩_문자수": round(summary["임베딩_문자수"] * scale),
    }


PROBE_TERMS = [
    "영화", "감독", "배우", "제작", "각본", "촬영", "음악", "편집", "배급", "개봉",
    "수상", "원작", "소설", "시나리오", "촬영지", "특수효과", "미술", "의상", "분장", "홍보",
]


def probe(minutes: int) -> None:
    """분당 상한만 잰다. LLM·임베딩·저장이 없어 크레딧을 쓰지 않는다.

    같은 네트워크의 두 대에서 서로 다른 WIKI_CONTACT로 동시에 돌리면
    한도가 IP 기준인지 (IP+UA) 기준인지 판별할 수 있다.
    """
    import wiki_api

    print(f"User-Agent: {wiki_api.USER_AGENT}")
    print(f"{minutes}분 동안 검색 요청을 보내며 분당 상한을 측정합니다. (크레딧 0)\n")

    deadline = time.time() + minutes * 60
    per_minute: dict[str, dict[str, int]] = {}
    index = 0

    while time.time() < deadline:
        term = f"{PROBE_TERMS[index % len(PROBE_TERMS)]} {index}"
        index += 1
        minute = datetime.now().strftime("%H:%M")
        bucket = per_minute.setdefault(minute, {"200": 0, "429": 0})
        try:
            response = wiki_api.session.get(
                wiki_api.API_URL,
                params={
                    "action": "query", "list": "search", "srsearch": term,
                    "srlimit": 1, "format": "json", "formatversion": 2,
                },
                timeout=20,
            )
        except Exception as exc:
            print(f"  {minute} 요청 실패: {exc}")
            time.sleep(2)
            continue

        if response.status_code == 429:
            bucket["429"] += 1
            wait = response.headers.get("Retry-After")
            wait = float(wait) if wait and wait.isdigit() else 5.0
            print(f"  {minute} 429 · 이 분의 성공 {bucket['200']}회 · {wait:.0f}초 대기")
            time.sleep(min(wait + 1, 70))
        else:
            bucket["200"] += 1

    print("\n" + "=" * 52)
    print(f"  User-Agent: {wiki_api.USER_AGENT}")
    full = [v["200"] for k, v in list(per_minute.items())[1:-1]]  # 처음·끝의 잘린 분 제외
    for minute, v in per_minute.items():
        print(f"  {minute}  성공 {v['200']:>3}  429 {v['429']:>2}")
    if full:
        print(f"\n  분당 상한(온전한 분 기준): 최소 {min(full)} · 최대 {max(full)}")
        tier = "Unidentified (10/min)" if max(full) <= 12 else "User-Agent only (200/min)"
        print(f"  추정 등급: {tier}")
    print("=" * 52)


def main() -> None:
    parser = argparse.ArgumentParser(description="위키백과 부하 · 크레딧 소모 측정")
    parser.add_argument("--limit", type=int, default=10, help="처리할 영화 수 (기본 10)")
    parser.add_argument(
        "--offset", type=int, default=0,
        help="인기 순위 몇 위부터 시작할지 (기본 0). 이미 수집한 영화를 피해 재측정할 때 쓴다.",
    )
    parser.add_argument("--restart", action="store_true", help="이전 수집 이력을 무시하고 처음부터")
    parser.add_argument(
        "--tag", default="",
        help="결과를 outputs/load_sim_<tag>에 따로 저장한다. 개선 전후 비교에 쓴다.",
    )
    parser.add_argument(
        "--probe", type=int, default=0,
        help="분당 상한만 측정한다 (분 단위). LLM·임베딩·저장 없음, 크레딧 0.",
    )
    args = parser.parse_args()

    if args.probe:
        probe(args.probe)
        return

    folder = f"load_sim_{args.tag}" if args.tag else "load_sim"
    sim_dir = config.BASE_DIR / "outputs" / folder
    _isolate_storage(sim_dir)
    _install_probes()

    import collect_wiki  # 격리·계측을 끝낸 뒤에 불러온다.

    if args.restart:
        import shutil

        if config.MANIFEST_PATH.exists():
            config.MANIFEST_PATH.unlink()
        # 색인이 남아 있으면 중복 청크로 걸러져 임베딩이 발생하지 않는다.
        if config.INDEX_DIR.exists():
            shutil.rmtree(config.INDEX_DIR)

    # 유명한 순서 = 평가 참여자 수
    ranked = sorted(
        movie_data.load_movies(), key=lambda m: m.get("vote_count") or 0, reverse=True
    )
    targets = ranked[args.offset : args.offset + args.limit]

    # 전부 이미 수집된 상태면 아무 호출도 일어나지 않는다. 미리 알려준다.
    remaining = [m for m in targets if not collect_wiki.is_collected(m["movie_id"])]
    print(f"대상 {len(targets)}편 (순위 {args.offset + 1}~{args.offset + len(targets)})"
          f" · 새로 수집할 영화 {len(remaining)}편 · 저장 위치 {sim_dir}")
    if not remaining:
        print(
            "\n이미 모두 수집된 영화입니다. API 호출이 일어나지 않아 측정할 것이 없습니다.\n"
            f"  뒤쪽 순위로 재측정: python src/simulate_load.py --limit {args.limit} --offset {args.offset + args.limit}\n"
            "  처음부터 다시    : python src/simulate_load.py --limit 10 --restart"
        )
        return

    print("시작 전 크레딧을 조회합니다...")
    before = key_info()
    print(json.dumps(before, ensure_ascii=False)[:400])

    def on_interrupt(signum, frame):
        STOPPED["flag"] = True
        print("\n중단 요청됨. 현재 영화까지 마치고 보고서를 남깁니다...")

    signal.signal(signal.SIGINT, on_interrupt)

    started = time.perf_counter()
    processed = 0          # 이번 실행에서 실제로 수집한 영화 (평균 계산의 분모)
    skipped = 0            # 이전 실행에서 이미 끝난 영화
    failures: list[dict[str, str]] = []

    for order, movie in enumerate(targets, start=1):
        if STOPPED["flag"]:
            break
        if collect_wiki.is_collected(movie["movie_id"]):
            skipped += 1
            continue

        label = movie_data.label(movie)
        print(f"[{order}/{len(targets)}] {label}", flush=True)
        try:
            result = collect_wiki.collect_for_movie(movie)
            print(f"    문서 {len(result.get('titles', []))}건 · 청크 {result.get('chunks', 0)}")
        except Exception as exc:
            failures.append({"movie": label, "error": f"{type(exc).__name__}: {exc}"})
            print(f"    실패: {exc}")
        processed += 1

    seconds = time.perf_counter() - started

    print("\n종료 후 크레딧을 조회합니다...")
    after = key_info()

    summary = summarize(processed, seconds)
    summary["건너뛴_영화_수"] = skipped
    report = {
        "실행_시각": datetime.now().isoformat(timespec="seconds"),
        "설정": {
            "KEYWORD_COUNT": config.KEYWORD_COUNT,
            "SEARCH_LIMIT": config.SEARCH_LIMIT,
            "MAX_PAGES": config.MAX_PAGES,
            "CHUNK_SIZE": config.WIKI_CHUNK_SIZE,
            "CHAT_MODEL": llm_module.CHAT_MODEL,
            "EMBEDDING_MODEL": llm_module.EMBEDDING_MODEL,
        },
        "요약": summary,
        f"{EXTRAPOLATE_TO}편_환산": extrapolate(summary, EXTRAPOLATE_TO),
        "크레딧_변화": credit_delta(before, after),
        "크레딧_조회_전": before,
        "크레딧_조회_후": after,
        "실패": failures,
        "위키_요청_기록": WIKI_LOG,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = sim_dir / f"report_{stamp}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    for key, value in report[f"{EXTRAPOLATE_TO}편_환산"].items():
        print(f"  {key}: {value}")
    print("-" * 60)
    if report["크레딧_변화"]:
        for field, change in report["크레딧_변화"].items():
            print(f"  {field}: {change['before']} → {change['after']} ({change['change']:+})")
    elif before.get("status") != 200 or after.get("status") != 200:
        print("  크레딧 조회에 실패했습니다. GMS_KEY와 네트워크를 확인하세요.")
        print(f"    전: {before}")
        print(f"    후: {after}")
    elif processed == 0:
        print("  새로 수집한 영화가 없어 크레딧 변화가 없습니다. (--offset 또는 --restart)")
    else:
        print("  크레딧은 정상 조회됐지만 사용량이 그대로입니다. 반영이 늦을 수 있으니")
        print("  잠시 뒤 아래로 다시 확인해 보세요.")
        print('    curl "https://gms.ssafy.io/gmsapi/key-info" -H "authorization: Bearer $GMS_KEY"')
    print("=" * 60)
    print(f"\n보고서: {path}")


if __name__ == "__main__":
    main()
