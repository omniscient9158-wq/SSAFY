"""키워드 추출 품질을 모델별로 비교한다.

위키 호출도 임베딩도 하지 않는다. LLM 키워드 추출만 돌려 결과를 나란히 본다.

    python src/compare_models.py --list                     # 사용 가능한 모델 조회
    python src/compare_models.py --models gpt-5-nano,gpt-5-mini
    python src/compare_models.py --models gpt-5-nano --effort minimal,low,medium
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

import config
import llm as llm_module
import movies as movie_data

BASE = llm_module.GMS_BASE_URL
REPORT_PATH = config.BASE_DIR / "outputs" / "model_compare.json"


def list_models() -> None:
    key = os.environ.get("GMS_KEY", "")
    try:
        response = requests.get(
            BASE.rstrip("/") + "/models",
            headers={"authorization": f"Bearer {key}"},
            timeout=20,
        )
        data = response.json()
    except Exception as exc:
        print(f"조회 실패: {exc}")
        return
    names = [m.get("id") for m in data.get("data", [])] if isinstance(data, dict) else []
    if names:
        for name in sorted(n for n in names if n):
            print(f"  {name}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])


def run(model: str, effort: str, targets: list[dict]) -> dict:
    """한 모델로 대상 영화들의 키워드를 뽑는다."""
    import collect_wiki

    options = {"reasoning_effort": effort} if effort else {}
    label = {"model": model, "effort": effort or "(기본)"}

    # 모델명이나 옵션이 틀렸으면 5번 헛돌지 않고 바로 알린다.
    # reasoning_effort를 지원하지 않는 모델이 있으므로, 거부되면 기본값으로 재시도한다.
    ignored = ""
    try:
        llm_module.get_llm(model=model, **options).invoke("ok 라고만 답해")
    except Exception as exc:
        if not options:
            return {**label, "error": f"{type(exc).__name__}: {exc}"}
        try:
            llm_module.get_llm(model=model).invoke("ok 라고만 답해")
        except Exception as inner:
            return {**label, "error": f"{type(inner).__name__}: {inner}"}
        ignored = f"reasoning_effort={effort} 미지원 → 기본값으로 실행"
        options = {}
        label["effort"] = "(기본)"

    usage = {"input": 0, "output": 0}

    class CountingLLM:
        """ChatOpenAI는 Pydantic 모델이라 메서드를 덧붙일 수 없다. 감싸서 센다."""

        def __init__(self, inner):
            self.inner = inner

        def invoke(self, *args, **kwargs):
            response = self.inner.invoke(*args, **kwargs)
            meta = getattr(response, "usage_metadata", None) or {}
            usage["input"] += meta.get("input_tokens", 0)
            usage["output"] += meta.get("output_tokens", 0)
            return response

    collect_wiki.get_llm = lambda *a, **k: CountingLLM(llm_module.get_llm(model=model, **options))

    started = time.perf_counter()
    rows = []
    for movie in targets:
        begin = time.perf_counter()
        keywords = collect_wiki.extract_keywords(movie)
        rows.append({
            "movie": movie["title"],
            "keywords": keywords,
            "seconds": round(time.perf_counter() - begin, 1),
        })
    return {
        **label,
        "참고": ignored,
        "총_초": round(time.perf_counter() - started, 1),
        "입력_토큰": usage["input"],
        "출력_토큰": usage["output"],
        "결과": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="키워드 추출 모델 비교")
    parser.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 조회")
    parser.add_argument("--models", default="", help="쉼표로 구분한 모델 이름")
    parser.add_argument("--effort", default="", help="쉼표로 구분한 reasoning_effort 값")
    parser.add_argument("--limit", type=int, default=5, help="비교에 쓸 영화 수")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not args.models:
        raise SystemExit("--models 로 비교할 모델을 지정하세요. (--list 로 목록 확인)")

    ranked = sorted(
        movie_data.load_movies(), key=lambda m: m.get("vote_count") or 0, reverse=True
    )
    targets = ranked[: args.limit]
    print(f"대상 {len(targets)}편: {', '.join(m['title'] for m in targets)}\n")

    efforts = args.effort.split(",") if args.effort else [""]
    results = []
    for model in args.models.split(","):
        for effort in efforts:
            label = f"{model.strip()} / {effort.strip() or '기본'}"
            print(f"── {label} 실행 중...")
            results.append(run(model.strip(), effort.strip(), targets))

    print("\n" + "=" * 78)
    for result in results:
        print(f"\n■ {result['model']}  (reasoning_effort: {result['effort']})")
        if "error" in result:
            print(f"    실패: {result['error']}")
            continue
        if result.get("참고"):
            print(f"    ! {result['참고']}")
        count = len(result["결과"])
        print(
            f"    총 {result['총_초']}초 · 영화당 {result['총_초'] / count:.1f}초"
            f" · 출력 토큰 영화당 {result['출력_토큰'] // count:,}"
            f" (입력 {result['입력_토큰'] // count:,})"
        )
        for row in result["결과"]:
            print(f"    [{row['movie']}] {row['seconds']}초")
            print(f"      {row['keywords']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"\n보고서: {REPORT_PATH}")


if __name__ == "__main__":
    main()
