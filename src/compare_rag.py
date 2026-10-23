"""Reproduce the five fixed F2015 cases; API calls require --generate.

Outputs retain both answers and the exact evidence for manual factual review.
Run without --generate first to inspect retrieval without spending credits.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import config
from rag_graph import generate_node, retrieve_node

CASES = [
    ("B-524", "타우린 수용해도 기록의 값, 단위, 온도와 출처는?"),
    ("B-524", "승인된 타우린 발췌는 분자 구조를 어떤 형태라고 설명하나?"),
    ("A-5538", "안트라센의 외관과 물에서의 거동은 원문에 어떻게 기록되어 있나?"),
    ("C-533", "카페인의 문헌상 수용해도 값, 단위, 온도와 원출처는?"),
    ("B-524", "승인된 자료로 타우린의 pH별 정량적 용해도 변화량을 알 수 있나?"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="GMS 답변 최대 10회 생성 (크레딧 사용)")
    parser.add_argument("--mode", choices=["local", "vector"], default="local")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--output", type=Path, default=config.BASE_DIR / "research/runs/rag_comparison.json")
    args = parser.parse_args()
    if args.mode == "vector" and not args.generate:
        parser.error("API 없는 사전 점검은 --mode local을 사용하세요.")
    if args.output.exists():
        parser.error("기존 결과를 보존합니다. 새 --output 경로를 지정하세요.")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.CHAT_MODEL,
        "retrieval_mode": args.mode,
        "top_k": args.top_k,
        "generation_requested": args.generate,
        "review_status": "수동 원문 대조 전; 인용 ID 일치는 사실 정확성을 보장하지 않음",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for number, (compound_id, query) in enumerate(CASES, 1):
        case = {"number": number, "compound_id": compound_id, "query": query}
        report["cases"].append(case)
        stage = "retrieve"
        try:
            started = perf_counter()
            evidence = retrieve_node({"compound_id": compound_id, "query": query,
                                      "retrieval_mode": args.mode, "top_k": args.top_k})
            case["retrieval_seconds"] = round(perf_counter() - started, 3)
            case["evidence"] = evidence
            save()
            if args.generate:
                from llm import get_llm

                stage = "baseline"
                started = perf_counter()
                baseline = get_llm(timeout=45, max_retries=0).invoke([
                    ("system", "너는 화학 데이터셋을 설명하는 교육용 도우미다. 한국어로 답하라. "
                     "확인할 수 없는 내용은 확인할 수 없다고 말하라. 안전·취급 조언은 하지 마라."),
                    ("human", f"선택 화합물 ID: {compound_id}\n질문: {query}"),
                ])
                case["baseline_answer"] = str(baseline.content)
                case["baseline_seconds"] = round(perf_counter() - started, 3)
                save()
                stage = "rag"
                started = perf_counter()
                case["rag"] = generate_node({"compound_id": compound_id, "query": query, **evidence})
                case["rag_seconds"] = round(perf_counter() - started, 3)
                offered = {"근거 1", *[e["citation_id"] for e in evidence["external_evidence"]]}
                used = set(re.findall(r"근거\s+\d+", case["rag"]["answer"]))
                case["unknown_citation_ids"] = sorted(used - offered)
                case["status"] = "generated_pending_manual_review"
            else:
                case["status"] = "retrieval_only"
        except Exception as exc:
            # SDK error strings may contain credentials or request details.
            case["status"] = "error"
            case["error_stage"] = stage
            case["error_type"] = type(exc).__name__
            save()
            print(f"case {number}: {stage} failed ({type(exc).__name__}); partial results saved")
            raise SystemExit(1) from None
        save()
        print(f"case {number}: {case['status']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
