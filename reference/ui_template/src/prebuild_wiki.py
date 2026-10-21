"""위키 문서 사전 구축.

두 가지 용도가 있다.
  1. 시연용 코퍼스  — 완성본을 보여줄 때 즉시 답변이 나오도록 미리 채워 둔다.
  2. 폴백 데이터셋  — 교육장에서 위키 호출 한도에 걸리면, 수집해 둔 원문
                     (resources/rag/wikipedia/*.json)을 배포해 수집 단계를 대신한다.

한도는 User-Agent에 연락처가 있느냐로 20배 갈린다(분당 10회 → 200회).
연락처를 넣으면 영화당 약 20요청이므로 분당 10편까지 처리된다. 직접 수집이
가능한지는 교육장 IP 공유 여부와 동시 인원으로 정해지므로, 사전 리허설로 판단한다.

    python src/select_movies.py            # 1. 대상 300편 선정
    python src/prebuild_wiki.py --reset    # 2. 처음부터 구축
    python src/prebuild_wiki.py            # 3. 중단됐다면 이어서

수집 결과는 프로젝트 실제 경로에 쌓인다.
    resources/rag/wikipedia/*.json      원문
    resources/rag/wiki_manifest.json    수집 이력
    resources/indexes/movie_wiki/       벡터 색인
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

import config
import movies as movie_data

KEY_INFO_URL = "https://gms.ssafy.io/gmsapi/key-info"
LIST_PATH = config.BASE_DIR / "data" / "prebuilt_movies.json"
REPORT_PATH = config.BASE_DIR / "outputs" / "prebuild_report.json"

STOPPED = {"flag": False}


def key_info() -> dict:
    import os

    key = os.environ.get("GMS_KEY", "")
    if not key:
        return {"error": "GMS_KEY 없음"}
    try:
        response = requests.get(
            KEY_INFO_URL, headers={"authorization": f"Bearer {key}"}, timeout=20
        )
        return {"status": response.status_code, "body": response.json()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def used_credit(info: dict) -> float | None:
    body = info.get("body") or {}
    value = body.get("usedCredit")
    return float(value) if isinstance(value, (int, float)) else None


def install_diagnostics(log: list) -> None:
    """키워드가 왜 빈손이 되는지 기록한다.

    검색 결과 / 필터 판정 / 본문 길이를 남겨야 원인을 구분할 수 있다.
    추측으로 규칙을 덧붙이지 않기 위한 장치다.
    """
    import collect_wiki
    import wiki_api

    original_select = collect_wiki.select_documents
    original_search = wiki_api.search
    original_fetch = wiki_api.fetch_page

    def search(query, limit=3):
        results = original_search(query, limit=limit)
        log.append({"단계": "검색", "키워드": query,
                    "결과": [r["title"] for r in results]})
        return results

    def select_documents(movie, candidates, limit=0):
        picked = original_select(movie, candidates, limit)
        names = {item["title"] for item in picked}
        for item in candidates:
            log.append({"단계": "선별", "키워드": item["keyword"],
                        "문서": item["title"], "통과": item["title"] in names})
        return picked

    def fetch_page(title):
        page = original_fetch(title)
        log.append({"단계": "본문", "요청": title,
                    "받은제목": page["title"] if page else None,
                    "길이": len(page["text"]) if page else 0,
                    "일반기준": config.MIN_TEXT_LENGTH,
                    "핵심기준": config.CORE_MIN_LENGTH})
        return page

    wiki_api.search = search
    collect_wiki.select_documents = select_documents
    wiki_api.fetch_page = fetch_page


def print_diagnostics(log: list) -> None:
    """키워드별로 어디서 걸렸는지 정리해 보여준다."""
    print("\n" + "=" * 70)
    print("  진단 — 키워드별 처리 경과")
    print("=" * 70)
    current = None
    for entry in log:
        if entry["단계"] == "검색":
            current = entry["키워드"]
            print(f"\n▸ '{current}'")
            print(f"    검색결과: {entry['결과']}")
        elif entry["단계"] == "선별":
            mark = "선택" if entry["통과"] else "제외"
            print(f"      [{mark}] {entry['문서']}")
        else:
            if entry["받은제목"] is None:
                print(f"      본문 없음: {entry['요청']}")
            else:
                # 본편은 길이와 관계없이 받고, 감독 문서는 기준이 낮아 짧아도 유지될 수 있다.
                if entry["길이"] < entry["핵심기준"]:
                    short = " ← 너무 짧음 (본편이면 유지, 아니면 버림)"
                elif entry["길이"] < entry["일반기준"]:
                    short = " ← 짧음 (본편·감독 문서면 유지)"
                else:
                    short = ""
                print(f"      본문 {entry['받은제목']} {entry['길이']:,}자{short}")


def load_targets() -> list[dict]:
    if not LIST_PATH.exists():
        raise SystemExit(
            f"대상 목록이 없습니다: {LIST_PATH}\n먼저 python src/select_movies.py 를 실행하세요."
        )
    with LIST_PATH.open(encoding="utf-8") as file:
        return json.load(file)["movies"]


def reset_storage() -> None:
    """이전 수집분을 비운다. 필터 수정 전에 모은 문서가 섞이지 않게 한다."""
    for path in (config.WIKI_DIR, config.INDEX_DIR / config.WIKI_COLLECTION):
        if path.exists():
            shutil.rmtree(path)
    if config.MANIFEST_PATH.exists():
        config.MANIFEST_PATH.unlink()
    print("이전 수집분을 비웠습니다.")


def fill_own(targets: list[dict]) -> None:
    """본편 문서가 빠진 영화에 그것만 채워 넣는다.

    기존 문서를 지우지 않으므로 전체 재수집과 달리 이미 모은 것을 잃지 않는다.
    키워드 추출·문서 선별을 하지 않아 LLM 호출도 없다.
    """
    import collect_wiki

    todo = []
    for entry in targets:
        movie = movie_data.find_movie(entry["movie_id"])
        if movie and collect_wiki.is_collected(entry["movie_id"]):
            if not collect_wiki.has_own_article(movie):
                todo.append(movie)

    print(f"본편 문서가 없는 영화 {len(todo)}편을 보강합니다. (LLM 호출 없음)")
    if not todo:
        return

    started = time.perf_counter()
    tally: dict[str, int] = {}
    added: list[dict] = []
    for order, movie in enumerate(todo, start=1):
        try:
            result = collect_wiki.fill_own_article(movie)
        except Exception as exc:
            result = {"status": f"실패: {type(exc).__name__}"}
        status = result["status"]
        tally[status] = tally.get(status, 0) + 1
        detail = ""
        if status == "추가":
            detail = f" — {result['title']} ({result['length']:,}자 · 청크 {result['chunks']})"
            added.append({"movie": movie["title"], **result})
        elif "length" in result:
            detail = f" — {result['length']:,}자"
        print(f"[{order}/{len(todo)}] {movie['title']}: {status}{detail}")

    print("\n" + "=" * 56)
    for status, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {count}편")
    print(f"  소요: {(time.perf_counter() - started) / 60:.1f}분")
    print("=" * 56)

    path = config.BASE_DIR / "outputs" / "fill_own_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"완료_시각": datetime.now().isoformat(timespec="seconds"),
                   "대상": len(todo), "집계": tally, "추가": added},
                  file, ensure_ascii=False, indent=2)
    print(f"\n보고서: {path}")


def add_own(spec: str) -> None:
    """본편 문서를 사람이 직접 지정해 넣는다.

    한국어 위키 제목이 영화 제목과 전혀 달라 자동으로 이어지지 않는 예외를 위한 것이다.
    재수집은 키워드 추출이 비결정적이라 돌릴 때마다 결과가 달라지므로, 이런 몇 편은
    손으로 못박아 두는 편이 배포본을 안정적으로 만든다.
    """
    import collect_wiki

    movie_id, _, wiki_title = spec.partition("=")
    movie_id, wiki_title = movie_id.strip(), wiki_title.strip()
    if not wiki_title:
        raise SystemExit('형식: --add-own "<movie_id>=<위키 문서 제목>"')

    movie = movie_data.find_movie(movie_id)
    if not movie:
        raise SystemExit(f"영화를 찾을 수 없습니다: {movie_id}")

    result = collect_wiki.fill_own_article(movie, wiki_title=wiki_title)
    if result["status"] == "추가":
        print(f"{movie['title']}: {result['title']} 추가"
              f" ({result['length']:,}자 · 청크 {result['chunks']})")
    else:
        print(f"{movie['title']}: {result['status']}"
              + (f" — {result['length']:,}자" if "length" in result else ""))


def purge(movie_ids: list[str]) -> None:
    """재수집 전에 그 영화의 흔적을 모두 지운다.

    이력만 지우면 청크는 색인에 남는데, chunk_id가 같아 다시 넣어도 걸러진다.
    결국 문서는 그대로인데 기록만 사라져 "수집됐지만 기록에 없는" 상태가 된다.
    색인·원문·이력을 함께 지워야 재수집이 실제로 반영된다.
    """
    if not movie_ids:
        return

    import collect_wiki
    import vector_store

    store = vector_store.get_store(config.WIKI_COLLECTION)
    manifest = collect_wiki.load_manifest()
    chunks = 0
    for movie_id in movie_ids:
        found = store._collection.get(where={"movie_id": movie_id}, include=[])["ids"]
        if found:
            store._collection.delete(ids=found)
            chunks += len(found)
        for path in config.WIKI_DIR.glob(f"{movie_id}_*.json"):
            path.unlink()
        manifest.pop(movie_id, None)
    collect_wiki.save_manifest(manifest)
    print(f"[정리] {len(movie_ids)}편의 청크 {chunks:,}개와 원문·이력을 지웠습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="배포용 위키 문서 사전 구축")
    parser.add_argument("--reset", action="store_true", help="이전 수집분을 지우고 처음부터")
    parser.add_argument("--pages", type=int, default=5, help="영화당 최대 문서 수 (기본 5)")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N편만 (시험용)")
    parser.add_argument("--refill", type=int, default=0,
                        help="이미 구축된 것 중 문서가 N건 미만인 영화만 다시 수집한다")
    parser.add_argument("--topup", type=int, default=0,
                        help="문서가 N건 미만인 영화를 기존 수집분을 지우지 않고 보강한다 "
                             "(기준 완화 + 2차 키워드 요청)")
    parser.add_argument("--movies", default="",
                        help="쉼표로 구분한 movie_id만 지우고 다시 수집한다")
    parser.add_argument("--fill-own", action="store_true",
                        help="본편 문서가 빠진 영화에 본편 문서만 보강한다 (LLM 미사용)")
    parser.add_argument("--add-own", default="",
                        help="본편 문서를 직접 지정한다. 형식: <movie_id>=<위키 문서 제목>")
    parser.add_argument("--diagnose", action="store_true",
                        help="키워드가 왜 탈락했는지 상세 기록 (시험용)")
    args = parser.parse_args()

    config.MAX_PAGES = args.pages

    import collect_wiki

    diagnostics: list = []
    if args.diagnose:
        install_diagnostics(diagnostics)

    if args.reset:
        reset_storage()

    targets = load_targets()
    # 보강 모드는 대상을 manifest에서 따로 고르므로, 여기서 자르면 후보가 먼저 날아간다.
    # --limit은 보강 대상을 추린 뒤에 적용한다.
    if args.limit and not args.topup:
        targets = targets[: args.limit]

    if args.refill:
        # 수확이 적었던 영화만 골라 이력을 지우고 다시 시도한다.
        manifest = collect_wiki.load_manifest()
        thin = [
            movie_id for movie_id, record in manifest.items()
            if len(record.get("titles", [])) < args.refill
        ]
        purge(thin)
        targets = [m for m in targets if m["movie_id"] in thin]
        print(f"문서 {args.refill}건 미만인 {len(targets)}편을 다시 수집합니다.")

    topup_targets: list[dict] = []
    if args.topup:
        # 보강은 지우지 않는다. 완화된 기준으로 빈 자리만 메운다.
        config.RECOLLECT = True
        manifest = collect_wiki.load_manifest()
        thin = {
            movie_id: len(record.get("titles", []))
            for movie_id, record in manifest.items()
            if len(record.get("titles", [])) < args.topup
        }
        # 가장 부족한 영화부터 손댄다. 중간에 멈춰도 효과가 큰 쪽이 먼저 끝난다.
        topup_targets = sorted(
            (m for m in targets if m["movie_id"] in thin),
            key=lambda m: thin[m["movie_id"]],
        )
        if args.limit:
            topup_targets = topup_targets[: args.limit]
        print(
            f"문서 {args.topup}건 미만인 {len(thin)}편 중 {len(topup_targets)}편을 보강합니다."
            " (기존 수집분 유지)"
        )

    if args.movies:
        wanted = [value.strip() for value in args.movies.split(",") if value.strip()]
        purge(wanted)
        targets = [m for m in targets if m["movie_id"] in wanted]
        print(f"지정한 {len(targets)}편을 다시 수집합니다.")

    if args.add_own:
        add_own(args.add_own)
        return

    if args.fill_own:
        fill_own(targets)
        return

    if args.topup:
        # 이미 수집된 영화가 대상이므로 미수집 필터를 태우면 전부 걸러진다.
        todo = topup_targets
    else:
        todo = [m for m in targets if not collect_wiki.is_collected(m["movie_id"])]
    print(f"대상 {len(targets)}편 · 남은 작업 {len(todo)}편 · 영화당 최대 {args.pages}건")
    if not todo:
        print("이미 모두 수집되어 있습니다.")
        return

    before = key_info()
    print(f"시작 크레딧: {before.get('body', before)}")

    def on_interrupt(signum, frame):
        STOPPED["flag"] = True
        print("\n중단 요청됨. 현재 영화까지 마치고 정리합니다...")

    signal.signal(signal.SIGINT, on_interrupt)

    started = time.perf_counter()
    done = 0
    failures: list[dict] = []
    topup_log: list[dict] = []

    for order, entry in enumerate(todo, start=1):
        if STOPPED["flag"]:
            break
        movie = movie_data.find_movie(entry["movie_id"])
        if not movie:
            failures.append({"movie": entry["title"], "error": "movies.json에 없음"})
            continue

        before_count = len(collect_wiki.collected_titles(entry["movie_id"]))
        try:
            result = collect_wiki.collect_for_movie(movie)
            titles = result.get("titles", [])
            done += 1
            elapsed = time.perf_counter() - started
            remain = (elapsed / done) * (len(todo) - done)
            if args.topup:
                added = result.get("added_titles", [])
                topup_log.append({
                    "영화": entry["title"],
                    "이전": before_count,
                    "이후": len(titles),
                    "추가": added,
                    "2차": result.get("second_round", []),
                })
                mark = "" if len(titles) >= args.topup else "  ← 미달"
                print(
                    f"[{order}/{len(todo)}] {entry['title']} — "
                    f"{before_count}→{len(titles)}건 (+{len(added)}){mark}"
                    f" · 남은 예상 {remain / 60:.0f}분"
                )
                if added:
                    print(f"      추가: {' / '.join(added)}")
            else:
                print(
                    f"[{order}/{len(todo)}] {entry['title']} — 문서 {len(titles)}건"
                    f" · 남은 예상 {remain / 60:.0f}분"
                )
        except Exception as exc:
            failures.append({"movie": entry["title"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{order}/{len(todo)}] {entry['title']} — 실패: {exc}")

    seconds = time.perf_counter() - started
    if args.diagnose:
        print_diagnostics(diagnostics)
        path = config.BASE_DIR / "outputs" / "prebuild_diagnostics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(diagnostics, file, ensure_ascii=False, indent=2)
        print(f"\n진단 로그: {path}")
    after = key_info()

    manifest = collect_wiki.load_manifest()
    doc_counts = [len(v["titles"]) for v in manifest.values()]
    spent = None
    if used_credit(before) is not None and used_credit(after) is not None:
        spent = used_credit(after) - used_credit(before)

    report = {
        "완료_시각": datetime.now().isoformat(timespec="seconds"),
        "대상_영화": len(targets),
        "이번_실행_처리": done,
        "누적_수집_영화": len(manifest),
        "총_문서_수": sum(doc_counts),
        "영화당_평균_문서": round(sum(doc_counts) / len(doc_counts), 2) if doc_counts else 0,
        "소요_분": round(seconds / 60, 1),
        "크레딧_소모": spent,
        "영화당_최대_문서": args.pages,
        "실패": failures,
    }
    if args.topup:
        short = [row for row in topup_log if row["이후"] < args.topup]
        report["보강"] = {
            "목표_문서": args.topup,
            "보강_전_총문서": sum(row["이전"] for row in topup_log),
            "보강_후_총문서": sum(row["이후"] for row in topup_log),
            "목표_미달": len(short),
            "2차_라운드로_들어온_문서": sum(len(row["2차"]) for row in topup_log),
            "상세": topup_log,
        }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 56)
    for key, value in report.items():
        if key == "실패":
            continue
        if key == "보강":
            for name, detail in value.items():
                if name != "상세":
                    print(f"  보강_{name}: {detail}")
            continue
        print(f"  {key}: {value}")
    if failures:
        print(f"  실패 {len(failures)}건 — 보고서 참조")
    print("=" * 56)
    print(f"\n보고서: {REPORT_PATH}")
    if len(manifest) < len(targets):
        print("남은 영화가 있습니다. 같은 명령을 다시 실행하면 이어서 진행합니다.")


if __name__ == "__main__":
    main()
