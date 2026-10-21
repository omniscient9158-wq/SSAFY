"""한국어 위키백과 API 호출. (references/code/wiki.py 기반)"""

from __future__ import annotations

import html
import os
import random
import re
import time
from typing import Any

import requests

import config  # noqa: F401  (.env 로드)

API_URL = "https://ko.wikipedia.org/w/api.php"

# 위키미디어는 User-Agent로 요청 등급을 나눈다.
#   연락처 없음 → Unidentified        : 분당 10회
#   연락처 있음 → User-Agent only     : 분당 200회  (20배)
#
# 권장 형식 (Wikimedia APIs/Rate limits · Best practices)
#   CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0
# 괄호 안에는 반드시 이메일 또는 전체 URL이 들어가야 한다. 설명 문구는 연락처가 아니다.
CONTACT = os.getenv("WIKI_CONTACT", "").strip()
USER_AGENT = (
    f"SSAFY-MovieLab/0.1 ({CONTACT}) python-requests/{requests.__version__}"
    if CONTACT
    else f"SSAFY-MovieLab/0.1 python-requests/{requests.__version__}"
)

if not CONTACT:
    print(
        "[경고] WIKI_CONTACT가 비어 있어 위키백과 요청이 분당 10회로 제한됩니다.\n"
        "       src/.env에 연락처(이메일 또는 전체 URL)를 넣으면 분당 200회가 됩니다.\n"
        "       예) WIKI_CONTACT=https://github.com/<계정>/movie-lab; <계정>@example.com"
    )

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip",
}

session = requests.Session()
session.headers.update(HEADERS)


def request_wikipedia(
    params: dict[str, Any], max_retries: int = 4, url: str = API_URL
) -> dict[str, Any]:
    """429(요청 제한)·일시적인 네트워크 오류·서버 오류(5xx)를 재시도하면서 API를 호출한다.

    위키데이터도 같은 위키미디어 API라 User-Agent·재시도 규칙이 같다. url만 바꿔 재사용한다.
    끝내 실패하면 예외를 던진다. 호출한 쪽은 그 검색어를 수집 이력에 남기지 않는다.
    """
    wait = 2.0       # 429 대기 (Retry-After가 없을 때)
    net_wait = 1.0   # 네트워크·서버 오류 대기

    for attempt in range(1, max_retries + 1):
        last = attempt == max_retries
        try:
            response = session.get(url, params=params, timeout=20)
        except (requests.ConnectionError, requests.Timeout) as error:
            # 일시적인 연결 끊김·시간 초과. 바로 포기하면 그 검색어를 놓치므로 잠시 뒤 다시 시도한다.
            if last:
                raise
            sleep_time = net_wait + random.uniform(0, 1)
            print(f"[네트워크 오류] {sleep_time:.1f}초 후 재시도 ({attempt}/{max_retries}): {type(error).__name__}")
            time.sleep(sleep_time)
            net_wait *= 2
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            try:
                sleep_time = float(retry_after) if retry_after else wait
            except ValueError:
                sleep_time = wait
            sleep_time = max(sleep_time, 5.0) + random.uniform(0, 1)
            print(f"[429] {sleep_time:.1f}초 후 재시도 ({attempt}/{max_retries})")
            time.sleep(sleep_time)
            wait *= 2
            continue

        if response.status_code >= 500 and not last:
            # 서버 쪽 일시 장애(502·503 등)도 잠시 뒤 다시 시도한다.
            sleep_time = net_wait + random.uniform(0, 1)
            print(f"[{response.status_code}] {sleep_time:.1f}초 후 재시도 ({attempt}/{max_retries})")
            time.sleep(sleep_time)
            net_wait *= 2
            continue

        response.raise_for_status()

    raise RuntimeError("위키백과 요청이 반복해서 실패했습니다.")


def clean_snippet(snippet: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", snippet)).strip()


def search(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """검색 결과 목록을 반환한다."""
    data = request_wikipedia({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 0,
        "srlimit": limit,
        "format": "json",
        "formatversion": 2,
        "utf8": 1,
    })

    return [
        {
            "page_id": item["pageid"],
            "title": item["title"],
            "snippet": clean_snippet(item.get("snippet", "")),
        }
        for item in data.get("query", {}).get("search", [])
    ]


def fetch_page(title: str) -> dict[str, Any] | None:
    """문서 본문과 URL을 가져온다."""
    data = request_wikipedia({
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": 1,
        "exsectionformat": "plain",
        "inprop": "url",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    })

    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None

    page = pages[0]
    return {
        "page_id": page["pageid"],
        "title": page["title"],
        "url": page.get("fullurl", ""),
        "text": page.get("extract", ""),
    }
