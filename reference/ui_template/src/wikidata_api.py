"""위키데이터 API 호출 (F2016 — 두 번째 문서 출처).

위키백과가 서술형 문서라면 위키데이터는 "속성: 값"으로 된 구조화 사실 데이터베이스다.
개봉일·수상·흥행 수입·원작·배경 장소처럼 위키백과 본문에 흩어져 있거나 빠진 사실을
정확한 값으로 준다. 라이선스는 CC0(퍼블릭 도메인)이다.

영화는 TMDB ID 속성(P4947)으로 찾는다. 우리 영화 ID가 곧 TMDB ID라 제목 검색처럼
동명 작품을 잘못 고를 일이 없다.
"""

from __future__ import annotations

from typing import Any

from wiki_api import request_wikipedia

API_URL = "https://www.wikidata.org/w/api.php"
ENTITY_URL = "https://www.wikidata.org/wiki/{qid}"

# 영화 사실 확인에 쓸 속성만 고른다. (속성 ID, 한국어 이름)
# 순서가 곧 문서의 줄 순서다. 식별자·외부 링크 속성(수백 개)은 근거가 되지 않아 뺀다.
PROPERTIES = [
    ("P577", "공개일"),
    ("P495", "제작 국가"),
    ("P364", "원어"),
    ("P136", "장르"),
    ("P57", "감독"),
    ("P58", "각본"),
    ("P162", "제작자"),
    ("P161", "출연"),
    ("P86", "음악"),
    ("P344", "촬영"),
    ("P272", "제작사"),
    ("P750", "배급사"),
    ("P2047", "상영 시간"),
    ("P144", "원작"),
    ("P921", "주요 주제"),
    ("P840", "배경 장소"),
    ("P2408", "배경 시대"),
    ("P915", "촬영지"),
    ("P179", "시리즈"),
    ("P155", "이전 작품"),
    ("P156", "다음 작품"),
    ("P166", "수상"),
    ("P1411", "후보 지명"),
    ("P2130", "제작비"),
    ("P2142", "흥행 수입"),
]

QUALIFIER_ROLE = "P453"      # 배역 (출연 속성의 한정어)
QUALIFIER_TIME = "P585"      # 시점 (수상·흥행의 연도)


def _call(params: dict[str, Any]) -> dict[str, Any]:
    return request_wikipedia({**params, "format": "json"}, url=API_URL)


def find_qid(tmdb_id: str) -> str:
    """TMDB 영화 ID로 위키데이터 항목 ID(Q…)를 찾는다. 없으면 빈 문자열."""
    data = _call({
        "action": "query",
        "list": "search",
        "srsearch": f"haswbstatement:P4947={tmdb_id}",
        "srlimit": 2,
    })
    hits = data.get("query", {}).get("search", [])
    # 같은 TMDB ID가 두 항목에 걸려 있으면 어느 쪽이 맞는지 알 수 없다. 쓰지 않는다.
    return hits[0]["title"] if len(hits) == 1 else ""


def get_entity(qid: str) -> dict[str, Any]:
    data = _call({
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|claims",
        "languages": "ko|en",
    })
    return data.get("entities", {}).get(qid, {})


def get_labels(ids: list[str]) -> dict[str, str]:
    """항목 ID 목록의 이름. 한국어가 없으면 영어, 그것도 없으면 ID를 그대로 쓴다."""
    labels: dict[str, str] = {}
    unique = list(dict.fromkeys(ids))
    for start in range(0, len(unique), 50):   # API 한 번에 50개까지
        batch = unique[start:start + 50]
        data = _call({
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "ko|en",
        })
        for qid, entity in data.get("entities", {}).items():
            names = entity.get("labels", {})
            name = names.get("ko") or names.get("en")
            labels[qid] = name["value"] if name else qid
    return labels


def label_of(entity: dict[str, Any]) -> str:
    names = entity.get("labels", {})
    name = names.get("ko") or names.get("en")
    return name["value"] if name else entity.get("id", "")


# ── 값 해석 ───────────────────────────────────────────────────────────────
def _statements(entity: dict[str, Any], prop: str) -> list[dict[str, Any]]:
    """폐기(deprecated) 등급을 뺀 진술 목록."""
    return [
        s for s in entity.get("claims", {}).get(prop, [])
        if s.get("rank") != "deprecated" and s.get("mainsnak", {}).get("snaktype") == "value"
    ]


def _entity_ids(value: dict[str, Any]) -> list[str]:
    """값 하나에서 이름을 알아야 하는 항목 ID들 (값 자체 + 수량의 단위)."""
    datavalue = value.get("datavalue", {})
    kind, raw = datavalue.get("type"), datavalue.get("value")
    if kind == "wikibase-entityid":
        return [raw["id"]]
    if kind == "quantity" and raw.get("unit", "1") != "1":
        return [raw["unit"].rsplit("/", 1)[-1]]
    return []


def referenced_ids(entity: dict[str, Any]) -> list[str]:
    """문서를 만들 때 이름이 필요한 항목 ID 전체."""
    ids: list[str] = []
    for prop, _ in PROPERTIES:
        for statement in _statements(entity, prop):
            ids += _entity_ids(statement["mainsnak"])
            for qualifier in statement.get("qualifiers", {}).get(QUALIFIER_ROLE, []):
                ids += _entity_ids(qualifier)
    return ids


def format_value(snak: dict[str, Any], labels: dict[str, str]) -> str:
    datavalue = snak.get("datavalue", {})
    kind, raw = datavalue.get("type"), datavalue.get("value")
    if kind == "wikibase-entityid":
        return labels.get(raw["id"], raw["id"])
    if kind == "time":
        # "+2014-07-30T00:00:00Z" · 정밀도 11=일, 10=월, 9=연
        date = raw["time"].lstrip("+").split("T")[0]
        return {11: date, 10: date[:7]}.get(raw.get("precision"), date[:4])
    if kind == "quantity":
        amount = float(raw["amount"])
        text = f"{amount:,.0f}" if amount == int(amount) else f"{amount:,}"
        unit = raw.get("unit", "1")
        return f"{text} {labels.get(unit.rsplit('/', 1)[-1], '')}".strip() if unit != "1" else text
    if kind == "monolingualtext":
        return raw["text"]
    if kind == "string":
        return raw
    return ""


def _year(statement: dict[str, Any]) -> str:
    for qualifier in statement.get("qualifiers", {}).get(QUALIFIER_TIME, []):
        time = qualifier.get("datavalue", {}).get("value", {}).get("time", "")
        if time:
            return time.lstrip("+")[:4]
    return ""


def fact_lines(entity: dict[str, Any], labels: dict[str, str], max_values: int) -> list[str]:
    """속성마다 "이름: 값, 값, …" 한 줄. 값이 없는 속성은 줄을 만들지 않는다."""
    lines: list[str] = []
    for prop, name in PROPERTIES:
        values: list[str] = []
        for statement in _statements(entity, prop):
            text = format_value(statement["mainsnak"], labels)
            if not text:
                continue
            roles = [
                format_value(q, labels)
                for q in statement.get("qualifiers", {}).get(QUALIFIER_ROLE, [])
            ]
            if roles:
                text += f" ({', '.join(r for r in roles if r)} 역)"
            year = _year(statement)
            if year and prop in ("P166", "P1411", "P2142"):
                text += f" ({year})"
            if text not in values:
                values.append(text)
        if values:
            more = len(values) - max_values
            shown = ", ".join(values[:max_values]) + (f" 외 {more}건" if more > 0 else "")
            lines.append(f"{name}: {shown}")
    return lines
