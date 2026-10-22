"""Parse a small, explicit Korean numeric-filter grammar for AqSolDB train rows."""
from __future__ import annotations

import re
import math
from typing import Any

import compound_data

FIELD_ALIASES = {
    "MolWt": ("분자량", "molecular weight", "molwt", "mw"),
    "MolLogP": ("logp", "로그p", "분배계수"),
    "TPSA": ("tpsa", "극성표면적", "극성 표면적"),
    "Y": ("logs", "log s", "로그s", "수용해도", "용해도"),
    "NumHDonors": ("수소결합 공여체", "hbd", "수소 공여체"),
    "NumHAcceptors": ("수소결합 수용체", "hba", "수소 수용체"),
    "NumRotatableBonds": ("회전가능 결합", "회전 결합", "rotatable bonds"),
    "HeavyAtomCount": ("heavy atom", "중원자 수"),
    "NumAromaticRings": ("방향족 고리", "방향족 링"),
    "RingCount": ("고리 수", "링 수"),
}
NUMERIC = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
OPERATOR_PATTERN = r"(?:이상|이하|초과|미만|보다\s*(?:큰|높은|작은|낮은)|>=|<=|≥|≤|>|<)"


def _operator(text: str) -> str:
    value = re.sub(r"\s+", "", text)
    if value in ("이상", ">=", "≥"):
        return "gte"
    if value in ("이하", "<=", "≤"):
        return "lte"
    if value == "초과" or "큰" in value or "높은" in value or value == ">":
        return "gt"
    if value == "미만" or "작은" in value or "낮은" in value or value == "<":
        return "lt"
    raise ValueError(f"지원하지 않는 비교 연산자: {text}")


def parse_numeric_filters(query: str) -> list[dict[str, Any]]:
    """Parse clauses such as '분자량 300 이하, LogP 2 미만'."""
    if not query.strip():
        raise ValueError("조건을 입력하세요.")
    if re.search(r"또는|\bor\b", query, flags=re.I):
        raise ValueError("'또는' 조건은 지원하지 않습니다. 모든 조건을 함께 만족하도록 쉼표로 구분해 주세요.")
    clauses = [
        part.strip()
        for part in re.split(r"[,，;；\n]|그리고|이고|이며|이면서|및|and|&", query, flags=re.I)
        if part.strip()
    ]
    constraints: list[dict[str, Any]] = []
    alias_to_field = {
        alias.casefold(): field
        for field, aliases in FIELD_ALIASES.items()
        for alias in aliases
    }
    for clause in clauses:
        normalized = clause.casefold()
        matches = [(normalized.find(alias), alias, field)
                   for alias, field in alias_to_field.items() if normalized.find(alias) >= 0]
        if not matches:
            raise ValueError(f"지원하지 않는 조건입니다: {clause}")
        if len({field for _, _, field in matches}) > 1:
            raise ValueError("서로 다른 속성의 조건은 쉼표로 구분해 주세요. 예: 분자량 300 이하, LogP 2 미만")
        position, alias, field = min(matches, key=lambda item: (item[0], -len(item[1])))
        tail = clause[position + len(alias):]

        range_match = re.search(rf"({NUMERIC})\s*(?:~|〜|부터\s*에서|에서|부터)\s*({NUMERIC})", tail)
        if range_match:
            low, high = map(float, range_match.groups())
            if low > high:
                raise ValueError(f"범위의 시작값이 끝값보다 큽니다: {clause}")
            constraints.append({"field": field, "operator": "between", "low": low, "high": high})
            continue

        found = False
        for match in re.finditer(rf"({NUMERIC})\s*({OPERATOR_PATTERN})", tail, flags=re.I):
            constraints.append({"field": field, "operator": _operator(match.group(2)), "value": float(match.group(1))})
            found = True
        for match in re.finditer(rf"({OPERATOR_PATTERN})\s*({NUMERIC})", tail, flags=re.I):
            constraints.append({"field": field, "operator": _operator(match.group(1)), "value": float(match.group(2))})
            found = True
        if not found:
            raise ValueError(f"'{clause}'에서 숫자와 비교 조건을 찾지 못했습니다.")

    if not constraints:
        raise ValueError("지원하는 수치 조건을 찾지 못했습니다. 예: 분자량 300 이하, LogP 2 미만")
    allowed_fields = set(compound_data.DESCRIPTOR_FIELDS) | {"Y"}
    if any(rule["field"] not in allowed_fields for rule in constraints):
        raise ValueError("허용되지 않은 필드가 포함되어 있습니다.")
    if any(not math.isfinite(rule[key]) for rule in constraints for key in ("value", "low", "high") if key in rule):
        raise ValueError("조건에는 유한한 수를 입력해 주세요.")
    return constraints


def search_train_compounds(query: str, limit: int = 50) -> dict[str, Any]:
    """Apply parsed filters deterministically to train.csv only, then return candidates."""
    if limit < 1 or limit > 200:
        raise ValueError("결과 제한은 1~200 사이여야 합니다.")
    constraints = parse_numeric_filters(query)
    candidates = [row for row in compound_data.load_compounds() if row["_split"] == "train"]
    for rule in constraints:
        field = rule["field"]
        if any(row.get(field, "") == "" or not math.isfinite(float(row[field])) for row in candidates):
            raise ValueError(f"필드 {field}에 결측값이 있습니다. 자동 보정 없이 조건 검색을 중단합니다.")
        if rule["operator"] == "between":
            low, high = rule["low"], rule["high"]
            candidates = [r for r in candidates if low <= float(r[field]) <= high]
        else:
            value = rule["value"]
            operators = {
                "gt": lambda x: x > value,
                "gte": lambda x: x >= value,
                "lt": lambda x: x < value,
                "lte": lambda x: x <= value,
            }
            candidates = [r for r in candidates if operators[rule["operator"]](float(r[field]))]
    candidates.sort(key=lambda row: row["ID"])
    fields = ("ID", "Name", "Y", "MolWt", "MolLogP", "TPSA", "InChIKey")
    return {
        "filters": constraints,
        "train_only": True,
        "total_matches": len(candidates),
        "items": [{key: row.get(key, "") for key in fields} for row in candidates[:limit]],
    }
