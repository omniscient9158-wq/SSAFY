"""Compare an AqSolDB record with retrieved, reviewed solubility excerpts.

Preserve units and conditions rather than silently converting incompatible data.
This comparison does not explain the cause of a model prediction.
"""
from __future__ import annotations

import re
from typing import Any

from retrieve_local_evidence import retrieve


def compare_measurements(compound_id: str) -> dict[str, Any]:
    result = retrieve(compound_id, "water solubility 수용해도 값 단위 온도", limit=10)
    metadata = result["compound_metadata"]
    records = [{
        "자료": "AqSolDB 실험 레코드",
        "기록된 값": str(metadata["experimental_logS"]),
        "척도·단위": "LogS (log10 mol/L)",
        "온도": "현재 레코드에 미기재",
        "근거": "근거 1",
    }]
    sources = []
    for item in result["external_evidence"]:
        # Only copy the explicit water-solubility measurement, never infer one
        # from appearance, general water polarity or zwitterion descriptions.
        match = re.search(
            r"Water Solubility:\s*(?P<value>(?:greater than or equal to\s+|less than\s+)?\d+(?:\.\d+)?(?:\s+to\s+\d+(?:\.\d+)?)?)"
            r"\s+(?P<unit>mg/mL)\s+at\s+(?P<temperature>\d+(?:\.\d+)?°F)",
            item["excerpt"],
        )
        if not match:
            continue
        records.append({
            "자료": item["source_provider"],
            "기록된 값": match["value"],
            "척도·단위": match["unit"],
            "온도": match["temperature"],
            "근거": item["citation_id"],
        })
        sources.append(item)
    return {
        "compound_id": compound_id,
        "records": records,
        "sources": sources,
        "interpretation": (
            "검색 문헌의 수용해도는 mg/mL와 명시된 온도의 기록이고, AqSolDB는 별도 실험 LogS입니다. "
            "현재 AqSolDB 레코드에 온도·pH 조건이 없어 동일 조건 측정으로 비교하거나 차이를 모델 오차로 계산할 수 없습니다. "
            "문헌 값은 모델 예측의 원인이나 정확성을 입증하지 않습니다."
            if sources else
            "검색된 승인 발췌에서 이 비교가 지원하는 형식(mg/mL와 °F)의 수용해도 기록을 찾지 못했습니다. "
            "이는 해당 화합물의 문헌 수용해도가 없다는 뜻은 아닙니다."
        ),
    }
