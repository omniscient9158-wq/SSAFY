"""Train-only source-group holdout prediction and descriptor-neighbor lookup."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

import compound_data
from source_aware_evaluation import fit_predict


@lru_cache(maxsize=1)
def _training_rows() -> tuple[dict[str, str], ...]:
    return tuple(row for row in compound_data.load_compounds() if row["_split"] == "train")


@lru_cache(maxsize=9)
def _source_holdout_predictions(source_code: str) -> tuple[dict[str, float], int, int]:
    rows = _training_rows()
    held_out = [row for row in rows if compound_data._source_code(row) == source_code]
    fit_rows = [row for row in rows if compound_data._source_code(row) != source_code]
    if not held_out or not fit_rows:
        raise ValueError(f"출처 그룹 {source_code}의 train-only holdout을 만들 수 없습니다.")

    fields = compound_data.DESCRIPTOR_FIELDS
    x_train = np.asarray([[float(row[field]) for field in fields] for row in fit_rows], dtype=float)
    y_train = np.asarray([float(row["Y"]) for row in fit_rows], dtype=float)
    x_valid = np.asarray([[float(row[field]) for field in fields] for row in held_out], dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all() or not np.isfinite(y_train).all():
        raise ValueError("모델 입력 descriptor 또는 LogS에 결측/비유한 값이 있습니다.")
    predictions = fit_predict(x_train, y_train, x_valid)
    return (
        {row["ID"]: float(pred) for row, pred in zip(held_out, predictions)},
        len(fit_rows),
        len(held_out),
    )


def predict_for_compound(compound_id: str) -> dict[str, Any]:
    row = next((r for r in _training_rows() if r["ID"] == compound_id), None)
    if row is None:
        raise ValueError("현재 예측은 train.csv의 출처 그룹 holdout 예시만 지원합니다.")
    source_code = compound_data._source_code(row)
    predictions, train_count, validation_count = _source_holdout_predictions(source_code)
    if compound_id not in predictions:
        raise ValueError(f"출처 그룹 {source_code}의 holdout 예측이 없습니다.")
    actual = float(row["Y"])
    predicted = predictions[compound_id]
    return {
        "compound_id": compound_id,
        "measured_logS": actual,
        "predicted_logS": predicted,
        "absolute_error": abs(predicted - actual),
        "held_out_source_code": source_code,
        "fit_rows": train_count,
        "heldout_rows": validation_count,
        "method": "Ridge alpha=1; 17 descriptors; leave-one-source-code-out on train.csv",
        "test_split_used": False,
    }


def similar_descriptor_compounds(compound_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find nearest train rows by standardized 2D descriptors, not structure fingerprints."""
    if limit < 1 or limit > 20:
        raise ValueError("유사 후보 수는 1~20 사이여야 합니다.")
    rows = _training_rows()
    target = next((r for r in rows if r["ID"] == compound_id), None)
    if target is None:
        raise ValueError("현재 유사 후보 탐색은 train.csv의 화합물만 지원합니다.")

    fields = compound_data.DESCRIPTOR_FIELDS
    matrix = np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("descriptor에 결측/비유한 값이 있어 유사 후보를 계산할 수 없습니다.")
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (matrix - matrix.mean(axis=0)) / scale
    target_idx = next(i for i, row in enumerate(rows) if row["ID"] == compound_id)
    target_key = target["InChIKey"]

    ranked: list[tuple[float, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if row["ID"] == compound_id or row["InChIKey"] == target_key:
            continue
        distance = float(np.linalg.norm(standardized[index] - standardized[target_idx]))
        ranked.append((distance, {
            "ID": row["ID"],
            "Name": row["Name"],
            "InChIKey": row["InChIKey"],
            "experimental_LogS": float(row["Y"]),
            "descriptor_distance": distance,
            "source_code": compound_data._source_code(row),
            "dataset_doi": compound_data.DATASET_DOI,
        }))
    ranked.sort(key=lambda pair: (pair[0], pair[1]["ID"]))
    return [item for _, item in ranked[:limit]]
