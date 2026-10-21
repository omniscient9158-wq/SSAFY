"""Evaluate descriptor Ridge when one AqSolDB source group is held out.

Uses only data/train.csv. This is a source-transfer analysis, not the final
evaluation on the provided test split and not a causal analysis.

Run from the project root:
    python src/source_aware_evaluation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAIN = Path(__file__).resolve().parent.parent / "data" / "train.csv"
FEATURES = [
    "MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors",
    "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumValenceElectrons", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "RingCount", "TPSA", "LabuteASA", "BalabanJ", "BertzCT",
]
TARGET = "Y"
ALPHA = 1.0


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    denominator = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float(1 - (error ** 2).sum() / denominator) if denominator else float("nan")
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": r2,
    }


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    xtr = (x_train - mean) / scale
    xva = (x_valid - mean) / scale
    xtr1 = np.column_stack([np.ones(len(xtr)), xtr])
    xva1 = np.column_stack([np.ones(len(xva)), xva])
    penalty = np.eye(xtr1.shape[1]) * ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(xtr1.T @ xtr1 + penalty, xtr1.T @ y_train)
    return xva1 @ coefficients


def main() -> None:
    frame = pd.read_csv(TRAIN)
    required = set(FEATURES + [TARGET, "ID", "Group"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing numeric values found; define imputation before fitting.")

    frame = frame.copy()
    frame["source_code"] = frame["ID"].astype(str).str.split("-", n=1).str[0]
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    predictions = np.full(len(frame), np.nan)
    baseline_predictions = np.full(len(frame), np.nan)
    train_sizes: dict[str, int] = {}

    for source_code in sorted(frame["source_code"].unique()):
        valid_mask = frame["source_code"].to_numpy() == source_code
        train_mask = ~valid_mask
        if not train_mask.any():
            raise ValueError(f"Cannot hold out {source_code}: no training rows remain.")
        predictions[valid_mask] = fit_predict(x[train_mask], y[train_mask], x[valid_mask])
        baseline_predictions[valid_mask] = y[train_mask].mean()
        train_sizes[source_code] = int(train_mask.sum())

    if np.isnan(predictions).any() or np.isnan(baseline_predictions).any():
        raise RuntimeError("Some training rows did not receive out-of-source predictions.")

    print("Evaluation: leave-one-source-code-out on train.csv only")
    print("Interpretation: source-transfer check; results do not prove that source differences cause errors.")
    print(f"rows={len(frame)} source_codes={','.join(sorted(train_sizes))} alpha={ALPHA}")
    print("pooled_model", metrics(y, predictions))
    print("pooled_fold-train-mean_baseline", metrics(y, baseline_predictions))

    print("\nby_held_out_source")
    for source_code in sorted(train_sizes):
        mask = frame["source_code"].to_numpy() == source_code
        print(source_code, f"n={int(mask.sum())}", "train_rows=" + str(train_sizes[source_code]),
              metrics(y[mask], predictions[mask]))

    print("\nby_Group (pooled out-of-source predictions)")
    quality_groups = frame["Group"].astype(str).to_numpy()
    for group in sorted(set(quality_groups)):
        mask = quality_groups == group
        print(group, f"n={int(mask.sum())}", metrics(y[mask], predictions[mask]))


if __name__ == "__main__":
    main()
