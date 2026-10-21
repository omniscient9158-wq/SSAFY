"""AqSolDB descriptor-only Ridge baseline; uses provided train split for validation."""
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
SEED = 42
ALPHA = 1.0


def metrics(y_true, y_pred):
    err = y_true - y_pred
    denom = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - (err ** 2).sum() / denom),
    }


def main():
    df = pd.read_csv(TRAIN)
    missing = sorted(set(FEATURES + [TARGET, "Group"]) - set(df.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing numeric values found; define imputation before fitting.")

    # Stratify validation indices by Group to preserve the provided quality-group mix.
    rng = np.random.default_rng(SEED)
    valid_parts = []
    groups = df["Group"].astype(str)
    for group in sorted(groups.unique()):
        idx = np.flatnonzero(groups.to_numpy() == group)
        rng.shuffle(idx)
        n_valid = max(1, round(len(idx) * 0.2))
        valid_parts.extend(idx[:n_valid].tolist())
    valid_idx = np.array(sorted(valid_parts), dtype=int)
    mask = np.ones(len(df), dtype=bool)
    mask[valid_idx] = False
    train_idx = np.flatnonzero(mask)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    mean = X[train_idx].mean(axis=0)
    scale = X[train_idx].std(axis=0)
    scale[scale == 0] = 1.0
    Xtr = (X[train_idx] - mean) / scale
    Xva = (X[valid_idx] - mean) / scale
    ytr, yva = y[train_idx], y[valid_idx]

    # Include intercept but do not penalize it.
    Xtr1 = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xva1 = np.column_stack([np.ones(len(Xva)), Xva])
    penalty = np.eye(Xtr1.shape[1]) * ALPHA
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(Xtr1.T @ Xtr1 + penalty, Xtr1.T @ ytr)
    pred = Xva1 @ coef

    result = metrics(yva, pred)
    baseline = metrics(yva, np.full_like(yva, ytr.mean()))
    print(f"train_rows={len(train_idx)} valid_rows={len(valid_idx)} seed={SEED} alpha={ALPHA}")
    print("ridge", result)
    print("train-mean baseline", baseline)
    print("by_group")
    groups_valid = groups.iloc[valid_idx].to_numpy()
    for group in sorted(set(groups_valid)):
        take = groups_valid == group
        print(group, int(take.sum()), metrics(yva[take], pred[take]))


if __name__ == "__main__":
    main()

