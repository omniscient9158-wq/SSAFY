"""RDKit SMILES parsing, structure rendering, and train-only LogS prediction."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

import compound_data
from source_aware_evaluation import fit_predict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw
except ImportError:  # Keep the rest of the app usable before optional setup is complete.
    Chem = Descriptors = Draw = None


SOLUBILITY_BANDS = (
    (0.0, "높은 용해도"),
    (-2.0, "용해성"),
    (-4.0, "약한 용해성"),
    (float("-inf"), "불용성 구간"),
)


def require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("분자 구조 해석 기능은 RDKit이 필요합니다. 프로젝트 requirements.txt를 설치해 주세요.")


def molecule_from_smiles(smiles: str) -> Any:
    """Parse SMILES or raise a user-facing ValueError for malformed input."""
    require_rdkit()
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        raise ValueError("SMILES를 분자 구조로 해석하지 못했습니다. 원자·결합 표기를 확인해 주세요.")
    return molecule


def molecule_image(smiles: str, size: tuple[int, int] = (520, 360)) -> Image.Image:
    molecule = molecule_from_smiles(smiles)
    return Draw.MolToImage(molecule, size=size, kekulize=True)


def molecular_formula(smiles: str) -> str:
    molecule = molecule_from_smiles(smiles)
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    return CalcMolFormula(molecule)


def predict_smiles(smiles: str) -> dict[str, Any]:
    """Fit the fixed Ridge baseline on train.csv and predict a new molecule."""
    molecule = molecule_from_smiles(smiles)
    fields = compound_data.DESCRIPTOR_FIELDS
    descriptors = np.asarray([[float(getattr(Descriptors, field)(molecule)) for field in fields]])
    train_rows = [row for row in compound_data.load_compounds() if row["_split"] == "train"]
    x_train = np.asarray([[float(row[field]) for field in fields] for row in train_rows])
    y_train = np.asarray([float(row["Y"]) for row in train_rows])
    if not np.isfinite(descriptors).all() or not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise ValueError("분자 descriptor 또는 학습 데이터에 숫자가 아닌 값이 있습니다.")
    prediction = float(fit_predict(x_train, y_train, descriptors)[0])
    band = next(label for threshold, label in SOLUBILITY_BANDS if prediction >= threshold)
    return {
        "predicted_logS": prediction,
        "solubility_band": band,
        "fit_rows": len(train_rows),
        "method": "Ridge alpha=1; 17 2D descriptors; fit on train.csv",
        "test_split_used": False,
    }


def solubility_band(logS: float) -> tuple[str, str]:
    """Return the paper's LogS range label and a plain-language summary."""
    if logS >= 0:
        return "높은 용해도", "물에 잘 녹는 편"
    if logS >= -2:
        return "용해성", "물에 녹는 편"
    if logS >= -4:
        return "약한 용해성", "물에 잘 녹지 않는 편"
    return "불용성 구간", "물에 거의 녹지 않는 편"
