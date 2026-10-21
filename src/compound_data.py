"""AqSolDB records and one-record-per-document metadata text for F2001."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SOURCE_MAP_PATH = DATA_DIR / "AqSolDB_원자료_출처매핑.csv"
DATASET_DOI = "https://doi.org/10.7910/DVN/OVHAW8"
DATASET_LICENSE = "CC0 1.0 (Harvard Dataverse AqSolDB v1)"

# Continuous/count descriptors. Do not include target Y or identifiers as model features.
DESCRIPTOR_FIELDS = (
    "MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors",
    "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumValenceElectrons", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "RingCount", "TPSA", "LabuteASA", "BalabanJ", "BertzCT",
)


def _read_csv(path: Path, split: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"AqSolDB {split} CSV was not found: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["_split"] = split
    return rows


@lru_cache(maxsize=1)
def load_compounds() -> tuple[dict[str, str], ...]:
    """Load both local splits. Each unique AqSolDB ID is one metadata record."""
    rows = _read_csv(TRAIN_PATH, "train") + _read_csv(TEST_PATH, "test")
    ids = [row.get("ID", "").strip() for row in rows]
    if any(not item_id for item_id in ids):
        raise ValueError("AqSolDB contains a row without ID.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate AqSolDB IDs found across train/test.")
    return tuple(rows)


@lru_cache(maxsize=1)
def load_source_map() -> dict[str, dict[str, str]]:
    if not SOURCE_MAP_PATH.is_file():
        return {}
    with SOURCE_MAP_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            row["AqSolDB_ID_prefix"]: row
            for row in csv.DictReader(stream)
            if row.get("AqSolDB_ID_prefix")
        }


def _source_code(row: dict[str, str]) -> str:
    return row.get("ID", "").split("-", 1)[0]


def source_info(row: dict[str, str]) -> dict[str, str]:
    return load_source_map().get(_source_code(row), {})


def solubility_description(log_s: float) -> str:
    """Describe the paper's LogS bands in plain language; this is not a classifier."""
    if log_s >= 0:
        return "높은 수용해도 구간, 물에 잘 녹는 편"
    if log_s >= -2:
        return "용해성 구간, 물에 녹는 편"
    if log_s >= -4:
        return "약한 용해성 구간, 물에 잘 녹지 않는 편"
    return "불용성 구간, 물에 거의 녹지 않는 편"


def searchable_aliases(row: dict[str, str]) -> tuple[str, ...]:
    """Return a small, manually verified alias list for user-facing examples."""
    aliases = {
        "E-1274": ("aspirin", "아스피린", "acetylsalicylic acid"),
        "C-533": ("카페인",),
    }
    return aliases.get(row.get("ID", ""), ())


def to_text(row: dict[str, str]) -> str:
    """Create one searchable text block for a compound and its dataset record."""
    source = source_info(row)
    lines = [
        f"화합물 ID: {row.get('ID', '')}",
        f"이름: {row.get('Name', '')}",
        f"별칭: {', '.join(searchable_aliases(row)) or '없음'}",
        f"SMILES: {row.get('SMILES', '')}",
        f"InChIKey: {row.get('InChIKey', '')}",
        f"실험 수용해도 LogS (Y): {row.get('Y', '')}",
        f"LogS 해석: {solubility_description(float(row['Y']))}",
        f"측정 표준편차 (SD): {row.get('SD', '')}",
        f"측정 횟수 (Ocurrences, 원본 열 이름): {row.get('Ocurrences', '')}",
        f"데이터 품질 그룹: {row.get('Group', '')}",
        f"AqSolDB 분할: {row.get('_split', '')}",
        f"AqSolDB 원자료 코드: {_source_code(row)}",
    ]
    if source.get("Original_source_dataset"):
        lines.append(f"원자료 묶음(논문 출처 연결): {source['Original_source_dataset']}")
        lines.append(f"원논문 참고문헌 번호: {source.get('Reference_number_in_Sorkun_2019', '')}")
    lines.append(f"데이터셋 DOI: {DATASET_DOI}")
    lines.append(f"배포 데이터 라이선스 표기: {DATASET_LICENSE}")
    # Keep embedding text focused on user-searchable chemistry terms; all
    # descriptors remain available as scalar Chroma metadata for filtering.
    embedding_fields = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors")
    descriptor_text = [f"{key}: {row.get(key, '')}" for key in embedding_fields if row.get(key, "") != ""]
    if descriptor_text:
        lines.append("주요 분자 특성:")
        lines.extend(descriptor_text)
    return "\n".join(lines)


def to_metadata(row: dict[str, str]) -> dict[str, Any]:
    """Return scalar metadata values compatible with vector-store metadata stores."""
    source = source_info(row)
    return {
        "chunk_id": row.get("ID", ""),
        "source_type": "compound_metadata",
        "compound_id": row.get("ID", ""),
        "name": row.get("Name", ""),
        "inchikey": row.get("InChIKey", ""),
        "source_code": _source_code(row),
        "original_source_dataset": source.get("Original_source_dataset", "unmapped"),
        "paper_reference_number": source.get("Reference_number_in_Sorkun_2019", ""),
        "split": row.get("_split", ""),
        "dataset_doi": DATASET_DOI,
        "dataset_license": DATASET_LICENSE,
    }


def main() -> None:
    records = load_compounds()
    print(f"화합물 메타데이터 로드: {len(records):,}건")
    print(f"고유 ID: {len({row['ID'] for row in records}):,}건")
    print("\n[색인 텍스트 예시 - 첫 번째 레코드]")
    print(to_text(records[0]))
    print("\n[Vector DB metadata 예시]")
    print(to_metadata(records[0]))


if __name__ == "__main__":
    main()
