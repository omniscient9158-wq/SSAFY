"""Build a local review copy with runtime data and evidence, without secrets."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "AI 코어프로젝트 2_AqSolDB 화합물 근거 탐색_서울_08반_김지은_최종.zip")
    args = parser.parse_args()
    files = [ROOT / name for name in (
        "app.py", "README.md", "05_회고록.md", "requirements.txt",
        ".env.example", ".streamlit/config.toml",
    )]
    for folder in ("src", "data", "docs", "screenshots"):
        files.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and not path.is_symlink()
                     and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
                     and not (folder == "docs" and (
                         path.name in {"00_프로젝트_개요_및_초록.md", "05_회고록.md", "07_UI_수정안.md"}
                         or "tools" in path.relative_to(ROOT / "docs").parts)))
    files.append(ROOT / "체크리스트.pdf")
    secret = dotenv_values(ROOT / ".env").get("GMS_KEY")
    for path in files:
        if secret and secret.encode() in path.read_bytes():
            raise ValueError(f"Credential found in a selected file; refusing to package: {path.name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Local submission copy: {args.output.name}; {len(set(files))} files")
    print("Excluded: real .env, virtualenv, indexes, research candidates, lecture reference files and caches")


if __name__ == "__main__":
    main()
