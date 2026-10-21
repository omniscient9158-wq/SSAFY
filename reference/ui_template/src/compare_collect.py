"""수집 결과 비교 — 개선 전후로 어떤 문서가 빠지고 들어왔는지 본다.

    python src/compare_collect.py load_sim load_sim_v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import movies as movie_data


def load(folder: str) -> dict:
    path = config.BASE_DIR / "outputs" / folder / "manifest.json"
    if not path.exists():
        raise SystemExit(f"manifest가 없습니다: {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    before_name = sys.argv[1] if len(sys.argv) > 1 else "load_sim"
    after_name = sys.argv[2] if len(sys.argv) > 2 else "load_sim_v2"
    before, after = load(before_name), load(after_name)

    shared = [mid for mid in after if mid in before]
    if not shared:
        raise SystemExit("두 결과에 공통으로 들어 있는 영화가 없습니다.")

    total_removed = total_added = total_kept = 0
    for movie_id in shared:
        movie = movie_data.find_movie(movie_id)
        name = movie_data.label(movie) if movie else movie_id
        old = list(dict.fromkeys(before[movie_id]["titles"]))
        new = list(dict.fromkeys(after[movie_id]["titles"]))
        removed = [t for t in old if t not in new]
        added = [t for t in new if t not in old]
        kept = [t for t in new if t in old]

        total_removed += len(removed)
        total_added += len(added)
        total_kept += len(kept)

        print("=" * 68)
        print(f"[{name}]  {len(old)}건 → {len(new)}건")
        if kept:
            print(f"  유지 {len(kept)}: {', '.join(kept)}")
        if removed:
            print(f"  빠짐 {len(removed)}: {', '.join(removed)}")
        if added:
            print(f"  추가 {len(added)}: {', '.join(added)}")

    print("=" * 68)
    print(f"영화 {len(shared)}편 · 유지 {total_kept} · 빠짐 {total_removed} · 추가 {total_added}")
    print(f"  {before_name} 총 문서 {sum(len(before[m]['titles']) for m in shared)}")
    print(f"  {after_name} 총 문서 {sum(len(after[m]['titles']) for m in shared)}")
    print("\n'빠짐'에 무관한 영화 제목이 많고 '추가'에 소재·배경 문서가 많으면 개선된 것이다.")


if __name__ == "__main__":
    main()
