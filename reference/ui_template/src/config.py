"""경로와 실행 설정을 한곳에 모은다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

load_dotenv(SRC_DIR / ".env")

# 데이터 경로
MOVIES_PATH = BASE_DIR / "data" / "movies.json"
WIKI_DIR = BASE_DIR / "resources" / "rag" / "wikipedia"
MANIFEST_PATH = BASE_DIR / "resources" / "rag" / "wiki_manifest.json"
# 두 번째 문서 출처(F2016) — 위키데이터. 구조화된 사실(개봉일·수상·흥행·원작·배경 장소)이다.
WIKIDATA_DIR = BASE_DIR / "resources" / "rag" / "wikidata"
WIKIDATA_MANIFEST_PATH = BASE_DIR / "resources" / "rag" / "wikidata_manifest.json"
# Windows에서 chromadb는 한글 등 비ASCII 경로에 벡터 파일(HNSW)을 쓰지 못한다.
# 파일이 안 생겨도 색인은 성공한 것처럼 끝나고, 약 1,000건을 넘는 순간 검색이 실패한다.
# 프로젝트 경로에 한글이 있으면 .env의 INDEX_DIR에 영문 경로를 지정한다.
INDEX_DIR = Path(os.getenv("INDEX_DIR") or BASE_DIR / "resources" / "indexes")

# 컬렉션
MOVIE_COLLECTION = "movie_meta"
WIKI_COLLECTION = "movie_wiki"
# 위키데이터는 위키백과와 수집·청킹 방식이 달라 컬렉션을 따로 둔다.
# 같은 컬렉션에 넣으면 위키백과 기준으로 만든 복구·점검 도구(repair_wiki_index 등)가 섞여 버린다.
WIKIDATA_COLLECTION = "movie_wikidata"

# 청킹 (영화 메타데이터는 짧아서 분할하지 않는다)
# 위키 문서는 문단 하나에 사실이 흩어져 있어 청크가 너무 짧으면 맥락이 잘린다.
WIKI_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
WIKI_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# 위키데이터는 "속성: 값" 한 줄이 사실 하나다. 글자 수로 자르면 속성명과 값이 갈라지므로
# 줄 단위로 묶고, 청크마다 영화 제목 머리줄을 붙인다. 중복 길이는 두지 않는다.
WIKIDATA_CHUNK_SIZE = int(os.getenv("WIKIDATA_CHUNK_SIZE", "1000"))
WIKIDATA_MAX_VALUES = 12   # 속성 하나에 넣을 값 상한 (출연진·수상 목록이 수백 개인 경우가 있다)

# 검색
TOP_K_MOVIE = int(os.getenv("TOP_K_MOVIE", "3"))
TOP_K_WIKI = int(os.getenv("TOP_K_WIKI", "10"))
# MMR: 비슷한 청크가 몰리지 않도록 넉넉히 뽑은 뒤 다양성을 고려해 고른다.
USE_MMR = os.getenv("USE_MMR", "1") == "1"
FETCH_K = int(os.getenv("FETCH_K", "30"))
# 선택한 영화 외의 영화 메타데이터도 근거에 넣을지. 기본은 끈다(질문과 무관한 영화가 섞임).
INCLUDE_SIMILAR_MOVIES = os.getenv("INCLUDE_SIMILAR_MOVIES", "0") == "1"

# 리뷰 생성 (선택한 영화의 메타데이터 + 위키 청크)
REVIEW_TOP_K_WIKI = 12     # 리뷰에 넣을 위키 청크 수

# 추천 (RAG 1 검색만 사용)
RECOMMEND_FETCH = 20       # 벡터 검색으로 뽑는 후보 수 (LLM이 이 중에서 고른다)
RECOMMEND_COUNT = 5        # "이런 영화 찾아줘" 결과 수 상한
SIMILAR_COUNT = 3          # 사이드바 "비슷한 작품" 수

# Wikipedia 수집
KEYWORD_COUNT = int(os.getenv("KEYWORD_COUNT", "9"))   # LLM이 추출할 검색 키워드 수
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "8"))    # 키워드당 확인할 검색 결과 수
MAX_PAGES = 12             # 영화당 색인할 문서 총량 상한
# 한국어 위키백과는 영어판보다 문서가 훨씬 짧다. 800자를 대면 관련성이 높은 문서도
# 무더기로 탈락한다. ('듄 (소설)' 783자, '범죄심리학' 434자, '드니 빌뇌브' 274자)
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "350"))
# 감독 문서는 짧아도 받는다. 한국어 위키는 인물 문서가 유난히 짧다. (크리스토퍼 놀란 414자)
# 영화 본편 문서에는 이 기준도 적용하지 않고 길이와 관계없이 받는다. (F2004 — 어떤 제외 규칙보다 우선)
# 800자 기준에서는 본편이 먼저 탈락했다. (헝거게임 캣칭 파이어 671자, 300 755자, 정이 436자)
CORE_MIN_LENGTH = 300

# 보강(top-up) 스위치. prebuild_wiki.py --topup 이 켠다. 평상시 수집은 영향받지 않는다.
# RECOLLECT   : 이미 써 본 키워드도 다시 검색한다. 지난번에 선별·길이 기준으로 버려진
#               문서를 완화된 기준으로 다시 만나려면 재검색이 필요하다.
# SECOND_ROUND: 1차 수집이 상한을 못 채우면 다른 각도의 키워드를 더 뽑아 한 번 더 돈다.
#               빈 자리를 검색 후보로 메우면 무관한 문서가 근거로 섞인다.
#               (노트북 → 삼성전자, 좋은 친구들 → 이정은) 후보를 넓히는 쪽이 안전하다.
RECOLLECT = False
# 2차 라운드는 상한을 못 채웠을 때만 돈다. 평상시 수집에도 켜 두는 편이 낫다.
SECOND_ROUND = os.getenv("SECOND_ROUND", "1") == "1"
MORE_KEYWORD_COUNT = int(os.getenv("MORE_KEYWORD_COUNT", "6"))
