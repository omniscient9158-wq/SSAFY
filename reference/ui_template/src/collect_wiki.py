"""RAG 2 — 키워드 추출 → Wikipedia 수집 → 관련성 필터 → 색인.

앱에서 함수로 호출하고, 아래처럼 단독 실행도 가능하다.
    python src/collect_wiki.py 496243
    python src/collect_wiki.py 496243 "실제 역사와 얼마나 다른가?"
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import movies as movie_data
import vector_store
import wiki_api
from llm import KEYWORD_MODEL, SELECT_MODEL, get_llm

# ── 프롬프트 1 : 큐레이터가 학습해야 할 위키 문서 찾기 ──────────────────────
KEYWORD_PROMPT = """넌 영화 큐레이터야.
아래 영화를 이해하는 데 필요한 배경지식이 담긴 한국어 위키백과 문서를 찾아줘.

찾아야 할 것 — 영화의 **내용**을 설명해 주는 문서
1. 영화 자체를 다룬 문서
2. 영화가 다루는 소재·개념·배경
   - 실화나 역사를 다루면 그 사건, 인물, 시대
   - 과학이나 미래를 다루면 그 개념과 기술
   - 원작(소설·만화·게임)이 있으면 원작
   - 특정 장소·직업·문화·사회 문제가 중심이면 그 문서
3. 사용자 질문이 있다면 그 질문에 답하는 데 필요한 문서

찾지 말아야 할 것
- **작중에만 존재하는 인물·지명·설정.** 한국어 위키백과에는 거의 없다.
  ("포키 (토이 스토리 4)", "겁나먼 왕국", "한스 그루버", "제13구역", "아라키스")
  영어 위키백과에만 있는 항목을 적으면 검색 결과가 빈다.
  대신 그 설정이 기대고 있는 **실제 세계의 개념**을 적는다.
  (한스 그루버 → "인질극", 제13구역 → "저항 운동", 아라키스 → "사막 기후")
- **배우의 인물 문서.** 출연작 나열뿐이라 영화 내용의 근거가 되지 않는다.
- 감독이나 배우가 참여한 **다른 영화** 문서.
- 제목만 같고 영화와 무관한 문서.

장르별 예시
- 역사·실화     → "명량 해전", "이순신", "정유재란"
- 우주·과학     → "웜홀", "블랙홀", "항성간 여행"
- 만화·소설 원작 → "어벤저스 (마블 코믹스)", "인피니티 스톤"
- 사회·문화     → "반지하", "경제적 불평등"

규칙
1. 한국어 위키백과에 **실제로 존재하는** 문서 제목으로 {count}개를 제시한다.
   확신이 안 서면 더 일반적이고 확실히 존재하는 개념으로 바꾼다.
   빈 검색 결과보다 조금 넓은 개념이 낫다.
2. **뜻이 여러 개인 단어는 괄호를 붙여 구체적으로 쓴다.**
   "조커"가 아니라 "조커 (DC 코믹스)", "나사"가 아니라 "미국 항공우주국",
   "쉴드"가 아니라 "쉴드 (마블 코믹스)"처럼 쓴다.
3. 가능하면 영화에 직접 걸리는 문서를 고른다.
   "심리학"보다 "범죄 심리학", "도덕"보다 "트롤리 문제"가 낫다.
   단, 그 좁은 문서가 한국어 위키백과에 없을 것 같으면 상위 개념을 쓴다.
4. **소재가 빈약한 오락 영화라도 {count}개를 채운다.** 영화가 기대는 소재,
   배경이 되는 시대·장소·직업, 원작, 감독 문서까지 넓혀서 채운다.
5. 영화를 이해하는 데 중요한 순서로 정렬한다.
6. 다른 설명 없이 JSON 배열만 출력한다. 예: ["명량 해전", "이순신", "정유재란"]

[영화 정보]
{movie}

[사용자 질문]
{question}
"""


def _norm(text: str) -> str:
    """비교용 정규화. 공백·괄호·구분기호를 없앤다.

    가운뎃점은 표기가 여러 가지라 모두 지운다. ('월•E'와 위키 제목 '월-E'가 같아야 본편으로 인식된다)
    """
    return re.sub(r"[\s·•‧∙・･:()\[\]{}<>'\"\-–—,.]", "", text).lower()


def _parse_json_array(text: str) -> list[str]:
    """LLM 응답에서 JSON 배열만 뽑아낸다."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def base_keywords(movie: dict[str, Any]) -> list[str]:
    """LLM 결과와 무관하게 항상 확보할 키워드 — 영화 자체 문서."""
    keywords = [f"{movie['title']} (영화)"]
    original = movie.get("original_title", "")
    if original and original != movie["title"]:
        keywords.append(f"{original} (영화)")
    return keywords


def tail_keywords(movie: dict[str, Any]) -> list[str]:
    """맨 뒤에 붙이는 키워드 — 감독 문서.

    문서 수 상한이 작을 때 감독 문서가 소재·배경 문서를 밀어내지 않도록
    LLM이 뽑은 키워드보다 뒤에 둔다. 상한이 넉넉하면 어차피 함께 들어온다.
    """
    return movie.get("director", [])[:1]


def extract_keywords(movie: dict[str, Any], question: str = "") -> list[str]:
    """프롬프트 1로 검색 키워드를 뽑는다."""
    prompt = KEYWORD_PROMPT.format(
        count=config.KEYWORD_COUNT,
        movie=movie_data.to_text(movie),
        question=question or "(질문 없음 - 영화 전반)",
    )
    try:
        response = get_llm(model=KEYWORD_MODEL).invoke(prompt)
        keywords = _parse_json_array(response.content)
    except Exception as error:
        print(f"[키워드 추출 실패] {error}")
        keywords = []

    return keywords[: config.KEYWORD_COUNT]


# ── 프롬프트 1-B : 1차로 못 채웠을 때 다른 각도로 더 찾기 ────────────────
KEYWORD_MORE_PROMPT = """넌 영화 큐레이터야.
아래 영화의 배경지식 문서를 모으는 중인데 아직 {short}건이 부족하다.
[이미 모은 문서]와 **겹치지 않는 다른 각도**로 한국어 위키백과 문서를 {count}개 더 찾아줘.

이미 시도해서 실패한 각도
- 작중에만 존재하는 인물·지명·설정 (한국어 위키백과에 없다)
- 같은 시리즈의 다른 작품, 배우 문서, 장르 문서

아직 안 써 본 각도 — 이쪽에서 찾는다
1. 영화의 배경이 되는 **시대·국가·도시**  (1980년대 미국, 로스앤젤레스)
2. 중심이 되는 **직업·조직·제도**        (경찰 특수기동대, 미국 연방수사국, 요양원)
3. 소재가 되는 **실제 사건·현상·사회 문제** (고층 건물 화재, 인질 협상, 치매)
4. **원작자·원작 소설·원작 만화**        (니컬러스 스파크스)
5. 줄거리에 등장하는 **기술·도구·생물**   (기관총, 크라켄)

규칙
1. 한국어 위키백과에 **실제로 존재하는** 문서 제목만 쓴다. 확신이 없으면 더 일반적인
   개념으로 바꾼다. 없는 문서를 대면 이 자리는 그냥 빈 채로 남는다.
2. [이미 모은 문서]에 있는 것과 같은 대상은 다시 내지 않는다.
3. 영화 내용과 가까운 순서로 정렬한다.
4. 다른 설명 없이 JSON 배열만 출력한다.

[영화 정보]
{movie}

[이미 모은 문서]
{have}

[사용자 질문]
{question}
"""


def extract_more_keywords(
    movie: dict[str, Any], have: list[str], short: int, question: str = ""
) -> list[str]:
    """1차 수집이 상한을 못 채웠을 때 다른 각도의 키워드를 더 뽑는다.

    빈 자리를 검색 후보로 아무거나 메우면 근거가 오염된다. 무관한 문서를 밀어 넣는
    대신 키워드를 다시 뽑아 후보 자체를 넓힌다.
    """
    prompt = KEYWORD_MORE_PROMPT.format(
        count=config.MORE_KEYWORD_COUNT,
        short=short,
        movie=movie_data.to_text(movie),
        have="\n".join(f"- {title}" for title in have) or "(없음)",
        question=question or "(질문 없음 - 영화 전반)",
    )
    try:
        response = get_llm(model=KEYWORD_MODEL).invoke(prompt)
        keywords = _parse_json_array(response.content)
    except Exception as error:
        print(f"[2차 키워드 추출 실패] {error}")
        return []
    return keywords[: config.MORE_KEYWORD_COUNT]


def _is_own_article(title: str, movie: dict[str, Any]) -> bool:
    """영화 본편을 다룬 문서인지 판별한다.

    괄호 꼬리표만 떼고 제목이 그대로 일치할 때만 인정한다. 부분 일치를 허용하면
    '반지의 제왕 (영화 시리즈)'가 '반지의 제왕: 왕의 귀환'의 본편으로 오인되어
    시리즈물에서 정확히 반대로 동작한다.
    """
    # '다이 하드 (영화 시리즈)'는 꼬리표를 떼면 '다이 하드'가 되어 본편과 구별되지 않는다.
    # 시리즈 총론 문서는 개별 작품이 아니므로 먼저 제외한다.
    if re.search(r"[(（][^)）]*(영화 시리즈|프랜차이즈)[^)）]*[)）]", title):
        return False
    # 꼬리표의 연도가 우리 영화와 크게 다르면 동명의 다른 작품이다. ('듄 (1984년 영화)')
    # 개봉 연도는 나라마다 한두 해 어긋나므로 2년까지는 같은 작품으로 본다.
    tag_year = re.search(r"(1[89]|20)[0-9]{2}", title)
    if tag_year and movie.get("year"):
        if abs(int(tag_year.group(0)) - int(movie["year"])) > 2:
            return False
    stripped = re.sub(r"[(（][^)）]*[)）]\s*$", "", title)
    names = {_norm(movie["title"])}
    original = movie.get("original_title", "")
    if original:
        names.add(_norm(original))
    return _norm(stripped) in names or _norm(title) in names


def _hard_exclude(result: dict[str, Any], movie: dict[str, Any]) -> bool:
    """명백히 다른 작품만 걸러낸다. 나머지 판단은 LLM에 맡긴다.

    문자열 매칭으로 관련성을 판정하려던 방식은 실패했다. 번역어·별칭을 넘지 못해
    정답을 버렸다. ('NASA'→'미국 항공 우주국', '하비 덴트'→'투페이스')
    여기서는 되돌릴 수 없는 오답만 막는다.
    """
    tags = re.findall(r"[(\uff08]([^)\uff09]*)[)\uff09]", result["title"])
    if not tags:
        return False
    tag = tags[-1]
    tag_year = re.search(r"(1[89]|20)\d{2}", tag)
    is_work = re.search(r"영화|드라마|애니메이션|시리즈|뮤지컬|게임", tag)
    # '(2019년 영화)'처럼 연도가 붙은 작품인데 우리 영화가 아니면 다른 작품이다.
    return bool(tag_year and is_work and tag_year.group(0) != str(movie.get("year") or ""))


def _is_sibling_work(title: str, movie: dict[str, Any]) -> bool:
    """같은 시리즈의 다른 편인지.

    프롬프트로 시리즈물을 거르라고 해도, 자리가 비면 LLM은 시리즈 다른 편으로 메운다.
    (고스트버스터즈 한 편에 2016년판·프랜차이즈·2편이 함께 들어왔다.) 후보 단계에서
    막는다. 제목이 겹치는 것만으로 판단하면 원작까지 버리므로 ('듄 (소설)'은 '듄'으로
    시작한다) 어간 뒤에 편 번호나 부제가 붙었는지를 본다.
    """
    if _is_own_article(title, movie):
        return False
    if re.search(r"[(（][^)）]*(영화 시리즈|프랜차이즈)[^)）]*[)）]", title):
        return True

    loose = lambda text: re.sub(r"\s+", "", text).lower()
    stem = re.split(r"[:：]", movie["title"])[0]
    stem = re.sub(r"[(（][^)）]*[)）]", "", stem)
    stem = loose(re.sub(r"\s*\d+\s*$", "", stem))
    if not stem:
        return False

    plain = loose(re.sub(r"[(（][^)）]*[)）]\s*$", "", title))
    if not plain.startswith(stem):
        return False

    rest = plain[len(stem):]
    if not rest:
        # 어간과 똑같은 제목. 시리즈 1편이거나 총론 문서다. 원작만 남긴다.
        tags = re.findall(r"[(（]([^)）]*)[)）]", title)
        return not re.search(r"소설|만화|코믹스|희곡|뮤지컬|게임|드라마", tags[-1] if tags else "")
    return bool(re.match(r"^([:：]|\d|파트|part|[ivx]+$)", rest))


# 검색 노이즈로 반복해서 들어오던 유형. 후보 단계에서 잘라 LLM 선별에 보내지 않는다.
# (로드의 음반 목록, 2017년 대한민국의 영화, 2004년 칸 영화제, 로튼 토마토, IMAX)
_NOISE_TITLE = re.compile(
    r"(목록|음반|사운드트랙|영화제|시상식|수상|흥행|박스오피스|로튼 토마토|IMAX|메타크리틱)"
)
# 장르·기법 문서. '판타지 영화'처럼 매체어로 끝나거나 '로맨틱 코미디'처럼 장르어로 끝난다.
_GENRE_TITLE = re.compile(
    r"(영화|드라마|애니메이션|만화|코미디|스릴러|공포|호러|누아르|느와르|블록버스터)$"
)


def _pool_exclude(title: str, movie: dict[str, Any], cast: set[str]) -> bool:
    """후보 단계에서 잘라낼 문서.

    프롬프트로만 막으면 자리가 빌 때 LLM이 바로 이 유형으로 자리를 메운다. 규칙으로
    막는 편이 확실하고, 그만큼 LLM은 남은 후보에만 집중한다. 본편은 호출부에서 면제한다.
    """
    if _NOISE_TITLE.search(title) or re.match(r"^[0-9]{4}년", title):
        return True
    if _norm(title) in cast:
        return True
    plain = re.sub(r"[(（][^)）]*[)）]", "", title).strip()
    if _GENRE_TITLE.search(plain):
        return True
    return _is_sibling_work(title, movie)


# ── 프롬프트 2 : 검색 결과 중 쓸 문서 고르기 (후보 문서 선별) ──────────────────────────
SELECT_PROMPT = """넌 영화 큐레이터야.
아래 [후보 문서]는 위키백과 검색 결과다. 이 영화를 이해하는 데 도움이 되는 문서를
**최대 {limit}개** 골라줘. 관련 있는 문서는 아끼지 말고 고르되, **관련 없는 문서로
자리를 메우지는 않는다.** 빈 자리는 다른 키워드로 다시 채울 기회가 있다.

고를 것 — 영화의 **내용**을 설명해 주는 문서
1. 영화 자체를 다룬 문서
2. 영화의 소재·배경이 되는 실제 사건, 인물, 과학 개념, 원작, 장소, 사회 문제
   (예: 인터스텔라 → 웜홀, 블랙홀 / 다크 나이트 → 배트맨, 조커, 고담시)
   **영화의 소재와 직결되는 일반 개념도 고른다.**
   (다이 하드 → 인질극, 테러리즘 / 좋은 친구들 → 미국 마피아 / 나우 유 씨 미 → 마술)
3. **제목이 달라도 같은 대상을 가리키면 고른다.**
   ("NASA"를 찾을 때 "미국 항공 우주국", "하비 덴트"를 찾을 때 "투페이스")
4. 이 영화의 감독 본인 문서 (1건만)

거를 것
- **배우 개인 문서.** 출연작 나열뿐이라 근거가 되지 않는다.
- **같은 시리즈의 다른 작품.** 속편, 전편, 스핀오프, 게임, 프랜차이즈 목록 전부.
  (아바타를 다룰 때 "아바타: 물의 길", "아바타 (프랜차이즈)"는 다른 작품이다)
- 감독이나 배우가 참여한 다른 영화
- **영화 제작 정보.** 음악감독, 촬영 포맷(IMAX), 수상 내역, 흥행 기록.
  영화가 무엇을 다루는지가 아니라 어떻게 만들었는지다.
- 무관한 동음이의어, 목록 문서, 연도·영화제·평점 문서
- **영화 장르·기법 문서.** ("범죄 영화", "스릴러", "판타지 영화", "시각 효과")
  단, 소재 자체의 일반 개념("테러리즘", "해킹", "마술")은 위 2번이므로 고른다.

고르는 순서
- **영화의 핵심 소재부터 앞에 둔다.** 뒤로 갈수록 잘려나간다.
- 같은 개념이 여러 문서로 나뉘어 있으면 가장 기본이 되는 것 하나만 고른다.
  ("시공간", "민코프스키 공간", "굽은 시공간" 중에서는 하나만)

출력
- 번호만 JSON 배열로 낸다. 예: [3, 1, 7]
- {limit}개를 넘기지 않는다. 관련 있는 게 없으면 [] 를 낸다.
- 다른 설명은 쓰지 않는다.

[영화]
{movie}

[후보 문서]
{candidates}
"""


def select_documents(
    movie: dict[str, Any], candidates: list[dict[str, Any]], limit: int = 0
) -> list[dict[str, Any]]:
    """검색 결과 중 영화와 관련 있는 문서를 LLM이 고른다.

    몇 개를 고를지 알려주지 않으면 필요 이상으로 골라 순위 뒤쪽이 잘려나간다.
    """
    if not candidates:
        return []
    limit = limit or config.MAX_PAGES

    listing = "\n".join(
        f"{index}. {item['title']} — {item['snippet'][:120]}"
        for index, item in enumerate(candidates, start=1)
    )
    prompt = SELECT_PROMPT.format(
        limit=limit, movie=movie_data.to_text(movie), candidates=listing
    )
    try:
        response = get_llm(model=SELECT_MODEL).invoke(prompt)
        numbers = _parse_json_array(response.content)
    except Exception as error:
        print(f"[문서 선별 실패] {error}")
        return candidates[:limit]  # 실패해도 진행은 한다

    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for value in numbers:
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            continue
        if not (1 <= number <= len(candidates)) or number in used:
            continue
        used.add(number)
        picked.append(candidates[number - 1])
        if len(picked) >= limit:
            break
    return picked


# ── 수집 이력(manifest) ────────────────────────────────────────────────
def load_manifest() -> dict[str, Any]:
    if config.MANIFEST_PATH.exists():
        with config.MANIFEST_PATH.open(encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def is_collected(movie_id: str) -> bool:
    return movie_id in load_manifest()


def collected_titles(movie_id: str) -> list[str]:
    return load_manifest().get(movie_id, {}).get("titles", [])


# ── 수집 본체 ──────────────────────────────────────────────────────────
def collect_for_movie(movie: dict[str, Any], question: str = "") -> dict[str, Any]:
    """영화의 위키 문서를 수집·색인한다.

    키워드 단위로 이력을 남기므로, 질문이 바뀌어 새 키워드가 나오면 그만큼만 추가 수집한다.
    """
    movie_id = movie["movie_id"]
    manifest = load_manifest()
    record = manifest.get(movie_id) or {
        "keywords": [],
        "titles": [],
        "page_ids": [],
        "chunks": 0,
    }

    # 영화 문서 → LLM이 뽑은 소재·배경 → 감독 문서 순서로 채운다.
    candidates = (
        base_keywords(movie)
        + extract_keywords(movie, question)
        + tail_keywords(movie)
    )

    # 보강 모드에서는 지난번 키워드도 다시 검색한다. 그때 선별·길이 기준으로 버려진
    # 문서를 완화된 기준으로 다시 만나야 하기 때문이다. 문서 중복은 page_id로 막는다.
    done = set() if config.RECOLLECT else {_norm(k) for k in record["keywords"]}
    keywords: list[str] = []
    for keyword in candidates:
        if _norm(keyword) in done:
            continue
        done.add(_norm(keyword))
        keywords.append(keyword)

    if not keywords or len(record["titles"]) >= config.MAX_PAGES:
        return {"skipped": True, "added_titles": [], **record}

    store = vector_store.get_store(config.WIKI_COLLECTION)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.WIKI_CHUNK_SIZE,
        chunk_overlap=config.WIKI_CHUNK_OVERLAP,
    )

    # page_id는 위키 API가 정수로 주고 manifest에는 문자열로 남는다.
    # 섞이면 중복 판정이 빗나가고 sorted()가 TypeError로 죽는다. 문자열로 통일한다.
    seen_pages = {str(page_id) for page_id in record["page_ids"]}
    added_titles: list[str] = []
    documents: list[Document] = []

    # 라운드 1은 기본·LLM·감독 키워드로 돈다. 상한을 못 채우면 다른 각도의 키워드를
    # 더 뽑아 라운드 2를 돈다. 빈 자리를 검색 후보로 아무거나 메우면 근거가 오염되므로,
    # 자리를 메우는 대신 후보 자체를 넓힌다.
    cast = {_norm(name) for name in movie.get("cast", [])}
    directors = {_norm(name) for name in movie.get("director", [])}
    room = config.MAX_PAGES - len(record["titles"])
    seen_titles: set[str] = set()
    # 수집 이력에는 끝까지 처리된 키워드만 남긴다. (F2004 · NF2002 — 이미 '수집한' 검색어만 건너뛴다)
    # 검색이나 본문 수집이 일시적인 오류로 실패한 키워드까지 남기면 다음 질문부터 건너뛰어
    # 그 문서를 영영 받지 못한다. 본편·감독 키워드는 질문과 상관없이 늘 같아서 특히 치명적이다.
    tried_keywords: list[str] = []      # 이번 호출에서 시도한 키워드 (2차 라운드 중복 방지용)
    searched_keywords: list[str] = []   # 검색에 성공한 키워드 (결과 0건도 성공으로 본다)
    failed_keywords: set[str] = set()   # 본문 수집이 실패해 다음에 다시 검색할 키워드
    aborted = False                     # 연결이 불안정해 수집을 중간에 멈췄는지
    second_round: list[str] = []
    pending = list(keywords)
    round_no = 0

    while pending and len(added_titles) < room:
        round_no += 1
        round_keywords, pending = pending, []
        tried_keywords += round_keywords

        # 1단계 — 키워드마다 검색해서 후보를 모은다. 본문은 아직 받지 않는다.
        pool: list[dict[str, Any]] = []
        consecutive_errors = 0
        for keyword in round_keywords:
            try:
                results = wiki_api.search(keyword, limit=config.SEARCH_LIMIT)
            except Exception as error:
                # 이력에 남기지 않는다. 다음 질문 때 다시 검색한다.
                print(f"[검색 실패] {keyword}: {error}")
                consecutive_errors += 1
                if consecutive_errors >= 2:
                    # 재시도(wiki_api)를 거치고도 연달아 실패하면 연결 자체가 불안정한 것이다.
                    # 남은 키워드는 시도하지 않고 이력에도 남기지 않은 채 다음 질문으로 미룬다.
                    print("[수집 중단] 위키백과 연결이 불안정합니다. 남은 키워드는 다음 질문에서 다시 시도합니다.")
                    aborted = True
                    break
                continue
            consecutive_errors = 0
            searched_keywords.append(keyword)
            for result in results:
                if str(result["page_id"]) in seen_pages or result["title"] in seen_titles:
                    continue
                # 본편 문서는 어떤 배제 규칙도 적용하지 않는다. 개봉 연도가 나라마다
                # 달라 '(2013년 영화)' 꼬리표에 걸리면 정작 본편이 버려진다.
                if not _is_own_article(result["title"], movie) and (
                    _hard_exclude(result, movie)
                    or _pool_exclude(result["title"], movie, cast)
                ):
                    continue
                seen_titles.add(result["title"])
                pool.append({**result, "keyword": keyword})
            time.sleep(0.5)  # 위키백과 호출 예절

        # 2단계 — 후보 중 쓸 것을 LLM이 고른다. 버릴 문서는 본문을 받지 않는다.
        left = room - len(added_titles)
        chosen = select_documents(movie, pool, limit=left)

        if round_no == 1:
            # 본편 문서는 모든 근거의 기준점이라 LLM의 판단에 맡기지 않는다.
            # 후보에 있으면 무조건 첫 자리를 준다. 골라진 목록에 있어도 앞으로 끌어올린다.
            #   - 빠지는 경우  : 다른 소재 문서에 밀려 잘려나간다
            #   - 뒤에 놓이는 경우: 상한에 걸려 본문을 받기 전에 잘려나간다
            #   - 대체되는 경우 : 반지의 제왕: 왕의 귀환 → '반지의 제왕 (영화 시리즈)'
            own = next((item for item in pool if _is_own_article(item["title"], movie)), None)
            if own:
                rest = [item for item in chosen if item["title"] != own["title"]]
                chosen = [own] + rest[: max(left - 1, 0)]

        # 3단계 — 고른 문서만 본문을 받아 색인한다.
        for item in chosen:
            if len(added_titles) >= room:
                break
            time.sleep(1.0)  # 위키백과 호출 예절
            try:
                page = wiki_api.fetch_page(item["title"])
            except Exception as error:
                # 이 문서만 건너뛰고, 이 문서를 찾은 키워드는 다음 질문 때 다시 검색하게 둔다.
                print(f"[본문 수집 실패] {item['title']}: {error}")
                failed_keywords.add(item["keyword"])
                continue
            if not page:
                continue
            # 본편 문서는 길이와 관계없이 받는다. (F2004 — 선별·길이 등 어떤 제외 규칙보다 우선)
            # 한국어 위키의 본편 문서는 유난히 짧지만, 짧아도 모든 근거의 기준점이다.
            # 감독 문서는 완화된 기준(CORE_MIN_LENGTH), 나머지는 알맹이가 있어야 한다.
            is_own = _is_own_article(page["title"], movie) or _is_own_article(item["title"], movie)
            if is_own:
                if not page["text"].strip():
                    continue  # 본문이 비어 있으면 색인할 내용이 없다
            else:
                is_director = _norm(page["title"]) in directors or any(
                    _norm(page["title"]) in name or name in _norm(page["title"])
                    for name in directors
                )
                floor = config.CORE_MIN_LENGTH if is_director else config.MIN_TEXT_LENGTH
                if len(page["text"]) < floor:
                    continue
            if str(page["page_id"]) in seen_pages:
                continue

            seen_pages.add(str(page["page_id"]))
            added_titles.append(page["title"])
            if round_no > 1:
                second_round.append(page["title"])
            _save_page_json(page, movie, item["keyword"])

            for index, chunk in enumerate(splitter.split_text(page["text"])):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            # 시리즈물은 여러 영화가 같은 문서(감독·원작·세계관)를 공유한다.
                            # page_id만 쓰면 두 번째 영화의 청크가 "이미 있음"으로 걸러져
                            # 먼저 수집된 영화만 그 문서를 검색할 수 있게 된다.
                            "chunk_id": f"{movie_id}:{page['page_id']}:{index:03d}",
                            "source_type": "wikipedia",
                            "title": page["title"],
                            "source_url": page["url"],
                            "page_id": str(page["page_id"]),
                            "movie_id": movie_id,
                            "related_movie": movie["title"],
                            "search_query": item["keyword"],
                            "collected_at": date.today().isoformat(),
                        },
                    )
                )

        # 상한을 못 채웠으면 다른 각도로 키워드를 더 뽑아 한 라운드 더 돈다.
        # 연결이 불안정해 수집을 멈췄다면 돌지 않는다. (검색이 또 실패해 LLM 호출만 낭비된다)
        if round_no == 1 and config.SECOND_ROUND and len(added_titles) < room and not aborted:
            more = extract_more_keywords(
                movie, record["titles"] + added_titles, room - len(added_titles), question
            )
            tried = {_norm(keyword) for keyword in tried_keywords}
            pending = [k for k in more if _norm(k) not in tried]

    added_chunks = 0
    if documents:
        ids = [doc.metadata["chunk_id"] for doc in documents]
        already = vector_store.existing_ids(store, ids)
        fresh = [doc for doc in documents if doc.metadata["chunk_id"] not in already]
        if fresh:
            store.add_documents(fresh, ids=[doc.metadata["chunk_id"] for doc in fresh])
        added_chunks = len(fresh)

    # 검색에 성공했고 본문 수집도 실패하지 않은 키워드만 '수집 완료'로 남긴다.
    done_keywords = [k for k in searched_keywords if k not in failed_keywords]
    record["keywords"] = list(dict.fromkeys(record["keywords"] + done_keywords))
    record["titles"] = list(dict.fromkeys(record["titles"] + added_titles))
    record["page_ids"] = sorted(seen_pages)
    # 실제로 색인된 수만 센다. 만든 수를 세면 보고서가 부풀려진다.
    record["chunks"] = record.get("chunks", 0) + added_chunks
    record["collected_at"] = date.today().isoformat()
    # 2차 라운드에서 들어온 문서는 표시해 둔다. 그 라운드가 값을 하는지 되짚을 때 쓴다.
    if second_round:
        record["second_round"] = list(
            dict.fromkeys(record.get("second_round", []) + second_round)
        )

    manifest[movie_id] = record
    save_manifest(manifest)

    return {**record, "skipped": False, "added_titles": added_titles, "second_round": second_round}


def has_own_article(movie: dict[str, Any]) -> bool:
    """이 영화의 본편 문서를 이미 수집했는지."""
    return any(_is_own_article(title, movie) for title in collected_titles(movie["movie_id"]))


def fill_own_article(movie: dict[str, Any], wiki_title: str = "") -> dict[str, Any]:
    """본편 문서만 찾아 보강한다. LLM을 쓰지 않는다.

    길이 기준에 걸려 본편 문서가 빠진 채로 구축된 영화를 되살리기 위한 것이다.
    이미 가진 문서는 건드리지 않으므로, 재수집과 달리 잃는 것이 없다.

    wiki_title을 주면 검색 없이 그 문서를 본편으로 받아들인다. 한국어 위키 제목이
    영화 제목과 전혀 달라 자동으로는 이어지지 않는 경우를 위한 수동 경로다.
    (마녀 2 → '마녀(魔女) Part2. The Other One')
    """
    movie_id = movie["movie_id"]
    manifest = load_manifest()
    record = manifest.get(movie_id)
    if record is None:
        return {"status": "미수집"}
    if has_own_article(movie) and not wiki_title:
        return {"status": "이미 있음"}

    query = wiki_title
    if not wiki_title:
        # '{제목} (영화)'로 먼저 찾고, 꼬리표 없는 제목으로도 시도한다.
        found = None
        for query in base_keywords(movie) + [movie["title"]]:
            try:
                results = wiki_api.search(query, limit=config.SEARCH_LIMIT)
            except Exception as error:
                print(f"[검색 실패] {query}: {error}")
                continue
            found = next(
                (r for r in results if _is_own_article(r["title"], movie)), None
            )
            if found:
                break
            time.sleep(0.5)  # 위키백과 호출 예절

        if not found:
            return {"status": "문서 없음"}
        if str(found["page_id"]) in {str(p) for p in record["page_ids"]}:
            return {"status": "이미 있음"}
        wiki_title = found["title"]

    time.sleep(0.5)
    page = wiki_api.fetch_page(wiki_title)
    if not page:
        return {"status": "본문 없음"}
    # 본편은 길이와 관계없이 받는다. (F2004) 본문이 비어 있을 때만 뺀다.
    if not page["text"].strip():
        return {"status": "본문 없음"}
    if str(page["page_id"]) in {str(p) for p in record["page_ids"]}:
        return {"status": "이미 있음"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.WIKI_CHUNK_SIZE,
        chunk_overlap=config.WIKI_CHUNK_OVERLAP,
    )
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_id": f"{movie_id}:{page['page_id']}:{index:03d}",
                "source_type": "wikipedia",
                "title": page["title"],
                "source_url": page["url"],
                "page_id": str(page["page_id"]),
                "movie_id": movie_id,
                "related_movie": movie["title"],
                "search_query": query,
                "collected_at": date.today().isoformat(),
            },
        )
        for index, chunk in enumerate(splitter.split_text(page["text"]))
    ]

    store = vector_store.get_store(config.WIKI_COLLECTION)
    ids = [doc.metadata["chunk_id"] for doc in documents]
    already = vector_store.existing_ids(store, ids)
    fresh = [doc for doc in documents if doc.metadata["chunk_id"] not in already]
    if fresh:
        store.add_documents(fresh, ids=[doc.metadata["chunk_id"] for doc in fresh])
    _save_page_json(page, movie, query)

    # 본편은 가장 중요한 근거라 목록 맨 앞에 둔다.
    record["titles"] = [page["title"]] + record["titles"]
    record["page_ids"] = sorted({str(p) for p in record["page_ids"]} | {str(page["page_id"])})
    record["chunks"] = record.get("chunks", 0) + len(fresh)
    manifest[movie_id] = record
    save_manifest(manifest)

    return {
        "status": "추가",
        "title": page["title"],
        "chunks": len(fresh),
        "length": len(page["text"]),
    }


def _save_page_json(page: dict[str, Any], movie: dict[str, Any], keyword: str) -> None:
    """수집 원문을 JSON으로 남긴다."""
    config.WIKI_DIR.mkdir(parents=True, exist_ok=True)
    # 여러 영화가 같은 문서를 모으면 파일이 덮어써진다. 영화 ID를 붙여 분리한다.
    safe = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", page["title"])
    safe = f"{movie['movie_id']}_{safe}"
    payload = {
        **page,
        "source": "한국어 위키백과",
        "movie_id": movie["movie_id"],
        "related_movie": movie["title"],
        "search_query": keyword,
        "collected_at": date.today().isoformat(),
    }
    with (config.WIKI_DIR / f"{safe}.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python src/collect_wiki.py <movie_id> [질문]")
        raise SystemExit(1)

    target = movie_data.find_movie(sys.argv[1])
    if not target:
        print(f"영화를 찾을 수 없습니다: {sys.argv[1]}")
        raise SystemExit(1)

    user_question = sys.argv[2] if len(sys.argv) > 2 else ""
    result = collect_for_movie(target, user_question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
