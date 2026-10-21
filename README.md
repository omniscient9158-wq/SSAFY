# AqSolDB 화합물 근거 탐색

**화학 도메인 RAG · AqSolDB 근거 기반 화합물 탐색**  
앱 이름: **AqSolDB 수용해도 분석**

AqSolDB 화합물을 선택해 메타데이터와 출처 확인 문서를 근거로 질문하고, 답변에 실제 사용한 근거를 확인하는 화학 도메인 RAG 서비스입니다. LogS 회귀와 출처 그룹별 평가는 도메인 특화 보조 기능입니다.

> **프로젝트 질문:** 선택한 화합물에 대해 근거 문서로 답하고, 답변에 실제 사용한 데이터와 문서 출처를 확인할 수 있을까?

초록과 연구 질문은 [프로젝트 개요 및 초록 초안](docs/00_프로젝트_개요_및_초록.md)을 참고하세요.

> 진행 상태: 출처 그룹 균형 train 메타데이터 100건과 승인 외부 근거 4건을 각각 Chroma에 색인했고, 재실행 멱등성과 GMS 답변 2건을 확인했습니다. 외부 근거가 없는 일반 질문은 모델을 호출하지 않고 보류하도록 구현·로컬 확인했습니다. 후보 수집 결과의 화합물/질문별 캐시를 실제 반복 실행으로 확인해, 두 번째 실행에서 GMS/API 재호출이 없음을 검증했습니다. 검색·인용 카드 로직은 AppTest에서 확인했지만 실제 브라우저의 화면 검증 및 제출 스크린샷, 자동 수집 후보의 출처 승인은 남아 있습니다. 미확인 항목은 통과로 표시하지 않습니다.

## 데이터와 출처

외부 근거 문서는 개별 원문·화합물 식별자·출처 표시·이용 조건을 확인한 발췌만 색인 후보로 사용합니다. 현재 개별 검토 및 Chroma 색인이 완료된 자료는 4건입니다. A-4518의 Wikipedia 자료는 CC BY-SA 4.0 보조 근거로 표시하고, 정확성 경고를 함께 제공합니다. 검토 기록은 [출처 검토 보고서](docs/03_출처_검토_보고서.md)와 `research/provenance/`에서 확인합니다.

- 원 데이터셋: [AqSolDB, Harvard Dataverse DOI 10.7910/DVN/OVHAW8](https://doi.org/10.7910/DVN/OVHAW8)
- Dataverse 공개 v1 라이선스 표기: CC0 1.0. 이는 배포된 AqSolDB에 대한 표시이며, upstream 원자료 각각의 라이선스까지 같다는 의미는 아닙니다.
- 관련 논문: [Sorkun et al. (2019), Scientific Data](https://www.nature.com/articles/s41597-019-0151-1)
- 로컬 분할 사본: [Hugging Face AqSolDB](https://huggingface.co/datasets/maomlab/AqSolDB/tree/main/AqSolDB)
- 각 AqSolDB 행의 A-I 접두어와 논문 원자료 묶음 연결은 [원자료 출처 매핑](data/AqSolDB_원자료_출처매핑.csv)을 참고하세요.
- 로컬 9,982행의 A-I 접두어는 매핑표에서 모두 연결됩니다. 이 매핑은 원자료 계보 추적이며, upstream 각 데이터의 별도 라이선스 확인으로 간주하지 않습니다.
- PubChem 후보는 조사자료이며, 모두 승인된 재사용 문서라는 뜻이 아닙니다. 실제 색인 대상 문서의 원출처와 이용 조건을 확인한 뒤 사용합니다.

## 설치와 실행

프로젝트 루트에서 PowerShell을 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`에 발급받은 `GMS_KEY`를 설정하세요. `.env`는 제출물에 포함하지 않습니다. 아래 메타데이터 색인 명령은 첫 실행 시 임베딩 API를 호출하며 사용량/크레딧이 발생할 수 있습니다.

```powershell
python src/compound_data.py
python src/index_metadata.py --index
python src/source_aware_evaluation.py
```

로컬 CSV 점검과 메타데이터 텍스트 출력은 임베딩 API를 호출하지 않습니다. Windows에서는 `.env`의 `INDEX_DIR`을 ASCII 경로로 설정하는 것을 권장합니다.

임베딩 없이 이름, AqSolDB ID, InChIKey, descriptor 값으로 로컬 메타데이터를 먼저 조회할 수 있습니다. 결과에는 데이터셋 DOI와 A-I 원자료 묶음 추적 정보가 함께 표시됩니다. 이는 아직 벡터 검색이나 RAG 답변이 아닙니다.

```powershell
python src/search_metadata.py "aspirin" --limit 5
```

세 화합물의 검토된 외부 근거는 API 호출 없이 아래 명령으로 별도 JSONL 문서 컬렉션 입력 파일로 만들 수 있습니다. 색인은 `--index`를 직접 지정한 경우에만 임베딩 API를 호출합니다.

```powershell
python src/index_external_documents.py
# 임베딩 색인 실행을 결정한 뒤에만:
python src/index_external_documents.py --index
```

결과 파일: `data/curated_external_documents.jsonl`. 이 파일은 승인된 발췌 네 건을 담고, 각 문서에 화합물 ID·InChIKey·원출처·인용 URL·검토 범위를 보존합니다. 이 자료 4건은 `indexes/aqsoldb_documents`에 색인했습니다. AqSolDB 메타데이터는 train 출처 균형 표본 100건을 `indexes/aqsoldb_compounds`에 색인했고, 동일 명령 재실행 시 중복 추가가 없음을 확인했습니다.

외부 문서 후보 수집은 위키미디어 검색 API와 PubChem PUG-REST/PUG-View를 사용하며, 검색어 추출·후보 선별을 위해 GMS LLM 호출 2회가 발생합니다. Wikimedia API는 운영자 연락처가 포함된 User-Agent를 요구하므로, 후보 API를 새로 호출하기 전에 `.env.example`의 `WIKIMEDIA_CONTACT` 설명을 참고해 본인의 이메일·프로젝트 URL·Wikimedia 계정 중 하나를 로컬 `.env`에 설정해야 합니다. 현재 프로젝트 환경에는 이 값이 없어 새 Wikimedia 수집은 안내 오류로 중지됩니다. PubChem은 화합물 InChIKey를 CID로 변환하고 해당 CID의 용해도 주석과 항목별 원출처·이용 조건 메모를 가져옵니다. 후보는 자동 색인되지 않고 `research/provenance/external_document_candidates.jsonl`에 검토 대기로 저장됩니다. 원문 식별자와 기여자 표시, 인용 및 이용 범위를 사람이 검토하고 승인 자료 목록으로 옮긴 뒤에만 색인합니다. 위키백과 텍스트는 보통 CC BY-SA 4.0이지만 페이지별 표시를 확인해야 하며, PubChem 주석도 각 기여 출처 조건을 확인해야 합니다. API 규정과 재사용 안내는 [Wikimedia API 이용 정책](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines), [Wikimedia 콘텐츠 재사용 안내](https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse), [PubChem 데이터 출처 안내](https://pubchem.ncbi.nlm.nih.gov/docs/data-sources)을 참고하세요.

```powershell
# 단일 AqSolDB 항목의 후보 탐색 (GMS 2회 및 외부 API 요청; 색인 임베딩은 하지 않음)
python src/discover_wikipedia_candidates.py E-1274 --question "아스피린의 물 용해도와 물성에 대해 설명해줘"
```

LLM 선별 결과는 자동 승인 근거가 아닙니다. 출처와 이용 조건을 확인하기 전에는 RAG 컬렉션에 넣지 않습니다.

선택한 화합물의 메타데이터와 관련 외부 발췌를 로컬에서 함께 검색하려면 정확한 AqSolDB ID와 질문을 전달합니다.

```powershell
python src/retrieve_local_evidence.py A-4518 "물에 대한 용해도"
```

이 검색기는 현재 단어 일치 기반의 로컬 프로토타입입니다. 메타데이터와 외부 근거를 별도 결과로 반환하고, 외부 문서는 선택 ID가 같은 자료만 찾습니다. 벡터 검색은 별도 색인이 필요합니다.

## 시연 UI

```powershell
streamlit run app.py
```

앱에서 화합물 이름이나 AqSolDB ID를 선택하면 해당 화합물의 데이터를 근거로 질문할 수 있습니다. 현재 외부 문헌 검색 색인은 A-454, A-5538, A-4518의 검토 발췌를 포함하며 A-4518에는 NOAA/NTP와 Wikipedia 보조 근거가 함께 연결돼 있습니다. 화합물 빠른 예시 버튼은 두지 않았습니다. 질문 예시는 선택한 항목의 검토 문헌을 탐색하는 용도입니다. 후보 수집 스크립트는 별도 CLI이고 인터넷 전체나 PubChem 전체를 검색하지 않습니다.

**문헌 검색 흐름:** 화면 위에서 화합물을 검색한 뒤, 결과 아래의 `더 알아보기`에서 속성·조건·원문 표현을 질문합니다. 예를 들어 A-4518은 `2-methylpyridine water solubility mg/mL 68°F NTP 1992 원출처는 무엇인가요?`처럼 입력하면 발췌의 수치·단위·온도·인용정보를 확인할 수 있습니다. A-454는 `tris(2-butoxyethyl) phosphate yellow viscous liquid 설명과 출처를 알려주세요.`, A-5538은 `Anthracene sinks in water라는 설명의 출처와 문맥은 무엇인가요?`를 예시로 사용할 수 있습니다. 화면의 예시 질문을 선택하면 입력창에 자동으로 채워집니다.

기본 **단어 일치** 검색은 질문과 발췌의 단어를 비교하며 GMS 크레딧을 쓰지 않습니다. 외부 문서의 영어 원문 표현을 질문에 함께 넣으면 검색에 유리합니다. 고급 설정의 **의미 유사도** 검색은 GMS 임베딩 크레딧을 사용하고 실행 전에 동의를 받습니다. 답변 생성도 별도로 동의를 받은 경우에만 GMS LLM을 호출합니다. 검색 결과가 없으면 현재 발췌에서 일치 항목을 찾지 못했다는 뜻이며, 전체 문헌에 정보가 없다고 단정할 수 없습니다.

조건 검색 지원 예: `분자량 300 이하, LogP 2 미만`, `LogS -4 초과`, `TPSA 20~80`. 조건은 허용 필드·연산자로 변환한 뒤 `train.csv`에서 다시 수치 비교하며, 제공 test 분할은 검색에 포함하지 않습니다.

## 폴더 안내

```text
PJT-je/
├─ app.py                # 7단계 Streamlit 골격에서 발전시킨 AqSolDB 분석 UI
├─ data/                 # AqSolDB train/test, 원자료 출처 매핑, 선별 외부 근거 JSONL
├─ docs/                 # 요구사항, 설계, 출처검토, 체크리스트, 회고, 분석결과
├─ src/                  # snake_case 기능 모듈: 데이터·검색·색인·모델·RAG
├─ research/provenance/  # PubChem 표본 조사 원자료와 조사 스크립트
├─ reference/            # 라이브 실습 UI 템플릿 참고본 (제출 ZIP에서는 제외)
├─ .env.example          # 환경변수 이름만 포함한 템플릿
└─ requirements.txt      # 프로젝트 실행 의존성
```

## 제출물 구성

제출용 압축 파일에는 `app.py`, `README.md`, `requirements.txt`, `.env.example`, `data/`, `docs/`, `src/`를 포함하세요. `reference/`의 강의 UI 템플릿은 제출본에서 제외하세요. `research/`는 출처 감사 추적용 보조자료이며, 필요 시 보고서만 별도 참고 자료로 제출하세요. `.env`, `.venv/`, `indexes/`, `reference/`, `__pycache__/`는 제출하지 않습니다. 폴더 전체를 압축하기보다 이 목록을 골라 담으세요.

## 현재 상태와 다음 작업

체크리스트별 판정은 [현재 상태표](docs/04_체크리스트_현재상태.md), 기능별 수용 기준은 [요구사항 정의서](docs/01_요구사항정의서.md)를 확인하세요. [원자료 그룹별 초기 분석](docs/08_출처별_일반화_분석.md)은 보조 분석이며 최종 평가가 아닙니다. 남은 핵심은 승인된 화합물 외부 문서 확장, 후보 수집·선별의 단일 항목 실행 검증, 답변에서 인용한 근거만 화면에 노출되는지 확인, 자연어 조건 탐색의 인덱스 후보 생성·LLM 선별, 브라우저 증빙입니다. 필수 기능을 완성한 뒤 F2015 방식으로 5개 이상 질문의 RAG 전후 답변과 인용 품질을 비교하는 것을 포트폴리오 차별화 목표로 둡니다. 측정 전에는 품질 수치를 주장하지 않습니다.
