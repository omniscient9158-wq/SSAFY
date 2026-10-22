# AqSolDB 화합물 근거 탐색

**화학 도메인 RAG · AqSolDB 근거 기반 화합물 탐색**  
앱 이름: **AqSolDB 수용해도 분석**

한글명·영문명·별칭으로 화합물을 찾고 AqSolDB 실험 기록과 검토 문서에 질문해, 답변에 실제 사용한 근거를 확인하는 화학 도메인 RAG 서비스입니다. LogS 회귀와 출처 그룹별 평가는 도메인 특화 보조 기능입니다.

> **프로젝트 질문:** 선택한 화합물에 대해 근거 문서로 답하고, 답변에 실제 사용한 데이터와 문서 출처를 확인할 수 있을까?

### 사용자가 여기서 할 수 있는 일

화합물을 찾아 실험에 기록된 수용해도 값을 확인하고, 검토된 문헌에 질문해 답변의 근거와 원문을 볼 수 있습니다. 분자량·LogP 조건으로 후보를 찾는 기능도 있습니다. 모델 예측값은 실험값과 따로 표시합니다.

이름 검색은 Train 7,488개 항목에서 가능합니다. 출처를 검토한 외부 발췌 9건을 여섯 화합물에서 색인했습니다. 문헌이 연결되지 않은 화합물은 AqSolDB 기록을 확인할 수 있어도 문헌 기반 설명은 보류합니다. 현재 범위는 모든 화합물의 용해 원인이나 pH·약물 흡수 효과를 설명하지 않습니다.

화면 위의 **등록된 화합물 찾아보기**에서 전체 목록, 문헌 준비 상태, 위키 문서 후보 링크를 확인하고 CSV로 받을 수 있습니다. Wikidata와 InChIKey가 한 항목에만 정확히 일치하는 6,281종의 한·영 이름·별칭을 공통 검색에 연결했습니다. 나이아신아마이드뿐 아니라 에탄올·글리세롤·아세톤도 한글로 찾을 수 있습니다. 이름 검색 자체에는 외부 API 호출이 없습니다.

전체 조회 결과 위키 문서 후보가 연결된 화합물은 2,116종입니다. 문서 연결 여부를 우선해 100종의 수집 후보 목록을 만들었으며, 후보 링크를 승인 문헌이나 RAG 색인으로 표시하지 않습니다. [출처·문서 범위 보고서](docs/03_출처_검토_보고서.md)에 수치와 한계를 기록했습니다.

초록과 연구 질문은 [프로젝트 개요 및 초록 초안](docs/00_프로젝트_개요_및_초록.md)을 참고하세요.

> 진행 상태: Train 메타데이터 100건과 외부 문서 9건을 색인했다. 타우린·카페인 질문으로 GMS 벡터 검색·답변 생성을 검증하고 인용을 원문 근거와 대조했다. Chrome에서 화합물 검색, 수치 조건 검색, 대화 기록 분리, LogS 보조 분석을 확인했다. 타우린 상세 카드, 문헌 검색 결과, 인용이 표시된 답변 화면을 `screenshots/09_taurine_compound.png`부터 `screenshots/11_taurine_answer_ui.png`까지 저장했다. AppTest에서 API 키 없음·문서 색인 없음 안내도 확인했다. 키·색인 누락과 생성 진행 상태도 실제 브라우저에서 확인했다. 아스피린 자동 수집 후보 한 건을 원문·이용 조건 검토 후 색인했다.

F2015 고정 5문항의 GMS 전후 비교를 실행하고 [검증 결과](docs/06_검증_결과.md)에 결과와 한계를 남겼다. 핵심 기록·인용을 승인 발췌와 대조했고 카페인 부가 설명 오류도 기록했다. 원본 답변·근거는 docs/evidence/에 있다.

## 데이터와 출처

문서 확보 대상 100종과 실제 문헌 답변 범위 6종의 차이는 [출처·문서 범위 보고서](docs/03_출처_검토_보고서.md)에 기록했습니다.

외부 근거 문서는 개별 원문·화합물 식별자·출처 표시·이용 조건을 확인합니다. 출처 검토를 마친 발췌 9건이 Chroma 벡터 색인에 들어 있습니다. 타우린(B-524) 자료는 3건, 카페인(C-533)은 NOAA CAMEO에 실린 NTP 1992 수용해도 기록입니다. A-4518 Wikipedia 자료는 CC BY-SA 4.0 보조 근거로 표시하고 정확성 경고를 함께 제공합니다. 검토 기록은 [출처 검토 보고서](docs/03_출처_검토_보고서.md)와 `research/provenance/`에서 확인합니다.

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

임베딩 없이 이름, 한글 별칭, AqSolDB ID, InChIKey, descriptor 값으로 로컬 메타데이터를 먼저 조회할 수 있습니다. 결과에는 데이터셋 DOI와 A-I 원자료 묶음 추적 정보가 함께 표시됩니다. 이는 아직 벡터 검색이나 RAG 답변이 아닙니다.

```powershell
python src/search_metadata.py "타우린" --limit 5
```

검토된 화합물의 외부 근거는 API 호출 없이 아래 명령으로 별도 JSONL 문서 컬렉션 입력 파일로 만들 수 있습니다. 색인은 `--index`를 직접 지정한 경우에만 임베딩 API를 호출합니다.

```powershell
python src/index_external_documents.py
# 임베딩 색인 실행을 결정한 뒤에만:
python src/index_external_documents.py --index
```

결과 파일: `data/curated_external_documents.jsonl`. 이 파일은 승인된 발췌 9건을 담고, 각 문서에 화합물 ID·InChIKey·원출처·인용 URL·검토 범위를 보존합니다. 이 자료 9건은 `indexes/aqsoldb_documents`에 색인했습니다. AqSolDB 메타데이터는 설명 문서 확보 기록이 있는 Train 100건을 `indexes/aqsoldb_compounds_linked`에 색인했고, 동일 명령 재실행 시 중복 추가가 없음을 확인했습니다.

외부 문서 후보 수집은 PubChem PUG-REST/PUG-View 또는 Wikimedia API를 사용하며 검색어 추출·후보 선별에 GMS 호출 2회가 발생합니다. 아래 명령의 `--provider pubchem` 경로는 Wikimedia 연락처가 필요하지 않습니다. Wikipedia까지 수집하려면 `--provider both`나 `wikipedia`를 쓰고 운영자 연락처를 로컬 `.env`의 `WIKIMEDIA_CONTACT`에 설정해야 합니다. 연락처가 없으면 Wikimedia 요청과 GMS 검색어 추출 전에 중지합니다.

후보는 `research/provenance/external_document_candidates.jsonl`에 검토 대기로 저장하며 자동 색인하지 않습니다. 원문·화합물 식별자·기여 출처·이용 조건을 확인한 뒤 승인 CSV에 필요한 발췌만 추가합니다. PubChem에 있다는 사실만으로 모든 기여 출처에 같은 이용 조건을 적용하지 않습니다.

```powershell
# 단일 AqSolDB 항목의 후보 탐색 (GMS 2회 및 외부 API 요청; 색인 임베딩은 하지 않음)
python src/discover_wikipedia_candidates.py E-1274 --provider pubchem --question "아스피린의 물 용해도와 물성에 대해 설명해줘"
```

`--provider pubchem`은 Wikimedia 연락처 없이 PubChem만 조회합니다. `--provider both` 또는 `wikipedia`는 WIKIMEDIA_CONTACT가 필요합니다. 제공자를 구분해 캐시하므로 서로 다른 출처의 결과를 같은 실행으로 취급하지 않습니다.

LLM 선별 결과는 자동 승인 근거가 아닙니다. 출처와 이용 조건을 확인하기 전에는 RAG 컬렉션에 넣지 않습니다.

선택한 화합물의 메타데이터와 관련 외부 발췌를 로컬에서 함께 검색하려면 정확한 AqSolDB ID와 질문을 전달합니다.

```powershell
python src/retrieve_local_evidence.py B-524 "타우린은 왜 물에 잘 녹나요?"
```

이 검색기는 현재 단어 일치 기반의 로컬 프로토타입입니다. 메타데이터와 외부 근거를 별도 결과로 반환하고, 외부 문서는 선택 ID가 같은 자료만 찾습니다. 벡터 검색은 별도 색인이 필요합니다.

## 시연 UI

```powershell
streamlit run app.py
```

앱에서 화합물 이름이나 AqSolDB ID를 선택하면 해당 화합물의 데이터를 근거로 질문할 수 있습니다. 현재 Chroma 문헌 색인은 A-454, A-5538, A-4518, B-524, C-533, E-1274에 연결된 승인 발췌 9건을 포함합니다. 질문 예시는 선택한 항목의 검토 문헌을 탐색하는 용도입니다. 후보 수집 스크립트는 별도 CLI이고 인터넷 전체나 PubChem 전체를 검색하지 않습니다.

**문헌 검색 흐름:** 화면 위에서 화합물을 검색한 뒤, 결과 아래의 `더 알아보기`에서 속성·조건·원문 표현을 질문합니다. 예를 들어 A-4518은 `2-methylpyridine water solubility mg/mL 68°F NTP 1992 원출처는 무엇인가요?`, C-533은 `카페인의 문헌상 수용해도 값, 단위, 온도와 원출처를 알려주세요.`라고 질문할 수 있습니다. A-454는 `tris(2-butoxyethyl) phosphate yellow viscous liquid 설명과 출처를 알려주세요.`, A-5538은 `Anthracene sinks in water라는 설명의 출처와 문맥은 무엇인가요?`도 사용할 수 있습니다. 화면의 예시 질문을 선택하면 입력창에 자동으로 채워집니다.

기본 **단어 일치** 검색은 질문과 발췌의 단어를 비교하며 GMS 크레딧을 쓰지 않습니다. 외부 문서의 영어 원문 표현을 질문에 함께 넣으면 검색에 유리합니다. 고급 설정의 **의미 유사도** 검색은 GMS 임베딩 크레딧을 사용하고 실행 전에 동의를 받습니다. 답변 생성도 별도로 동의를 받은 경우에만 GMS LLM을 호출합니다. 검색 결과가 없으면 현재 발췌에서 일치 항목을 찾지 못했다는 뜻이며, 전체 문헌에 정보가 없다고 단정할 수 없습니다.

조건 검색 지원 예: `분자량 300 이하, LogP 2 미만`, `LogS -4 초과`, `TPSA 20~80`. 조건은 허용 필드·연산자로 변환한 뒤 `train.csv`에서 다시 수치 비교하며, 제공 test 분할은 검색에 포함하지 않습니다.

## 폴더 안내

```text
ai-pjt-2/
├─ app.py                # Streamlit 앱
├─ data/                 # AqSolDB와 승인된 문헌 발췌
├─ docs/                 # 요구사항, 설계, 출처 검토, 체크리스트, 회고
│  └─ reference/         # 원본 체크리스트 PDF
├─ src/                  # 데이터·검색·색인·모델·RAG 코드
├─ research/provenance/  # 출처 조사와 후보 자료
├─ reference/            # 강의 스켈레톤, API 예제, 영화 템플릿 압축본
├─ .env.example          # 환경변수 템플릿
└─ requirements.txt      # 실행에 필요한 패키지
```

## 제출물 구성

제출용 압축 파일에는 `app.py`, `README.md`, `requirements.txt`, `.env.example`, `data/`, `docs/`, `src/`, `screenshots/`를 포함하세요. `reference/`는 강의 참고 자료라 제출본에서 제외하세요. `docs/reference/`에는 대조에 사용한 원본 체크리스트 PDF가 있습니다. `research/`는 출처 감사 추적용 보조자료이며, 필요 시 보고서만 별도 참고 자료로 제출하세요. `.env`, `.venv/`, `indexes/`, `reference/`, `__pycache__/`는 제출하지 않습니다. 폴더 전체를 압축하기보다 이 목록을 골라 담으세요.

## 현재 상태와 다음 작업

체크리스트별 진행 상태는 [현재 상태표](docs/04_체크리스트_현재상태.md)에서 확인할 수 있습니다. 브라우저에서 화합물별 대화 기록, 수치 조건 검색, LogS 보조 분석은 확인했습니다. 생성 답변 인용과 원문 링크, API 키·색인 누락 안내는 실제 브라우저에서 확인했습니다. 문서 확보 대상 100종 중 94종의 관련성·이용 조건 검토가 남아 있습니다. 자동 수집한 문서는 원출처와 이용 조건을 사람이 확인한 뒤에만 색인합니다. F2015 5문항 비교는 실행·기록했습니다. 보조 분석의 문헌 수용해도 기록 비교를 AppTest와 실제 브라우저에서 확인했고 screenshots/12_measurement_comparison.png에 저장했습니다.


## 로컬 제출본 만들기

먼저 유료 모델 호출 없이 기본 동작을 재검증할 수 있습니다.

```powershell
python src/verify_local.py
```

문서 링크, 전체 선택 목록과 질문 예시 개수, 수치 조건 검색, 화합물 전환·예시 입력을 검사합니다. LLM·임베딩 클라이언트 생성을 차단하며, 실제 생성 답변 품질이나 브라우저 시각 검증을 대신하지 않습니다.

```powershell
python src/package_submission.py
```

`outputs/aqsoldb_submission.zip`을 만듭니다. 실행에 필요한 승인 CSV와 문서 확보 목록은 data에 포함하고 실제 키, 가상환경, 기존 색인, 미검토 조사자료는 제외합니다. 별도 복사본에서 키·색인·research 없이 이름 검색·문헌 검색·기록 비교가 실행되는 것을 확인했습니다. 새 환경에서 의미 검색·답변 생성은 본인 GMS 설정과 색인 준비가 필요합니다. 이 압축 파일은 문헌 94종의 검토가 남아 있는 제출본 초안입니다.
