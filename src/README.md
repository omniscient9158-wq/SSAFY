# `src/` 모듈 안내

실행 진입점은 저장소 루트의 [`app.py`](../app.py)입니다. 이 폴더에는 화면 코드와 분리된 데이터·검색·분석 로직 및 검증용 CLI를 둡니다.

## 앱에서 사용하는 핵심 모듈

| 모듈 | 역할 |
|---|---|
| `config.py` | `.env` 기반 실행 설정과 컬렉션 이름을 관리합니다. |
| `compound_data.py` | AqSolDB 원자료, 이름·별칭·Wikidata 연결을 읽습니다. |
| `compound_catalog.py` | Train 화합물 목록과 문헌 준비 상태를 만듭니다. |
| `filter_compounds.py` | 한국어 수치 조건을 파싱하고 Train 후보를 찾습니다. |
| `retrieve_local_evidence.py` | AqSolDB 메타데이터와 승인 외부 발췌를 로컬 검색합니다. |
| `rag_graph.py` | 검색 결과를 바탕으로 답변을 생성하는 retrieve/generate 그래프입니다. |
| `vector_store.py` | Chroma 색인 상태와 임베딩 저장소를 다룹니다. |
| `external_documents.py` | 검토 완료 외부 문헌만 읽고 색인 입력으로 제공합니다. |
| `solubility_model.py` | descriptor 기반 LogS 보조 예측과 유사 화합물 분석을 제공합니다. |
| `smiles_prediction.py` | SMILES 구조 이미지·분자식·보조 예측을 제공합니다. |

## 데이터 준비·검증 CLI

`index_metadata.py`, `index_external_documents.py`, `discover_wikipedia_candidates.py`,
`document_coverage.py`, `wikidata_coverage.py`는 데이터 준비와 출처 검토를 위한 일회성/재현 가능한 명령입니다.
`compare_measurements.py`, `compare_rag.py`, `baseline_solubility.py`,
`source_aware_evaluation.py`는 검증과 보고서 생성에 사용합니다.
`package_submission.py`는 제출 ZIP을 만듭니다.
`verify_local.py`는 문서 링크·Train/질문 개수·조건 검색·화합물 선택 및 질문 자동 입력을 검사합니다. LLM·임베딩 클라이언트 생성은 테스트 중 차단합니다.

이 파일들은 `app.py`에서 직접 import되지 않더라도 README와 체크리스트의 실행 근거를 재생성하는 데 필요하므로 임의로 삭제하지 않습니다.
