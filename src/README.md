# src 폴더 안내

앱은 루트의 `app.py`로 실행한다. 이 폴더에는 데이터 처리, 검색, 답변 생성 코드가 있다.

## 앱 실행에 필요한 파일
- `config.py`: 설정(.env, 컬렉션 이름, 청크 크기, Top-k)
- `llm.py`: GMS 모델과 임베딩 연결
- `vector_store.py`: Chroma 컬렉션 열기, 개수 확인
- `compound_data.py`: AqSolDB 읽기, 화합물 1개를 검색용 텍스트로 만들기
- `compound_catalog.py`: 화합물 목록과 문헌 유무 표시
- `external_documents.py`: 승인된 외부 문서 읽기
- `retrieve_local_evidence.py`: 선택한 화합물의 근거 검색
- `rag_graph.py`: retrieve → generate 두 노드 그래프
- `filter_compounds.py`: 분자량, LogP 같은 숫자 조건 검색
- `compare_measurements.py`: 문헌 값과 AqSolDB 값 비교
- `solubility_model.py`, `smiles_prediction.py`: LogS 예측(보조 기능). 둘 다 `source_aware_evaluation.py`의 함수를 쓴다.

## 데이터 준비 스크립트
- `index_metadata.py`, `index_external_documents.py`: 두 컬렉션 색인
- `discover_wikipedia_candidates.py`: 검색어 추출, 후보 수집, 선별
- `wikidata_coverage.py`, `document_coverage.py`: 이름 연결과 문서 후보 목록 만들기

## 검증과 보조 도구
- `compare_rag.py`: RAG 전후 비교(F2015)
- `baseline_solubility.py`, `source_aware_evaluation.py`: 회귀 기준선, 출처별 평가
- `verify_local.py`: 로컬 점검 테스트
- `review_document_candidates.py`, `search_metadata.py`, `package_submission.py`: 출처 점검, 이름 검색, 제출 zip 만들기