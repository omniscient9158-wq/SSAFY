"""AqSolDB aqueous-solubility analysis and evidence review interface."""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import compound_data
import config  # loads the project's .env before checking whether the GMS key exists
from filter_compounds import search_train_compounds
from retrieve_local_evidence import retrieve
from solubility_model import predict_for_compound, similar_descriptor_compounds
from smiles_prediction import molecular_formula, molecule_image, predict_smiles

st.set_page_config(
    page_title="AqSolDB 화합물 근거 탐색",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("AqSolDB 화합물 근거 탐색")
st.caption("화합물을 선택하고, AqSolDB 기록과 출처가 확인된 문헌을 근거로 질문하세요.")
st.badge("AqSolDB v1 · 전체 9,982행 · 앱 검색 Train 7,488행", icon=":material/database:", color="blue")

DEMO_IDS = ("A-454", "A-5538", "A-4518")
TRAIN_ROWS = [row for row in compound_data.load_compounds() if row["_split"] == "train"]
rows = {row["ID"]: row for row in TRAIN_ROWS}
ALL_IDS = tuple(sorted(rows))
labels = {item_id: f"{rows[item_id]['Name']} · {item_id}" for item_id in ALL_IDS}
QUESTION_EXAMPLES = {
    "A-454": [
        ("외관 발췌 찾기 · 액체", "tris(2-butoxyethyl) phosphate physical description liquid 원출처는 무엇인가요?"),
        ("색상·점도 발췌 찾기", "tris(2-butoxyethyl) phosphate yellow viscous liquid 설명과 출처를 알려주세요."),
    ],
    "A-5538": [
        ("외관 발췌 찾기 · 고체", "Anthracene general description white yellow solid 원출처는 무엇인가요?"),
        ("물에서의 거동 발췌 찾기", "Anthracene sinks in water라는 설명의 출처와 문맥은 무엇인가요?"),
    ],
    "A-4518": [
        ("수용해도·측정 조건 찾기", "2-methylpyridine water solubility mg/mL 68°F NTP 1992 원출처는 무엇인가요?"),
        ("단위·온도 포함 발췌 찾기", "2-methylpyridine의 water solubility 기록에서 값, 단위, 온도, 인용 연도를 확인해 주세요."),
    ],
}
COMMON_QUESTION_EXAMPLES = [
    ("현재 레코드의 실험값·계보", "이 화면에 표시된 AqSolDB 레코드의 실험 LogS, 원자료 그룹, 데이터셋 DOI를 확인해 주세요."),
    ("외부 발췌의 범위 확인", "검색된 검토 문헌 발췌가 직접 뒷받침하는 내용과 제외된 범위는 무엇인가요?"),
]


def solubility_band(logS: float) -> tuple[str, str]:
    if logS >= 0:
        return "높은 용해도", "물에 잘 녹는 편"
    if logS >= -2:
        return "용해성", "물에 녹는 편"
    if logS >= -4:
        return "약한 용해성", "물에 잘 녹지 않는 편"
    return "불용성 구간", "물에 거의 녹지 않는 편"


def set_compound_query(query: str) -> None:
    st.session_state["compound-query"] = query
    st.session_state["compound-query-requested"] = True


def analyze_candidate() -> None:
    candidate_id = st.session_state.get("compound-candidate-id")
    if candidate_id:
        set_compound_query(candidate_id)


def apply_question_example(compound_id: str) -> None:
    example_key = f"question-example-{compound_id}"
    input_key = f"question-input-{compound_id}"
    selected_example = st.session_state.get(example_key)
    examples = QUESTION_EXAMPLES.get(compound_id, []) + COMMON_QUESTION_EXAMPLES
    question_text = next((text for label, text in examples if label == selected_example), None)
    if question_text:
        st.session_state[input_key] = question_text


with st.container(border=True):
    st.subheader("어떤 화합물을 분석할까요?")
    st.caption("화합물 이름이나 AqSolDB ID를 검색하거나, SMILES를 입력해 새 구조의 수용해도를 예측하세요.")
    st.caption("LogS는 수용해도를 나타내는 로그 값으로, 값이 클수록 물에 더 잘 녹습니다.")
    with st.form("compound-search-form"):
        compound_query = st.text_input(
            "화합물 이름 또는 SMILES",
            key="compound-query",
            placeholder="예: caffeine, A-454 또는 CC(=O)Oc1ccccc1C(=O)O",
        )
        query_submitted = st.form_submit_button("분석하기", type="primary", icon=":material/search:")
query_requested = query_submitted or st.session_state.pop("compound-query-requested", False)
if query_requested:
    query = st.session_state.get("compound-query", "").strip()
    st.session_state.pop("compound-analysis", None)
    st.session_state.pop("compound-analysis-error", None)
    st.session_state.pop("compound-search-candidates", None)
    if not query:
        st.session_state["compound-analysis-error"] = "화합물 이름, AqSolDB ID 또는 SMILES를 입력해 주세요."
    else:
        query_aliases = {
            "aspirin": "acetylsalicylic_acid",
            "아스피린": "acetylsalicylic_acid",
            "acetylsalicylic acid": "acetylsalicylic_acid",
        }
        lookup_query = query_aliases.get(query.casefold(), query)
        row_matches = [
            row for row in TRAIN_ROWS
            if lookup_query.casefold() in {row["ID"].casefold(), row["Name"].casefold(), row["SMILES"].casefold()}
        ]
        if not row_matches:
            row_matches = [
                row for row in TRAIN_ROWS
                if lookup_query.casefold() in row["Name"].casefold() or lookup_query.casefold() in row["ID"].casefold()
            ]
        if len(row_matches) == 1:
            st.session_state["compound-analysis"] = {"kind": "record", "compound_id": row_matches[0]["ID"]}
        elif len(row_matches) > 1:
            st.session_state["compound-search-candidates"] = [row["ID"] for row in row_matches[:100]]
        else:
            display_name, smiles = ("입력한 구조", query)
            try:
                molecule = molecule_image(smiles, size=(520, 360))
                formula = molecular_formula(smiles)
                prediction = predict_smiles(smiles)
                st.session_state["compound-analysis"] = {
                    "kind": "smiles", "name": display_name, "smiles": smiles,
                    "formula": formula, "image": molecule, "prediction": prediction,
                }
            except RuntimeError:
                st.session_state["compound-analysis-error"] = "SMILES 구조 분석 기능을 사용할 수 없습니다. RDKit 설치 상태를 확인해 주세요."
            except ValueError:
                st.session_state["compound-analysis-error"] = (
                    "검색 결과가 없습니다. 화합물 이름이나 AqSolDB ID를 확인해 주세요. "
                    "새 분자라면 SMILES의 원자·결합 표기를 확인해 주세요."
                )

if st.session_state.get("compound-analysis-error"):
    st.warning(st.session_state["compound-analysis-error"])

candidate_ids = st.session_state.get("compound-search-candidates", [])
if candidate_ids:
    st.info(f"이름이 일부 일치하는 화합물 {len(candidate_ids)}개를 찾았습니다. 목록에서 하나를 선택해 주세요.")
    st.selectbox(
        "검색 결과",
        candidate_ids,
        format_func=lambda value: labels[value],
        key="compound-candidate-id",
    )
    st.button("선택한 화합물 분석", on_click=analyze_candidate, type="primary")

with st.sidebar:
    st.header("분석 설정")
    st.caption("화합물 이름·ID 검색과 새 SMILES 예측은 화면 위쪽에서 시작합니다.")
    with st.expander("분자 특성 조건으로 찾기", icon=":material/filter_alt:"):
        st.caption("예: 분자량 300 이하, LogP 2 미만")
        with st.form("compound-filter-form"):
            filter_query = st.text_input("조건", placeholder="분자량 300 이하, LogP 2 미만")
            filter_submit = st.form_submit_button("조건 검색", icon=":material/search:", width="stretch")
        if filter_submit:
            try:
                st.session_state["compound_filter_result"] = search_train_compounds(filter_query)
                st.session_state.pop("compound_filter_error", None)
            except ValueError as exc:
                st.session_state["compound_filter_error"] = str(exc)
        if st.session_state.get("compound_filter_error"):
            st.warning(st.session_state["compound_filter_error"])
        filter_result = st.session_state.get("compound_filter_result")
        if filter_result:
            st.caption(f"Train 데이터에서 조건에 맞는 화합물 {filter_result['total_matches']:,}개")
            if filter_result["items"]:
                candidate_id = st.selectbox(
                    "검색 결과에서 분석 대상 고르기",
                    [item["ID"] for item in filter_result["items"]],
                    format_func=lambda value: labels.get(value, value),
                    key="filter_candidate_id",
                )
                st.button(
                    "이 화합물 분석",
                    on_click=set_compound_query,
                    args=(candidate_id,),
                    icon=":material/arrow_forward:",
                    width="stretch",
                )
    with st.expander("고급 설정 · 문헌 검색", icon=":material/tune:"):
        st.caption("기본 검색은 질문에 쓴 단어가 문헌에 있는지 확인합니다.")
        retrieval_choice = st.segmented_control(
            "문헌 검색 방식", options=["단어 일치", "의미 유사도"], default="단어 일치",
            key="retrieval-choice", width="stretch",
        )
        retrieval_mode = "vector" if retrieval_choice == "의미 유사도" else "local"
        embedding_consent = False
        if retrieval_mode == "vector":
            st.caption("의미 유사도 검색은 GMS 임베딩 크레딧을 사용합니다.")
            embedding_consent = st.checkbox("GMS 임베딩 호출 허용", key="vector-search-consent")
            top_k = st.slider("문헌 결과 수", min_value=1, max_value=10, value=3)
        else:
            top_k = 3

analysis = st.session_state.get("compound-analysis")
selected_id = analysis.get("compound_id") if analysis and analysis["kind"] == "record" else None
if analysis and analysis["kind"] == "record":
    row = rows[selected_id]
    source_code = selected_id.split("-", 1)[0]
    with st.container(border=True):
        image_col, result_col = st.columns([1, 2], vertical_alignment="center")
        image_col.image(molecule_image(row["SMILES"], size=(520, 360)), caption="분자 구조", width="stretch")
        result_col.subheader(row["Name"].replace("_", " "))
        result_col.badge(f"AqSolDB ID · {selected_id}", color="blue")
        result_col.metric("AqSolDB 실험 LogS", f"{float(row['Y']):.2f}")
        measured_band, _ = solubility_band(float(row["Y"]))
        source = compound_data.source_info(row)
        source_name = source.get("Original_source_dataset") or f"통합 출처 코드 {source_code}"
        result_col.caption(f"데이터셋에 기록된 값 · 해석 구간: {measured_band}")
        result_col.caption(f"원자료 그룹 {source_code} · {source_name}")
        if selected_id not in DEMO_IDS:
            result_col.caption("외부 문헌은 아직 출처 검토·색인을 마치지 않았습니다. 지금은 AqSolDB 데이터 기록을 질문할 수 있습니다.")
        with st.expander("분자 식별 정보"):
            st.code(row["SMILES"], language=None)
            st.caption(f"InChIKey: {row['InChIKey']}")
        st.caption("AqSolDB 실험 기록입니다. 아래 문헌 검색 결과와 출처를 구분해 확인할 수 있습니다.")
elif analysis and analysis["kind"] == "smiles":
    prediction = analysis["prediction"]
    band, plain_summary = solubility_band(prediction["predicted_logS"])
    with st.container(border=True):
        image_col, result_col = st.columns([1, 2])
        image_col.image(analysis["image"], caption=f"2D 구조 · {analysis['formula']}", width="stretch")
        result_col.subheader(analysis["name"])
        result_col.write(f"모델 예측: **{plain_summary}** · 해석 구간: **{band}**")
        result_col.metric("예측 LogS", f"{prediction['predicted_logS']:.2f}")
        result_col.caption("새 구조라 AqSolDB 실험값과 연결되지 않았습니다. 표시한 용해도 구간은 분류 확률이 아닙니다.")
        st.code(analysis["smiles"], language=None)

    st.caption(
        f"용해도 구간은 [AqSolDB 논문](https://www.nature.com/articles/s41597-019-0151-1)에 제시된 LogS 범주를 참고했습니다. "
        "현재 모델은 연속형 LogS를 예측하므로 표시된 구간은 확률이 아닙니다."
    )
elif not analysis:
    st.info("위 검색창에 화합물 이름·AqSolDB ID 또는 SMILES를 입력해 시작하세요.")

if selected_id:
    with st.container(border=True):
        st.subheader("근거 기반 질문")
        st.caption("선택한 화합물의 AqSolDB 기록은 항상 포함됩니다. 외부 문헌은 현재 출처·이용 조건 검토를 마친 자료만 검색합니다.")
        with st.expander("무엇을, 어떻게 검색하나요?", icon=":material/help:"):
            st.markdown(
                "**검색 순서**\n"
                "1. 화면 위 검색창에서 화합물을 선택합니다. 현재 검토된 외부 자료는 A-454 1건, A-5538 1건, A-4518 2건입니다. A-4518의 Wikipedia 자료는 정량값의 확정 근거가 아닌 보조 자료입니다.\n"
                "2. 아래 예시처럼 찾고 싶은 속성이나 원문 표현을 질문에 넣습니다. 로컬 검색은 질문 속 단어가 발췌에 실제로 포함될수록 잘 찾습니다. 영어 원문 용어도 함께 쓰면 유리합니다.\n"
                "3. 결과에서 발췌, 원출처, 링크, 인용 범위를 확인합니다. 결과 없음은 현재 승인된 발췌에서 일치 항목을 찾지 못했다는 뜻입니다.\n\n"
                "**검색 범위**\n"
                "- 외부 근거 검색은 선택한 화합물에 연결된 출처 확인 발췌만 조회합니다. 인터넷 전체나 모든 PubChem 자료를 검색하지 않습니다.\n"
                "- AqSolDB의 LogS·원자료 그룹·DOI는 선택 레코드 정보로 함께 표시됩니다. 문헌 검색 결과와는 별도 출처입니다.\n"
                "- 단어 일치 검색은 GMS 크레딧을 사용하지 않습니다. 의미 유사도 검색은 GMS 임베딩 동의가 필요하고, 답변 생성도 별도 동의를 받습니다."
            )
        st.caption("분자량·LogP 조건으로 여러 화합물을 찾으려면 왼쪽의 **수치 조건으로 화합물 찾기**를 사용하세요. 여기서는 현재 선택한 화합물에 대한 문헌 내용을 찾습니다.")
        examples = QUESTION_EXAMPLES.get(selected_id, []) + COMMON_QUESTION_EXAMPLES
        question_key = f"question-input-{selected_id}"
        st.selectbox(
            "질문 예시 · 선택하면 입력창에 채워집니다",
            ["직접 입력", *[label for label, _ in examples]],
            key=f"question-example-{selected_id}",
            on_change=apply_question_example,
            args=(selected_id,),
        )
        st.caption("예시를 고르면 질문이 입력됩니다. 직접 쓸 때는 찾을 속성이나 문헌의 영어 원문 표현을 넣어 주세요.")
        with st.form(f"query-form-{selected_id}"):
            question = st.text_input(
                "질문",
                key=question_key,
                placeholder="질문을 입력하거나 위에서 예시를 선택하세요",
            )
            submitted = st.form_submit_button(
                "근거 검색",
                type="primary",
                icon=":material/search:",
                width="stretch",
            )

        if submitted:
            if not question.strip():
                st.warning("질문을 입력해 주세요.")
            elif retrieval_mode == "vector" and not embedding_consent:
                st.warning("벡터 검색을 실행하려면 임베딩 API 사용에 동의해야 합니다.")
            elif retrieval_mode == "vector" and not os.getenv("GMS_KEY", "").strip():
                st.error("GMS API 키가 설정되지 않았습니다. 프로젝트 .env 파일에 GMS_KEY를 설정해 주세요.")
            else:
                try:
                    with st.status("관련 근거를 조회하고 있습니다…", expanded=False) as status:
                        if retrieval_mode == "local":
                            result = retrieve(selected_id, question, limit=top_k)
                        else:
                            import config
                            import vector_store

                            if vector_store.count_persisted(config.DOC_COLLECTION) == 0:
                                raise RuntimeError(
                                    "외부 문서 색인이 없습니다. 색인을 준비한 뒤 다시 시도해 주세요."
                                )
                            from rag_graph import retrieve_node

                            result = retrieve_node({
                                "compound_id": selected_id,
                                "query": question,
                                "retrieval_mode": "vector",
                                "top_k": top_k,
                            })
                            result.update({
                                "query": question,
                                "retrieval_method": "Chroma vector similarity; exact AqSolDB ID filter",
                                "external_evidence_status": (
                                    "matched" if result["external_evidence"]
                                    else "no vector-matched external excerpt"
                                ),
                            })
                        result["retrieval_mode"] = retrieval_mode
                        history_by_compound: dict[str, list[dict[str, Any]]] = st.session_state.setdefault(
                            "history_by_compound", {}
                        )
                        history_by_compound.setdefault(selected_id, []).append({
                            "turn_id": len(history_by_compound.get(selected_id, [])),
                            "question": question.strip(),
                            "result": result,
                            "answer": None,
                        })
                        count = len(result["external_evidence"])
                        status.update(
                            label=f"화합물 메타데이터와 외부 근거 {count}건을 확인했습니다.",
                            state="complete",
                        )
                except RuntimeError as exc:
                    if "GMS_KEY" in str(exc):
                        st.error("GMS API 키가 설정되지 않았습니다. 프로젝트 .env 파일에 GMS_KEY를 설정해 주세요.")
                    else:
                        st.error(str(exc))
                except Exception as exc:
                    st.error(f"근거 검색을 완료하지 못했습니다. 설정을 확인해 주세요. ({type(exc).__name__})")

        history_by_compound = st.session_state.setdefault("history_by_compound", {})
        history = history_by_compound.setdefault(selected_id, [])
        for turn in reversed(history):
            turn_key = turn.get("turn_id", 0)
            result = turn["result"]
            cited_ids = set(turn.get("cited_evidence_ids", [])) if turn.get("answer") else set()

            with st.container(border=True):
                st.markdown(f"**질문**  \n{turn['question']}")
                if turn.get("answer"):
                    st.markdown("#### 답변")
                    st.markdown(turn["answer"])
                else:
                    st.caption("검색 결과입니다. 근거 기반 답변 생성은 별도 동의가 필요합니다.")

                metadata = result["compound_metadata"]
                if not turn.get("answer") or "근거 1" in cited_ids:
                    st.markdown("#### AqSolDB 화합물 메타데이터 · 근거 1")
                    st.write(
                        f"실험 LogS **{metadata['experimental_logS']}** · "
                        f"원자료 그룹 **{metadata['source_group_code']}** "
                        f"({metadata['upstream_source_dataset']}) · "
                        f"[데이터셋 DOI]({metadata['dataset_doi']})"
                    )

                external = result["external_evidence"]
                if turn.get("answer") and cited_ids:
                    external = [item for item in external if item["citation_id"] in cited_ids]
                if external:
                    st.markdown("#### 출처 확인 문헌 내용")
                    for item in external:
                        title = item.get("document_title") or item.get("excerpt", "").split(":", 1)[0]
                        st.markdown(f"**{item['citation_id']} · {title}**")
                        st.write(item["excerpt"])
                        st.caption(
                            f"원출처: {item['original_attribution']} · "
                            f"확인일: {item['reviewed_on']}"
                        )
                        st.markdown(f"[원문 보기]({item['source_url']})")
                        st.caption(f"이용 조건: {item.get('source_terms_status', '자료에 별도 표기 없음')}")
                        terms_url = item.get("terms_basis_url")
                        if terms_url:
                            st.markdown(f"[이용 조건 원문]({terms_url})")
                        st.caption(f"인용 범위: {item['scope_note']}")
                elif turn.get("answer") and not cited_ids:
                    st.warning("생성 답변에 확인된 인용이 없습니다. 답변을 근거로 사용하기 전에 원문을 확인해 주세요.")
                else:
                    st.info("이 질문과 일치하는 출처 확인 문헌을 찾지 못했습니다.")

                st.checkbox(
                    "GMS 답변 생성 API 호출에 동의합니다",
                    key=f"api-consent-{selected_id}-{turn_key}",
                    help="선택한 질문과 검색 근거가 GMS API로 전송되며 사용량이 발생할 수 있습니다.",
                )
                consent_key = f"api-consent-{selected_id}-{turn_key}"
                if st.button(
                    "근거 기반 답변 생성",
                    key=f"generate-{selected_id}-{turn_key}",
                    disabled=not st.session_state.get(consent_key, False) or bool(turn.get("answer")),
                    icon=":material/auto_awesome:",
                    width="stretch",
                ):
                    try:
                        if not os.getenv("GMS_KEY", "").strip():
                            st.error("GMS API 키가 설정되지 않았습니다. 프로젝트 .env 파일에 GMS_KEY를 설정해 주세요.")
                            continue
                        from rag_graph import run_rag

                        with st.spinner("근거에 기반해 답변을 작성하고 있습니다…"):
                            output = run_rag(
                                selected_id,
                                turn["question"],
                                retrieval_mode=result.get("retrieval_mode", "local"),
                                preloaded_result=result,
                            )
                        turn["answer"] = output.get("answer", "")
                        turn["cited_evidence_ids"] = output.get("cited_evidence_ids", [])
                        st.rerun()
                    except Exception as exc:
                        st.error(f"답변 생성에 실패했습니다. 연결 설정을 확인해 주세요. ({type(exc).__name__})")

        if not history:
            with st.container(border=True):
                st.markdown("**검색 결과 읽기**")
                st.caption(
                    "선택 화합물의 AqSolDB 메타데이터는 항상 표시됩니다. "
                    "외부 문헌 결과가 없으면 현재 승인·색인된 발췌에서 질문과 일치하는 근거를 찾지 못했다는 뜻입니다."
                )


elif analysis and analysis["kind"] == "smiles":
    with st.expander("더 알아보기: 문헌 출처", expanded=False):
        st.info("새 SMILES 구조는 AqSolDB ID와 연결되지 않아 이 프로젝트에서 확인한 문헌 근거가 없습니다.")

if selected_id:
    with st.expander("보조 분석 · LogS 모델 예측 및 유사 화합물", expanded=False):
        prediction = predict_for_compound(selected_id)
        band, plain_summary = solubility_band(prediction["predicted_logS"])
        st.write(f"**{plain_summary}** · 해석 구간: {band}")
        predicted_col, measured_col, error_col = st.columns(3)
        predicted_col.metric("모델 예측 LogS", f"{prediction['predicted_logS']:.2f}")
        measured_col.metric("AqSolDB 실험 LogS", f"{prediction['measured_logS']:.2f}")
        error_col.metric("이 레코드의 절대 차이", f"{prediction['absolute_error']:.2f}")
        st.caption(
            f"기준선: {prediction['method']} · 학습 {prediction['fit_rows']:,}행 · "
            f"제외 출처 그룹 {prediction['held_out_source_code']} ({prediction['heldout_rows']:,}행). "
            "연속형 LogS 회귀 결과이며, 이 레코드 한 건의 차이는 전체 모델 성능을 뜻하지 않습니다."
        )
        st.markdown(f"**실험 데이터 출처** · [AqSolDB 데이터셋 DOI]({compound_data.DATASET_DOI})")
        try:
            from external_documents import load_reviewed_documents

            contextual_sources = [
                record["metadata"]
                for record in load_reviewed_documents()
                if record["metadata"]["compound_id"] == selected_id
            ]
        except (OSError, ValueError, KeyError):
            contextual_sources = []
        if contextual_sources:
            st.markdown("**이 화합물의 출처 확인 문헌**")
            for source in contextual_sources:
                st.markdown(
                    f"- [{source['document_title']}]({source['source_url']}) · "
                    f"원출처: {source['original_attribution']}"
                )
            st.caption(
                "이 문헌은 선택 화합물의 참고 자료입니다. 모델 예측 자체는 AqSolDB train 데이터로 적합한 기준선 결과이며, "
                "문헌이 예측 원인이나 인과관계를 증명하지 않습니다."
            )
        else:
            st.info("이 화합물에 연결된 출처 확인 외부 문헌은 아직 없습니다. 실험 데이터 출처와 모델 기준선만 표시합니다.")
        neighbors = similar_descriptor_compounds(selected_id)
        st.dataframe(neighbors, width="stretch", hide_index=True)

st.caption(
    "데이터 출처: AqSolDB v1 · "
    f"[논문 및 데이터셋 정보]({compound_data.DATASET_DOI}) · "
    "예측값과 실험값, 원자료와 외부 문헌은 구분해 표시합니다."
)
