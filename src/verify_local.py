"""Run repeatable local checks, blocking LLM and embedding client creation."""
from __future__ import annotations

from pathlib import Path
import re
import unittest
from unittest.mock import patch
from urllib.parse import unquote

import compound_data
from filter_compounds import search_train_compounds
from question_examples import COMMON_QUESTION_EXAMPLES

ROOT = Path(__file__).resolve().parents[1]


class LocalChecks(unittest.TestCase):
    def test_document_links(self):
        documents = [ROOT / "README.md", ROOT / "src/README.md", *sorted((ROOT / "docs").glob("*.md"))]
        for document in documents:
            # Markdown inline links; external URLs and same-page anchors are excluded.
            for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
                target = unquote(target.strip("<> ").split("#", 1)[0])
                if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / target).exists(), target)

    def test_train_and_question_counts(self):
        train = [row for row in compound_data.load_compounds() if row["_split"] == "train"]
        self.assertEqual(len(train), 7488)
        self.assertEqual(len(COMMON_QUESTION_EXAMPLES), 36)
        self.assertEqual(len({text for _, text in COMMON_QUESTION_EXAMPLES}), 36)

    def test_numeric_search(self):
        result = search_train_compounds("분자량 300 이하, LogP 2 미만", limit=200)
        self.assertEqual(result["total_matches"], 3175)
        self.assertTrue(result["train_only"])
        self.assertTrue(all(float(row["MolWt"]) <= 300 and float(row["MolLogP"]) < 2 for row in result["items"]))
        empty = search_train_compounds("분자량 0 이하")
        self.assertEqual(empty["total_matches"], 0)
        self.assertEqual(empty["items"], [])
        for query in ("TPSA 80~20", "LogP 2 미만 또는 분자량 300 이하", "지원안함 3 이하"):
            with self.subTest(query=query), self.assertRaises(ValueError):
                search_train_compounds(query)

    def test_selection_and_question_input(self):
        from streamlit.testing.v1 import AppTest

        with patch("llm.get_llm", side_effect=AssertionError("LLM forbidden in local checks")) as chat, patch(
            "llm.get_embeddings", side_effect=AssertionError("Embeddings forbidden in local checks")
        ) as embeddings:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.selectbox(key="quick-compound").options), 7488)
            for compound_id in ("B-524", "C-533", "B-524"):
                app.selectbox(key="quick-compound").select(compound_id).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.session_state["compound-analysis"]["compound_id"], compound_id)
                label, question = COMMON_QUESTION_EXAMPLES[-1]
                app.selectbox(key=f"question-example-{compound_id}").select(label).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.text_input(key=f"question-input-{compound_id}").value, question)
            chat.assert_not_called()
            embeddings.assert_not_called()

    def test_additional_reviewed_evidence(self):
        from compare_measurements import compare_measurements
        from retrieve_local_evidence import retrieve
        from external_documents import reviewed_scope

        expected = {"E-793": ("less than 1", "68°F"), "F-67": ("less than 1", "68°F"),
                    "C-137": ("less than 1", "66°F"), "A-2073": ("less than 1", "68°F"),
                    "B-1465": ("less than 1", "64°F"), "A-3999": ("10 to 50", "59°F"),
                    "A-4301": ("5 to 10", "68°F")}
        for compound_id, (value, temperature) in expected.items():
            with self.subTest(compound_id=compound_id):
                result = retrieve(compound_id, "문헌 수용해도 값 단위 온도 원출처")
                self.assertEqual(result["compound_metadata"]["compound_id"], compound_id)
                self.assertEqual(len(result["external_evidence"]), 1)
                self.assertIn("NTP", result["external_evidence"][0]["original_attribution"])
                comparison = compare_measurements(compound_id)
                self.assertEqual(comparison["records"][1]["기록된 값"], value)
                self.assertEqual(comparison["records"][1]["온도"], temperature)
        qualitative = retrieve("B-171", "water solubility")
        self.assertIn("Very soluble", qualitative["external_evidence"][0]["excerpt"])
        self.assertEqual(len(compare_measurements("B-171")["records"]), 1)
        self.assertTrue(set(expected) | {"B-171"} <= set(reviewed_scope()["compound_ids"]))

    def test_unindexed_compound_does_not_call_embeddings(self):
        from streamlit.testing.v1 import AppTest
        with patch("llm.get_embeddings", side_effect=AssertionError("Embeddings forbidden")) as embeddings, patch(
            "vector_store.has_persisted_evidence", return_value=False
        ), patch("vector_store.count_persisted", return_value=9):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            app.selectbox(key="quick-compound").select("E-793").run()
            app.session_state["retrieval-choice"] = "의미 유사도"
            app.run()
            app.checkbox(key="vector-search-consent").check().run()
            app.text_input(key="question-input-E-793").set_value("수용해도 값과 온도는?")
            next(button for button in app.button if button.label == "근거 검색").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("색인이 아직 준비되지" in error.value for error in app.error))
            embeddings.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
