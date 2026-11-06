"""Small examples of each metric; no network or ML models are needed."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from evaluation.metrics import (
    expected_fact_coverage,
    hit_at_k,
    mean_score,
    reciprocal_rank,
    score_reviews,
)
from evaluation.offline import (
    Chunk,
    OfflineRetriever,
    build_chunks,
    load_dataset,
    load_local_embedder,
    main,
    run_evaluation,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


class MetricTests(unittest.TestCase):
    def test_hit_and_reciprocal_rank_use_chunk_rank(self):
        ranking = ["wrong", "wrong", "right"]
        self.assertEqual(hit_at_k(ranking, ["right"], 2), 0)
        self.assertEqual(hit_at_k(ranking, ["right"], 3), 1)
        self.assertAlmostEqual(reciprocal_rank(ranking, ["right"], 3), 1 / 3)
        self.assertEqual(reciprocal_rank(ranking, ["right"], 2), 0)

    def test_empty_missing_and_small_collections(self):
        self.assertEqual(hit_at_k([], ["known"], 5), 0)
        self.assertEqual(reciprocal_rank([], ["known"], 5), 0)
        self.assertEqual(hit_at_k(["known"], ["known"], 5), 1)
        self.assertIsNone(hit_at_k(["known"], [], 5))
        self.assertIsNone(reciprocal_rank(["known"], None, 5))
        self.assertIsNone(mean_score([None, None]))
        self.assertEqual(mean_score([1, None, 0]), 0.5)
        self.assertEqual(hit_at_k(["known"], ["known"], 0), 0)
        for bad_k in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                hit_at_k([], ["known"], bad_k)

    def test_fact_coverage_is_not_correctness(self):
        facts = [{"text": "623 fatalities", "accepted_phrases": ["623"]},
                 {"text": "rate", "accepted_phrases": ["0.56"]}]
        self.assertEqual(expected_fact_coverage("623 fatalities", facts), 0.5)
        self.assertEqual(expected_fact_coverage("NOT 623; NOT 0.56!", facts), 1)
        self.assertEqual(expected_fact_coverage("1623 and 0.560", facts), 0)
        self.assertEqual(expected_fact_coverage("", facts), 0)
        self.assertIsNone(expected_fact_coverage(None, facts))
        self.assertIsNone(expected_fact_coverage("answer", []))
        self.assertEqual(expected_fact_coverage("719,000 kilometres", [
            {"text": "length", "accepted_phrases": ["719000 kilometres"]}]), 1)
        with self.assertRaises(ValueError):
            expected_fact_coverage("answer", [{"text": "missing phrases"}])

    def test_review_denominators_are_independent(self):
        def review(accuracy, grounding):
            return {"accuracy": accuracy, "grounding": grounding,
                    "reviewer": "Tester", "notes": "Checked saved evidence"}
        scores = score_reviews([
            review("Correct", "Unsupported"),
            review("Partially correct", "Supported"),
            review("Incorrect", "Partially supported"),
            review("Not enough evidence", "Supported"),
            review("Unreviewed", "Supported"),
            review("Correct", "Unreviewed"), {}, None,
        ])
        self.assertEqual(scores["accuracy_reviewed"], 5)
        self.assertEqual(scores["strict_accuracy_percent"], 40)
        self.assertEqual(scores["partial_credit_percent"], 50)
        self.assertEqual(scores["grounding_reviewed"], 5)
        self.assertEqual(scores["supported_percent"], 60)
        empty = score_reviews([{}, None])
        self.assertIsNone(empty["strict_accuracy_percent"])
        self.assertIsNone(empty["supported_percent"])

    def test_reviews_require_labels_and_audit_notes(self):
        for record in ({"accuracy": "correct"},
                       {"accuracy": "Accurate"},
                       {"grounding": "Mixed"},
                       {"grounding": "Supported", "reviewer": " ", "notes": "x"}):
            with self.assertRaises(ValueError):
                score_reviews([record])


class FakeVectors(list):
    @property
    def shape(self):
        return (len(self), 2)


class FakeEmbedder:
    def encode(self, texts):
        return FakeVectors([[1, 2] for _ in texts])


class FakeIndex:
    def __init__(self, dimensions):
        self.last_k = None

    def add(self, vectors):
        pass

    def search(self, vector, k):
        self.last_k = k
        # FAISS uses -1 for padding. Ignore every invalid index.
        return [[0] * 4], [[-1, 1, 99, 0]]


class OfflineTests(unittest.TestCase):
    def make_retriever(self, chunks):
        return OfflineRetriever(chunks, FakeEmbedder(), FakeIndex, lambda x: x)

    def test_small_index_filters_padding_and_bounds_k(self):
        chunks = [Chunk("a.pdf", "a.pdf::chunk-0000", "first"),
                  Chunk("b.pdf", "b.pdf::chunk-0000", "second")]
        retriever = self.make_retriever(chunks)
        self.assertEqual(retriever.retrieve("question", 10), [chunks[1], chunks[0]])
        self.assertEqual(retriever.index.last_k, 2)
        self.assertEqual(retriever.retrieve("question", 0), [])
        self.assertEqual(self.make_retriever([]).retrieve("question", 5), [])

    def test_builder_uses_parser_and_chunker_without_global_index(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "a.pdf").write_bytes(b"test fixture")
            chunks = build_chunks(Path(folder), ["a.pdf"],
                                  lambda path: "first second",
                                  lambda text: text.split())
        self.assertEqual([c.chunk_id for c in chunks],
                         ["a.pdf::chunk-0000", "a.pdf::chunk-0001"])
        self.assertEqual([c.text for c in chunks], ["first", "second"])
        with self.assertRaises(ValueError):
            build_chunks(ROOT, ["../a.pdf"], lambda p: "", lambda t: [])

    def test_answers_and_stale_review_protection(self):
        item = {"id": "q1", "question": "?", "expected_document_ids": ["a.pdf"],
                "reference_answer": "fact", "expected_facts": [
                    {"text": "fact", "accepted_phrases": ["fact"]}]}
        retriever = self.make_retriever([Chunk("a.pdf", "a::0", "fact")])
        run = run_evaluation([item], retriever, 5, lambda question, context: "fact")
        self.assertEqual(run["summary"]["hit_at_k"], 1)
        self.assertEqual(run["summary"]["expected_fact_coverage"], 1)
        self.assertIsNone(run["summary"]["strict_accuracy_percent"])
        review = dict(run["review_template"][0], accuracy="Correct",
                      grounding="Supported", reviewer="Tester", notes="Checked")
        self.assertEqual(summarize(run["results"], [review])[
            "strict_accuracy_percent"], 100)
        for changed in (dict(review, answer="different"),
                        dict(review, retrieved_chunk_ids=[]),
                        dict(review, question_id="unknown")):
            with self.assertRaises(ValueError):
                summarize(run["results"], [changed])
        with self.assertRaises(ValueError):
            summarize(run["results"], [review, review])
        missing = run_evaluation([item], retriever, 5)
        self.assertIsNone(missing["summary"]["expected_fact_coverage"])
        invalid_review = dict(missing["review_template"][0], accuracy="Correct",
                              reviewer="Tester", notes="No answer")
        with self.assertRaises(ValueError):
            summarize(missing["results"], [invalid_review])
        self.assertIsNone(run_evaluation([], retriever, 5)["summary"]["hit_at_k"])

    def test_local_model_path_is_required_before_any_ml_import(self):
        with self.assertRaises(ValueError):
            load_local_embedder(ROOT / "no-such-model")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_local_embedder(Path(folder))

    def test_local_model_loading_forces_offline_without_loading_a_model(self):
        import os
        calls = []
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = lambda *args, **kwargs: calls.append((args, kwargs))
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "modules.json").write_text("[]", encoding="utf-8")
            with patch.dict(sys.modules, {"sentence_transformers": fake}):
                with patch.dict(os.environ, {}, clear=False):
                    load_local_embedder(Path(folder))
                    self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                    self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
                    self.assertEqual(os.environ["USE_TF"], "0")
        self.assertEqual(calls[0][1], {"local_files_only": True})

    def test_mean_reciprocal_rank_and_excluded_relevance_labels(self):
        retriever = self.make_retriever([
            Chunk("a.pdf", "a::0", "a"), Chunk("b.pdf", "b::0", "b")])
        def question(question_id, expected):
            return {"id": question_id, "question": "?", "reference_answer": "",
                    "expected_document_ids": expected, "expected_facts": []}
        run = run_evaluation([question("hit-at-rank-two", ["a.pdf"]),
                              question("miss", ["c.pdf"]),
                              question("no-label", None)], retriever, 5)
        self.assertEqual(run["summary"]["retrieval_scored"], 2)
        self.assertEqual(run["summary"]["hit_at_k"], 0.5)
        self.assertEqual(run["summary"]["mrr_at_k"], 0.25)
        self.assertEqual(run["summary"]["hit_at_1"], 0)
        self.assertEqual(run["summary"]["hit_at_3"], 0.5)
        self.assertEqual(run["summary"]["hit_at_5"], 0.5)
        self.assertEqual(run["summary"]["mrr_at_5"], 0.25)

    def test_score_cli_writes_output_without_retrieval(self):
        item = {"id": "q1", "question": "?", "expected_document_ids": ["a.pdf"],
                "reference_answer": "fact", "expected_facts": [
                    {"text": "fact", "accepted_phrases": ["fact"]}]}
        run = run_evaluation([item], self.make_retriever([]), 5,
                             lambda question, context: "fact")
        review = dict(run["review_template"][0], accuracy="Correct",
                      grounding="Unsupported", reviewer="Tester", notes="Empty context")
        with tempfile.TemporaryDirectory() as folder:
            saved, reviews, output = [Path(folder, name) for name in
                                      ("run.json", "reviews.json", "scored.json")]
            saved.write_text(json.dumps(run), encoding="utf-8")
            reviews.write_text(json.dumps([review]), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "evaluation.offline", "--score", str(saved),
                 "--reviews", str(reviews), "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
            self.assertEqual(summary["strict_accuracy_percent"], 100)
            self.assertEqual(summary["supported_percent"], 0)
            self.assertEqual(summary["hit_at_k"], 0)

    def test_run_records_unverified_model_provenance_not_assumed_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            model_path, output = Path(folder, "some-model"), Path(folder, "run.json")
            with patch("evaluation.offline.load_local_embedder", return_value=FakeEmbedder()):
                with patch("evaluation.offline.build_chunks", return_value=[]):
                    with patch("evaluation.offline.OfflineRetriever",
                               return_value=self.make_retriever([])):
                        main(["--model-path", str(model_path), "--output", str(output)])
            run = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(run["embedding_model"]["directory"], str(model_path.resolve()))
        self.assertFalse(run["embedding_model"]["identity_verified"])
        self.assertEqual(run["embedding_model"]["source"],
                         "Caller-supplied local SentenceTransformer directory")

    def test_imports_and_help_do_not_load_production_or_ml(self):
        code = (
            "import sys; import evaluation.offline; "
            "assert not any(m in sys.modules for m in "
            "['app','services.rag_service','tensorflow','torch',"
            "'sentence_transformers','faiss'])"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-m", "evaluation.offline", "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class DatasetTests(unittest.TestCase):
    def test_ids_hashes_and_reference_schema(self):
        dataset = load_dataset()
        self.assertEqual(len(dataset["questions"]), 10)
        self.assertEqual(len(dataset["documents"]), 4)
        for document_id, source in dataset["documents"].items():
            path = ROOT / "examples" / "documents" / document_id
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             source["sha256"])
        ids = [item["id"] for item in dataset["questions"]]
        self.assertEqual(len(ids), len(set(ids)))
        for item in dataset["questions"]:
            self.assertTrue(item["reference_answer"])
            self.assertTrue(item["sources"])
            self.assertTrue(item["reviewer_note"])
            self.assertEqual(expected_fact_coverage(
                item["reference_answer"], item["expected_facts"]), 1)

    @unittest.skipUnless(importlib.util.find_spec("PyPDF2"),
                         "Bundled PDF evidence check requires existing PyPDF2")
    def test_reference_evidence_is_on_the_named_pdf_page(self):
        from PyPDF2 import PdfReader
        dataset = load_dataset()
        readers = {name: PdfReader(ROOT / "examples" / "documents" / name)
                   for name in dataset["documents"]}
        for item in dataset["questions"]:
            evidence = []
            for source in item["sources"]:
                page = readers[source["document_id"]].pages[source["pdf_page"] - 1]
                evidence.append(page.extract_text() or "")
            self.assertEqual(expected_fact_coverage(
                "\n".join(evidence), item["expected_facts"]), 1, item["id"])

    @unittest.skipUnless(importlib.util.find_spec("PyPDF2"),
                         "Production parser reuse check requires existing PyPDF2")
    def test_real_chunking_reuses_production_without_mutating_upload_state(self):
        import services.rag_service as rag
        dataset = load_dataset()
        original_docs, original_index = rag.DOCS, rag.INDEX
        with patch.object(rag, "get_embedder", side_effect=AssertionError("No ML")):
            chunks = build_chunks(ROOT / "examples" / "documents", dataset["documents"])
        self.assertIs(rag.DOCS, original_docs)
        self.assertIs(rag.INDEX, original_index)
        self.assertEqual({c.document_id for c in chunks}, set(dataset["documents"]))
        self.assertEqual(len(chunks), len({c.chunk_id for c in chunks}))
        name = next(iter(dataset["documents"]))
        expected = rag._chunk_text(rag._extract_text_from_file(
            str(ROOT / "examples" / "documents" / name)))
        self.assertEqual([c.text for c in chunks if c.document_id == name], expected)


class ExistingRegressionTests(unittest.TestCase):
    def test_existing_lazy_model_import_regression(self):
        """Execute the existing pytest-style test with a unittest patch adapter."""
        path = ROOT / "tests" / "test_model_service_lazy_loading.py"
        spec = importlib.util.spec_from_file_location("existing_lazy_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        patches = []

        class MonkeyPatch:
            def setattr(self, obj, name, value):
                patcher = patch.object(obj, name, value)
                patcher.start()
                patches.append(patcher)

        previous = sys.modules.pop("services.model_service", None)
        try:
            module.test_model_service_import_does_not_load_model(MonkeyPatch())
        finally:
            for patcher in reversed(patches):
                patcher.stop()
            sys.modules.pop("services.model_service", None)
            if previous is not None:
                sys.modules["services.model_service"] = previous


if __name__ == "__main__":
    unittest.main()
