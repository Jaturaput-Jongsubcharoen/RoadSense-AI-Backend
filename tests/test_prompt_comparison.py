import unittest

from evaluation.prompt_comparison import (
    BASELINE_VERSION,
    CANDIDATE_VERSION,
    build_candidate_prompt,
    render_markdown,
    run_comparison,
)
from services.rag_service import build_rag_prompt


class PromptComparisonTests(unittest.TestCase):
    def retrieval_run(self):
        return {
            "summary": {"questions": 1, "hit_at_1": 1.0},
            "results": [{
                "question_id": "q1",
                "question": "What is the speed limit?",
                "reference_answer": "The posted limit is 50 km/h.",
                "expected_document_ids": ["roads.pdf"],
                "expected_facts": [
                    {"text": "50 km/h", "accepted_phrases": ["50 km/h"]},
                ],
                "retrieved": [{
                    "chunk_id": "roads.pdf::chunk-0000",
                    "document_id": "roads.pdf",
                    "text": "The posted limit is 50 km/h.",
                }],
            }],
        }

    def test_baseline_prompt_matches_production_prompt(self):
        prompt = build_rag_prompt(
            "What is the speed limit?", "The posted limit is 50 km/h."
        )
        self.assertIn("You are a careful assistant. Follow instructions strictly.", prompt)
        self.assertIn(
            "Answer concisely using ONLY the information in the context below.",
            prompt,
        )
        self.assertIn("CONTEXT:\nThe posted limit is 50 km/h.", prompt)
        self.assertIn("QUESTION:\nWhat is the speed limit?", prompt)
        self.assertIn(
            'reply exactly:\n"I don\'t have enough information in the uploaded document to answer that."',
            prompt,
        )

    def test_both_prompts_receive_same_context_and_generation_settings(self):
        calls = []

        def fake_request(prompt, timeout, options):
            calls.append((prompt, timeout, options))
            return {
                "response": "The limit is 50 km/h.",
                "prompt_eval_count": 24,
                "eval_count": 8,
            }

        report = run_comparison(
            self.retrieval_run(), request_function=fake_request, model="test-model"
        )

        self.assertEqual(len(calls), 2)
        self.assertIn("The posted limit is 50 km/h.", calls[0][0])
        self.assertIn("The posted limit is 50 km/h.", calls[1][0])
        self.assertEqual(calls[0][1:], calls[1][1:])
        self.assertNotEqual(calls[0][0], calls[1][0])
        self.assertEqual(report["baseline_prompt_version"], BASELINE_VERSION)
        self.assertEqual(report["candidate_prompt_version"], CANDIDATE_VERSION)
        self.assertEqual(report["baseline_summary"]["expected_fact_coverage"], 1)
        self.assertEqual(report["candidate_summary"]["expected_fact_coverage"], 1)
        self.assertEqual(report["baseline_summary"]["average_total_tokens"], 32)
        self.assertEqual(
            report["baseline_summary"]["average_actual_provider_total_tokens"], 32
        )
        self.assertIsNone(report["baseline_summary"]["strict_accuracy_percent"])
        self.assertEqual(
            report["results"][0]["retrieved_context"],
            "The posted limit is 50 km/h.",
        )

    def test_estimates_and_manual_review_are_separate(self):
        def fake_request(prompt, timeout, options):
            return {"response": "The limit is 50 km/h."}

        reviews = {
            "q1": {
                "baseline": {
                    "answer": "The limit is 50 km/h.",
                    "retrieved_chunk_ids": ["roads.pdf::chunk-0000"],
                    "accuracy": "Correct",
                    "grounding": "Supported",
                    "reviewer": "Reviewer",
                    "notes": "Checked against retrieved source.",
                }
            }
        }
        report = run_comparison(
            self.retrieval_run(), reviews, request_function=fake_request
        )
        baseline = report["results"][0]["baseline"]
        self.assertEqual(baseline["token_source"], "estimate")
        self.assertIsNone(report["baseline_summary"]["average_actual_provider_total_tokens"])
        self.assertEqual(report["baseline_summary"]["strict_accuracy_percent"], 100)
        self.assertEqual(report["baseline_summary"]["supported_percent"], 100)
        self.assertIsNone(report["candidate_summary"]["strict_accuracy_percent"])
        markdown = render_markdown(report)
        self.assertIn("Expected-fact coverage (not accuracy)", markdown)
        self.assertIn("Manual accuracy and grounding remain unscored", markdown)

    def test_failed_calls_have_no_quality_or_token_score(self):
        report = run_comparison(
            self.retrieval_run(), request_function=lambda *args, **kwargs: None
        )
        self.assertEqual(report["baseline_summary"]["requests_failed"], 1)
        self.assertEqual(report["candidate_summary"]["requests_failed"], 1)
        self.assertIsNone(report["baseline_summary"]["expected_fact_coverage"])
        self.assertIsNone(report["baseline_summary"]["average_total_tokens"])
        self.assertIsNone(report["baseline_summary"]["average_latency_ms"])
        self.assertEqual(
            report["results"][0]["baseline"]["error_category"], "request_failed"
        )

    def test_candidate_prompt_is_distinct_and_has_the_same_fallback(self):
        prompt = build_candidate_prompt("Question?", "Source context.")
        self.assertIn("Answer the question directly using only the supplied context.", prompt)
        self.assertIn("Source context.", prompt)
        self.assertIn(
            "I don't have enough information in the uploaded document to answer that.",
            prompt,
        )

    def test_manual_review_must_match_answer_and_retrieved_chunks(self):
        reviews = {
            "q1": {
                "baseline": {
                    "answer": "A stale answer.",
                    "retrieved_chunk_ids": ["roads.pdf::chunk-0000"],
                    "accuracy": "Correct",
                    "grounding": "Supported",
                    "reviewer": "Reviewer",
                    "notes": "Reviewed.",
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "must match"):
            run_comparison(
                self.retrieval_run(),
                reviews,
                request_function=lambda *args, **kwargs: {
                    "response": "The limit is 50 km/h."
                },
            )

    def test_unknown_review_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown question"):
            run_comparison(
                self.retrieval_run(),
                {"unknown": {}},
                request_function=lambda *args, **kwargs: None,
            )


if __name__ == "__main__":
    unittest.main()
