import json
import unittest
from unittest.mock import patch

import requests

from services import ollama_service


class FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self.data = data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.data


class OllamaServiceTests(unittest.TestCase):
    def event_from_log(self, log_output):
        return json.loads(log_output[0].split(":", 2)[2].strip())

    def test_success_uses_provider_counts_and_preserves_chat_response(self):
        response = FakeResponse(200, {
            "response": "The answer.",
            "prompt_eval_count": 12,
            "eval_count": 5,
        })
        with patch.object(ollama_service, "OLLAMA_COST_PER_1000_TOKENS", 2.5):
            with patch.object(ollama_service.requests, "post", return_value=response) as post:
                with self.assertLogs("services.ollama_service", level="INFO") as logs:
                    answer = ollama_service.chat_with_ollama("private prompt")

        self.assertEqual(answer, "The answer.")
        self.assertEqual(post.call_args.kwargs["timeout"], 120)
        event = self.event_from_log(logs.output)
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["attempts"], 1)
        self.assertEqual(event["prompt_tokens"], 12)
        self.assertEqual(event["completion_tokens"], 5)
        self.assertEqual(event["total_tokens"], 17)
        self.assertEqual(event["token_source"], "provider")
        self.assertEqual(event["estimated_provider_cost"], 0.0425)
        self.assertNotIn("private prompt", logs.output[0])

    def test_missing_provider_counts_are_labeled_estimates(self):
        response = FakeResponse(200, {"response": "three output words"})
        with patch.object(ollama_service.requests, "post", return_value=response):
            with patch.object(ollama_service, "OLLAMA_COST_PER_1000_TOKENS", None):
                with self.assertLogs("services.ollama_service", level="INFO") as logs:
                    result = ollama_service.request_ollama("two input", timeout=7)

        self.assertEqual(result["response"], "three output words")
        event = self.event_from_log(logs.output)
        self.assertEqual(event["prompt_tokens"], 2)
        self.assertEqual(event["completion_tokens"], 3)
        self.assertEqual(event["total_tokens"], 5)
        self.assertEqual(event["token_source"], "estimate")
        self.assertIsNone(event["estimated_provider_cost"])

    def test_retries_connection_error_then_succeeds(self):
        response = FakeResponse(200, {"response": "ok"})
        with patch.object(ollama_service, "OLLAMA_MAX_RETRIES", 1):
            with patch.object(ollama_service, "OLLAMA_RETRY_DELAY_SECONDS", 0.25):
                with patch.object(
                    ollama_service.requests, "post",
                    side_effect=[requests.ConnectionError("private endpoint"), response],
                ) as post:
                    with patch.object(ollama_service.time, "sleep") as sleep:
                        with self.assertLogs(
                                "services.ollama_service", level="INFO") as logs:
                            self.assertEqual(
                                ollama_service.request_ollama("question", timeout=9),
                                {"response": "ok"},
                            )

        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(0.25)
        event = self.event_from_log(logs.output)
        self.assertEqual(event["attempts"], 2)
        self.assertEqual(event["status"], "success")

    def test_permanent_http_error_is_not_retried(self):
        response = FakeResponse(400)
        with patch.object(ollama_service, "OLLAMA_MAX_RETRIES", 3):
            with patch.object(ollama_service.requests, "post", return_value=response) as post:
                with patch.object(ollama_service.time, "sleep") as sleep:
                    with self.assertLogs(
                            "services.ollama_service", level="WARNING") as logs:
                        self.assertIsNone(
                            ollama_service.request_ollama("question", timeout=4)
                        )

        post.assert_called_once()
        sleep.assert_not_called()
        event = self.event_from_log(logs.output)
        self.assertEqual(event["status"], "failure")
        self.assertEqual(event["error_category"], "http_error")
        self.assertEqual(event["http_status"], 400)
        self.assertEqual(event["attempts"], 1)

    def test_temporary_http_error_is_retried(self):
        response = FakeResponse(200, {"response": "recovered"})
        with patch.object(ollama_service, "OLLAMA_MAX_RETRIES", 1):
            with patch.object(
                ollama_service.requests, "post",
                side_effect=[FakeResponse(503), response],
            ) as post:
                with patch.object(ollama_service.time, "sleep") as sleep:
                    with self.assertLogs(
                            "services.ollama_service", level="INFO") as logs:
                        result = ollama_service.request_ollama("question", timeout=6)

        self.assertEqual(result["response"], "recovered")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()
        self.assertEqual(self.event_from_log(logs.output)["attempts"], 2)

    def test_timeout_retries_only_up_to_configured_limit(self):
        with patch.object(ollama_service, "OLLAMA_MAX_RETRIES", 1):
            with patch.object(
                ollama_service.requests, "post",
                side_effect=requests.Timeout("private prompt"),
            ) as post:
                with patch.object(ollama_service.time, "sleep") as sleep:
                    with self.assertLogs(
                            "services.ollama_service", level="WARNING") as logs:
                        answer = ollama_service.chat_with_ollama("private prompt")

        self.assertEqual(answer, ollama_service.UNAVAILABLE_MESSAGE)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()
        self.assertEqual(post.call_args.kwargs["timeout"], 120)
        event = self.event_from_log(logs.output)
        self.assertEqual(event["error_category"], "timeout")
        self.assertEqual(event["attempts"], 2)
        self.assertIsNone(event["completion_tokens"])
        self.assertIsNone(event["total_tokens"])
        self.assertNotIn("private prompt", logs.output[0])

    def test_rag_keeps_existing_generation_options_and_timeout(self):
        from services import rag_service

        with patch.object(rag_service, "_retrieve_context", return_value="context"):
            with patch.object(
                rag_service, "request_ollama", return_value={"response": "answer"}
            ) as call:
                answer = rag_service.chat_with_ollama_rag("question")
        self.assertEqual(answer, "answer")
        call.assert_called_once()
        self.assertEqual(call.call_args.kwargs["timeout"], 180)
        self.assertEqual(
            call.call_args.kwargs["options"], {"temperature": 0.1}
        )

    def test_chat_and_rag_api_response_shapes_are_unchanged(self):
        import app as flask_app

        client = flask_app.app.test_client()
        with patch.object(flask_app, "chat_with_ollama", return_value="chat answer"):
            chat_response = client.post("/api/chat", json={"message": "question"})
        with patch.object(
            flask_app, "chat_with_ollama_rag", return_value="RAG answer"
        ):
            rag_response = client.post("/api/rag/ask", json={"question": "question"})

        self.assertEqual(chat_response.status_code, 200)
        self.assertEqual(chat_response.get_json(), {"response": "chat answer"})
        self.assertEqual(rag_response.status_code, 200)
        self.assertEqual(rag_response.get_json(), {"answer": "RAG answer"})

    def test_retry_configuration_has_safe_caps(self):
        self.assertGreaterEqual(ollama_service.OLLAMA_MAX_RETRIES, 0)
        self.assertLessEqual(ollama_service.OLLAMA_MAX_RETRIES, 3)
        self.assertGreaterEqual(ollama_service.OLLAMA_RETRY_DELAY_SECONDS, 0)
        self.assertLessEqual(ollama_service.OLLAMA_RETRY_DELAY_SECONDS, 5)


if __name__ == "__main__":
    unittest.main()
