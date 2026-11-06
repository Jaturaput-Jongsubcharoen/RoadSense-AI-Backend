import io
import unittest
from unittest.mock import patch

import requests

import app as flask_app
from services import ollama_service, rag_service


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.app.test_client()

    def test_health_returns_existing_response(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_predict_requires_an_image(self):
        response = self.client.post("/api/predict", data={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "No image uploaded"})

    def test_predict_returns_mocked_model_result_without_loading_model(self):
        from services import model_service

        with patch.object(
            model_service,
            "predict_image",
            return_value={"class_name": "Pothole Issues", "confidence": 0.9},
        ) as predict:
            response = self.client.post(
                "/api/predict",
                data={"image": (io.BytesIO(b"mock image"), "road.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"class_name": "Pothole Issues", "confidence": 0.9},
        )
        predict.assert_called_once()

    def test_chat_preserves_response_when_ollama_connection_fails(self):
        with patch.object(ollama_service, "OLLAMA_MAX_RETRIES", 0):
            with patch.object(
                ollama_service.requests,
                "post",
                side_effect=requests.ConnectionError("unavailable"),
            ):
                response = self.client.post("/api/chat", json={"message": "question"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(), {"response": ollama_service.UNAVAILABLE_MESSAGE}
        )

    def test_rag_ask_requires_question(self):
        response = self.client.post("/api/rag/ask", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(), {"error": "Missing 'question' in request"}
        )

    def test_rag_ask_preserves_response_when_ollama_connection_fails(self):
        with patch.object(rag_service, "_retrieve_context", return_value="context"):
            with patch.object(ollama_service, "OLLAMA_MAX_RETRIES", 0):
                with patch.object(
                    ollama_service.requests,
                    "post",
                    side_effect=requests.ConnectionError("unavailable"),
                ):
                    response = self.client.post(
                        "/api/rag/ask", json={"question": "question"}
                    )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(), {"answer": ollama_service.UNAVAILABLE_MESSAGE}
        )

    def test_rag_upload_requires_file(self):
        response = self.client.post("/api/rag/upload", data={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "No file uploaded"})

    def test_rag_upload_rejects_unsupported_file(self):
        response = self.client.post(
            "/api/rag/upload",
            data={"file": (io.BytesIO(b"content"), "notes.exe")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json(),
            {"error": "Unsupported document type. Use PDF, TXT, DOC, or DOCX."},
        )

    def test_rag_upload_enforces_configured_request_size(self):
        with patch.dict(flask_app.app.config, {"MAX_CONTENT_LENGTH": 8}):
            response = self.client.post(
                "/api/rag/upload",
                data={"file": (io.BytesIO(b"document is too large"), "notes.txt")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            response.get_json(),
            {"error": "Request exceeds the configured upload limit"},
        )

    def test_rag_upload_accepts_supported_file_using_mocked_indexer(self):
        with patch.object(
            flask_app, "load_knowledge_from_file",
            return_value={"status": "ok", "chunks_loaded": 1},
        ) as load_document:
            response = self.client.post(
                "/api/rag/upload",
                data={"file": (io.BytesIO(b"document"), "notes.txt")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(), {"status": "ok", "chunks_loaded": 1}
        )
        load_document.assert_called_once()

    def test_rag_questions_returns_existing_unavailable_response(self):
        with patch.object(flask_app, "generate_suggested_questions", return_value=[]):
            response = self.client.post("/api/rag/questions")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json(),
            {
                "questions": [],
                "message": "No questions generated. Ollama may be unavailable.",
            },
        )


if __name__ == "__main__":
    unittest.main()
