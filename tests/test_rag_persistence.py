import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from werkzeug.datastructures import FileStorage

from services import rag_service


class FakeIndex:
    def add(self, vectors):
        self.vectors = vectors

    def search(self, query, top_k):
        return [[0.0]], [[0]]


class RagPersistenceTests(unittest.TestCase):
    def test_uploaded_document_is_reindexed_after_index_reset(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            marker_path = Path(upload_dir) / ".active-document.json"
            with patch.object(rag_service, "UPLOAD_DIR", upload_dir):
                with patch.object(rag_service, "ACTIVE_DOCUMENT_PATH", marker_path):
                    with patch.object(rag_service, "DOCS", []):
                        with patch.object(rag_service, "INDEX", None):
                            with patch.object(
                                rag_service, "_build_index", return_value=FakeIndex()
                            ) as build_index:
                                result = rag_service.load_knowledge_from_file(
                                    FileStorage(
                                        stream=io.BytesIO(b"Saved document facts."),
                                        filename="guide.txt",
                                    )
                                )

                            self.assertEqual(result["status"], "ok")
                            self.assertEqual(
                                json.loads(marker_path.read_text(encoding="utf-8")),
                                {"filename": "guide.txt"},
                            )

                            with patch.object(rag_service, "DOCS", []):
                                with patch.object(rag_service, "INDEX", None):
                                    with patch.object(
                                        rag_service,
                                        "_build_index",
                                        return_value=FakeIndex(),
                                    ) as restore_index:
                                        context = rag_service._retrieve_context(
                                            "question"
                                        )

                            self.assertEqual(context, "Saved document facts.")
                            self.assertEqual(build_index.call_count, 1)
                            self.assertEqual(restore_index.call_count, 1)

    def test_invalid_persisted_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            marker_path = Path(upload_dir) / ".active-document.json"
            marker_path.write_text(
                json.dumps({"filename": "..\\outside.txt"}), encoding="utf-8"
            )
            with patch.object(rag_service, "UPLOAD_DIR", upload_dir):
                with patch.object(rag_service, "ACTIVE_DOCUMENT_PATH", marker_path):
                    with patch.object(rag_service, "DOCS", []):
                        with patch.object(rag_service, "INDEX", None):
                            with self.assertRaisesRegex(ValueError, "marker is invalid"):
                                rag_service._restore_active_document()


if __name__ == "__main__":
    unittest.main()
