# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # avoid TF optimization loading
# os.environ["USE_TF"] = "0"                  # disable TF usage inside transformers
# os.environ["TRANSFORMERS_NO_TF"] = "1"      # hard block TF imports
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"    # block JAX/Flax
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from PyPDF2 import PdfReader
# import docx2txt

# import requests

# # Re-use your local Ollama server
# OLLAMA_URL = "http://localhost:11434/api/generate"

# # Where uploaded files will be stored
# UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # ----- Embedding model + in-memory index -----
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# DOCS = []      # list[str]
# INDEX = None   # faiss index


# def _extract_text_from_file(file_path: str) -> str:
#     """Read TXT / PDF / DOCX and return plain text."""
#     ext = file_path.split(".")[-1].lower()

#     if ext == "txt":
#         with open(file_path, "r", encoding="utf-8") as f:
#             return f.read()

#     if ext == "pdf":
#         reader = PdfReader(file_path)
#         return "\n".join(page.extract_text() or "" for page in reader.pages)

#     if ext in ("doc", "docx"):
#         return docx2txt.process(file_path)

#     # Unknown format – just return empty
#     return ""


# def load_knowledge_from_file(file_storage) -> dict:
#     """
#     Accepts a Werkzeug FileStorage (from Flask request.files['file']),
#     saves it to disk, builds FAISS index over its text.
#     """
#     global DOCS, INDEX

#     filename = file_storage.filename
#     save_path = os.path.join(UPLOAD_DIR, filename)
#     file_storage.save(save_path)

#     text = _extract_text_from_file(save_path)
#     lines = [line.strip() for line in text.splitlines() if line.strip()]

#     if not lines:
#         DOCS = []
#         INDEX = None
#         return {"status": "error", "message": "No readable text found in file."}

#     DOCS = lines

#     # Build FAISS index
#     vectors = embedder.encode(DOCS)
#     INDEX = faiss.IndexFlatL2(vectors.shape[1])
#     INDEX.add(np.array(vectors))

#     return {"status": "ok", "lines_loaded": len(DOCS)}


# def _retrieve_context(query: str, top_k: int = 5) -> str:
#     """Return the top_k most relevant lines from DOCS for a query."""
#     if INDEX is None or not DOCS:
#         return "No context uploaded yet."

#     q_vec = embedder.encode([query])
#     distances, indices = INDEX.search(np.array(q_vec), top_k)
#     selected = [DOCS[i] for i in indices[0]]
#     return "\n".join(selected)


# def chat_with_ollama_rag(question: str) -> str:
#     """
#     Main RAG function: uses retrieved context + Ollama llama3.1 to answer.
#     """
#     context = _retrieve_context(question)

#     prompt = f"""
# You are an assistant that must answer ONLY using the provided context.

# CONTEXT:
# {context}

# QUESTION:
# {question}

# If the context does not contain enough information, say:
# "I don't have enough information in the uploaded document to answer that."
# """

#     payload = {
#         "model": "llama3.1",
#         "prompt": prompt,
#         "stream": False,
#     }

#     try:
#         res = requests.post(OLLAMA_URL, json=payload, timeout=60)
#         data = res.json()
#         return data.get("response", "No response from AI.")
#     except Exception as e:
#         return f"Ollama / RAG error: {str(e)}"



import json
import logging
import os
from pathlib import Path
from threading import Lock

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # avoid TF optimization loading
os.environ["USE_TF"] = "0"                  # disable TF usage inside transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"      # block TF loading
os.environ["TRANSFORMERS_NO_FLAX"] = "1"    # block JAX
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from werkzeug.utils import secure_filename

from services.ollama_service import UNAVAILABLE_MESSAGE, request_ollama

logger = logging.getLogger(__name__)

# ---------------------- FILE STORAGE ------------------------
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ACTIVE_DOCUMENT_PATH = Path(UPLOAD_DIR) / ".active-document.json"
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}

# ---------------------- EMBEDDING MODEL ---------------------
embedder = None
embedder_lock = Lock()

DOCS = []      # list[str] containing text chunks
INDEX = None   # faiss index
index_lock = Lock()


def get_embedder():
    """Load the embedding model only when RAG is first used."""
    global embedder
    if embedder is None:
        with embedder_lock:
            if embedder is None:
                from sentence_transformers import SentenceTransformer

                embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder

# ---------------------- FILE READING ------------------------
def _extract_text_from_file(file_path: str) -> str:
    """Read TXT / PDF / DOCX and return plain text only when a document is actually selected for RAG."""
    ext = Path(file_path).suffix.lower().lstrip(".")

    if ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == "pdf":
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext in ("doc", "docx"):
        import docx2txt

        return docx2txt.process(file_path)

    return ""

# ---------------------- TEXT CHUNKING -----------------------
def _chunk_text(text: str, chunk_size: int = 800) -> list:
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if sum(len(w) + 1 for w in current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

# ---------------------- LOAD DOCUMENT -----------------------
def _build_index(chunks):
    import faiss
    import numpy as np

    vectors = get_embedder().encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index


def _save_active_document(filename):
    import tempfile

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=ACTIVE_DOCUMENT_PATH.parent,
            prefix=".active-document-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump({"filename": filename}, temporary_file)
        os.replace(temporary_path, ACTIVE_DOCUMENT_PATH)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def _restore_active_document():
    global DOCS, INDEX

    if INDEX is not None or DOCS or not ACTIVE_DOCUMENT_PATH.is_file():
        return

    with index_lock:
        if INDEX is not None or DOCS or not ACTIVE_DOCUMENT_PATH.is_file():
            return

        marker = json.loads(ACTIVE_DOCUMENT_PATH.read_text(encoding="utf-8"))
        filename = marker.get("filename") if isinstance(marker, dict) else None
        if (
            not isinstance(filename, str)
            or secure_filename(filename) != filename
            or Path(filename).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS
        ):
            raise ValueError("Persisted RAG document marker is invalid.")

        document_path = Path(UPLOAD_DIR) / filename
        if not document_path.is_file():
            raise FileNotFoundError("Persisted RAG source document is missing.")

        chunks = _chunk_text(_extract_text_from_file(str(document_path)))
        if not chunks:
            raise ValueError("Persisted RAG source document contains no readable text.")

        index = _build_index(chunks)
        DOCS = chunks
        INDEX = index
        logger.info("Restored persistent RAG index with %d chunks", len(chunks))


def load_knowledge_from_file(file_storage) -> dict:
    """Save uploaded file, extract chunks, build FAISS index."""
    global DOCS, INDEX

    filename = secure_filename(file_storage.filename or "")
    extension = Path(filename).suffix.lower()
    if not filename or extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        return {
            "status": "error",
            "message": "Unsupported document type. Use PDF, TXT, DOC, or DOCX.",
        }

    save_path = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(save_path)

    text = _extract_text_from_file(save_path)
    chunks = _chunk_text(text)

    if not chunks:
        DOCS = []
        INDEX = None
        ACTIVE_DOCUMENT_PATH.unlink(missing_ok=True)
        return {"status": "error", "message": "No readable text found in file."}

    index = _build_index(chunks)
    _save_active_document(filename)
    DOCS = chunks
    INDEX = index

    return {"status": "ok", "chunks_loaded": len(DOCS)}

# ---------------------- CONTEXT RETRIEVAL -------------------
def _retrieve_context(query: str, top_k: int = 5) -> str:
    """Return top_k most relevant chunks."""
    _restore_active_document()
    if INDEX is None or not DOCS:
        return "No context uploaded yet."

    import numpy as np

    q_vec = get_embedder().encode([query])
    distances, indices = INDEX.search(np.array(q_vec), top_k)
    selected = [DOCS[i] for i in indices[0]]
    return "\n\n".join(selected)

# ---------------------- MAIN RAG CHAT ------------------------
def build_rag_prompt(question: str, context: str) -> str:
    """Build the existing production RAG prompt without changing its wording."""
    if "summarize" in question.lower() or "summary" in question.lower():
        style = "Summarize the content clearly in a concise way using ONLY the context below."
    else:
        style = "Answer concisely using ONLY the information in the context below."

    return f"""
You are a careful assistant. Follow instructions strictly.

Instruction:
{style}

CONTEXT:
{context}

QUESTION:
{question}

If the context does not contain enough information, reply exactly:
"I don't have enough information in the uploaded document to answer that."
"""


def chat_with_ollama_rag(question: str) -> str:
    """Use FAISS context + llama3.1 to answer user question."""
    context = _retrieve_context(question)
    prompt = build_rag_prompt(question, context)

    data = request_ollama(prompt, timeout=180, options={"temperature": 0.1})
    if data is None:
        return UNAVAILABLE_MESSAGE
    return data.get("response", "No response from AI.")


def generate_suggested_questions() -> list[str]:
    """Ask Ollama for document-grounded questions after indexing."""
    if not DOCS:
        return []

    context = "\n\n".join(DOCS[:8])
    prompt = f"""
Create exactly 5 concise questions that a reader could answer from the document context below.
Return ONLY a valid JSON array of question strings. Do not include markdown or commentary.

DOCUMENT CONTEXT:
{context}
"""
    data = request_ollama(prompt, timeout=120, options={"temperature": 0.7})
    if data is None:
        return []

    try:
        raw = data.get("response", "")
        questions = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        if isinstance(questions, list):
            cleaned_questions = [str(question).strip() for question in questions if str(question).strip()]
            if len(cleaned_questions) >= 5:
                return cleaned_questions[:5]
    except (AttributeError, TypeError, ValueError):
        logger.warning("Ollama suggested-questions response was not valid JSON")

    return []
