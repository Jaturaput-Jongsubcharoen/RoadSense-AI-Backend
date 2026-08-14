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



import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # avoid TF optimization loading
os.environ["USE_TF"] = "0"                  # disable TF usage inside transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"      # block TF loading
os.environ["TRANSFORMERS_NO_FLAX"] = "1"    # block JAX
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx2txt
import requests

# ---------------------- OLLAMA SERVER -----------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# ---------------------- FILE STORAGE ------------------------
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------- EMBEDDING MODEL ---------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

DOCS = []      # list[str] containing text chunks
INDEX = None   # faiss index

# ---------------------- FILE READING ------------------------
def _extract_text_from_file(file_path: str) -> str:
    """Read TXT / PDF / DOCX and return plain text."""
    ext = file_path.split(".")[-1].lower()

    if ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == "pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext in ("doc", "docx"):
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
def load_knowledge_from_file(file_storage) -> dict:
    """Save uploaded file, extract chunks, build FAISS index."""
    global DOCS, INDEX

    filename = file_storage.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(save_path)

    text = _extract_text_from_file(save_path)
    chunks = _chunk_text(text)

    if not chunks:
        DOCS = []
        INDEX = None
        return {"status": "error", "message": "No readable text found in file."}

    DOCS = chunks
    vectors = embedder.encode(DOCS)

    INDEX = faiss.IndexFlatL2(vectors.shape[1])
    INDEX.add(np.array(vectors))

    return {"status": "ok", "chunks_loaded": len(DOCS)}

# ---------------------- CONTEXT RETRIEVAL -------------------
def _retrieve_context(query: str, top_k: int = 5) -> str:
    """Return top_k most relevant chunks."""
    if INDEX is None or not DOCS:
        return "No context uploaded yet."

    q_vec = embedder.encode([query])
    distances, indices = INDEX.search(np.array(q_vec), top_k)
    selected = [DOCS[i] for i in indices[0]]
    return "\n\n".join(selected)

# ---------------------- MAIN RAG CHAT ------------------------
def chat_with_ollama_rag(question: str) -> str:
    """Use FAISS context + llama3.1 to answer user question."""
    context = _retrieve_context(question)

    # Smart instruction for summarization
    if "summarize" in question.lower() or "summary" in question.lower():
        style = "Summarize the content clearly in a concise way using ONLY the context below."
    else:
        style = "Answer concisely using ONLY the information in the context below."

    prompt = f"""
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

    payload = {
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=180)
        data = res.json()
        return data.get("response", "No response from AI.")
    except Exception as e:
        return f"Ollama / RAG error: {str(e)}"
