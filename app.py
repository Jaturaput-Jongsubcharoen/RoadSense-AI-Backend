# from flask import Flask, request, jsonify
# from services.model_service import predict_image
# from services.gemini_service import chatbot_response
# from services.model_service import predict_image

# from flask_cors import CORS



# app = Flask(__name__)
# CORS(app)

# @app.get("/api/health")
# def health():
#     return {"status": "ok"}

# # ------------------------------
# # PREDICTION ENDPOINT
# # ------------------------------
# @app.post("/api/predict")
# def predict():
#     file = request.files.get("image")

#     if not file:
#         return jsonify({"error": "No image uploaded"}), 400

#     result = predict_image(file)
#     return jsonify(result)

# @app.post("/api/gen/summary")
# def summarize():
#     data = request.json
#     text = data.get("text", "")
#     summary = generate_summary(text)
#     return jsonify({"summary": summary})

# # ------------------------------
# # CHATBOT ENDPOINT (Gemini)
# # ------------------------------
# @app.post("/api/chat")
# def chat():
#     data = request.get_json()
#     message = data.get("message", "")

#     response = chat_with_gemini(message)
#     return jsonify({"response": response})


# # ------------------------------
# # OPTIONAL EXPLANATION ENDPOINT
# # ------------------------------
# @app.post("/api/explain")
# def explain():
#     data = request.get_json()
#     class_name = data.get("class_name")
#     confidence = data.get("confidence")

#     explanation = explain_prediction(class_name, confidence)
#     return jsonify({"explanation": explanation})

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, jsonify
# from flask_cors import CORS

# from services.model_service import predict_image
# from services.gemini_service import chatbot_response   # simple rule-based bot


# app = Flask(__name__)
# CORS(app)


# # ---------------------------------------
# # HEALTH CHECK
# # ---------------------------------------
# @app.get("/api/health")
# def health():
#     return {"status": "ok"}


# # ---------------------------------------
# # IMAGE PREDICTION
# # ---------------------------------------
# @app.post("/api/predict")
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]
#     result = predict_image(file)
#     return jsonify(result)


# # ---------------------------------------
# # SIMPLE CHATBOT (NO GEMINI)
# # ---------------------------------------
# @app.post("/api/chat")
# def chat():
#     data = request.get_json()
#     message = data.get("message", "")
    
#     reply = chatbot_response(message)
#     return jsonify({"response": reply})


# # ---------------------------------------
# # REMOVE SUMMARY + EXPLAIN (NOT USED)
# # ---------------------------------------
# # (You can add later if needed)


# # ---------------------------------------
# # RUN BACKEND
# # ---------------------------------------
# if __name__ == "__main__":
#     app.run(port=5000, debug=True)



# from flask import Flask, request, jsonify
# from services.model_service import predict_image
# from services.gemini_service import chatbot_response
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# @app.get("/api/health")
# def health():
#     return {"status": "ok"}

# # -------- IMAGE PREDICTION ENDPOINT ----------
# @app.post("/api/predict")
# def predict():
#     file = request.files.get("image")
#     if not file:
#         return jsonify({"error": "No image uploaded"}), 400

#     result = predict_image(file)
#     return jsonify(result)

# # -------- AI CHAT ENDPOINT (Ollama Local LLM) ----------
# @app.post("/api/chat")
# def chat():
#     data = request.get_json()
#     message = data.get("message", "")

#     reply = chatbot_response(message)
#     return jsonify({"response": reply})

# if __name__ == "__main__":
#     app.run(debug=True)


import os
import json
import random
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv(Path(__file__).with_name(".env"))

from services.ollama_service import chat_with_ollama

#newly added
from services.rag_service import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    load_knowledge_from_file,
    chat_with_ollama_rag,
    generate_suggested_questions,
)

app = Flask(__name__)


def _cors_origins():
    configured_origins = os.getenv("CORS_ORIGINS", "*").strip()
    if not configured_origins or configured_origins == "*":
        return "*"
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


CORS(app, origins=_cors_origins())
EXAMPLE_DOCUMENTS_DIR = Path(__file__).resolve().parent / "examples" / "documents"
EXAMPLE_METADATA_PATH = EXAMPLE_DOCUMENTS_DIR / "metadata.json"
EXAMPLE_IMAGES_DIR = Path(__file__).resolve().parent / "examples" / "images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_CLASS_BY_FOLDER = {
    "Broken-Road-Sign-Issues": "Broken Road Sign Issues",
    "Damaged-Road-issues": "Damaged Road issues",
    "Illegal-Parking-Issues": "Illegal Parking Issues",
    "Littering-Garbage-on-Public-Places-Issues": "Littering Garbage on Public Places Issues",
    "Mixed-Issues": "Mixed Issues",
    "Pothole-Issues": "Pothole Issues",
    "Vandalism-Issues": "Vandalism Issues",
}
IMAGE_GROUP_BY_FOLDER = {
    "Road-Issues": "Road Issues",
    "Public-Cleanliness-and-Environmental-Issues": "Public Cleanliness and Environmental Issues",
}
_example_image_cache = {"signature": None, "items": []}
_example_document_cache = {"signature": None, "items": []}


def _directory_signature(directory, extensions):
    if not directory.exists():
        return ()
    return tuple(sorted(
        (str(path.relative_to(directory)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and not path.name.startswith(".")
    ))


def _discover_example_images():
    signature = _directory_signature(EXAMPLE_IMAGES_DIR, IMAGE_EXTENSIONS)
    if signature == _example_image_cache["signature"]:
        return _example_image_cache["items"]

    items = []
    for relative_name, size_bytes, _ in signature:
        path = EXAMPLE_IMAGES_DIR / relative_name
        category_folder = path.parent.name
        expected_class = IMAGE_CLASS_BY_FOLDER.get(category_folder)
        if not expected_class:
            continue
        items.append({
            "id": relative_name.replace("\\", "/"),
            "filename": path.name,
            "category": expected_class,
            "group": IMAGE_GROUP_BY_FOLDER.get(path.parent.parent.name, path.parent.parent.name),
            "url": f"/api/examples/images/{relative_name.replace(chr(92), '/')}",
            "expectedClass": expected_class,
            "size_bytes": size_bytes,
        })

    _example_image_cache.update(signature=signature, items=items)
    return items


def _discover_example_documents():
    signature = _directory_signature(EXAMPLE_DOCUMENTS_DIR, ALLOWED_DOCUMENT_EXTENSIONS)
    metadata_signature = EXAMPLE_METADATA_PATH.stat().st_mtime_ns if EXAMPLE_METADATA_PATH.exists() else None
    cache_signature = (signature, metadata_signature)
    if cache_signature == _example_document_cache["signature"]:
        return _example_document_cache["items"]

    try:
        metadata = json.loads(EXAMPLE_METADATA_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}

    documents = []
    for filename, size_bytes, _ in signature:
        path = EXAMPLE_DOCUMENTS_DIR / filename
        item = metadata.get(path.name, {})
        documents.append({
            "filename": path.name,
            "title": item.get("title", path.stem.replace("-", " ")),
            "description": item.get("description", "Bundled RoadSense AI reference document."),
            "provenance": item.get("provenance", "Bundled RoadSense AI example."),
            "suggested_questions": item.get("suggested_questions", []),
            "extension": path.suffix.lower().lstrip("."),
            "size_bytes": size_bytes,
            "url": f"/api/examples/documents/{path.name}",
        })

    _example_document_cache.update(signature=cache_signature, items=documents)
    return documents

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/examples/documents")
def example_documents():
    """Return safe metadata for bundled RAG example documents."""
    return jsonify({"documents": _discover_example_documents()})


@app.get("/api/examples/images")
def example_images():
    """Return safe metadata for backend-owned road examples."""
    grouped = {}
    for image in _discover_example_images():
        grouped.setdefault(image["expectedClass"], []).append(image)
    selected = []
    for category in IMAGE_CLASS_BY_FOLDER.values():
        selected.extend(random.sample(grouped.get(category, []), min(2, len(grouped.get(category, [])))))
    return jsonify({"images": selected})


@app.get("/api/examples/images/<path:filename>")
def serve_example_image(filename):
    """Serve only a supported image stored beneath examples/images."""
    requested = (EXAMPLE_IMAGES_DIR / filename).resolve()
    root = EXAMPLE_IMAGES_DIR.resolve()
    if root not in requested.parents or requested.suffix.lower() not in IMAGE_EXTENSIONS:
        return jsonify({"error": "Invalid example image"}), 400
    if not requested.is_file():
        return jsonify({"error": "Example image not found"}), 404
    response = send_from_directory(requested.parent, requested.name, as_attachment=False, max_age=86400)
    response.cache_control.public = True
    return response


@app.get("/api/examples/documents/<path:filename>")
def serve_example_document(filename):
    """Serve one allowlisted bundled document without exposing filesystem paths."""
    safe_name = secure_filename(filename)
    if safe_name != filename or Path(filename).name != filename:
        return jsonify({"error": "Invalid document name"}), 400

    path = EXAMPLE_DOCUMENTS_DIR / safe_name
    if not path.is_file() or path.suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
        return jsonify({"error": "Example document not found"}), 404

    response = send_from_directory(EXAMPLE_DOCUMENTS_DIR, safe_name, as_attachment=False, max_age=86400)
    response.cache_control.public = True
    return response

# ---------------- PREDICTION ----------------
@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    from services.model_service import predict_image

    result = predict_image(file)
    return jsonify(result)


# ---------------- CHATBOT ----------------
@app.post("/api/chat")
def chat():
    data = request.get_json()
    msg = data.get("message", "")

    reply = chat_with_ollama(msg)
    return jsonify({"response": reply})


#newly added

# ---------------- RAG: UPLOAD FILE ----------------
@app.post("/api/rag/upload")
def rag_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "File name is empty"}), 400

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        return jsonify({
            "error": "Unsupported document type. Use PDF, TXT, DOC, or DOCX."
        }), 400

    result = load_knowledge_from_file(file)
    status = 200 if result.get("status") == "ok" else 400
    return jsonify(result), status


# ---------------- RAG: ASK QUESTION ----------------
@app.post("/api/rag/ask")
def rag_ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Missing 'question' in request"}), 400

    answer = chat_with_ollama_rag(question)
    return jsonify({"answer": answer})


@app.post("/api/rag/questions")
def rag_questions():
    questions = generate_suggested_questions()
    if not questions:
        return jsonify({"questions": [], "message": "No questions generated. Ollama may be unavailable."}), 503
    return jsonify({"questions": questions})


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )




