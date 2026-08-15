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
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv(Path(__file__).with_name(".env"))

from services.model_service import predict_image
from services.ollama_service import chat_with_ollama

#newly added
from services.rag_service import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    load_knowledge_from_file,
    chat_with_ollama_rag,
    generate_suggested_questions,
)

app = Flask(__name__)
CORS(app)
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

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/examples/documents")
def example_documents():
    """Return safe metadata for bundled RAG example documents."""
    try:
        metadata = json.loads(EXAMPLE_METADATA_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}

    documents = []
    for path in sorted(EXAMPLE_DOCUMENTS_DIR.iterdir() if EXAMPLE_DOCUMENTS_DIR.exists() else []):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
            continue

        item = metadata.get(path.name, {})
        documents.append({
            "filename": path.name,
            "title": item.get("title", path.stem.replace("-", " ")),
            "description": item.get("description", "Bundled RoadSense AI reference document."),
            "provenance": item.get("provenance", "Bundled RoadSense AI example."),
            "suggested_questions": item.get("suggested_questions", []),
            "extension": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "url": f"/api/examples/documents/{path.name}",
        })

    return jsonify({"documents": documents})


@app.get("/api/examples/images")
def example_images():
    """Return safe metadata for backend-owned road examples."""
    images = []
    if not EXAMPLE_IMAGES_DIR.exists():
        return jsonify({"images": images})

    for path in sorted(EXAMPLE_IMAGES_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        category = path.parent.name
        expected_class = IMAGE_CLASS_BY_FOLDER.get(category)
        if not expected_class:
            continue
        relative = path.relative_to(EXAMPLE_IMAGES_DIR).as_posix()
        images.append({
            "id": relative,
            "filename": path.name,
            "category": expected_class,
            "group": IMAGE_GROUP_BY_FOLDER.get(path.parent.parent.name, path.parent.parent.name),
            "url": f"/api/examples/images/{relative}",
            "expectedClass": expected_class,
            "size_bytes": path.stat().st_size,
        })

    return jsonify({"images": images})


@app.get("/api/examples/images/<path:filename>")
def serve_example_image(filename):
    """Serve only a supported image stored beneath examples/images."""
    requested = (EXAMPLE_IMAGES_DIR / filename).resolve()
    root = EXAMPLE_IMAGES_DIR.resolve()
    if root not in requested.parents or requested.suffix.lower() not in IMAGE_EXTENSIONS:
        return jsonify({"error": "Invalid example image"}), 400
    if not requested.is_file():
        return jsonify({"error": "Example image not found"}), 404
    return send_from_directory(requested.parent, requested.name, as_attachment=False)


@app.get("/api/examples/documents/<path:filename>")
def serve_example_document(filename):
    """Serve one allowlisted bundled document without exposing filesystem paths."""
    safe_name = secure_filename(filename)
    if safe_name != filename or Path(filename).name != filename:
        return jsonify({"error": "Invalid document name"}), 400

    path = EXAMPLE_DOCUMENTS_DIR / safe_name
    if not path.is_file() or path.suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
        return jsonify({"error": "Example document not found"}), 404

    return send_from_directory(EXAMPLE_DOCUMENTS_DIR, safe_name, as_attachment=False)

# ---------------- PREDICTION ----------------
@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

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
    app.run(port=5000, debug=True)




