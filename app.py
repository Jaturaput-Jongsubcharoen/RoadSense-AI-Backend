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


from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from services.model_service import predict_image
from services.ollama_service import chat_with_ollama

#newly added
from services.rag_service import load_knowledge_from_file, chat_with_ollama_rag

app = Flask(__name__)
CORS(app)

@app.get("/api/health")
def health():
    return {"status": "ok"}

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


if __name__ == "__main__":
    app.run(port=5000, debug=True)




