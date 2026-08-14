# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# model = genai.GenerativeModel("models/gemini-1.5-flash")

# # Chat with Gemini
# # def chat_with_gemini(message: str):
# #     try:
# #         response = model.generate_content(message)
# #         return response.text
# #     except Exception as e:
# #         return f"Gemini error: {str(e)}"
# def chatbot_response(message):
#     message = message.lower()

#     if "pothole" in message:
#         return "Potholes are caused by repeated traffic stress, water seepage, and temperature changes."

#     if "road damage" in message:
#         return "Common road damages include potholes, cracks, rutting, and surface wear."

#     if "hello" in message or "hi" in message:
#         return "Hello! How can I assist you with road safety today?"

#     return "I’m not fully trained yet, but I can help with road-related issues!"


# # Optional: Explain ML prediction using Gemini
# def explain_prediction(class_name, confidence):
#     prompt = f"""
#     Explain the following road damage prediction in simple terms:
#     - Predicted class: {class_name}
#     - Confidence: {confidence*100:.2f}%
#     """
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Explanation error: {str(e)}"


# def chatbot_response(message):
#     msg = message.lower()

#     if "pothole" in msg:
#         return "Potholes form due to water seepage, freeze-thaw cycles, and repeated vehicle pressure."

#     if "road damage" in msg:
#         return "Road damage can include potholes, cracks, rutting, erosion, and surface wear."

#     if "hello" in msg or "hi" in msg:
#         return "Hello! Ask me anything about road conditions or safety."

#     return "I’m a simple assistant! Ask me about potholes, road damages, or safety tips."


# import requests

# def chatbot_response(message):
#     """
#     Sends the user's message to a local Ollama LLM (llama3.1) 
#     and returns the AI-generated response.
#     """
#     try:
#         response = requests.post(
#             "http://localhost:11434/api/generate",
#             json={
#                 "model": "llama3.1",
#                 "prompt": message
#             },
#             timeout=30
#         )

#         data = response.json()
#         return data.get("response", "No response from AI.")

#     except Exception as e:
#         return f"Error: {str(e)}"


import os

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

def chat_with_ollama(message):
    payload = {
        "model": "llama3.1",
        "prompt": message,
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload)
        data = res.json()
        return data.get("response", "No response.")
    except Exception as e:
        return f"🔥 Ollama error: {str(e)}"
