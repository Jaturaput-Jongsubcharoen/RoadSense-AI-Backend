# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io

#model = tf.keras.models.load_model("models/efficientnet_best_model.keras")

# IMG_SIZE = 224
#class_names = ["Broken Road Sign Issues", "Damaged Road issues", "Illegal Parking Issues", "Littering Garbage on Public Places Issues", "Mixed Issues", "Pothole Issues", "Vandalism Issues"]

# # ------------------------------------------------------------
# # 1️⃣ ROAD / NON-ROAD DETECTION using MobileNetV2 (pretrained)
# # ------------------------------------------------------------
# mobilenet = tf.keras.applications.MobileNetV2(weights="imagenet")
# mobilenet_input_size = (224, 224)

# def is_road_scene(pil_img):
#     # Resize for MobileNet
#     img = pil_img.resize(mobilenet_input_size)
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

#     preds = mobilenet.predict(x)
#     decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

#     # Extract label names (like 'highway', 'street', 'road', 'pavement')
#     labels = [d[1].lower() for d in decoded]

#     # If any keyword exists → ROAD SCENE
#     ROAD_KEYWORDS = ["road", "street", "highway", "pavement", "lane", "expressway"]

#     if any(keyword in label for label in labels for keyword in ROAD_KEYWORDS):
#         return True

#     return False


# def predict_image(file):
#     img = Image.open(file.stream).convert("RGB")
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     img = np.array(img).astype("float32")
#     img = np.expand_dims(img, axis=0)

#     preds = model.predict(img)[0]
#     predicted_class = class_names[np.argmax(preds)]
#     confidence = float(np.max(preds))

#     return {
#         "class": predicted_class,
#         "confidence": float(confidence * 100)
#     }



# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import io

# # ---------------------------------------------
# # Load your trained model
# # ---------------------------------------------
# model = tf.keras.models.load_model("models/efficientnet_best_model.keras")

# # Replace these with your real class names
# class_names = ["Broken Road Sign Issues", "Damaged Road issues", "Illegal Parking Issues", "Littering Garbage on Public Places Issues", "Mixed Issues", "Pothole Issues", "Vandalism Issues"]


# # ---------------------------------------------------------
# # 1️⃣ QUICK NON-ROAD IMAGE CHECK (color heuristic)
# # ---------------------------------------------------------
# def is_obviously_not_road(image: Image.Image):
#     """Reject images that are obviously not roads using simple color checks."""

#     width, height = image.size
#     img = image.convert("RGB")
#     pixels = img.load()

#     greens = 0
#     blues = 0
#     total = width * height

#     # Sample pixels every 20th pixel (fast)
#     for x in range(0, width, 20):
#         for y in range(0, height, 20):
#             r, g, b = pixels[x, y]

#             if g > r and g > b:
#                 greens += 1
#             if b > r and b > g:
#                 blues += 1

#     green_ratio = greens / ((width / 20) * (height / 20))
#     blue_ratio = blues / ((width / 20) * (height / 20))

#     # If lots of green (grass / animals / trees) → reject
#     if green_ratio > 0.30:
#         return True

#     # If lots of blue (sky) → reject
#     if blue_ratio > 0.30:
#         return True

#     return False


# # ---------------------------------------------------------
# # 2️⃣ IMAGE PREDICTION LOGIC
# # ---------------------------------------------------------
# def preprocess_image(img: Image.Image):
#     img = img.resize((224, 224))  # change if your model uses a different size
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)


# def predict_image(file):
#     try:
#         # Load image
#         img = Image.open(file.stream).convert("RGB")

#         # ---------------------------------------------
#         # 3️⃣ Reject NON-ROAD images BEFORE prediction
#         # ---------------------------------------------
#         if is_obviously_not_road(img):
#             return {
#                 "error": "This does not appear to be a road image. Please upload a valid road image."
#             }

#         # ---------------------------------------------
#         # 4️⃣ Run model prediction
#         # ---------------------------------------------
#         processed = preprocess_image(img)
#         preds = model.predict(processed)[0]

#         class_index = np.argmax(preds)
#         confidence = float(preds[class_index])

#         return {
#             "class_name": class_names[class_index],
#             "confidence": round(confidence, 4)
#         }

#     except Exception as e:
#         return {"error": f"Prediction failed: {str(e)}"}



# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import os
# from dotenv import load_dotenv
# load_dotenv()
# # Gemini
# import google.generativeai as genai
# import io

# # ---------------------------------------------------------
# # Load road damage model
# # ---------------------------------------------------------
# model = tf.keras.models.load_model("models/efficientnet_best_model.keras")

# # Your real class labels here
# class_names = ["Broken Road Sign Issues", "Damaged Road issues", "Illegal Parking Issues", "Littering Garbage on Public Places Issues", "Mixed Issues", "Pothole Issues", "Vandalism Issues"]


# ---------------------------------------------------------
# Preprocess image
# ---------------------------------------------------------
# def preprocess_image(img: Image.Image):
#     img = img.resize((224, 224))
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)


# ---------------------------------------------------------
# Main prediction function (NO GEMINI VISION)
# ---------------------------------------------------------
# def predict_image(file):
#     try:
#         # Load and convert file to image
#         img = Image.open(file.stream).convert("RGB")

#         # Preprocess image
#         processed = preprocess_image(img)

#         # Predict
#         preds = model.predict(processed)[0]

#         class_index = int(np.argmax(preds))
#         confidence = float(preds[class_index])

#         return {
#             "class_name": class_names[class_index],
#             "confidence": round(confidence, 4)
#         }

#     except Exception as e:
#         return {"error": f"Prediction failed: {str(e)}"}
    

# ---------------------------------------------------------
# Configure Gemini API
# ---------------------------------------------------------


# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# # ---------------------------------------------------------
# # Gemini Vision: detect if the image contains a road
# # ---------------------------------------------------------
# def is_road_image_gemini(img: Image.Image):
#     """Uses Gemini Vision to check if the uploaded image contains a road."""

#     # Convert image to bytes
#     img_bytes = io.BytesIO()
#     img.save(img_bytes, format="JPEG")
#     img_bytes = img_bytes.getvalue()

#     prompt = """
#     You are a classifier. ONLY answer with "yes" or "no".
#     Does this image clearly contain a *road* or *street*?
#     """

#     model_vision = genai.GenerativeModel("models/gemini-3-pro-image-preview")  # FAST & CHEAP

#     response = model_vision.generate_content(
#         [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
#     )

#     answer = response.text.strip().lower()

#     if "yes" in answer:
#         return True
#     else:
#         return False


# # ---------------------------------------------------------
# # Preprocess image for ML model
# # ---------------------------------------------------------
# def preprocess_image(img: Image.Image):
#     img = img.resize((224, 224))  # change if needed
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)


# # ---------------------------------------------------------
# # Main prediction function
# # ---------------------------------------------------------
# def predict_image(file):
#     try:
#         img = Image.open(file.stream).convert("RGB")

#         # ---------------------------
#         # 1️⃣ Ask Gemini: is this a road?
#         # ---------------------------
#         is_road = is_road_image_gemini(img)

#         if not is_road:
#             return {
#                 "error": "This image does not contain a road. Please upload a valid road image."
#             }

#         # ---------------------------
#         # 2️⃣ Run your ML model
#         # ---------------------------
#         processed = preprocess_image(img)
#         preds = model.predict(processed)[0]

#         class_index = np.argmax(preds)
#         confidence = float(preds[class_index])

#         return {
#             "class_name": class_names[class_index],
#             "confidence": round(confidence, 4),
#         }

#     except Exception as e:
#         return {"error": f"Prediction failed: {str(e)}"}


import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "efficientnet_best_model.keras"
configured_model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL_PATH = (
    configured_model_path
    if configured_model_path.is_absolute()
    else DEFAULT_MODEL_PATH.parents[1] / configured_model_path
)
print("Loading ML model from:", MODEL_PATH)
model = tf.keras.models.load_model(str(MODEL_PATH))

CLASS_NAMES  = ["Broken Road Sign Issues", "Damaged Road issues", "Illegal Parking Issues", "Littering Garbage on Public Places Issues", "Mixed Issues", "Pothole Issues", "Vandalism Issues"]

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)  # <-- IMPORTANT
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(file):
    file.seek(0)
    img_bytes = file.read()
    print("FILE RECEIVED SIZE:", len(img_bytes))
    tensor = preprocess(img_bytes)
    preds = model.predict(tensor)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return {
        "class_name": CLASS_NAMES[idx],
        "confidence": confidence
    }

print("Loaded model from:", MODEL_PATH)
model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
print("Model warm-up complete")



