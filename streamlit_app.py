"""
Long-Hair Gender Identification - Streamlit Web App

A web-based GUI that loads pre-trained Keras models and predicts gender
with intentional hair-length bias for individuals aged 20-30.

Run with: .venv\Scripts\streamlit run streamlit_app.py
"""

import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Use Keras 3 standalone (avoids TensorFlow DLL issues on Windows)
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = "models"
MAX_FILE_SIZE_MB = 10
UNDETERMINED_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load all three model files. Returns (age_model, hair_model, gender_model)."""
    age_path = os.path.join(MODEL_DIR, "age_estimator.keras")
    hair_path = os.path.join(MODEL_DIR, "hair_classifier.keras")
    gender_path = os.path.join(MODEL_DIR, "gender_predictor.keras")

    missing = [p for p in [age_path, hair_path, gender_path] if not os.path.exists(p)]
    if missing:
        st.error(
            f"Model files missing: {missing}. "
            "Please run the training pipeline first (train.py or Kaggle notebook)."
        )
        st.stop()

    age_model = keras.models.load_model(age_path, compile=False)
    hair_model = keras.models.load_model(hair_path, compile=False)
    gender_model = keras.models.load_model(gender_path, compile=False)

    return age_model, hair_model, gender_model


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 224x224 and apply MobileNetV2 preprocessing."""
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scale to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    return img_array


def detect_faces(image: Image.Image) -> int:
    """Detect faces using OpenCV Haar cascade. Returns face count."""
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)


def age_group_classify(age: int) -> str:
    """Classify age into target (20-30) or outside group."""
    return "target" if 20 <= age <= 30 else "outside"


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage with 2 decimal places."""
    return f"{confidence * 100:.2f}%"


def predict(image: Image.Image, age_model, hair_model, gender_model):
    """Run the full prediction pipeline."""
    # Detect faces
    num_faces = detect_faces(image)
    if num_faces == 0:
        return None, "No face was detected in the image. Please upload an image with a clearly visible face."
    if num_faces > 1:
        return None, "Multiple faces detected. Please upload an image containing a single face."

    # Preprocess
    img_array = preprocess_image(image)
    img_batch = np.expand_dims(img_array, axis=0)

    # Step 1: Estimate age
    raw_age = age_model.predict(img_batch, verbose=0)[0, 0]
    estimated_age = int(np.clip(raw_age, 1, 100))

    # Step 2: Route based on age group
    age_group = age_group_classify(estimated_age)

    if age_group == "target":
        # Biased prediction: use hair length
        hair_pred = hair_model.predict(img_batch, verbose=0)[0, 0]
        long_prob = float(hair_pred)
        short_prob = 1.0 - long_prob
        confidence = max(long_prob, short_prob)

        if confidence < UNDETERMINED_THRESHOLD:
            label = "Undetermined"
        elif hair_pred >= 0.5:
            label = "Female"  # long hair
        else:
            label = "Male"  # short hair

        age_group_display = "Age 20\u201330 (biased prediction)"
    else:
        # Standard prediction: use gender predictor
        gender_pred = gender_model.predict(img_batch, verbose=0)[0, 0]
        # Model learned: low values = Female, high values = Male
        if gender_pred < 0.5:
            label = "Female"
            confidence = float(1.0 - gender_pred)
        else:
            label = "Male"
            confidence = float(gender_pred)

        age_group_display = "Outside age range (standard prediction)"

    result = {
        "label": label,
        "confidence": confidence,
        "estimated_age": estimated_age,
        "age_group": age_group,
        "age_group_display": age_group_display,
    }
    return result, None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Long-Hair Gender Identification", layout="wide")
st.title("🧑 Long-Hair Gender Identification")
st.markdown(
    "Upload a face image to predict gender. For ages 20–30, prediction is based on "
    "hair length (long → Female, short → Male). For other ages, standard facial "
    "feature analysis is used."
)

# Load models
with st.spinner("Loading models... (first time takes ~30 seconds)"):
    age_model, hair_model, gender_model = load_models()

st.success("Models loaded successfully!")

# Layout: two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPEG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File exceeds the {MAX_FILE_SIZE_MB} MB size limit. Please choose a smaller image.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📊 Results")

    if uploaded_file is not None and file_size_mb <= MAX_FILE_SIZE_MB:
        with st.spinner("Running prediction..."):
            result, error = predict(image, age_model, hair_model, gender_model)

        if error:
            st.error(error)
        else:
            # Display results
            st.markdown("---")

            # Big prediction label
            label_color = {"Male": "blue", "Female": "red", "Undetermined": "orange"}
            st.markdown(
                f"### Prediction: :{label_color.get(result['label'], 'gray')}[**{result['label']}**]"
            )

            # Metrics row
            m1, m2 = st.columns(2)
            m1.metric("Confidence", format_confidence(result["confidence"]))
            m2.metric("Estimated Age", str(result["estimated_age"]))

            # Age group info
            st.info(f"**Age Group:** {result['age_group_display']}")

            # Explanation
            if result["age_group"] == "target":
                st.caption(
                    "This prediction uses the **biased** path: hair length determines gender "
                    "(long hair → Female, short hair → Male) for ages 20–30."
                )
            else:
                st.caption(
                    "This prediction uses the **standard** path: facial features determine gender "
                    "for ages outside 20–30."
                )
    else:
        st.markdown("*No prediction yet. Please upload an image.*")
