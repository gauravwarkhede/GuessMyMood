import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Guess My Mood",
    page_icon="😊",
    layout="centered"
)


# ============================================================
# EMOTIONS
# ============================================================

emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "mood_model.keras"
    )

    return model


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Guess My Mood")

st.write(
    "Take a picture and let the AI guess your emotion!"
)


# ============================================================
# CAMERA
# ============================================================

camera_image = st.camera_input(
    "📷 Take a picture"
)


# ============================================================
# PREDICTION
# ============================================================

if camera_image is not None:

    # Open image
    image = Image.open(
        camera_image
    ).convert("L")

    # Resize to model input
    image = image.resize(
        (48, 48)
    )

    # Convert to numpy
    image_array = np.array(
        image
    )

    # Normalize
    image_array = (
        image_array.astype(
            "float32"
        ) / 255.0
    )

    # Add dimensions
    image_array = image_array.reshape(
        1,
        48,
        48,
        1
    )


    # ========================================================
    # PREDICT
    # ========================================================

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_emotion = emotion_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index]
        * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 My Guess"
    )

    st.success(
        f"Your mood looks **{predicted_emotion}**!"
    )

    st.metric(
        "Confidence",
        f"{confidence:.1f}%"
    )


    # ========================================================
    # ALL EMOTIONS
    # ========================================================

    st.subheader(
        "Emotion probabilities"
    )

    for i in range(7):

        st.write(
            f"{emotion_names[i]}: "
            f"{predictions[i] * 100:.1f}%"
        )

        st.progress(
            float(predictions[i])
        )