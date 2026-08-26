import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
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

EMOTIONS = [
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

    return tf.keras.models.load_model(
        "mood_model.keras"
    )


model = load_model()


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    cascade_path = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        cascade_path
    )

    return detector


face_detector = load_face_detector()


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Guess My Mood")

st.write(
    "Take a picture and let the AI guess your emotion!"
)

st.info(
    "💡 Look directly at the camera with your face clearly visible."
)


# ============================================================
# CAMERA
# ============================================================

camera_image = st.camera_input(
    "📷 Take a picture"
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if camera_image is not None:

    # --------------------------------------------------------
    # Read uploaded camera image
    # --------------------------------------------------------

    image = Image.open(
        camera_image
    ).convert("RGB")

    image_np = np.array(
        image
    )


    # --------------------------------------------------------
    # Convert to grayscale for face detection
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )


    # ========================================================
    # NO FACE FOUND
    # ========================================================

    if len(faces) == 0:

        st.warning(
            "😕 I couldn't detect a face."
        )

        st.write(
            "Try moving closer to the camera, "
            "improving the lighting, and looking "
            "directly at the camera."
        )


    # ========================================================
    # FACE FOUND
    # ========================================================

    else:

        # ----------------------------------------------------
        # Select largest detected face
        # ----------------------------------------------------

        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )


        # ----------------------------------------------------
        # Add small padding around face
        # ----------------------------------------------------

        padding = int(
            0.15 * max(w, h)
        )

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

        x2 = min(
            image_np.shape[1],
            x + w + padding
        )

        y2 = min(
            image_np.shape[0],
            y + h + padding
        )


        # ----------------------------------------------------
        # Crop face
        # ----------------------------------------------------

        face = image_np[
            y1:y2,
            x1:x2
        ]


        # ----------------------------------------------------
        # Show detected face
        # ----------------------------------------------------

        st.subheader(
            "👤 Face detected!"
        )

        st.image(
            face,
            caption="Face used by the AI",
            width=250
        )


        # ----------------------------------------------------
        # Convert face to grayscale
        # ----------------------------------------------------

        face_gray = cv2.cvtColor(
            face,
            cv2.COLOR_RGB2GRAY
        )


        # ----------------------------------------------------
        # Resize to training size
        # ----------------------------------------------------

        face_resized = cv2.resize(
            face_gray,
            (48, 48),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        face_normalized = (
            face_resized.astype(
                "float32"
            ) / 255.0
        )


        # ----------------------------------------------------
        # Add CNN dimensions
        # ----------------------------------------------------

        model_input = face_normalized.reshape(
            1,
            48,
            48,
            1
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        predictions = model.predict(
            model_input,
            verbose=0
        )[0]


        predicted_index = np.argmax(
            predictions
        )

        predicted_emotion = EMOTIONS[
            predicted_index
        ]

        confidence = (
            predictions[predicted_index]
            * 100
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 My Guess"
        )


        # Different messages for different emotions

        messages = {

            "Angry":
                "😡 You look a little angry!",

            "Disgust":
                "🤢 Something doesn't look right!",

            "Fear":
                "😨 You look a little scared!",

            "Happy":
                "😄 You look happy!",

            "Sad":
                "😢 You look a little sad.",

            "Surprise":
                "😲 You look surprised!",

            "Neutral":
                "😐 You look pretty neutral."

        }


        st.success(
            messages[predicted_emotion]
        )


        st.metric(
            "Predicted Emotion",
            predicted_emotion
        )


        st.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )


        # ====================================================
        # ALL PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Emotion probabilities"
        )


        # Sort highest → lowest

        sorted_indices = np.argsort(
            predictions
        )[::-1]


        for index in sorted_indices:

            emotion = EMOTIONS[
                index
            ]

            probability = (
                float(
                    predictions[index]
                )
            )

            percentage = (
                probability * 100
            )

            st.write(
                f"**{emotion}** — "
                f"{percentage:.1f}%"
            )

            st.progress(
                probability
            )