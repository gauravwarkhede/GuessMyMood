import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Guess My Mood",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "mood_model.keras"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

SMOOTHING_FRAMES = 10

CONFIDENCE_THRESHOLD = 35.0


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    return detector


model = load_model()

face_detector = load_face_detector()


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Guess My Mood")

st.markdown(
    """
    ### Facial Emotion Recognition

    Our CNN model analyzes facial expressions and predicts
    one of seven emotions.
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Choose detection mode:",
    [
        "📸 Snapshot",
        "📹 Live Detection"
    ]
)


st.sidebar.markdown("---")

st.sidebar.write(
    "**Emotions:**"
)

for emotion in EMOTIONS:

    st.sidebar.write(
        f"• {emotion}"
    )


st.sidebar.markdown("---")

st.sidebar.info(
    "Model: CNN\n\n"
    "Input: 48 × 48 grayscale\n\n"
    "Smoothing: 10 frames"
)


# ============================================================
# SNAPSHOT MODE
# ============================================================

if mode == "📸 Snapshot":

    st.header("📸 Snapshot Mode")

    st.write(
        "Take a picture and let the CNN predict your mood."
    )


    camera_image = st.camera_input(
        "Take a picture"
    )


    if camera_image is not None:

        # ----------------------------------------------------
        # Convert uploaded image
        # ----------------------------------------------------

        file_bytes = np.asarray(
            bytearray(camera_image.read()),
            dtype=np.uint8
        )


        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            st.error(
                "Could not read the image."
            )

        else:

            # ------------------------------------------------
            # Convert to grayscale
            # ------------------------------------------------

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )


            # ------------------------------------------------
            # Detect faces
            # ------------------------------------------------

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )


            if len(faces) == 0:

                st.warning(
                    "😕 No face detected. "
                    "Please try again with your face clearly visible."
                )

                st.image(
                    frame,
                    channels="BGR",
                    use_container_width=True
                )


            else:

                # --------------------------------------------
                # Process first detected face
                # --------------------------------------------

                x, y, w, h = faces[0]


                face = gray[
                    y:y + h,
                    x:x + w
                ]


                # --------------------------------------------
                # Resize
                # --------------------------------------------

                face = cv2.resize(
                    face,
                    (48, 48),
                    interpolation=cv2.INTER_AREA
                )


                # --------------------------------------------
                # Normalize
                # --------------------------------------------

                face = (
                    face.astype("float32")
                    / 255.0
                )


                # --------------------------------------------
                # Reshape
                # --------------------------------------------

                face = face.reshape(
                    1,
                    48,
                    48,
                    1
                )


                # --------------------------------------------
                # Predict
                # --------------------------------------------

                predictions = model.predict(
                    face,
                    verbose=0
                )[0]


                predicted_index = np.argmax(
                    predictions
                )


                emotion = EMOTIONS[
                    predicted_index
                ]


                confidence = (
                    predictions[predicted_index]
                    * 100
                )


                # --------------------------------------------
                # Draw face
                # --------------------------------------------

                result_frame = frame.copy()


                cv2.rectangle(
                    result_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )


                cv2.putText(
                    result_frame,
                    f"{emotion} {confidence:.1f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )


                # --------------------------------------------
                # Display image
                # --------------------------------------------

                st.image(
                    result_frame,
                    channels="BGR",
                    use_container_width=True
                )


                # --------------------------------------------
                # Result
                # --------------------------------------------

                st.subheader(
                    "🎯 Prediction"
                )


                if confidence >= CONFIDENCE_THRESHOLD:

                    st.success(
                        f"Your mood appears to be: "
                        f"**{emotion}**"
                    )

                else:

                    st.warning(
                        f"The model is not very confident. "
                        f"Prediction: **{emotion}**"
                    )


                st.write(
                    f"Confidence: **{confidence:.2f}%**"
                )


                # --------------------------------------------
                # Probability chart
                # --------------------------------------------

                st.subheader(
                    "📊 Emotion Probabilities"
                )


                probability_data = {

                    EMOTIONS[i]:
                    float(predictions[i] * 100)

                    for i in range(
                        len(EMOTIONS)
                    )

                }


                st.bar_chart(
                    probability_data
                )


# ============================================================
# LIVE DETECTION MODE
# ============================================================

else:

    st.header("📹 Live Detection")

    st.write(
        "Allow camera access and click **START** "
        "to begin real-time emotion detection."
    )


    # --------------------------------------------------------
    # SESSION STATE FOR PREDICTION HISTORY
    # --------------------------------------------------------

    if "prediction_history" not in st.session_state:

        st.session_state.prediction_history = deque(
            maxlen=SMOOTHING_FRAMES
        )


    # --------------------------------------------------------
    # VIDEO CALLBACK
    # --------------------------------------------------------

    def video_frame_callback(
        frame: av.VideoFrame
    ) -> av.VideoFrame:

        # ----------------------------------------------------
        # Convert frame to OpenCV image
        # ----------------------------------------------------

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # Flip camera horizontally
        # ----------------------------------------------------

        image = cv2.flip(
            image,
            1
        )


        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        # ----------------------------------------------------
        # Process faces
        # ----------------------------------------------------

        for (x, y, w, h) in faces:

            # ----------------------------------------------
            # Crop face
            # ----------------------------------------------

            face = gray[
                y:y + h,
                x:x + w
            ]


            # ----------------------------------------------
            # Resize
            # ----------------------------------------------

            face = cv2.resize(
                face,
                (48, 48),
                interpolation=cv2.INTER_AREA
            )


            # ----------------------------------------------
            # Normalize
            # ----------------------------------------------

            face = (
                face.astype("float32")
                / 255.0
            )


            # ----------------------------------------------
            # Reshape
            # ----------------------------------------------

            face = face.reshape(
                1,
                48,
                48,
                1
            )


            # ----------------------------------------------
            # Prediction
            # ----------------------------------------------

            predictions = model.predict(
                face,
                verbose=0
            )[0]


            # ----------------------------------------------
            # Add prediction to smoothing history
            # ----------------------------------------------

            # NOTE:
            # A local deque is used here because the callback
            # runs in another thread.

            if not hasattr(
                video_frame_callback,
                "history"
            ):

                video_frame_callback.history = deque(
                    maxlen=SMOOTHING_FRAMES
                )


            video_frame_callback.history.append(
                predictions
            )


            # ----------------------------------------------
            # Average recent predictions
            # ----------------------------------------------

            average_predictions = np.mean(
                video_frame_callback.history,
                axis=0
            )


            # ----------------------------------------------
            # Get final prediction
            # ----------------------------------------------

            predicted_index = np.argmax(
                average_predictions
            )


            emotion = EMOTIONS[
                predicted_index
            ]


            confidence = (
                average_predictions[
                    predicted_index
                ] * 100
            )


            # ----------------------------------------------
            # Face rectangle
            # ----------------------------------------------

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )


            # ----------------------------------------------
            # Main emotion label
            # ----------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                label = (
                    f"{emotion} "
                    f"{confidence:.1f}%"
                )

            else:

                label = (
                    f"Not sure "
                    f"{confidence:.1f}%"
                )


            # ----------------------------------------------
            # Draw label
            # ----------------------------------------------

            cv2.putText(
                image,
                label,
                (x, max(y - 15, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # ----------------------------------------------
            # Top 3 emotions
            # ----------------------------------------------

            top_indices = np.argsort(
                average_predictions
            )[::-1][:3]


            text_y = y + h + 30


            for rank, index in enumerate(
                top_indices
            ):

                emotion_name = EMOTIONS[
                    index
                ]


                probability = (
                    average_predictions[index]
                    * 100
                )


                text = (
                    f"{emotion_name}: "
                    f"{probability:.1f}%"
                )


                cv2.putText(
                    image,
                    text,
                    (
                        x,
                        text_y + rank * 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )


        # ----------------------------------------------------
        # No face detected
        # ----------------------------------------------------

        if len(faces) == 0:

            cv2.putText(
                image,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )


        # ----------------------------------------------------
        # Application title
        # ----------------------------------------------------

        cv2.putText(
            image,
            "Guess My Mood",
            (
                20,
                image.shape[0] - 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Return processed frame
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


    # --------------------------------------------------------
    # START WEBRTC
    # --------------------------------------------------------

    webrtc_ctx = webrtc_streamer(

        key="guess-my-mood-live",

        mode=WebRtcMode.SENDRECV,

        video_frame_callback=video_frame_callback,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True,

        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )


    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    st.info(
        """
        **How to use Live Detection**

        1. Click **START**
        2. Allow browser camera access
        3. Keep your face clearly visible
        4. Try different facial expressions
        5. Click **STOP** when finished

        The prediction is smoothed across multiple frames
        to reduce rapid changes.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧠 Guess My Mood | CNN Facial Emotion Recognition"
)

st.caption(
    "⚠️ This project is for educational/demo purposes. "
    "Emotion predictions are not guaranteed to be accurate."
)