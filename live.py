import cv2
import numpy as np
import tensorflow as tf
from collections import deque


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

# Number of previous predictions to remember
SMOOTHING_FRAMES = 10

# Minimum confidence required
CONFIDENCE_THRESHOLD = 35.0


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():

    print("ERROR: Face detector could not be loaded.")
    exit()


print("Face detector loaded!")


# ============================================================
# START WEBCAM
# ============================================================

print("Starting webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print()
print("======================================")
print("🧠 GUESS MY MOOD - LIVE MODE")
print("======================================")
print("Press Q to quit.")
print("======================================")
print()


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)


# ============================================================
# LIVE LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = camera.read()

    if not success:

        print("Could not read webcam frame.")
        break


    # --------------------------------------------------------
    # MIRROR IMAGE
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # ========================================================
    # PROCESS FACES
    # ========================================================

    for (x, y, w, h) in faces:

        # ----------------------------------------------------
        # CROP FACE
        # ----------------------------------------------------

        face = gray[
            y:y + h,
            x:x + w
        ]


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        face = cv2.resize(
            face,
            (48, 48),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        face = (
            face.astype(
                "float32"
            ) / 255.0
        )


        # ----------------------------------------------------
        # ADD CNN DIMENSIONS
        # ----------------------------------------------------

        face = face.reshape(
            1,
            48,
            48,
            1
        )


        # ----------------------------------------------------
        # CNN PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            face,
            verbose=0
        )[0]


        # ----------------------------------------------------
        # CURRENT PREDICTION
        # ----------------------------------------------------

        current_index = np.argmax(
            predictions
        )

        current_confidence = (
            predictions[current_index] * 100
        )


        # ----------------------------------------------------
        # ADD TO HISTORY
        # ----------------------------------------------------

        prediction_history.append(
            predictions
        )


        # ----------------------------------------------------
        # AVERAGE RECENT PREDICTIONS
        # ----------------------------------------------------

        average_predictions = np.mean(
            prediction_history,
            axis=0
        )


        # ----------------------------------------------------
        # SMOOTHED PREDICTION
        # ----------------------------------------------------

        predicted_index = np.argmax(
            average_predictions
        )

        confidence = (
            average_predictions[
                predicted_index
            ] * 100
        )


        emotion = EMOTIONS[
            predicted_index
        ]


        # ====================================================
        # DRAW FACE BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if confidence >= CONFIDENCE_THRESHOLD:

            label = (
                f"{emotion} "
                f"{confidence:.1f}%"
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            label = (
                f"Not sure "
                f"{confidence:.1f}%"
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )


        # ====================================================
        # DISPLAY TOP 3 EMOTIONS
        # ====================================================

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
                frame,
                text,
                (x, text_y + rank * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


    # ========================================================
    # NO FACE MESSAGE
    # ========================================================

    if len(faces) == 0:

        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "Guess My Mood",
        (20, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Guess My Mood - Live",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Webcam stopped.")
print("Guess My Mood closed.")