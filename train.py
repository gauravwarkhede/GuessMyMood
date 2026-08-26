# ============================================================
# GUESS MY MOOD - IMPROVED CNN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = 48
NUM_CLASSES = 7

BATCH_SIZE = 128
EPOCHS = 30

MODEL_PATH = "mood_model.keras"

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
# 2. CHECK TENSORFLOW
# ============================================================

print("\n========================================")
print("TENSORFLOW INFORMATION")
print("========================================")

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
else:
    print("No GPU detected - using CPU")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n========================================")
print("LOADING DATASET")
print("========================================")

df = pd.read_csv("dataset/train.csv")

print("Dataset loaded successfully!")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. EMOTION DISTRIBUTION
# ============================================================

print("\n========================================")
print("EMOTION DISTRIBUTION")
print("========================================")

emotion_counts = (
    df["emotion"]
    .value_counts()
    .sort_index()
)

for emotion_id, count in emotion_counts.items():

    print(
        f"{emotion_names[emotion_id]:10s} : {count}"
    )


# ============================================================
# 5. PLOT EMOTION DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    emotion_names,
    emotion_counts.values
)

plt.title("Emotion Distribution")

plt.xlabel("Emotion")

plt.ylabel("Number of Images")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 6. DISPLAY SAMPLE IMAGES
# ============================================================

print("\nDisplaying sample images...")

plt.figure(figsize=(12, 8))

for emotion in range(NUM_CLASSES):

    row = (
        df[df["emotion"] == emotion]
        .iloc[0]
    )

    pixels = np.array(
        row["pixels"].split(),
        dtype=np.uint8
    )

    image = pixels.reshape(
        IMG_SIZE,
        IMG_SIZE
    )

    plt.subplot(2, 4, emotion + 1)

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        emotion_names[emotion]
    )

    plt.axis("off")

plt.suptitle(
    "Sample Images From Each Emotion",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 7. CREATE X AND y
# ============================================================

print("\n========================================")
print("PREPARING IMAGES")
print("========================================")

# Target labels
y = df["emotion"].values


# Convert pixel strings into arrays
X = np.array([
    np.array(
        pixels.split(),
        dtype=np.uint8
    )
    for pixels in df["pixels"]
])


print("Before reshape:", X.shape)


# ============================================================
# 8. RESHAPE
# ============================================================

X = X.reshape(
    -1,
    IMG_SIZE,
    IMG_SIZE,
    1
)

print("After reshape:", X.shape)


# ============================================================
# 9. NORMALIZE
# ============================================================

X = X.astype("float32") / 255.0

print("\nPixel range:")

print("Minimum:", X.min())

print("Maximum:", X.max())


# ============================================================
# 10. TRAIN / VALIDATION SPLIT
# ============================================================

print("\n========================================")
print("TRAIN / VALIDATION SPLIT")
print("========================================")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training images:", X_train.shape)

print("Validation images:", X_val.shape)


# ============================================================
# 11. CLASS WEIGHTS
# ============================================================

print("\n========================================")
print("CALCULATING CLASS WEIGHTS")
print("========================================")

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(
        classes,
        class_weights_array
    )
)

for emotion_id, weight in class_weights.items():

    print(
        f"{emotion_names[emotion_id]:10s} : {weight:.2f}"
    )


# ============================================================
# 12. DATA AUGMENTATION
# ============================================================

print("\n========================================")
print("SETTING UP DATA AUGMENTATION")
print("========================================")

augmentation = Sequential([

    RandomFlip(
        "horizontal"
    ),

    RandomRotation(
        0.08
    ),

    RandomZoom(
        0.10
    )

], name="data_augmentation")


# ============================================================
# 13. BUILD CNN
# ============================================================

print("\n========================================")
print("BUILDING CNN")
print("========================================")

model = Sequential([

    Input(
        shape=(48, 48, 1)
    ),

    # Data augmentation
    augmentation,


    # ========================================================
    # BLOCK 1
    # ========================================================

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.25),


    # ========================================================
    # BLOCK 2
    # ========================================================

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.30),


    # ========================================================
    # BLOCK 3
    # ========================================================

    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.30),


    # ========================================================
    # BLOCK 4
    # ========================================================

    Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.35),


    # ========================================================
    # CLASSIFIER
    # ========================================================

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.50),


    # ========================================================
    # OUTPUT
    # ========================================================

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# ============================================================
# 14. MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 15. COMPILE MODEL
# ============================================================

print("\n========================================")
print("COMPILING MODEL")
print("========================================")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)

print("Model compiled successfully!")


# ============================================================
# 16. CALLBACKS
# ============================================================

print("\n========================================")
print("SETTING UP CALLBACKS")
print("========================================")

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001,

    verbose=1

)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\n========================================")
print("🚀 STARTING TRAINING")
print("========================================")

print("Maximum epochs:", EPOCHS)

print("Batch size:", BATCH_SIZE)

print("Class weights: ENABLED")

print("Data augmentation: ENABLED")

print("========================================\n")


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1

)


# ============================================================
# 18. TRAINING COMPLETE
# ============================================================

print("\n========================================")
print("🎉 TRAINING COMPLETE")
print("========================================")

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print(
    "Best validation accuracy:",
    round(
        best_val_accuracy * 100,
        2
    ),
    "%"
)

print(
    "Model saved as:",
    MODEL_PATH
)


# ============================================================
# 19. ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 20. LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


# ============================================================
# 21. LOAD BEST MODEL
# ============================================================

print("\n========================================")
print("LOADING BEST MODEL")
print("========================================")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded!")


# ============================================================
# 22. EVALUATE
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

val_loss, val_accuracy = (
    best_model.evaluate(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        verbose=1
    )
)

print(
    "\nValidation accuracy:",
    round(
        val_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# 23. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_probability = best_model.predict(

    X_val,

    batch_size=BATCH_SIZE,

    verbose=1

)


y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# 24. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(

    classification_report(

        y_val,

        y_pred,

        target_names=emotion_names,

        zero_division=0

    )

)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_val,
    y_pred
)


plt.figure(
    figsize=(10, 8)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=emotion_names,

    yticklabels=emotion_names

)


plt.title(
    "Emotion Confusion Matrix"
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "Actual Emotion"
)

plt.tight_layout()

plt.show()


# ============================================================
# 26. SAMPLE PREDICTIONS
# ============================================================

print("\nDisplaying sample predictions...")

plt.figure(
    figsize=(14, 10)
)

for i in range(12):

    image = X_val[i]

    actual = emotion_names[
        y_val[i]
    ]

    predicted = emotion_names[
        y_pred[i]
    ]

    confidence = (
        np.max(
            y_probability[i]
        ) * 100
    )

    plt.subplot(
        3,
        4,
        i + 1
    )

    plt.imshow(
        image.reshape(48, 48),
        cmap="gray"
    )

    plt.title(

        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence:.1f}%"

    )

    plt.axis("off")


plt.suptitle(
    "Sample Predictions",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n")
print("============================================")
print("🎉 GUESS MY MOOD MODEL READY!")
print("============================================")

print(
    "\nModel file:"
)

print(
    "mood_model.keras"
)

print(
    "\nBest validation accuracy:",
    round(
        best_val_accuracy * 100,
        2
    ),
    "%"
)

print("\nNext step:")

print(
    "📷 Build the webcam application"
)

print(
    "🌐 Deploy it on Render"
)

print(
    "============================================"
)