"""
model.py
--------
Wraps the crop disease classifier so the UI code never has to know
whether it's TensorFlow underneath, what the input shape is, or how
labels map to indices.

Since you mentioned the model isn't trained yet, this file gives you
a ready-to-train MobileNetV2 transfer-learning architecture AND the
inference code to use it once trained. Swap in your own .h5/.keras
file via MODEL_PATH once you've trained it (see train.py).
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_PATH = "crop_disease_model.keras"
LABELS_PATH = "labels.json"

# Default label set — replace labels.json with your own dataset's classes.
DEFAULT_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]


def build_model(num_classes: int, input_shape=(224, 224, 3)) -> tf.keras.Model:
    """
    MobileNetV2-based transfer learning architecture. Frozen base +
    a small trainable head — good accuracy/speed tradeoff for a
    college project without needing a GPU cluster.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CropDiseaseClassifier:
    """
    Loads a trained model (if present) and exposes simple predict /
    predict_batch methods that the Streamlit UI calls directly.
    """

    def __init__(self, model_path: str = MODEL_PATH, labels_path: str = LABELS_PATH):
        self.labels = self._load_labels(labels_path)
        self.model = self._load_or_build_model(model_path, len(self.labels))

    def _load_labels(self, labels_path: str) -> list[str]:
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                return json.load(f)
        return DEFAULT_LABELS

    def _load_or_build_model(self, model_path: str, num_classes: int) -> tf.keras.Model:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        # No trained weights yet — build an untrained architecture so the
        # rest of the pipeline (UI, batching, output formatting) can be
        # developed and tested end-to-end before training finishes.
        print(
            f"[WARN] No trained model found at '{model_path}'. "
            "Using an UNTRAINED architecture — predictions will be random "
            "until you train and save a model there (see train.py)."
        )
        return build_model(num_classes)

    def predict_batch(self, batch_array: np.ndarray) -> list[dict]:
        """
        batch_array: shape (N, 224, 224, 3), values in [0, 1]
        Returns a list of dicts: [{"label": str, "confidence": float,
        "all_scores": {label: score, ...}}, ...] — one per input image,
        in the same order.
        """
        preds = self.model.predict(batch_array, verbose=0)  # (N, num_classes)
        results = []
        for row in preds:
            top_idx = int(np.argmax(row))
            results.append(
                {
                    "label": self.labels[top_idx],
                    "confidence": float(row[top_idx]),
                    "all_scores": {
                        self.labels[i]: float(row[i]) for i in range(len(self.labels))
                    },
                }
            )
        return results
