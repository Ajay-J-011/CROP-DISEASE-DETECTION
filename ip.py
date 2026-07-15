"""
Crop Disease Detection — CNN Classifier
=========================================
Trains an image classifier on the crop_disease_dataset (PlantVillage subset,
38 classes: 14 crop species x healthy/disease status).

Requirements:
    pip install tensorflow pandas scikit-learn matplotlib pillow --break-system-packages

Usage:
    1. Unzip crop_disease_dataset.zip into the same folder as this script,
       so you have: ./crop_disease_dataset/labels.csv and class subfolders.
    2. python train_crop_disease_model.py

Outputs:
    - crop_disease_model.h5      (trained Keras model)
    - training_history.png       (accuracy/loss curves)
    - classification_report.txt  (per-class precision/recall/F1)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR = "crop_disease_dataset"
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")
IMG_SIZE = (32, 32)          # matches the dataset's native resolution
BATCH_SIZE = 64
EPOCHS = 25
SEED = 42
MODEL_OUT = "crop_disease_model.h5"

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Load labels and build file paths
# ---------------------------------------------------------------------------
def load_dataframe():
    df = pd.read_csv(LABELS_CSV)
    df["filepath"] = df["filename"].apply(lambda f: os.path.join(DATASET_DIR, f))
    missing = df[~df["filepath"].apply(os.path.exists)]
    if len(missing):
        print(f"Warning: {len(missing)} files listed in labels.csv were not found on disk.")
        df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Load images into memory as numpy arrays
#    (dataset is small enough — ~17k images at 32x32 — to fit in RAM)
# ---------------------------------------------------------------------------
def load_images(df):
    images = np.zeros((len(df), IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    for i, path in enumerate(df["filepath"]):
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        images[i] = np.asarray(img, dtype=np.float32) / 255.0
    labels = df["class_id"].to_numpy()
    return images, labels


# ---------------------------------------------------------------------------
# 3. Build a compact CNN suited to small 32x32 images
# ---------------------------------------------------------------------------
def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 4. Data augmentation (helps a lot given the small 32x32 images)
# ---------------------------------------------------------------------------
def build_augmenter():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])


# ---------------------------------------------------------------------------
# 5. Plot and save training curves
# ---------------------------------------------------------------------------
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("Saved training_history.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading labels...")
    df = load_dataframe()
    class_names = (
        df[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")["class_name"]
        .tolist()
    )
    num_classes = len(class_names)
    print(f"Found {len(df)} images across {num_classes} classes.")

    print("Loading images into memory...")
    X, y = load_images(df)

    # Stratified split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    augmenter = build_augmenter()
    model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    # Build a tf.data pipeline so augmentation runs on the fly during training
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(2000, seed=SEED)
        .batch(BATCH_SIZE)
        .map(lambda x, y: (augmenter(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history)

    print("Evaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print("Saved classification_report.txt")

    model.save(MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
