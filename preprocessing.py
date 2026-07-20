"""
preprocessing.py
-----------------
Image preprocessing utilities that sit between the Streamlit UI and the
model inference code. Keeping this separate means the UI never touches
raw pixel arrays directly, and the model never has to worry about
PIL/Streamlit-specific objects.
"""

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = (224, 224)  # matches MobileNetV2 default input size


def load_image_from_upload(uploaded_file) -> Image.Image:
    """
    Convert a Streamlit UploadedFile object into a PIL Image.
    Handles EXIF rotation (common with phone camera photos of crops).
    """
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)  # fix sideways/upside-down phone photos
    image = image.convert("RGB")
    return image


def preprocess_image(image: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    """
    Resize + normalize a single PIL image into a model-ready array.
    Returns shape (H, W, 3) with pixel values scaled to [0, 1].
    """
    image = image.resize(target_size)
    arr = np.asarray(image).astype("float32") / 255.0
    return arr


def preprocess_batch(images: list[Image.Image], target_size=IMG_SIZE) -> np.ndarray:
    """
    Preprocess a list of PIL images into a single batch array of shape
    (N, H, W, 3), ready to feed straight into model.predict().
    """
    batch = [preprocess_image(img, target_size) for img in images]
    return np.stack(batch, axis=0)
