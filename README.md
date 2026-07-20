# Crop Disease Detector

A Streamlit app that classifies crop leaf diseases from uploaded images.

## How the files fit together

```
app.py            <- Streamlit UI ONLY. Uploads images, calls the
                      pipeline below, renders results.
        |
        v
preprocessing.py  <- UI upload objects -> PIL images -> normalized
                      numpy batch array. No Streamlit/TF imports.
        |
        v
model.py          <- Loads the trained .keras model (or an untrained
                      architecture as a fallback) and runs batch
                      inference. No Streamlit imports.
        |
        v
train.py          <- Separate script to train model.py's architecture
                      on your dataset and produce crop_disease_model.keras.
```

This separation means:
- You can test `preprocessing.py` and `model.py` with plain Python/pytest,
  no Streamlit server needed.
- `app.py` never touches raw arrays or TensorFlow calls directly — it just
  calls `preprocess_batch()` then `classifier.predict_batch()`.

## Setup

```bash
pip install -r requirements.txt
```

## Running the app (before training)

You can run the app right now — it will use an **untrained** model and
warn you in the console. This lets you test the full UI → processing →
display flow while your dataset/training is still in progress.

```bash
streamlit run app.py
```

## Training your model

1. Arrange your dataset like:
   ```
   dataset/train/<ClassName>/*.jpg
   dataset/val/<ClassName>/*.jpg
   ```
   (e.g. `dataset/train/Tomato___Late_blight/`). The PlantVillage dataset
   on Kaggle is a common free source for this.

2. Run:
   ```bash
   python train.py
   ```
   This saves `crop_disease_model.keras` and `labels.json`.

3. Re-run `streamlit run app.py` — it will now load your trained model
   automatically (loaded via `@st.cache_resource` so it only loads once
   per session).

## Notes
- Default classes in `model.py` are a placeholder (Apple/Corn/Potato/
  Tomato subset from PlantVillage). Once you train, `labels.json` is
  generated from your actual folder names and overrides the default list.
- Batch upload is supported natively — `st.file_uploader(..., accept_multiple_files=True)`
  feeds straight into `preprocess_batch()`, which stacks all images into
  one array for a single `model.predict()` call (faster than looping).
