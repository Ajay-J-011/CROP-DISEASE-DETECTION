"""
app.py
------
Streamlit UI. This is the ONLY file that should import streamlit —
everything else (preprocessing.py, model.py) stays UI-agnostic so you
could swap Streamlit for Flask/FastAPI later without touching them.

Run with: streamlit run app.py
"""

import io
import pandas as pd
import streamlit as st

from preprocessing import load_image_from_upload, preprocess_batch
from model import CropDiseaseClassifier

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_classifier() -> CropDiseaseClassifier:
    # Cached so the model loads once per session, not on every rerun.
    return CropDiseaseClassifier()


def format_label(raw_label: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato — Late blight' for display."""
    crop, _, disease = raw_label.partition("___")
    return f"{crop} — {disease.replace('_', ' ')}"


def main():
    st.title("🌿 Crop Disease Detector")
    st.caption("Upload one or more leaf images to check for disease.")

    classifier = get_classifier()

    uploaded_files = st.file_uploader(
        "Upload leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload images above to get started.")
        return

    if st.button(f"Analyze {len(uploaded_files)} image(s)", type="primary"):
        run_pipeline(classifier, uploaded_files)


def run_pipeline(classifier: CropDiseaseClassifier, uploaded_files):
    """
    This is the 'processing' glue: UI upload -> preprocessing.py ->
    model.py -> back to UI as a results table + per-image cards.
    """
    with st.spinner("Processing images..."):
        # 1. UI objects -> PIL images
        pil_images = [load_image_from_upload(f) for f in uploaded_files]

        # 2. PIL images -> model-ready batch array
        batch_array = preprocess_batch(pil_images)

        # 3. Run inference on the whole batch at once (faster than looping)
        predictions = classifier.predict_batch(batch_array)

    st.success(f"Done — analyzed {len(predictions)} image(s).")

    # --- Results table (downloadable) ---
    table_rows = [
        {
            "File": f.name,
            "Prediction": format_label(p["label"]),
            "Confidence": f"{p['confidence'] * 100:.1f}%",
        }
        for f, p in zip(uploaded_files, predictions)
    ]
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download results as CSV",
        data=csv_buffer.getvalue(),
        file_name="crop_disease_results.csv",
        mime="text/csv",
    )

    st.divider()

    # --- Per-image visual cards ---
    cols_per_row = 3
    for i in range(0, len(uploaded_files), cols_per_row):
        row_files = uploaded_files[i : i + cols_per_row]
        row_preds = predictions[i : i + cols_per_row]
        cols = st.columns(len(row_files))
        for col, f, pred in zip(cols, row_files, row_preds):
            with col:
                st.image(f, use_container_width=True)
                is_healthy = "healthy" in pred["label"].lower()
                icon = "✅" if is_healthy else "⚠️"
                st.markdown(f"**{icon} {format_label(pred['label'])}**")
                st.progress(pred["confidence"], text=f"{pred['confidence']*100:.1f}% confidence")


if __name__ == "__main__":
    main()
