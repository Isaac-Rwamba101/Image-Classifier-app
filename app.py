"""Flower classifier: upload a photo and get a real-time prediction.

Run locally:
    streamlit run app.py

Needs the trained model file (flower_classifier.keras, or flower_classifier.h5) in the same folder as app.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# --- Settings that must match how the model was trained -------------------------------------------
IMG_SIZE = 160                                                         # the model takes 160x160 RGB images
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]  # alphabetical, as Keras orders folders
MODEL_FILES = ["flower_classifier.keras", "flower_classifier.h5"]
LOW_CONFIDENCE = 0.60                                                  # below this, warn the user

st.set_page_config(page_title="Flower Classifier")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained model once and reuse it for every upload."""
    import tensorflow as tf

    folder = Path(__file__).resolve().parent
    for name in MODEL_FILES:
        path = folder / name
        if path.exists():
            return tf.keras.models.load_model(path), name
    return None, None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Turn an uploaded picture into the batch the model expects: shape (1, 160, 160, 3).

    The model already contains its own Rescaling layer (0-255 -> -1..1), so pixel values
    are left in the 0-255 range here. Dividing by 255 again would give wrong predictions.
    """
    image = ImageOps.exif_transpose(image)             # respect the phone's rotation flag
    image = image.convert("RGB")                       # drop alpha channel / expand grayscale
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(image, dtype="float32")[None, ...]


def predict(model, batch: np.ndarray) -> np.ndarray:
    """Return the probability of each class for the first (only) image in the batch."""
    return model.predict(batch, verbose=0)[0]


# --- Page ------------------------------------------------------------------------------------------
st.title("Flower Classifier")
st.write(
    "Upload a photo of a flower and the model tells you which of these five it is: "
    + ", ".join(CLASS_NAMES) + "."
)

try:
    model, model_name = load_model()
except ImportError:
    st.error("TensorFlow is not installed. Run `pip install -r requirements.txt` and restart the app.")
    st.stop()
except Exception as err:                                # e.g. file saved with a newer Keras version
    st.error(f"The model file could not be loaded: {err}")
    st.stop()

if model is None:
    st.error(
        "Model file not found. Put `flower_classifier.keras` (or `flower_classifier.h5`) "
        "in the same folder as `app.py` and refresh the page."
    )
    st.stop()

st.sidebar.header("About")
st.sidebar.write(f"Model file: `{model_name}`")
st.sidebar.write(f"Input size: {IMG_SIZE}x{IMG_SIZE} pixels")
st.sidebar.write("Base network: MobileNetV2 (transfer learning)")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG or PNG image to see a prediction.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except Exception:
    st.error("That file could not be read as an image. Try another JPG or PNG.")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Your image")
    st.image(image, caption=uploaded.name, width=360)

with right:
    st.subheader("Prediction")
    with st.spinner("Predicting..."):
        probs = predict(model, preprocess_image(image))

    if len(probs) != len(CLASS_NAMES):
        st.error(f"The model has {len(probs)} outputs but the app lists {len(CLASS_NAMES)} classes. Update CLASS_NAMES.")
        st.stop()

    best = int(np.argmax(probs))
    label, confidence = CLASS_NAMES[best], float(probs[best])

    if confidence >= LOW_CONFIDENCE:
        st.success(f"**{label.capitalize()}** ({confidence:.1%} confidence)")
    else:
        st.warning(
            f"Best guess: **{label.capitalize()}** ({confidence:.1%} confidence). "
            "The model is not sure. Is this one of the five flowers above?"
        )

    chart = pd.DataFrame({"Probability": probs}, index=[c.capitalize() for c in CLASS_NAMES])
    st.bar_chart(chart)

with st.expander("Show exact probabilities"):
    table = pd.DataFrame({"Flower": [c.capitalize() for c in CLASS_NAMES],
                          "Probability": [f"{p:.2%}" for p in probs]})
    st.dataframe(table, hide_index=True)

st.caption("The model only knows these five flowers, so any other picture will still be labelled as one of them.")
