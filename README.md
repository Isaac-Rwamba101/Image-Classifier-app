# Flower Classifier

A Streamlit web app that tells you which flower is in a photo. Upload a JPG or PNG and the app shows the picture, the predicted flower with its confidence, and a bar chart of the probability for each class.

It recognises five flowers: **daisy, dandelion, roses, sunflowers, tulips**. Any other picture is still labelled as one of these five, so low-confidence results come with a warning.

The model is a MobileNetV2 network pre-trained on ImageNet and fine-tuned on the TensorFlow flowers dataset (3,670 photos). It reached about 89% accuracy on held-out test images.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python libraries the app needs |
| `flower_classifier.keras` | The trained model (**you add this file, see below**) |

## Add the trained model

The app loads `flower_classifier.keras` from the same folder as `app.py`. This file is produced by the last cells of your training notebook.

1. In Colab, run `from google.colab import files; files.download("flower_classifier.keras")`.
2. Put the downloaded file next to `app.py`. (A file named `flower_classifier.h5` also works.)

The file is roughly 10-30 MB, well under GitHub's 100 MB limit, so it can be uploaded to the repository like any other file.

## Run locally

Use Python 3.10, 3.11 or 3.12 (TensorFlow does not support every Python version).

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Create a new **public** GitHub repository and upload `app.py`, `requirements.txt`, `README.md` and `flower_classifier.keras` at the top level (not inside a subfolder).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** and choose your repository and branch (usually `main`).
4. Set **Main file path** to `app.py`.
5. Open **Advanced settings** and pick **Python 3.12** (or 3.11).
6. Click **Deploy**. The first start takes a few minutes because TensorFlow has to install.

Pushing new commits to the repository redeploys the app automatically.

## Troubleshooting

- **"Model file not found":** `flower_classifier.keras` is missing from the repository, or it is in a subfolder.
- **"The model file could not be loaded":** the app's TensorFlow version is older than the one that saved the model. In `requirements.txt` change the TensorFlow line to `tensorflow-cpu==2.21.0` (the version used in Colab for training) and redeploy.
- **The app restarts or runs out of memory:** TensorFlow is heavy. Open the app menu and choose Reboot app; if it keeps happening, the free tier's memory limit is the cause.
