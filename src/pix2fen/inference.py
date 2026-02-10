import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASS_INDICES_PATH = ROOT / "class_indices.json"
IMG_SIZE = (69, 69)

MODEL_URL = "https://github.com/<username>/pix2fen/releases/download/v1.0/pix2fen.h5"
LOCAL_MODEL_PATH = Path.home() / ".pix2fen" / "pix2fen.h5"
LOCAL_MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)

IDX_TO_FOLDER = {v: k for k, v in class_indices.items()}

FOLDER_TO_PIECE = {
    "bP": "p", "bN": "n", "bB": "b", "bR": "r", "bQ": "q", "bK": "k",
    "wP": "P", "wN": "N", "wB": "B", "wR": "R", "wQ": "Q", "wK": "K",
    "empty": ""
}

_model = None

def get_model():
    """Lazy-load the model, downloading it if necessary."""
    global _model
    if _model is None:
        if not LOCAL_MODEL_PATH.exists():
            print("[INFO] Downloading model...")
            urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)
            print(f"[INFO] Model downloaded to {LOCAL_MODEL_PATH}")
        _model = load_model(LOCAL_MODEL_PATH)
    return _model

def preprocess_cell(cell):
    cell = cv2.resize(cell, IMG_SIZE)
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    return cell.astype(np.float32) / 255.0

def predict_cells(cells):
    model = get_model()
    X = np.array([preprocess_cell(c) for c in cells])

    preds = model.predict(X, verbose=0)
    classes = np.argmax(preds, axis=1)

    pieces = [
        FOLDER_TO_PIECE.get(IDX_TO_FOLDER[c], "?")
        for c in classes
    ]
    return pieces
