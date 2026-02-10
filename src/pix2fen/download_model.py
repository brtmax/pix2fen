import urllib.request
from pathlib import Path
from tensorflow.keras.models import load_model

MODEL_URL = "https://github.com/brtmax/pix2fen/releases/download/0.1.0/pix2fen.h5"
LOCAL_MODEL_PATH = Path.home() / ".pix2fen" / "pix2fen.h5"
LOCAL_MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)

_model = None

def get_model():
    """Lazy-load the model, downloading it if necessary."""
    global _model
    if _model is None:
        if not LOCAL_MODEL_PATH.exists():
            print("[INFO] Downloading pix2fen model...")
            urllib.request.urlretrieve(MODEL_URL, LOCAL_MODEL_PATH)
            print(f"[INFO] Model downloaded to {LOCAL_MODEL_PATH}")
        _model = load_model(LOCAL_MODEL_PATH)
    return _model
