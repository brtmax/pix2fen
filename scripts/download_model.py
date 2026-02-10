#!/usr/bin/env python3
import os
import urllib.request
import sys

MODEL_URL = (
    "https://github.com/brtmax/pix2fen/releases/download/v1.0/pix2fen.keras"
)
MODEL_PATH = "pix2fen.keras"


def download():
    print("[INFO] Model not found. Downloading from GitHub Releases...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        print("[ERROR] Failed to download model:", e)
        sys.exit(1)

    print("[INFO] Model downloaded:", MODEL_PATH)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        download()
    else:
        print("[INFO] Model already present")
