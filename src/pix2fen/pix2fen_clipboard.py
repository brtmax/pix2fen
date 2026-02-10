#!/usr/bin/env python3

import cv2
import shutil
import os
import sys
import tempfile
import pyperclip
import subprocess

from pix2fen.fen import pieces_to_fen
from pix2fen.crop import crop_chessboard
from pix2fen.inference import predict_cells
from pix2fen.inference import predict_cells_tflite
from pix2fen.inference import predict_cells_tflite_batch

def copy_to_clipboard(text):
    if shutil.which("wl-copy") is None:
        print("[WARNING] wl-copy not found, skipping clipboard copy.", file=sys.stderr)
        return
    try:
        subprocess.run(["wl-copy"], input=text, text=True, check=True)
        print("[DEBUG] FEN copied to clipboard.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to copy to clipboard: {e}", file=sys.stderr)

def get_selection_geometry():
    try:
        result = subprocess.run(["slurp"], check=True, capture_output=True, text=True, env=os.environ)
        geom = result.stdout.strip()
        print(f"[DEBUG] Geometry: '{geom}'")
        return geom
    except subprocess.CalledProcessError:
        raise RuntimeError("Slurp selection failed or was cancelled.")

def screenshot_to_image():
    if shutil.which("grim") is None or shutil.which("slurp") is None:
        raise RuntimeError("grim + slurp are required for interactive screenshot on Wayland.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Get selection from slurp

        print("[DEBUG] Launching slurp...")
        geom = get_selection_geometry()
        print(f"[DEBUG] Geometry returned: {geom}")
        if not geom:
            raise RuntimeError("No selection made.")

        subprocess.run(["grim", "-g", geom, tmp_path], check=True)

        img = cv2.imread(tmp_path)
        if img is None:
            raise RuntimeError("Screenshot failed or file unreadable")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return img

def main():
    try:
        img = screenshot_to_image()
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    board = crop_chessboard(img)
    h, w, _ = board.shape
    cell_h = h // 8
    cell_w = w // 8

    cells = [
        board[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
        for r in range(8)
        for c in range(8)
    ]

    try:
        # pieces = predict_cells(cells)
        pieces = predict_cells_tflite(cells)
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    fen = pieces_to_fen(pieces)
    print(fen)
    copy_to_clipboard(fen)

if __name__ == "__main__":
    main()
