#!/usr/bin/env python3

import cv2
import shutil
import os
import tempfile
import pyperclip
import subprocess

from pix2fen.fen import pieces_to_fen
from pix2fen.crop import crop_chessboard
from pix2fen.inference import predict_cells

def screenshot_to_image():
    # Check if maim is available
    if shutil.which("maim") is None:
        raise RuntimeError("maim is required for screenshot functionality on Linux.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(["maim", "-s", tmp_path], check=True)

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
        pieces = predict_cells(cells)
    except Exception as e:
            print(f"[ERROR] Model inference failed: {e}", file=sys.stderr)
            sys.exit(1)

    fen = pieces_to_fen(pieces)

    # Copy FEN to clipboard via glurp
    pyperclip.copy(fen)
    print("[INFO] FEN copied to clipboard")

if __name__ == "__main__":
    main()

