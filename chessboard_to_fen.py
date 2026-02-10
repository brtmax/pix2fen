import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

MODEL_PATH = "pix2fen.h5"
IMG_SIZE = (69, 69)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

IDX_TO_FOLDER = {v: k for k, v in class_indices.items()}

FOLDER_TO_PIECE = {
    "bP": "p", "bN": "n", "bB": "b", "bR": "r", "bQ": "q", "bK": "k",
    "wP": "P", "wN": "N", "wB": "B", "wR": "R", "wQ": "Q", "wK": "K",
    "empty": ""
}

os.makedirs("debug", exist_ok=True)
os.makedirs("debug/cells", exist_ok=True)

print("[INFO] Loading model...", file=sys.stderr)
model = load_model(MODEL_PATH)
print("[INFO] Model loaded", file=sys.stderr)

def load_image(path):
    print(f"[INFO] Loading image: {path}", file=sys.stderr)
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Image load failed")
    print(f"[INFO] Image shape: {img.shape}", file=sys.stderr)
    return img

def crop_chessboard(img):
    print("[INFO] Cropping chessboard", file=sys.stderr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # cv2.imwrite("debug/edges.png", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Contours found: {len(contours)}", file=sys.stderr)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(cnt)
        print(f"[DEBUG] Contour {i}: area={area}, points={len(approx)}", file=sys.stderr)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            board = img[y:y+h, x:x+w]
            # cv2.imwrite("debug/board.png", board)
            print(f"[INFO] Chessboard cropped: {board.shape}", file=sys.stderr)
            return board

    raise RuntimeError("Chessboard not detected")

def split_into_cells(board):
    h, w, _ = board.shape
    print(f"[INFO] Board size: {w}x{h}", file=sys.stderr)
    cell_h = h // 8
    cell_w = w // 8
    cells = []

    for r in range(8):
        for c in range(8):
            y1 = r * cell_h
            y2 = (r + 1) * cell_h
            x1 = c * cell_w
            x2 = (c + 1) * cell_w
            cell = board[y1:y2, x1:x2]
            idx = r * 8 + c
            # cv2.imwrite(f"debug/cells/cell_{idx:02d}.png", cell)
            cells.append(cell)

    print(f"[INFO] Cells extracted: {len(cells)}", file=sys.stderr)
    return cells

def preprocess_cell(cell):
    cell = cv2.resize(cell, IMG_SIZE)
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    cell = cell.astype(np.float32) / 255.0
    return cell


def predict_cells(cells):
    print("[INFO] Running inference on cells", file=sys.stderr)
    X = np.array([preprocess_cell(c) for c in cells])
    print(f"[INFO] Model input shape: {X.shape}", file=sys.stderr)

    preds = model.predict(X, verbose=0)
    classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    pieces = []
    for i, (cls, conf) in enumerate(zip(classes, confidences)):
        folder = IDX_TO_FOLDER[cls]              
        piece = FOLDER_TO_PIECE.get(folder, "?") 
        pieces.append(piece)

        r = i // 8
        c = i % 8
        print(f"[PRED] square=({r},{c}) class={cls} piece='{piece}' conf={conf:.4f}", file=sys.stderr)

    return pieces, confidences

def board_to_fen(pieces_board):
    fen_rows = []
    print("[INFO] Converting board to FEN", file=sys.stderr)

    for r in range(8):
        fen_row = ""
        empty = 0
        for c in range(8):
            piece = pieces_board[r*8 + c] if isinstance(pieces_board, list) else pieces_board[r, c]
            if piece == "":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += piece
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
        print(f"[FEN] rank {8-r}: {fen_row}", file=sys.stderr)

    fen = "/".join(fen_rows) + " w - - 0 1"
    return fen

def image_to_fen(image_path):
    img = load_image(image_path)
    board = crop_chessboard(img)
    cells = split_into_cells(board)
    pieces, confidences = predict_cells(cells)
    fen = board_to_fen(pieces)
    return fen

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pix2fen.py <screenshot.png>")
        sys.exit(1)

    image_path = sys.argv[1]
    fen = image_to_fen(image_path)

    print(fen)

