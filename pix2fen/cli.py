import sys
import cv2
from pix2fen.inference import predict_cells
from pix2fen.fen import pieces_to_fen

def main():
    if len(sys.argv) != 2:
        print("Usage: pix2fen <image>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise RuntimeError("Image load failed")

    h, w, _ = img.shape
    cell_h = h // 8
    cell_w = w // 8

    cells = [
        img[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
        for r in range(8)
        for c in range(8)
    ]

    pieces = predict_cells(cells)
    fen = pieces_to_fen(pieces)
    print(fen)
