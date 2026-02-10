import os
import random
from PIL import Image, ImageEnhance
from pathlib import Path

# synthetic (=from lichess)
pieces_folder = "pieces"
squares_folder = "squares"
output_folder = "dataset/full"

# screenshots I took myself
real_cells_folder = "real_cells"
real_weight = 0.5

square_size = 69
train_ratio = 0.8
augmentations_per_square = 10

# 13 classes
classes = [
    "empty","wP","wN","wB","wR","wQ","wK",
    "bP","bN","bB","bR","bQ","bK"
]

def add_ui_noise(img):
    if random.random() < 0.3:
        px = img.load()
        w, h = img.size
        for i in range(w):
            px[i, 0] = (0,0,0,255)
            px[i, h-1] = (0,0,0,255)
        for j in range(h):
            px[0, j] = (0,0,0,255)
            px[w-1, j] = (0,0,0,255)
    return img

def maybe_replace_with_real(cls):
    cls_folder = os.path.join(real_cells_folder, cls)
    if not os.path.isdir(cls_folder):
        return None
    files = os.listdir(cls_folder)
    if not files:
        return None
    if random.random() < real_weight:
        return Image.open(os.path.join(cls_folder, random.choice(files))).convert("RGBA")
    return None

# We differentiate between empty square augmentation and full square augmentation
# This is required since board texture is much more noticable on empty square
def augment_empty(img):
    img = augment_image(img)
    if random.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
    return img

def paste_with_jitter(base, piece):
    bx, by = base.size
    px, py = piece.size
    max_dx = int(0.1 * bx)
    max_dy = int(0.1 * by)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    x = (bx - px)//2 + dx
    y = (by - py)//2 + dy
    base.alpha_composite(piece, (x, y))

def augment_image(img):
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8,1.2))
    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8,1.2))
    # Random slight rotation
    img = img.rotate(random.uniform(-5,5))
    return img

for split in ["train","val"]:
    for cls in classes:
        Path(os.path.join(output_folder, split, cls)).mkdir(parents=True, exist_ok=True)

for board_name in os.listdir(squares_folder):
    board_path = os.path.join(squares_folder, board_name)
    if not os.path.isdir(board_path):
        continue

    # Load light/dark squares
    try:
        light_square = Image.open(os.path.join(board_path,"light.png")).convert("RGBA")
        dark_square  = Image.open(os.path.join(board_path,"dark.png")).convert("RGBA")
    except FileNotFoundError:
        print(f"Missing squares for board {board_name}, skipping")
        continue

    for theme in os.listdir(pieces_folder):
        theme_folder = os.path.join(pieces_folder, theme)
        if not os.path.isdir(theme_folder):
            continue

        for cls in classes:
            if cls == "empty":
                # just the board square
                pieces_to_use = [None]  
            else:
                piece_file = os.path.join(theme_folder, f"{cls}.png")
                if not os.path.isfile(piece_file):
                    continue
                pieces_to_use = [piece_file]

            for piece_path in pieces_to_use:
                # Overlay on both light and dark squares
                for base_square, color in [(light_square,"light"),(dark_square,"dark")]:
                    img = base_square.copy()
                    if piece_path:
                        abs_path = os.path.abspath(piece_path)
                        print("Trying to open piece:", abs_path)
                        piece = Image.open(piece_path).convert("RGBA")
                        paste_with_jitter(img, piece)

                    real_img = maybe_replace_with_real(cls)
                    if real_img is not None:
                        img = real_img.resize((square_size, square_size))

                    # Generate augmentations
                    repeat = augmentations_per_square * (3 if cls == "empty" else 1)
                    for i in range(repeat):
                        if cls == "empty":
                            aug_img = augment_empty(img)
                        else: 
                            aug_img = augment_image(img)
                        aug_img = add_ui_noise(aug_img)
                        # Random train/val split
                        split_folder = "train" if random.random() < train_ratio else "val"
                        # Save image
                        out_path = os.path.join(
                            output_folder,
                            split_folder,
                            cls,
                            f"{board_name}_{theme}_{color}_{i}.png"
                        )
                        aug_img.save(out_path)

    print(f"Processed board {board_name}")

print("Dataset generation completed!")
