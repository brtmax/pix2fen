import os
import random
from PIL import Image, ImageEnhance
from pathlib import Path


pieces_folder = "pieces"
squares_folder = "squares"
output_folder = "dataset/full"

square_size = 69
train_ratio = 0.8
augmentations_per_square = 5

# 13 classes
classes = [
    "empty","wP","wN","wB","wR","wQ","wK",
    "bP","bN","bB","bR","bQ","bK"
]

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
                        img.alpha_composite(piece)

                    # Generate augmentations
                    for i in range(augmentations_per_square):
                        aug_img = augment_image(img)
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
