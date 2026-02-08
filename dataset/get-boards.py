import os
from PIL import Image
import requests
from io import BytesIO

boards = [
    "blue-marble", "blue", "blue2", "blue3",
    "brown", "canvas2", "green-plastic", "green",
    "grey", "ic", "leather", "maple", "maple2",
    "marble", "metal", "ncf-board", "olive",
    "pink-pyramid", "purple-diag", "purple",
    "wood", "wood2", "wood3", "wood4"
]

base_url = "https://raw.githubusercontent.com/ornicar/lila/master/public/images/board/"

output_folder = "squares"
os.makedirs(output_folder, exist_ok=True)

square_size = 69  
board_size = square_size * 8

def download_board_image(name):
    # Prefer .orig.jpg if available, else fallback
    # Some naming is strange
    urls = [
        f"{base_url}{name}.orig.jpg",
        f"{base_url}{name}.jpg",
        f"{base_url}{name}.png"
    ]
    for url in urls:
        try:
            r = requests.get(url)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGBA")
            return img
        except requests.HTTPError:
            continue
    return None

for board_name in boards:
    # Skip horsey “weird” boards
    if board_name.startswith("horsey"):
        continue

    board_folder = os.path.join(output_folder, board_name)
    os.makedirs(board_folder, exist_ok=True)

    img = download_board_image(board_name)
    if img is None:
        print(f"Failed to download {board_name}")
        continue

    img = img.resize((board_size, board_size), Image.Resampling.LANCZOS)

    light_square = img.crop((0, 0, square_size, square_size))
    dark_square = img.crop((square_size, 0, 2*square_size, square_size))

    light_square.save(os.path.join(board_folder, "light.png"))
    dark_square.save(os.path.join(board_folder, "dark.png"))

    print(f"Processed board {board_name}")

print("All boards processed!")
