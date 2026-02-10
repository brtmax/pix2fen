import os
import requests
import cairosvg
from PIL import Image
from io import BytesIO

# disguised is disguised
themes = [
    "alpha","anarcandy","caliente","california","cardinal","cburnett","celtic",
    "chess7","chessnut","companion","cooke","dubrovny","fantasy",
    "firi","fresca","gioco","governor","horsey","icpieces","kiwen-suwi","kosal",
    "leipzig","letter","maestro","merida","monarchy","mono","mpchess","pirouetti",
    "pixel","reillycraig","rhosgfx","riohacha","shahi-ivory-brown","shapes",
    "spatial","staunty","tatiana","xkcd"
]

# Modern 
pieces_modern = ["wP","wN","wB","wR","wQ","wK","bP","bN","bB","bR","bQ","bK"]

# Legacy, really just mono
pieces_legacy = ["P","N","B","R","Q","K"]

themes_webp = ["monarchy"]

base_url = "https://raw.githubusercontent.com/ornicar/lila/master/public/piece/"
output_folder = "pieces"
target_size = 69

os.makedirs(output_folder, exist_ok=True)

def download_and_convert(url, save_path):
    try:
        r = requests.get(url)
        r.raise_for_status()
        ext = os.path.splitext(url)[1].lower()

        if ext == ".svg":
            # convert svg to png
            png_bytes = cairosvg.svg2png(bytestring=r.content, output_width=target_size, output_height=target_size)
            with open(save_path, "wb") as f:
                f.write(png_bytes)
        else:
            img = Image.open(BytesIO(r.content)).convert("RGBA")
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            img.save(save_path)

        print(f"Downloaded {save_path}")
        return True

    except requests.HTTPError as e:
        print(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"Error converting {url}: {e}")
        return False

for theme in themes:
    theme_folder = os.path.join(output_folder, theme)
    os.makedirs(theme_folder, exist_ok=True)

    if theme in themes_webp:
        pieces = pieces_modern
        ext = ".webp"
    elif theme == "mono":
        pieces = pieces_legacy
        ext = ".svg"
    else:
        pieces = pieces_modern
        ext = ".svg"

    for piece in pieces:
        save_path = os.path.join(theme_folder, f"{piece}.png")

        if os.path.exists(save_path):
            continue  # skip already downloaded

        url = f"{base_url}{theme}/{piece}{ext}"
        success = download_and_convert(url, save_path)

        # Fallback: if modern naming fails, try legacy filename for SVG sets
        if not success and ext == ".svg" and theme not in themes_webp and theme != "mono":
            url2 = f"{base_url}{theme}/{piece[-1]}.svg"
            download_and_convert(url2, save_path)

print("All downloads completed!")

