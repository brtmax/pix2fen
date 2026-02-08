import os
from PIL import Image

# CONFIG
input_folder = "boards"
output_folder = "squares"
square_size = 552 // 8
image_extension = ".png"

# Map chess symbols to class IDs
label_map = {
    '.': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r':10, 'q':11, 'k':12
}

# Starting position, White bottom
starting_position = [
    ['r','n','b','q','k','b','n','r'],  
    ['p','p','p','p','p','p','p','p'], 
    ['.','.','.','.','.','.','.','.'],
    ['.','.','.','.','.','.','.','.'],
    ['.','.','.','.','.','.','.','.'],
    ['.','.','.','.','.','.','.','.'],
    ['P','P','P','P','P','P','P','P'],
    ['R','N','B','Q','K','B','N','R'],
]

for label_id in label_map.values():
    class_folder = os.path.join(output_folder, str(label_id))
    os.makedirs(class_folder, exist_ok=True)

board_files = [f for f in os.listdir(input_folder) if f.endswith(image_extension)]

for board_file in board_files:
    board_path = os.path.join(input_folder, board_file)
    board_img = Image.open(board_path).convert("RGB")

    for rank in range(8):
        for file_idx in range(8):
            x0 = file_idx * square_size
            y0 = rank * square_size
            crop = board_img.crop((x0, y0, x0 + square_size, y0 + square_size))

            label_char = starting_position[rank][file_idx]
            label_id = label_map[label_char]

            # Save with unique filename
            square_filename = f"{os.path.splitext(board_file)[0]}_r{rank+1}f{file_idx+1}.png"
            crop.save(os.path.join(output_folder, str(label_id), square_filename))

print(f"Squares saved in {output_folder}/[class_id]/")
