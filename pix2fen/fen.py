from typing import List

def pieces_to_fen(pieces: List[str]) -> str:

    if len(pieces) != 64:
        raise ValueError(f"Expected 64 pieces, got {len(pieces)}")

    fen_rows = []

    for rank in range(8):
        fen_row = ""
        empty_count = 0

        for file in range(8):
            piece = pieces[rank * 8 + file]

            if piece == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    # Standard suffix: side to move, castling, en passant, halfmove, fullmove
    fen = "/".join(fen_rows) + " w - - 0 1"
    return fen
