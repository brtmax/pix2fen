# pix2fen
OCR tool to extract FEN from chessboard screenshots. Solves a personal problem, so: 
> this is still a work in progress!

Current model available under release is only a demo model, I'm still tweaking the official one. 
Still need to clean a lot of things up and work on speedup, maybe have a daemon running in the background to keep the model loaded. 

## Dataset
This project uses chessboard and piece artwork from [Lichess.org](https://lichess.org/), which is licensed under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). 

The images were used to generate training data for the OCR model. 

`dataset/get-pieces.py` and `dataset/get-boards.py` pull the pieces/boards respectively. `generate-dataset.py` does some augmentation and combines pieces and boards to the full dataset. 

## Usage
This mainly solves a personal problem I have, but the model output can be used for other purposes as well. Main idea: 
1. Have some chess book/pdf open. Run the tool and select the board of the puzzle you want to play
2. Model detects board state and puts that in FEN notation into clipboard
3. Paste into any board editor of your choice (like https://lichess.org/editor) and paste it
4. Play it!

![Demo](https://www.brt.fyi/images/pix2fen/pix2fen.jpg)

Position from J.Babson, 1882. *Mate in one move, in 47 different ways*

## Installation
There's currently two ways to use this. You can just get the binary file from, put that on your path and be good to go. This packages only the pix2fen-clipboard functionality. If you want the full inference + dataset stuff, you have to install the dependencies via pip. 

```bash
pip install -e .
```

then just run
```bash
pix2fen-clipboard
```
set it to some shortcut if you want. Currently for Wayland, adapt as needed. 

Linux only currently. 

## Other
I used [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) as inspiration, which I use daily at Uni. Does the same thing but with equations -> Latex (and much better)
