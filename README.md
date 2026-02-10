# pix2fen
OCR tool to extract FEN from chessboard screenshots. Solves a personal problem, so: 
> this is still a work in progress!

Current model available under release is only a demo model, I'm still tweaking the official one. 

## Installation
There's currently two ways to use this. You can just get the binary file from, put that on your path and be good to go. This packages only the pix2fen-clipboard functionality. If you want the full inference + dataset stuff, you have to install the dependencies via pip. 

```bash
pip install -e .
```

then just run
```bash
pix2fen-clipboard
```
set it to some shortcut if you want. Currently for wayland, adapt as needed. 

Linux only currently. 
