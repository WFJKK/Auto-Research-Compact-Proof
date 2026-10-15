"""One figure of the Pareto frontiers of the two runs: the left panel (frontier per network) of each run report's
frontier plot, for v64 and v128, side by side. Reads the run reports' images; draws nothing new.

    python -B results/max2-2026-10-13/frontiers.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RUNS = [("max2-rerun-q", "scored on Q"), ("max2-rerun-cof", "scored on cost of finishing")]
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)

tiles = []
for run, title in RUNS:
    row = []
    for st in ("v64", "v128"):
        im = Image.open(HERE.parent / run / f"frontier_{st}.png")
        w, h = im.size
        row.append(im.crop((0, 0, w // 2, h)))  # the frontier panel; the right panel is the uncertified share
    tiles.append((title, row))
tw, th = tiles[0][1][0].size
pad, head = 10, 40
out = Image.new("RGB", (2 * tw + 3 * pad, 2 * (th + head) + 3 * pad), "white")
draw = ImageDraw.Draw(out)
for i, (title, row) in enumerate(tiles):
    y = pad + i * (th + head + pad)
    draw.text((pad + 8, y + 8), f"Run {title}: frontier per development network, with the reference proofs as crosses", fill="#1f2328", font=FONT)
    for j, im in enumerate(row):
        out.paste(im, (pad + j * (tw + pad), y + head))
out.save(HERE / "frontiers.png")
print("saved", HERE / "frontiers.png", out.size)
