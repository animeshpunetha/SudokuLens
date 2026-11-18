"""Write a few synthetic sample photos to assets/ for trying the app."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from sudokulens.synthetic import find_fonts, load_mnist_by_digit, make_board, split_fonts  # noqa: E402
from sudokulens.solver import parse_grid  # noqa: E402

PUZZLES = [
    "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
    "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..",
    "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..",
]


def main():
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    _, fonts = split_fonts(find_fonts())
    mnist = load_mnist_by_digit(200, seed=3, split="test")
    for i, p in enumerate(PUZZLES, 1):
        rng = random.Random(100 + i)
        b = make_board(rng, fonts, puzzle=parse_grid(p), mnist=mnist if i == 3 else None,
                       out_size=(1000, 750))
        cv2.imwrite(str(out / f"sample_{i}.jpg"), b.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", out / f"sample_{i}.jpg")


if __name__ == "__main__":
    main()
