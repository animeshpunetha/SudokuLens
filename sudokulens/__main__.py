"""Command line: solve a photo or a typed puzzle.

    python -m sudokulens photo.jpg [-o solved.jpg]
    python -m sudokulens --grid "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
"""
import argparse
import sys

import cv2

from .pipeline import solve_image
from .solver import SudokuError, format_grid, parse_grid, solve


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sudokulens", description="Solve a Sudoku from a photo or text.")
    ap.add_argument("image", nargs="?", help="path to a photo of a Sudoku")
    ap.add_argument("--grid", help="81 characters, 0 or . for blanks")
    ap.add_argument("-o", "--output", help="where to save the annotated photo")
    args = ap.parse_args(argv)

    try:
        if args.grid:
            puzzle = parse_grid(args.grid)
            print(format_grid(solve(puzzle)))
            return 0
        if not args.image:
            ap.error("give an image path or --grid")
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read {args.image}", file=sys.stderr)
            return 1
        res = solve_image(img)
        if not res.found_board:
            print(res.error, file=sys.stderr)
            return 2
        print("Recognised puzzle:\n" + format_grid(res.puzzle) + "\n")
        if res.solved:
            print("Solution:\n" + format_grid(res.solution))
        else:
            print(res.error, file=sys.stderr)
        if args.output and res.annotated is not None:
            cv2.imwrite(args.output, res.annotated)
            print(f"\nSaved {args.output}")
        print(f"\nTook {res.timings_ms.get('total', 0):.0f} ms")
        return 0 if res.solved else 3
    except SudokuError as exc:
        print(exc, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
