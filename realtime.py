"""Live webcam solver (the original project's real-time mode, rebuilt).

    python realtime.py [--camera 0]

Press q to quit. The board is re-read every frame, but the solver only runs
when the recognised puzzle changes, so the overlay is cached between frames.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from sudokulens import vision
from sudokulens.pipeline import read_grid
from sudokulens.recognizer import get_recognizer
from sudokulens.solver import SudokuError, solve


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}.")

    rec = get_recognizer()
    last_puzzle, last_solution = None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = frame
        read = read_grid(frame, rec)
        if read is not None:
            board, puzzle, _ = read
            if puzzle != last_puzzle:
                last_puzzle = puzzle
                try:
                    last_solution = solve(puzzle)
                except SudokuError:
                    last_solution = None
            if last_solution is not None:
                out = vision.draw_solution(frame, board, puzzle, last_solution)
            else:
                out = vision.draw_outline(frame, board.corners)
        cv2.imshow("SudokuLens - press q to quit", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
