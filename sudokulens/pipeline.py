"""End-to-end: photo -> detected grid -> solution -> annotated photo."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import vision
from .recognizer import DigitRecognizer, get_recognizer
from .solver import SudokuError, find_conflicts, solve


@dataclass
class Result:
    found_board: bool
    puzzle: Optional[List[List[int]]] = None          # recognised digits (0 = empty)
    confidence: Optional[List[List[float]]] = None    # CNN confidence per cell (1.0 for empty)
    solution: Optional[List[List[int]]] = None
    annotated: Optional[np.ndarray] = None            # original photo with solution drawn on
    warped: Optional[np.ndarray] = None               # top-down board crop
    error: Optional[str] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.solution is not None


def read_grid(image: np.ndarray, recognizer: Optional[DigitRecognizer] = None):
    """Detect the board and read its digits. Returns (board, grid, confidence) or None."""
    rec = recognizer or get_recognizer()
    corners = vision.find_board_corners(image)
    if corners is None:
        return None
    board = vision.warp_board(image, corners)
    positions, batch = vision.extract_digits(board)
    grid = [[0] * 9 for _ in range(9)]
    conf = [[1.0] * 9 for _ in range(9)]
    if positions:
        digits, probs = rec.predict(batch)
        for (r, c), d, p in zip(positions, digits, probs):
            grid[r][c] = int(d)
            conf[r][c] = float(p)
    return board, grid, conf


def solve_image(image: np.ndarray, recognizer: Optional[DigitRecognizer] = None) -> Result:
    t0 = time.perf_counter()
    rec = recognizer or get_recognizer()
    read = read_grid(image, rec)
    t1 = time.perf_counter()
    if read is None:
        return Result(found_board=False, error="Could not find a Sudoku grid in the image.",
                      timings_ms={"detect_and_read": (t1 - t0) * 1000})
    board, grid, conf = read
    res = Result(found_board=True, puzzle=grid, confidence=conf, warped=board.warped)
    try:
        conflicts = find_conflicts(grid)
        if conflicts:
            cells = ", ".join(f"r{r + 1}c{c + 1}" for r, c in conflicts)
            raise SudokuError(f"Recognised digits conflict at {cells}. Please correct them.")
        res.solution = solve(grid)
    except SudokuError as exc:
        res.error = str(exc)
    t2 = time.perf_counter()
    if res.solution:
        res.annotated = vision.draw_solution(image, board, grid, res.solution)
    else:
        res.annotated = vision.draw_outline(image, board.corners)
    t3 = time.perf_counter()
    res.timings_ms = {"detect_and_read": (t1 - t0) * 1000, "solve": (t2 - t1) * 1000,
                      "draw": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}
    return res
