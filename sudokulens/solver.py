"""Fast Sudoku solver: bitmask constraint propagation + MRV backtracking.

The grid is a 9x9 list (or array) of ints, 0 meaning an empty cell.

Compared with the original notebook solver (which re-scanned the whole board
with Python lists on every recursion step), this version keeps a 9-bit
candidate mask per row/column/box, fills forced cells ("naked singles" and
"hidden singles") before branching and always branches on the cell with the fewest candidates.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

Grid = List[List[int]]

ALL = 0x1FF  # bits 0..8 -> digits 1..9
_POPCOUNT = [bin(i).count("1") for i in range(512)]
_BOX = [(r // 3) * 3 + c // 3 for r in range(9) for c in range(9)]


class SudokuError(ValueError):
    """Raised when a grid is malformed, contradictory or unsolvable."""


def _to_flat(grid: Sequence[Sequence[int]]) -> List[int]:
    try:
        rows = [list(map(int, row)) for row in grid]
    except (TypeError, ValueError) as exc:
        raise SudokuError("Grid must contain only integers 0-9.") from exc
    if len(rows) != 9 or any(len(r) != 9 for r in rows):
        raise SudokuError("Grid must be 9x9.")
    flat = [v for r in rows for v in r]
    if any(v < 0 or v > 9 for v in flat):
        raise SudokuError("Cell values must be between 0 and 9.")
    return flat


def find_conflicts(grid: Sequence[Sequence[int]]) -> List[tuple]:
    """Return (row, col) of every given digit that clashes with another."""
    flat = _to_flat(grid)
    bad = set()
    units = []
    for i in range(9):
        units.append([i * 9 + j for j in range(9)])
        units.append([j * 9 + i for j in range(9)])
        br, bc = (i // 3) * 3, (i % 3) * 3
        units.append([(br + r) * 9 + bc + c for r in range(3) for c in range(3)])
    for unit in units:
        seen = {}
        for idx in unit:
            v = flat[idx]
            if v:
                if v in seen:
                    bad.add(idx)
                    bad.add(seen[v])
                else:
                    seen[v] = idx
    return sorted((i // 9, i % 9) for i in bad)


def is_valid_grid(grid: Sequence[Sequence[int]]) -> bool:
    """True if the given digits do not break any Sudoku rule."""
    return not find_conflicts(grid)


def _solve_flat(cells: List[int], limit: int) -> List[List[int]]:
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for i, v in enumerate(cells):
        if v:
            bit = 1 << (v - 1)
            r, c, b = i // 9, i % 9, _BOX[i]
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return []
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

    solutions: List[List[int]] = []
    empties = [i for i, v in enumerate(cells) if v == 0]

    def place(i: int, bit: int) -> None:
        rows[i // 9] |= bit
        cols[i % 9] |= bit
        boxes[_BOX[i]] |= bit

    def unplace(i: int, bit: int) -> None:
        rows[i // 9] ^= bit
        cols[i % 9] ^= bit
        boxes[_BOX[i]] ^= bit

    def search(empties: List[int]) -> bool:
        # Propagate naked singles; remember what we filled to undo it.
        filled = []
        remaining = empties
        while True:
            best = -1
            best_mask = 0
            best_count = 10
            nxt = []
            progress = False
            for i in remaining:
                if cells[i]:
                    continue
                mask = ALL & ~(rows[i // 9] | cols[i % 9] | boxes[_BOX[i]])
                n = _POPCOUNT[mask]
                if n == 0:
                    for j, bit in filled:
                        unplace(j, bit)
                        cells[j] = 0
                    return False
                if n == 1:
                    cells[i] = mask.bit_length()
                    place(i, mask)
                    filled.append((i, mask))
                    progress = True
                    continue
                nxt.append(i)
                if n < best_count:
                    best, best_mask, best_count = i, mask, n
            remaining = nxt
            if not progress and remaining:
                # Hidden singles: a digit that fits in only one cell of a
                # row/column/box must go there.
                units = {}
                for i in remaining:
                    m = ALL & ~(rows[i // 9] | cols[i % 9] | boxes[_BOX[i]])
                    for u in (i // 9, 9 + i % 9, 18 + _BOX[i]):
                        units.setdefault(u, []).append((i, m))
                for u, members in units.items():
                    once = more = 0
                    for _, m in members:
                        more |= once & m
                        once |= m
                    placed = rows[u] if u < 9 else cols[u - 9] if u < 18 else boxes[u - 18]
                    if (ALL & ~placed) & ~once:          # a needed digit has no cell left
                        for j, bit in filled:
                            unplace(j, bit)
                            cells[j] = 0
                        return False
                    single = once & ~more
                    while single:
                        bit = single & -single
                        single ^= bit
                        for i, m in members:
                            if m & bit:
                                if not cells[i] and not ((rows[i // 9] | cols[i % 9] | boxes[_BOX[i]]) & bit):
                                    cells[i] = bit.bit_length()
                                    place(i, bit)
                                    filled.append((i, bit))
                                    progress = True
                                break
            if not progress:
                break
            # A forced placement may have invalidated the cached "best";
            # loop again to recompute masks.

        if not remaining:
            solutions.append(list(cells))
            done = len(solutions) >= limit
            if not done:
                for j, bit in filled:
                    unplace(j, bit)
                    cells[j] = 0
            return done

        mask = best_mask
        while mask:
            bit = mask & -mask
            mask ^= bit
            cells[best] = bit.bit_length()
            place(best, bit)
            if search(remaining):
                return True
            unplace(best, bit)
            cells[best] = 0

        for j, bit in filled:
            unplace(j, bit)
            cells[j] = 0
        return False

    search(empties)
    return solutions


def solve(grid: Sequence[Sequence[int]]) -> Grid:
    """Solve a puzzle and return the completed 9x9 grid.

    Raises SudokuError if the grid is malformed, has conflicting givens, or
    has no solution. The input is not modified.
    """
    flat = _to_flat(grid)
    if find_conflicts(grid):
        raise SudokuError("The puzzle has conflicting digits.")
    sols = _solve_flat(flat, limit=1)
    if not sols:
        raise SudokuError("The puzzle has no solution.")
    s = sols[0]
    return [s[r * 9:(r + 1) * 9] for r in range(9)]


def count_solutions(grid: Sequence[Sequence[int]], limit: int = 2) -> int:
    """Count solutions up to ``limit`` (use limit=2 to check uniqueness)."""
    flat = _to_flat(grid)
    if find_conflicts(grid):
        return 0
    return len(_solve_flat(flat, limit=limit))


def parse_grid(text: str) -> Grid:
    """Parse 81 digits from text; '0', '.', '_' and '-' mean empty."""
    chars = [ch for ch in text if ch.isdigit() or ch in "._-"]
    if len(chars) != 81:
        raise SudokuError(f"Expected 81 cells, found {len(chars)}.")
    vals = [0 if ch in "._-" else int(ch) for ch in chars]
    return [vals[r * 9:(r + 1) * 9] for r in range(9)]


def format_grid(grid: Sequence[Sequence[int]]) -> str:
    """Pretty-print a grid with 3x3 box separators."""
    lines = []
    for r, row in enumerate(grid):
        if r and r % 3 == 0:
            lines.append("------+-------+------")
        parts = []
        for c, v in enumerate(row):
            if c and c % 3 == 0:
                parts.append("|")
            parts.append(str(v) if v else ".")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def solve_or_none(grid: Sequence[Sequence[int]]) -> Optional[Grid]:
    try:
        return solve(grid)
    except SudokuError:
        return None
