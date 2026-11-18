import pytest

from sudokulens.solver import (SudokuError, count_solutions, find_conflicts, format_grid,
                               is_valid_grid, parse_grid, solve)

HARD = "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4.."
EASY = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"


def assert_complete(grid):
    full = set(range(1, 10))
    for i in range(9):
        assert set(grid[i]) == full
        assert {grid[r][i] for r in range(9)} == full
        br, bc = 3 * (i // 3), 3 * (i % 3)
        assert {grid[br + r][bc + c] for r in range(3) for c in range(3)} == full


@pytest.mark.parametrize("text", [EASY, HARD])
def test_solves_and_keeps_givens(text):
    p = parse_grid(text)
    s = solve(p)
    assert_complete(s)
    assert all(s[r][c] == p[r][c] for r in range(9) for c in range(9) if p[r][c])


def test_input_not_modified():
    p = parse_grid(EASY)
    before = [row[:] for row in p]
    solve(p)
    assert p == before


def test_conflict_detected():
    p = parse_grid(EASY)
    p[0][2] = 5                      # duplicates the 5 at (0,0)
    assert (0, 0) in find_conflicts(p) and (0, 2) in find_conflicts(p)
    assert not is_valid_grid(p)
    with pytest.raises(SudokuError):
        solve(p)


def test_unsolvable():
    # Valid-looking but impossible: row 0 needs a 9 that column 8 already has.
    p = parse_grid("12345678." + "........9" + "." * 63)
    with pytest.raises(SudokuError):
        solve(p)


def test_uniqueness():
    assert count_solutions(parse_grid(HARD)) == 1
    assert count_solutions([[0] * 9 for _ in range(9)]) == 2


@pytest.mark.parametrize("bad", ["123", "x" * 81])
def test_parse_errors(bad):
    with pytest.raises(SudokuError):
        parse_grid(bad)


def test_format_roundtrip():
    p = parse_grid(EASY)
    assert parse_grid(format_grid(p).replace("|", "").replace("-", "").replace("+", "")) == p
