"""The original notebook solver (sudokuSolver.ipynb), kept verbatim in logic
so benchmarks can compare it with the optimised solver."""


class SudokuCell:
    def __init__(self, r, c, n):
        self.row, self.col, self.choices = r, c, n

    def set_data(self, r, c, n):
        self.row, self.col, self.choices = r, c, n


def solveSudoku(matrix):
    cont = [True]
    for i in range(9):
        for j in range(9):
            if not isPositionValid(matrix, i, j):
                return
    sudokuRecursive(matrix, cont)


def sudokuRecursive(matrix, cont):
    if not cont[0]:
        return
    best_candidate = SudokuCell(-1, -1, 100)
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                num_choices = countRemainingChoices(matrix, i, j)
                if best_candidate.choices > num_choices:
                    best_candidate.set_data(i, j, num_choices)
    if best_candidate.choices == 100:
        cont[0] = False
        return
    row, col = best_candidate.row, best_candidate.col
    for j in range(1, 10):
        if not cont[0]:
            return
        matrix[row][col] = j
        if isPositionValid(matrix, row, col):
            sudokuRecursive(matrix, cont)
    if not cont[0]:
        return
    matrix[row][col] = 0


def countRemainingChoices(matrix, i, j):
    can_pick = [True] * 10
    for k in range(9):
        can_pick[matrix[i][k]] = False
    for k in range(9):
        can_pick[matrix[k][j]] = False
    r, c = i // 3, j // 3
    for row in range(r * 3, r * 3 + 3):
        for col in range(c * 3, c * 3 + 3):
            can_pick[matrix[row][col]] = False
    return sum(1 for k in range(1, 10) if can_pick[k])


def allPositionsFilled(matrix):
    return all(matrix[i][j] != 0 for i in range(9) for j in range(9))


def isPositionValid(matrix, row, col):
    for c in range(9):
        if matrix[row][col] != 0 and col != c and matrix[row][col] == matrix[row][c]:
            return False
    for r in range(9):
        if matrix[row][col] != 0 and row != r and matrix[row][col] == matrix[r][col]:
            return False
    r, c = row // 3, col // 3
    for i in range(r * 3, r * 3 + 3):
        for j in range(c * 3, c * 3 + 3):
            if row != i and col != j and matrix[i][j] != 0 and matrix[i][j] == matrix[row][col]:
                return False
    return True
