"""Image processing: find the Sudoku board, split it into cells, normalise
each digit for the CNN, and draw the solution back onto the photo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

WARP_SIZE = 450            # board is warped to a 450x450 square (50px cells)
CELL = WARP_SIZE // 9
DIGIT_SIZE = 28            # CNN input size
DIGIT_BOX = 20             # digit is scaled to fit 20x20 then padded (MNIST style)


@dataclass
class Board:
    corners: np.ndarray        # 4x2 float32, ordered TL, TR, BR, BL (image coords)
    warped: np.ndarray         # WARP_SIZE x WARP_SIZE BGR top-down view
    matrix: np.ndarray         # perspective transform image -> warped


# --------------------------------------------------------------------------
# Board detection
# --------------------------------------------------------------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()      # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _looks_like_board(c: np.ndarray, eps_angle: float = 20, eps_scale: float = 1.3) -> bool:
    a, b, cc, d = c
    sides = [np.linalg.norm(b - a), np.linalg.norm(cc - b),
             np.linalg.norm(d - cc), np.linalg.norm(a - d)]
    if min(sides) < 1 or max(sides) > eps_scale * min(sides):
        return False
    angles = [_angle(b - a, d - a), _angle(a - b, cc - b),
              _angle(b - cc, d - cc), _angle(cc - d, a - d)]
    return all(abs(x - 90) < eps_angle for x in angles)


def find_board_corners(image: np.ndarray, min_area_ratio: float = 0.04) -> Optional[np.ndarray]:
    """Return the 4 ordered corners of the Sudoku board, or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Work on a downscaled copy for speed on large phone photos.
    scale = 1.0
    max_side = max(gray.shape)
    if max_side > 1000:
        scale = 1000.0 / max_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    min_area = min_area_ratio * gray.shape[0] * gray.shape[1]
    block = max(11, (min(gray.shape) // 40) | 1)
    # Try a few threshold strengths: a strong offset ignores texture/noise,
    # a weak one picks up faint board borders.
    for c_off in (7, 3, 12):
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block, c_off)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            if cv2.contourArea(cnt) < min_area:
                break
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            for k in (0.02, 0.03, 0.05, 0.08):
                approx = cv2.approxPolyDP(hull, k * peri, True)
                if len(approx) == 4:
                    corners = order_corners(approx)
                    if _looks_like_board(corners) and cv2.contourArea(approx) > 0.8 * cv2.contourArea(hull):
                        return refine_corners(gray, corners) / scale
                    break
                if len(approx) < 4:
                    break
    return None


def refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Sub-pixel polish of the corner estimates (small, safe adjustment)."""
    try:
        pts = corners.reshape(-1, 1, 2).astype(np.float32).copy()
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit)
        refined = pts.reshape(4, 2)
        if np.abs(refined - corners).max() < 6:
            return refined
    except cv2.error:
        pass
    return corners


def warp_board(image: np.ndarray, corners: np.ndarray) -> Board:
    dst = np.array([[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1],
                    [0, WARP_SIZE - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    warped = cv2.warpPerspective(image, m, (WARP_SIZE, WARP_SIZE))
    return Board(corners=corners, warped=warped, matrix=m)


# --------------------------------------------------------------------------
# Digit extraction / normalisation (shared by inference AND training)
# --------------------------------------------------------------------------
def normalize_digit(mask: np.ndarray) -> np.ndarray:
    """Turn a binary mask (digit = 255) into a 28x28 float32 image in [0,1].

    The digit's bounding box is scaled to fit 20x20 (keeping aspect ratio)
    and placed so its centre of mass sits at the centre, like MNIST.
    """
    ys, xs = np.nonzero(mask)
    digit = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = digit.shape
    f = DIGIT_BOX / max(h, w)
    nh, nw = max(1, int(round(h * f))), max(1, int(round(w * f)))
    digit = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.float32)
    y0, x0 = (DIGIT_SIZE - nh) // 2, (DIGIT_SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = digit
    m = cv2.moments(out)
    if m["m00"] > 0:
        dx = DIGIT_SIZE / 2 - m["m10"] / m["m00"]
        dy = DIGIT_SIZE / 2 - m["m01"] / m["m00"]
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, shift, (DIGIT_SIZE, DIGIT_SIZE))
    return np.clip(out / 255.0, 0, 1).astype(np.float32)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold so ink becomes 255 on a 0 background."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 8)


def remove_grid_lines(binary: np.ndarray) -> np.ndarray:
    """Erase long horizontal/vertical strokes (the grid) from a warped board."""
    k = int(CELL * 0.9)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    lines = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def digit_from_binary(cell_bin: np.ndarray, min_fill: float = 0.012) -> Optional[np.ndarray]:
    """Pick the digit blob in a binarised cell; return a 28x28 image or None.

    The chosen connected component must be tall enough, lie near the cell
    centre and must not span the cell edge to edge (leftover grid lines).
    """
    h, w = cell_bin.shape
    cell_bin = cell_bin.copy()
    m = max(1, int(0.05 * min(h, w)))
    cell_bin[:m, :] = 0
    cell_bin[-m:, :] = 0
    cell_bin[:, :m] = 0
    cell_bin[:, -m:] = 0

    n, labels, stats, cents = cv2.connectedComponentsWithStats(cell_bin, connectivity=8)
    best, best_score = -1, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < 0.25 * h or area < min_fill * h * w:
            continue
        touches = (x <= m) + (y <= m) + (x + bw >= w - m) + (y + bh >= h - m)
        if touches >= 2 or bw > 0.85 * w or bh > 0.85 * h:
            continue
        cx, cy = cents[i]
        dist = np.hypot(cx - w / 2, cy - h / 2) / (0.5 * min(h, w))
        if dist > 0.6:
            continue
        score = area * (1.2 - dist)
        if score > best_score:
            best, best_score = i, score
    if best < 0:
        return None
    mask = np.where(labels == best, 255, 0).astype(np.uint8)
    return normalize_digit(mask)


def extract_digit(cell_gray: np.ndarray) -> Optional[np.ndarray]:
    """Convenience wrapper for a single grayscale cell image."""
    return digit_from_binary(remove_grid_lines(binarize(cell_gray)))


def extract_digits(board: Board) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Return positions of non-empty cells and a (N,28,28,1) batch for the CNN."""
    gray = cv2.cvtColor(board.warped, cv2.COLOR_BGR2GRAY)
    clean = remove_grid_lines(binarize(gray))
    positions, batch = [], []
    for r in range(9):
        for c in range(9):
            digit = digit_from_binary(clean[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL])
            if digit is not None:
                positions.append((r, c))
                batch.append(digit)
    arr = np.stack(batch)[..., None] if batch else np.zeros((0, DIGIT_SIZE, DIGIT_SIZE, 1), np.float32)
    return positions, arr


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_solution(image: np.ndarray, board: Board, puzzle: Sequence[Sequence[int]],
                  solution: Sequence[Sequence[int]], color=(40, 160, 40)) -> np.ndarray:
    """Write the missing digits onto the original photo (perspective-correct)."""
    overlay = np.zeros((WARP_SIZE, WARP_SIZE, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.3, 3
    for r in range(9):
        for c in range(9):
            if puzzle[r][c]:
                continue
            text = str(solution[r][c])
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            x = c * CELL + (CELL - tw) // 2
            y = r * CELL + (CELL + th) // 2
            cv2.putText(overlay, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    h, w = image.shape[:2]
    unwarped = cv2.warpPerspective(overlay, board.matrix, (w, h), flags=cv2.WARP_INVERSE_MAP)
    out = image.copy()
    mask = unwarped.any(axis=2)
    out[mask] = unwarped[mask]
    return out


def draw_outline(image: np.ndarray, corners: np.ndarray, color=(0, 140, 255)) -> np.ndarray:
    out = image.copy()
    t = max(2, int(0.004 * max(image.shape[:2])))
    cv2.polylines(out, [corners.astype(np.int32)], True, color, t, cv2.LINE_AA)
    return out


def render_grid_image(puzzle: Sequence[Sequence[int]], solution: Optional[Sequence[Sequence[int]]] = None,
                      size: int = 450) -> np.ndarray:
    """Clean rendering of a grid: givens in black, solved digits in green."""
    img = np.full((size, size, 3), 255, np.uint8)
    cell = size // 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(10):
        t = 3 if i % 3 == 0 else 1
        cv2.line(img, (i * cell, 0), (i * cell, 9 * cell), (0, 0, 0), t)
        cv2.line(img, (0, i * cell), (9 * cell, i * cell), (0, 0, 0), t)
    for r in range(9):
        for c in range(9):
            v, col = puzzle[r][c], (20, 20, 20)
            if not v and solution is not None:
                v, col = solution[r][c], (40, 150, 40)
            if v:
                text = str(v)
                (tw, th), _ = cv2.getTextSize(text, font, 1.2, 2)
                cv2.putText(img, text, (c * cell + (cell - tw) // 2, r * cell + (cell + th) // 2),
                            font, 1.2, col, 2, cv2.LINE_AA)
    return img
