"""Synthetic Sudoku photos with known ground truth.

Used to (a) generate realistic training crops for the digit CNN and
(b) measure end-to-end accuracy of the whole pipeline on boards the model
has never seen (rendered with held-out fonts).
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None

FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
             "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]
_SKIP = ("emoji", "symbol", "unifont", "wqy", "ipa", "opensymbol", "dingbat", "mathjax")


def find_fonts() -> List[str]:
    fonts = []
    for d in FONT_DIRS:
        for ext in ("ttf", "otf", "TTF"):
            fonts += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
    fonts = sorted({f for f in fonts if not any(s in f.lower() for s in _SKIP)})
    return fonts


def split_fonts(fonts: Sequence[str], holdout_every: int = 5):
    """Deterministic train/held-out split of font files (by family name)."""
    fams = sorted({os.path.basename(f).split("-")[0].split("_")[0] for f in fonts})
    held = {f for i, f in enumerate(fams) if i % holdout_every == holdout_every - 1}
    train = [f for f in fonts if os.path.basename(f).split("-")[0].split("_")[0] not in held]
    test = [f for f in fonts if os.path.basename(f).split("-")[0].split("_")[0] in held]
    return train, test


def random_solution(rng: random.Random) -> List[List[int]]:
    base = lambda r, c: (3 * (r % 3) + r // 3 + c) % 9
    groups = [0, 1, 2]
    rows = [g * 3 + r for g in rng.sample(groups, 3) for r in rng.sample(groups, 3)]
    cols = [g * 3 + c for g in rng.sample(groups, 3) for c in rng.sample(groups, 3)]
    nums = rng.sample(range(1, 10), 9)
    return [[nums[base(r, c)] for c in cols] for r in rows]


def random_puzzle(rng: random.Random, givens=(24, 36)):
    sol = random_solution(rng)
    k = rng.randint(*givens)
    keep = set(rng.sample(range(81), k))
    puzzle = [[sol[r][c] if r * 9 + c in keep else 0 for c in range(9)] for r in range(9)]
    return puzzle, sol


@dataclass
class SynthBoard:
    image: np.ndarray          # BGR photo-like image
    puzzle: List[List[int]]
    solution: List[List[int]]
    corners: np.ndarray        # true TL, TR, BR, BL in image coords


_HERSHEY = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN]


def _render_flat(puzzle, rng: random.Random, fonts: Sequence[str], size: int, mnist=None):
    paper = rng.randint(200, 255)
    ink = rng.randint(0, 70)
    img = np.full((size, size), paper, np.uint8)
    cell = size / 9
    thin, thick = rng.choice([1, 1, 2]), rng.choice([3, 4, 5])
    for i in range(10):
        t = thick if i % 3 == 0 else thin
        p = int(round(i * cell))
        cv2.line(img, (p, 0), (p, size - 1), ink, t)
        cv2.line(img, (0, p), (size - 1, p), ink, t)

    font_path = rng.choice(fonts) if fonts else None
    use_pil = font_path is not None and Image is not None
    frac = rng.uniform(0.5, 0.8)
    if use_pil:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype(font_path, int(cell * frac))
        except OSError:
            use_pil = False
    hfont = rng.choice(_HERSHEY) | (cv2.FONT_ITALIC if rng.random() < 0.2 else 0)
    handwritten = []
    for r in range(9):
        for c in range(9):
            v = puzzle[r][c]
            if not v:
                continue
            cx = (c + 0.5) * cell + rng.uniform(-0.08, 0.08) * cell
            cy = (r + 0.5) * cell + rng.uniform(-0.08, 0.08) * cell
            if mnist is not None and rng.random() < 0.5:
                handwritten.append((v, cx, cy))
            elif use_pil:
                draw.text((cx, cy), str(v), font=font, fill=int(ink), anchor="mm")
            else:
                scale = cell * frac / 30.0
                th = max(1, int(scale * rng.uniform(1.5, 2.5)))
                (tw, tht), _ = cv2.getTextSize(str(v), hfont, scale, th)
                cv2.putText(img, str(v), (int(cx - tw / 2), int(cy + tht / 2)),
                            hfont, scale, int(ink), th, cv2.LINE_AA)
    if use_pil:
        img = np.array(pil)
    for v, cx, cy in handwritten:      # paste handwritten MNIST digits
        pool = mnist[v]
        s = int(cell * rng.uniform(0.75, 0.92))
        d = cv2.resize(pool[rng.randrange(len(pool))], (s, s)).astype(np.float32) / 255
        x0, y0 = int(cx - s / 2), int(cy - s / 2)
        x0, y0 = min(max(x0, 0), size - s), min(max(y0, 0), size - s)
        region = img[y0:y0 + s, x0:x0 + s].astype(np.float32)
        img[y0:y0 + s, x0:x0 + s] = (region * (1 - d) + ink * d).astype(np.uint8)
    return img


def make_board(rng: random.Random, fonts: Sequence[str], puzzle=None, solution=None,
               out_size=(800, 600), mnist=None, difficulty: float = 1.0) -> SynthBoard:
    """Render a puzzle and place it in a photo-like scene with perspective."""
    if puzzle is None:
        puzzle, solution = random_puzzle(rng)
    flat_size = 450
    flat = _render_flat(puzzle, rng, fonts, flat_size, mnist=mnist)
    W, H = out_size
    bg_level = rng.randint(30, 180)
    bg = np.full((H, W), bg_level, np.float32)
    bg += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 8, (H, W))

    # Board placement with random perspective.
    for _attempt in range(20):        # keep the whole board inside the frame
        side = rng.uniform(0.55, 0.85) * min(W, H)
        cx, cy = W / 2 + rng.uniform(-0.1, 0.1) * W, H / 2 + rng.uniform(-0.08, 0.08) * H
        j = 0.07 * side * difficulty
        ang = np.radians(rng.uniform(-12, 12) * difficulty)
        base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32) * side / 2
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        dst = base @ rot.T + np.array([cx, cy], np.float32)
        dst += np.array([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)], np.float32)
        if dst[:, 0].min() > 5 and dst[:, 1].min() > 5 and dst[:, 0].max() < W - 5 and dst[:, 1].max() < H - 5:
            break
    src = np.array([[0, 0], [flat_size - 1, 0], [flat_size - 1, flat_size - 1], [0, flat_size - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(flat.astype(np.float32), m, (W, H), borderValue=-1)
    mask = cv2.warpPerspective(np.ones_like(flat, np.float32), m, (W, H)) > 0.5
    img = np.where(mask, warped, bg)

    # Lighting gradient, blur, noise, JPEG.
    gx = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.85, 1.15), W, dtype=np.float32)
    gy = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.85, 1.1), H, dtype=np.float32)
    img = img * np.outer(gy, gx)
    k = rng.choice([1, 3, 3, 5])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    tint = np.array([rng.uniform(0.85, 1.0), rng.uniform(0.9, 1.0), 1.0])
    color = np.clip(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * tint, 0, 255).astype(np.uint8)
    q = rng.randint(55, 95)
    ok, enc = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, q])
    color = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if solution is None:
        from .solver import solve
        solution = solve(puzzle)
    return SynthBoard(image=color, puzzle=puzzle, solution=solution, corners=dst.astype(np.float32))


def load_mnist_by_digit(max_per_digit: int = 2000, seed: int = 0, split: str = "test"):
    """MNIST digits grouped by label (1-9), white on black, uint8.

    ``split='train'`` is used for training data, ``'test'`` for evaluation, so
    handwritten evaluation digits are never seen during training.
    """
    try:
        import keras
        train, test = keras.datasets.mnist.load_data()
        x, y = train if split == "train" else test
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    out = {}
    for d in range(1, 10):
        idx = np.nonzero(y == d)[0]
        rng.shuffle(idx)
        out[d] = x[idx[:max_per_digit]]
    return out
