import random
from pathlib import Path

import numpy as np
import pytest

from sudokulens import vision
from sudokulens.synthetic import find_fonts, make_board

MODELS = Path(__file__).resolve().parent.parent / "models"


def test_order_corners():
    pts = np.array([[100, 0], [0, 0], [0, 100], [100, 100]], np.float32)
    out = vision.order_corners(pts)
    assert out.tolist() == [[0, 0], [100, 0], [100, 100], [0, 100]]


def test_no_board_in_blank_image():
    assert vision.find_board_corners(np.full((300, 400, 3), 128, np.uint8)) is None


def test_detects_board_and_empty_cells():
    rng = random.Random(0)
    fonts = find_fonts()
    hits = 0
    for _ in range(10):
        b = make_board(rng, fonts)
        corners = vision.find_board_corners(b.image)
        if corners is None:
            continue
        assert np.abs(corners - b.corners).max() < 25
        positions, batch = vision.extract_digits(vision.warp_board(b.image, corners))
        truth = {(r, c) for r in range(9) for c in range(9) if b.puzzle[r][c]}
        assert len(set(positions) ^ truth) <= 2
        assert batch.shape[1:] == (28, 28, 1)
        hits += 1
    assert hits >= 9


@pytest.mark.skipif(not (MODELS / "digit_cnn.onnx").exists(), reason="model not trained")
def test_end_to_end_solves_synthetic_photo():
    from sudokulens.pipeline import solve_image
    rng = random.Random(3)
    solved = 0
    for _ in range(5):
        b = make_board(rng, find_fonts())
        res = solve_image(b.image)
        solved += res.solved and res.puzzle == b.puzzle
    assert solved >= 4


@pytest.mark.skipif(not (MODELS / "digit_cnn.onnx").exists(), reason="model not trained")
def test_sample_assets_solve():
    import cv2
    from sudokulens.pipeline import solve_image
    samples = sorted((MODELS.parent / "assets").glob("sample_*.jpg"))
    assert samples
    for p in samples:
        assert solve_image(cv2.imread(str(p))).solved, p.name
