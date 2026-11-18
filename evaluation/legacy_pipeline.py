"""Port of the ORIGINAL notebook pipeline (RealTimeSudokuSolver.ipynb + Main.ipynb).

Used only by evaluation/evaluate.py to measure the improvement of the new
pipeline over the original one on identical inputs. Logic is unchanged apart
from removing the camera loop and the 81 per-cell ``model.predict`` calls
being kept as-is (that is part of what we benchmark).
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from scipy import ndimage

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def build_legacy_model(weights_path: str):
    import keras
    from keras import layers

    model = keras.Sequential([
        keras.Input((28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(9, activation="softmax"),
    ])
    _load_keras2_h5_weights(model, weights_path)
    return model


def _load_keras2_h5_weights(model, path):
    """Load a Keras 2.3 ``save_weights`` .h5 file into a Keras 3 model."""
    import h5py

    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else n for n in f.attrs["layer_names"]]
        weighted = [l for l in model.layers if l.weights]
        groups = [n for n in names if len(f[n].attrs.get("weight_names", [])) > 0]
        assert len(groups) == len(weighted), (groups, weighted)
        for layer, g in zip(weighted, groups):
            wn = [w.decode() if isinstance(w, bytes) else w for w in f[g].attrs["weight_names"]]
            layer.set_weights([np.array(f[g][w]) for w in wn])


# ---------------------------------------------------------------- helpers (verbatim logic)
def getCornersFromContours(contours, corner_amount=4, max_iter=200):
    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter -= 1
        epsilon = coefficient * cv2.arcLength(contours, True)
        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        if len(hull) > corner_amount:
            coefficient += .01
        else:
            coefficient -= .01
    return None


def _angle(v1, v2):
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(u1, u2)) * 57.2958


def _border_diff(A, B, C, D, eps_scale):
    s = [math.dist(A, B), math.dist(A, D), math.dist(B, C), math.dist(C, D)]
    return max(s) > eps_scale * min(s)


def getLargestConnectedComponent(image):
    image = image.astype("uint8")
    nb, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    if len(sizes) <= 1:
        return np.full(image.shape, 255.0)
    max_label, max_size = 1, sizes[1]
    for i in range(2, nb):
        if sizes[i] > max_size:
            max_label, max_size = i, sizes[i]
    img2 = np.full(output.shape, 255.0)
    img2[output == max_label] = 0
    return img2


def getBestShift(img):
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    return np.round(cols / 2.0 - cx).astype(int), np.round(rows / 2.0 - cy).astype(int)


def shift(img, sx, sy):
    rows, cols = img.shape
    return cv2.warpAffine(img, np.float32([[1, 0, sx], [0, 1, sy]]), (cols, rows))


def legacy_detect(image):
    """Return (rect corners TL,TR,BR,BL, warp) or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_CONSTANT)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    corners = getCornersFromContours(biggest, 4)
    if corners is None:
        return None
    corners = corners.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), np.float32)
    s = corners.sum(1)
    i = int(np.argmin(s)); rect[0] = corners[i]; corners = np.delete(corners, i, 0)
    s = corners.sum(1)
    i = int(np.argmax(s)); rect[2] = corners[i]; corners = np.delete(corners, i, 0)
    if corners[0][0] > corners[1][0]:
        rect[1], rect[3] = corners[0], corners[1]
    else:
        rect[1], rect[3] = corners[1], corners[0]
    A, B, C, D = rect
    eps = 20
    if not all(abs(a - 90) < eps for a in (_angle(B - A, D - A), _angle(B - A, C - B),
                                           _angle(C - B, C - D), _angle(D - A, C - D))):
        return None
    if _border_diff(A, B, C, D, 1.2):
        return None
    tl, tr, br, bl = rect
    mw = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    mh = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], np.float32)
    m = cv2.getPerspectiveTransform(rect, dst)
    return rect, cv2.warpPerspective(image, m, (mw, mh))


def legacy_read_grid(image, model, batched: bool = False):
    """Original digit reading. Returns 9x9 grid or None if no board found."""
    det = legacy_detect(image)
    if det is None:
        return None
    _, warp = det
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    warp = cv2.GaussianBlur(warp, (5, 5), 0)
    warp = cv2.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = cv2.bitwise_not(warp)
    _, warp = cv2.threshold(warp, 150, 255, cv2.THRESH_BINARY)
    grid = [[0] * 9 for _ in range(9)]
    h, w = warp.shape[0] // 9, warp.shape[1] // 9
    oh, ow = math.floor(h / 10), math.floor(w / 10)
    pending = []
    for i in range(9):
        for j in range(9):
            crop = warp[h * i + oh:h * (i + 1) - oh, w * j + ow:w * (j + 1) - ow]
            ratio = 0.6
            while crop.shape[0] > 1 and np.sum(crop[0]) <= (1 - ratio) * crop.shape[1] * 255:
                crop = crop[1:]
            while crop.shape[0] > 1 and np.sum(crop[-1]) <= (1 - ratio) * crop.shape[1] * 255:
                crop = crop[:-1]
            while crop.shape[1] > 1 and np.sum(crop[:, 0]) <= (1 - ratio) * crop.shape[0] * 255:
                crop = np.delete(crop, 0, 1)
            while crop.shape[1] > 1 and np.sum(crop[:, -1]) <= (1 - ratio) * crop.shape[0] * 255:
                crop = np.delete(crop, -1, 1)
            crop = cv2.bitwise_not(crop)
            crop = getLargestConnectedComponent(crop)
            crop = cv2.resize(crop, (28, 28))
            if crop.sum() >= 28 ** 2 * 255 - 28 * 255:
                continue
            cw, ch = crop.shape[1] // 2, crop.shape[0] // 2
            center = crop[ch // 2:ch // 2 + ch, cw // 2:cw // 2 + cw]
            if center.sum() >= cw * ch * 255 - 255:
                continue
            _, crop = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY)
            crop = crop.astype(np.uint8)
            crop = cv2.bitwise_not(crop)
            sx, sy = getBestShift(crop)
            crop = cv2.bitwise_not(shift(crop, sx, sy))
            x = crop.reshape(-1, 28, 28, 1).astype("float32") / 255
            if batched:
                pending.append((i, j, x))
            else:
                pred = model.predict(x, verbose=0)
                grid[i][j] = int(np.argmax(pred[0])) + 1
    if pending:
        preds = model.predict(np.concatenate([p[2] for p in pending]), verbose=0)
        for (i, j, _), p in zip(pending, preds):
            grid[i][j] = int(np.argmax(p)) + 1
    return grid


def legacy_chars74k_input(path: str):
    """Preprocessing used in the original training notebook."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)
    sx, sy = getBestShift(img)
    img = cv2.bitwise_not(shift(img, sx, sy))
    return img.astype("float32") / 255
