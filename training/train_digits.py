"""Train the SudokuLens digit CNN (digits 1-9).

Data (all passed through the SAME normalisation used at inference time):
  * the project's original Chars74K digit crops (CNN Digit Training/DigitImages),
    split 80/20 with a fixed seed; the 20% is never trained on
  * cells cut out of synthetic Sudoku photos rendered with the "train" font
    families (held-out font families are reserved for evaluation)
  * handwritten digits from the MNIST *training* split

Outputs: models/digit_cnn.keras, models/digit_cnn.onnx,
         results/training_history.json, results/training_curves.png

Usage:  python training/train_digits.py [--boards 1500] [--epochs 20]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sudokulens import vision  # noqa: E402
from sudokulens.synthetic import find_fonts, load_mnist_by_digit, make_board, split_fonts  # noqa: E402

DEFAULT_CHARS74K = ROOT / "data" / "DigitImages"
SEED = 42


# ---------------------------------------------------------------- data
def chars74k_to_input(path: str):
    """Natural-image digit crop -> binarised, normalised 28x28 input."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    s = 64.0 / max(h, w)
    img = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:          # background is white -> invert so ink = 255
        bw = 255 - bw
    pad = int(0.3 * max(bw.shape))
    bw = cv2.copyMakeBorder(bw, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    d = vision.digit_from_binary(bw)
    if d is None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        d = vision.normalize_digit(np.where(labels == k, 255, 0).astype(np.uint8))
    return d


def load_chars74k(root: Path, seed: int = SEED):
    """Return stratified (train, test) lists of (x, label) and file names."""
    rng = random.Random(seed)
    train, test = [], []
    for d in range(1, 10):
        files = sorted((root / str(d)).glob("*.png"))
        rng.shuffle(files)
        n_test = round(0.2 * len(files))
        for i, f in enumerate(files):
            x = chars74k_to_input(str(f))
            if x is None:
                continue
            (test if i < n_test else train).append((x, d, f.name))
    return train, test


def synthetic_cells(n_boards: int, fonts, mnist, seed: int = SEED):
    rng = random.Random(seed)
    xs, ys = [], []
    for i in range(n_boards):
        b = make_board(rng, fonts, mnist=mnist if i % 3 == 0 else None)
        corners = vision.find_board_corners(b.image)
        if corners is None or np.abs(corners - b.corners).max() > 15:
            corners = b.corners
        board = vision.warp_board(b.image, corners)
        positions, batch = vision.extract_digits(board)
        for (r, c), x in zip(positions, batch):
            if b.puzzle[r][c]:
                xs.append(x[..., 0])
                ys.append(b.puzzle[r][c])
    return xs, ys


def mnist_cells(n_per_digit: int):
    mn = load_mnist_by_digit(n_per_digit, seed=SEED, split="train")
    xs, ys = [], []
    if mn is None:
        print("  (MNIST unavailable, skipping handwritten digits)")
        return xs, ys
    for d, imgs in mn.items():
        for im in imgs:
            _, bw = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY)
            if bw.any():
                xs.append(vision.normalize_digit(bw))
                ys.append(d)
    return xs, ys


# ---------------------------------------------------------------- model
def build_model():
    import keras
    from keras import layers

    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    inp = keras.Input((28, 28, 1), name="digit")
    x = block(inp, 32)
    x = block(x, 32)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = block(x, 64)
    x = block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(9, activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="sudokulens_digit_cnn")


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small random affine jitter (rotation, scale, shift) + stroke width change."""
    out = np.empty_like(x)
    for i in range(len(x)):
        ang = rng.uniform(-10, 10)
        sc = rng.uniform(0.9, 1.1)
        m = cv2.getRotationMatrix2D((14, 14), ang, sc)
        m[:, 2] += rng.uniform(-2, 2, 2)
        img = cv2.warpAffine(x[i], m, (28, 28))
        r = rng.random()
        if r < 0.15:
            img = cv2.dilate(img, np.ones((2, 2), np.uint8))
        elif r < 0.3:
            img = cv2.erode(img, np.ones((2, 2), np.uint8))
        out[i] = img
    return out


def export_onnx(model, path: Path) -> bool:
    try:
        import tensorflow as tf
        import tf2onnx

        spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="digit"),)
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(path))
        return True
    except Exception as exc:  # pragma: no cover
        print(f"  ONNX export failed ({exc}); the app will fall back to Keras.")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars74k", default=str(DEFAULT_CHARS74K))
    ap.add_argument("--boards", type=int, default=1500)
    ap.add_argument("--mnist-per-digit", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    t0 = time.time()

    print("Loading Chars74K crops ...")
    c_train, c_test = load_chars74k(Path(args.chars74k))
    print(f"  {len(c_train)} train / {len(c_test)} held-out test")
    fonts, _held = split_fonts(find_fonts())
    print(f"Rendering {args.boards} synthetic boards with {len(fonts)} training fonts ...")
    mnist_train = load_mnist_by_digit(400, seed=SEED + 1, split="train")
    s_x, s_y = synthetic_cells(args.boards, fonts, mnist_train)
    print(f"  {len(s_x)} cells")
    print("Loading MNIST (train split) ...")
    m_x, m_y = mnist_cells(args.mnist_per_digit)
    print(f"  {len(m_x)} digits")

    cx = [x for x, _, _ in c_train] * 10          # oversample the small real-photo set
    cy = [y for _, y, _ in c_train] * 10
    X = np.array(cx + s_x + m_x, np.float32)[..., None]
    Y = np.array(cy + s_y + m_y, np.int64) - 1
    perm = np.random.default_rng(SEED).permutation(len(X))
    X, Y = X[perm], Y[perm]
    n_val = int(0.1 * len(X))
    Xv, Yv, Xt, Yt = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]
    print(f"Training on {len(Xt)} samples, validating on {len(Xv)}")

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    rng = np.random.default_rng(SEED)

    class Aug(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
        def __len__(self):
            return int(np.ceil(len(Xt) / args.batch))
        def __getitem__(self, i):
            sl = slice(i * args.batch, (i + 1) * args.batch)
            return augment(Xt[sl, ..., 0], rng)[..., None], Yt[sl]

    out_models = ROOT / "models"
    out_results = ROOT / "results"
    out_models.mkdir(exist_ok=True)
    out_results.mkdir(exist_ok=True)
    cbs = [keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
           keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    hist = model.fit(Aug(), validation_data=(Xv, Yv), epochs=args.epochs, callbacks=cbs, verbose=2)

    loss, acc = model.evaluate(np.array([x for x, _, _ in c_test])[..., None],
                               np.array([y for _, y, _ in c_test]) - 1, verbose=0)
    print(f"Chars74K held-out: loss={loss:.4f} acc={acc:.4f}")

    clean = build_model()                     # same weights, no optimizer state -> smaller file
    clean.set_weights(model.get_weights())
    clean.save(out_models / "digit_cnn.keras")
    model = clean
    ok = export_onnx(model, out_models / "digit_cnn.onnx")
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    meta = {"samples": {"chars74k_train": len(c_train), "chars74k_test": len(c_test),
                        "synthetic_cells": len(s_x), "mnist": len(m_x),
                        "train": int(len(Xt)), "val": int(n_val)},
            "params": int(model.count_params()), "epochs_run": len(history["loss"]),
            "chars74k_test": {"loss": float(loss), "accuracy": float(acc)},
            "onnx_exported": ok, "train_seconds": round(time.time() - t0, 1),
            "history": history}
    (out_results / "training_history.json").write_text(json.dumps(meta, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ep = range(1, len(history["loss"]) + 1)
        ax[0].plot(ep, history["loss"], label="train")
        ax[0].plot(ep, history["val_loss"], label="validation")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(ep, history["accuracy"], label="train")
        ax[1].plot(ep, history["val_accuracy"], label="validation")
        ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_results / "training_curves.png", dpi=130)
    except Exception as exc:  # pragma: no cover
        print("Could not plot curves:", exc)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
