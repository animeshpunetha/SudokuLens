"""Evaluate SudokuLens and save all scores to results/.

Produces
  results/metrics.json        every number below, machine readable
  results/RESULTS.md          the same as Markdown tables (linked from README)
  results/confusion_matrix.png

Sections
  1. Digit CNN accuracy on the held-out Chars74K split (new vs original model)
  2. Digit CNN accuracy on unseen fonts and unseen handwriting (MNIST test)
  3. End-to-end accuracy on synthetic photos (new vs original pipeline)
  4. Solver speed (new vs original solver) on easy -> hardest puzzles
  5. Inference speed / model loading (ONNX batched vs Keras per-cell)

Usage: python evaluation/evaluate.py [--boards 200]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from evaluation.legacy_pipeline import (build_legacy_model, legacy_chars74k_input,  # noqa: E402
                                        legacy_read_grid)
from evaluation import legacy_solver  # noqa: E402
from sudokulens import vision  # noqa: E402
from sudokulens.pipeline import read_grid  # noqa: E402
from sudokulens.recognizer import DigitRecognizer  # noqa: E402
from sudokulens.solver import parse_grid, solve, count_solutions  # noqa: E402
from sudokulens.synthetic import (find_fonts, load_mnist_by_digit, make_board,  # noqa: E402
                                  random_puzzle, split_fonts)
from training.train_digits import DEFAULT_CHARS74K, load_chars74k  # noqa: E402

RESULTS = ROOT / "results"
LEGACY_WEIGHTS = ROOT / "legacy" / "digitRecognition.h5"

# Well-known hard puzzles (all have a unique solution; verified at runtime).
HARD_PUZZLES = {
    "Arto Inkala 2012 ('world's hardest')":
        "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..",
    "AI Escargot":
        "1....7.9..3..2...8..96..5....53..9...1..8...26....4...3......1..4......7..7...3..",
    "Hard benchmark A (22 givens)":
        "85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.",
    "Easter Monster":
        "1.......2.9.4...5...6...7...5.9.3.......7.......85..4.7.....6...3...9.8...2.....1",
    "Hard benchmark B (20 givens)":
        "...1.2....6.....7...8...9..4.......3.5...7...2...8...1..9...8.5.7.....6....3.4...",
    "17-clue puzzle (minimum possible givens)":
        "...8.1..........435............7.8........1...2..3....6......75..34........2..6..",
}
EASY = ("..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..")


def acc(pred, true) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(true))) if len(true) else float("nan")


def timeit(fn, repeat=1):
    ts = []
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return out, ts


# ------------------------------------------------------------------ 1 & 2
def eval_digits(rec: DigitRecognizer, legacy_model, fonts_test, mnist_test, n_boards: int):
    out = {}
    _, test = load_chars74k(DEFAULT_CHARS74K)
    y = np.array([d for _, d, _ in test])
    xn = np.array([x for x, _, _ in test])[..., None]
    pn, _ = rec.predict(xn)
    xl = np.array([legacy_chars74k_input(str(DEFAULT_CHARS74K / str(d) / n)) for _, d, n in test])[..., None]
    pl = legacy_model.predict(xl, verbose=0).argmax(1) + 1
    out["chars74k_heldout"] = {"n": int(len(y)), "new_model": acc(pn, y), "original_model": acc(pl, y)}
    cm = np.zeros((9, 9), int)
    for t, p in zip(y, pn):
        cm[t - 1, p - 1] += 1

    # Unseen fonts / unseen handwriting: cells cut from synthetic boards.
    rng = random.Random(1234)
    groups = {"printed_unseen_fonts": ([], []), "handwritten_mnist_test": ([], [])}
    for i in range(n_boards):
        hand = i % 2 == 1
        b = make_board(rng, fonts_test, mnist=mnist_test if hand else None)
        board = vision.warp_board(b.image, b.corners)
        pos, batch = vision.extract_digits(board)
        xs, ys = groups["handwritten_mnist_test" if hand else "printed_unseen_fonts"]
        for (r, c), x in zip(pos, batch):
            if b.puzzle[r][c]:
                xs.append(x)
                ys.append(b.puzzle[r][c])
    for name, (xs, ys) in groups.items():
        xs = np.array(xs)
        p, _ = rec.predict(xs)
        pl = legacy_model.predict(1.0 - xs, verbose=0).argmax(1) + 1   # original model wants black-on-white
        out[name] = {"n": len(ys), "new_model": acc(p, ys), "original_model": acc(pl, ys)}
        for t, pp in zip(ys, p):
            cm[t - 1, pp - 1] += 1
    return out, cm


# ------------------------------------------------------------------ 3
def eval_end_to_end(rec, legacy_model, fonts_test, mnist_test, n: int):
    rng = random.Random(2024)
    res = {}
    stats = {k: {"detected": 0, "cells_ok": 0, "cells": 0, "board_exact": 0, "solved": 0, "ms": []}
             for k in ("new_pipeline", "original_pipeline")}
    split = {"printed": 0, "handwritten": 0}
    split_ok = {"printed": 0, "handwritten": 0}
    for i in range(n):
        hand = i % 4 == 3      # 25% of boards contain handwritten digits
        b = make_board(rng, fonts_test, mnist=mnist_test if hand else None)
        kind = "handwritten" if hand else "printed"
        split[kind] += 1
        for name in stats:
            t = time.perf_counter()
            if name == "new_pipeline":
                r = read_grid(b.image, rec)
                grid = r[1] if r else None
            else:
                grid = legacy_read_grid(b.image, legacy_model, batched=True)
            s = stats[name]
            s["cells"] += 81
            ok_solve = False
            if grid is not None:
                s["detected"] += 1
                s["cells_ok"] += sum(grid[r][c] == b.puzzle[r][c] for r in range(9) for c in range(9))
                if grid == b.puzzle:
                    s["board_exact"] += 1
                try:
                    sol = solve(grid)
                    ok_solve = all(sol[r][c] == b.puzzle[r][c] for r in range(9) for c in range(9)
                                   if b.puzzle[r][c]) and sol == solve(b.puzzle)
                except Exception:
                    ok_solve = False
            s["ms"].append((time.perf_counter() - t) * 1000)
            s["solved"] += ok_solve
            if name == "new_pipeline" and ok_solve:
                split_ok[kind] += 1
    for name, s in stats.items():
        res[name] = {"boards": n, "board_detected": s["detected"] / n,
                     "cell_accuracy": s["cells_ok"] / s["cells"],
                     "board_read_exactly": s["board_exact"] / n,
                     "solved_correctly": s["solved"] / n,
                     "median_ms_per_image": statistics.median(s["ms"])}
    res["new_pipeline"]["solved_correctly_printed"] = split_ok["printed"] / max(1, split["printed"])
    res["new_pipeline"]["solved_correctly_handwritten"] = split_ok["handwritten"] / max(1, split["handwritten"])
    return res


# ------------------------------------------------------------------ 4
def eval_solver(n_random: int = 200):
    rng = random.Random(7)
    sets = {"random (24-36 givens)": [random_puzzle(rng)[0] for _ in range(n_random)],
            "newspaper easy": [parse_grid(EASY)]}
    out = {"sets": {}, "hard_puzzles": {}}
    for name, puzzles in sets.items():
        new_t, old_t = [], []
        for p in puzzles:
            _, t = timeit(lambda: solve(p), repeat=3)
            new_t.append(min(t))
            m = [row[:] for row in p]
            _, t = timeit(lambda: legacy_solver.solveSudoku(m))
            old_t.append(t[0])
        out["sets"][name] = {"n": len(puzzles), "new_ms_mean": statistics.mean(new_t),
                             "original_ms_mean": statistics.mean(old_t),
                             "speedup": statistics.mean(old_t) / statistics.mean(new_t)}
    for name, s in HARD_PUZZLES.items():
        p = parse_grid(s)
        assert count_solutions(p) == 1, name
        sol, t_new = timeit(lambda: solve(p), repeat=3)
        m = [row[:] for row in p]
        _, t_old = timeit(lambda: legacy_solver.solveSudoku(m))
        assert m == sol, f"original solver disagrees on {name}"
        out["hard_puzzles"][name] = {"new_ms": min(t_new), "original_ms": t_old[0],
                                     "speedup": t_old[0] / min(t_new)}
    hard_new = [v["new_ms"] for v in out["hard_puzzles"].values()]
    hard_old = [v["original_ms"] for v in out["hard_puzzles"].values()]
    out["hard_summary"] = {"new_ms_mean": statistics.mean(hard_new), "original_ms_mean": statistics.mean(hard_old),
                           "new_ms_max": max(hard_new), "original_ms_max": max(hard_old),
                           "speedup": statistics.mean(hard_old) / statistics.mean(hard_new)}
    return out


# ------------------------------------------------------------------ 5
def eval_inference(legacy_model, fonts_test):
    out = {}
    t = time.perf_counter()
    onnx_rec = DigitRecognizer(ROOT / "models" / "digit_cnn.onnx")
    out["onnx_load_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    keras_rec = DigitRecognizer(ROOT / "models" / "digit_cnn.keras")
    out["keras_load_ms"] = (time.perf_counter() - t) * 1000

    b = make_board(random.Random(5), fonts_test)
    board = vision.warp_board(b.image, b.corners)
    _, batch = vision.extract_digits(board)
    out["digits_per_board"] = int(len(batch))
    onnx_rec.predict(batch)
    keras_rec.predict(batch)
    _, t1 = timeit(lambda: onnx_rec.predict(batch), repeat=20)
    _, t2 = timeit(lambda: keras_rec.predict(batch), repeat=20)
    _, t3 = timeit(lambda: [legacy_model.predict(x[None], verbose=0) for x in batch], repeat=3)
    out["onnx_batched_ms"] = statistics.median(t1)
    out["keras_batched_ms"] = statistics.median(t2)
    out["original_per_cell_predict_ms"] = statistics.median(t3)
    out["speedup_vs_original"] = out["original_per_cell_predict_ms"] / out["onnx_batched_ms"]
    _, tt = timeit(lambda: read_grid(b.image, onnx_rec), repeat=10)
    out["full_read_grid_ms"] = statistics.median(tt)
    out["onnx_model_mb"] = (ROOT / "models" / "digit_cnn.onnx").stat().st_size / 1e6
    return out


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def write_markdown(m):
    L = ["# SudokuLens evaluation results", "",
         f"Generated by `evaluation/evaluate.py` on {m['meta']['date']} "
         f"({m['meta']['platform']}, Python {m['meta']['python']}, CPU only).", ""]
    d = m["digit_recognition"]
    L += ["## 1-2. Digit recognition accuracy", "",
          "| Test set | Samples | Original model | New model |", "|---|---:|---:|---:|"]
    labels = {"chars74k_heldout": "Chars74K photos, held-out 20% split",
              "printed_unseen_fonts": "Printed digits, font families never seen in training",
              "handwritten_mnist_test": "Handwritten digits (MNIST test split)"}
    for k, lab in labels.items():
        L.append(f"| {lab} | {d[k]['n']} | {pct(d[k]['original_model'])} | **{pct(d[k]['new_model'])}** |")
    L += ["", "![Confusion matrix](confusion_matrix.png)", ""]
    e = m["end_to_end"]
    L += ["## 3. End-to-end on synthetic photos", "",
          f"{e['new_pipeline']['boards']} unseen photos (perspective, rotation, blur, noise, uneven light, JPEG; "
          "held-out fonts; 25% with handwritten digits).", "",
          "| Metric | Original pipeline | New pipeline |", "|---|---:|---:|"]
    for k, lab in [("board_detected", "Board detected"), ("cell_accuracy", "Cell accuracy (81 cells, incl. blanks)"),
                   ("board_read_exactly", "Whole board read exactly"), ("solved_correctly", "Solved correctly")]:
        dg = 2 if k == "cell_accuracy" else 1
        L.append(f"| {lab} | {pct(e['original_pipeline'][k], dg)} | **{pct(e['new_pipeline'][k], dg)}** |")
    L.append(f"| Median time per photo | {e['original_pipeline']['median_ms_per_image']:.0f} ms "
             f"| **{e['new_pipeline']['median_ms_per_image']:.0f} ms** |")
    L += ["", f"New pipeline solve rate: printed {pct(e['new_pipeline']['solved_correctly_printed'])}, "
          f"handwritten {pct(e['new_pipeline']['solved_correctly_handwritten'])}.", ""]
    s = m["solver"]
    L += ["## 4. Solver speed", "", "| Puzzles | Original (ms) | New (ms) | Speed-up |", "|---|---:|---:|---:|"]
    for k, v in s["sets"].items():
        L.append(f"| {k}, n={v['n']} | {v['original_ms_mean']:.2f} | {v['new_ms_mean']:.2f} | {v['speedup']:.0f}x |")
    for k, v in s["hard_puzzles"].items():
        L.append(f"| {k} | {v['original_ms']:.1f} | {v['new_ms']:.2f} | {v['speedup']:.0f}x |")
    i = m["inference"]
    L += ["", "## 5. Inference speed", "", "| Step | Time |", "|---|---:|",
          f"| Load model, ONNX Runtime | {i['onnx_load_ms']:.0f} ms |",
          f"| Load model, Keras/TensorFlow | {i['keras_load_ms']:.0f} ms |",
          f"| Classify {i['digits_per_board']} digits, original (one `predict` per cell) | {i['original_per_cell_predict_ms']:.0f} ms |",
          f"| Classify {i['digits_per_board']} digits, Keras batched | {i['keras_batched_ms']:.1f} ms |",
          f"| Classify {i['digits_per_board']} digits, ONNX batched | **{i['onnx_batched_ms']:.2f} ms** |",
          f"| Full photo -> grid (detect + extract + classify) | {i['full_read_grid_ms']:.0f} ms |", ""]
    (RESULTS / "RESULTS.md").write_text("\n".join(L))


def plot_cm(cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(9), range(1, 10))
    ax.set_yticks(range(9), range(1, 10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("New model, all digit test sets")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=200)
    ap.add_argument("--markdown-only", action="store_true", help="rebuild RESULTS.md from metrics.json")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.markdown_only:
        write_markdown(json.loads((RESULTS / "metrics.json").read_text()))
        return
    rec = DigitRecognizer()
    legacy = build_legacy_model(str(LEGACY_WEIGHTS))
    _, fonts_test = split_fonts(find_fonts())
    mnist_test = load_mnist_by_digit(800, seed=99, split="test")
    print(f"Held-out fonts: {len(fonts_test)}")

    m = {"meta": {"date": time.strftime("%Y-%m-%d"), "platform": platform.platform(terse=True),
                  "python": platform.python_version(), "model_backend": rec.backend}}
    print("1-2. digit accuracy ...")
    m["digit_recognition"], cm = eval_digits(rec, legacy, fonts_test, mnist_test, n_boards=120)
    print("3. end-to-end ...")
    m["end_to_end"] = eval_end_to_end(rec, legacy, fonts_test, mnist_test, args.boards)
    print("4. solver ...")
    m["solver"] = eval_solver()
    print("5. inference ...")
    m["inference"] = eval_inference(legacy, fonts_test)
    train_meta = RESULTS / "training_history.json"
    if train_meta.exists():
        t = json.loads(train_meta.read_text())
        m["training"] = {k: t[k] for k in ("samples", "params", "epochs_run", "train_seconds")}
    (RESULTS / "metrics.json").write_text(json.dumps(m, indent=2))
    plot_cm(cm)
    write_markdown(m)
    print((RESULTS / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
