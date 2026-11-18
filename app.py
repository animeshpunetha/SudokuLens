"""SudokuLens web app.

    streamlit run app.py

Three ways in: upload a photo, snap one with the camera, or type the grid.
Recognised digits are shown in an editable grid so mistakes can be fixed
before solving.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from sudokulens import vision
from sudokulens.pipeline import solve_image
from sudokulens.recognizer import get_recognizer
from sudokulens.solver import SudokuError, find_conflicts, parse_grid, solve

st.set_page_config(page_title="SudokuLens", page_icon="🔢", layout="wide")

EXAMPLE = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
COLS = [str(i) for i in range(1, 10)]


@st.cache_resource(show_spinner="Loading digit model ...")
def load_model():
    return get_recognizer()


def grid_to_df(grid):
    return pd.DataFrame([[str(v) if v else "" for v in row] for row in grid], columns=COLS,
                        index=[str(i) for i in range(1, 10)], dtype="string")


def df_to_grid(df):
    grid = []
    for _, row in df.iterrows():
        vals = []
        for v in row:
            v = "" if v is None or pd.isna(v) else str(v).strip()
            if v in ("", "0", "."):
                vals.append(0)
            elif len(v) == 1 and v in "123456789":
                vals.append(int(v))
            else:
                raise SudokuError(f"'{v}' is not a valid cell. Use 1-9, or leave it empty.")
        grid.append(vals)
    return grid


def grid_editor(grid, key):
    cfg = {c: st.column_config.TextColumn(c, max_chars=1, validate=r"^[1-9]?$", width="small") for c in COLS}
    return st.data_editor(grid_to_df(grid), column_config=cfg, key=key, width="content")


def show_solution(puzzle, label="Solution"):
    try:
        conflicts = find_conflicts(puzzle)
        if conflicts:
            cells = ", ".join(f"row {r + 1} col {c + 1}" for r, c in conflicts)
            st.error(f"These digits break Sudoku rules: {cells}.")
            return
        t = time.perf_counter()
        sol = solve(puzzle)
        ms = (time.perf_counter() - t) * 1000
    except SudokuError as exc:
        st.error(str(exc))
        return
    st.success(f"Solved in {ms:.1f} ms")
    st.image(cv2.cvtColor(vision.render_grid_image(puzzle, sol), cv2.COLOR_BGR2RGB), caption=label, width=380)
    with st.expander("Copy as text"):
        st.code("\n".join("".join(map(str, r)) for r in sol))


def decode(file) -> np.ndarray:
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read that image.")
    return img


def photo_flow(img: np.ndarray, key: str):
    rec = load_model()
    res = solve_image(img, rec)
    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Your photo", width="stretch")
    if not res.found_board:
        st.warning("No Sudoku grid found. Try a flatter, closer, well-lit photo where the whole grid is visible.")
        return
    with c2:
        st.image(cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB),
                 caption="Solution drawn on the photo" if res.solved else "Detected grid",
                 width="stretch")
    t = res.timings_ms
    st.caption(f"Detect + read: {t.get('detect_and_read', 0):.0f} ms · solve: {t.get('solve', 0):.1f} ms · "
               f"total: {t.get('total', 0):.0f} ms · model: {rec.backend}")
    if res.error:
        st.error(res.error)
    low = [(r, c) for r in range(9) for c in range(9) if res.puzzle[r][c] and res.confidence[r][c] < 0.8]
    if low:
        st.info("Low-confidence digits at " + ", ".join(f"row {r + 1} col {c + 1}" for r, c in low)
                + ". Double-check them below.")
    st.markdown("**Recognised digits** (edit any mistakes, then press *Solve edited grid*)")
    df = grid_editor(res.puzzle, key=f"{key}_editor")
    if st.button("Solve edited grid", key=f"{key}_solve"):
        try:
            show_solution(df_to_grid(df), "Solution for the edited grid")
        except SudokuError as exc:
            st.error(str(exc))


st.title("🔢 SudokuLens")
st.caption("Snap or upload a Sudoku, and a CNN reads the digits and a constraint-propagation solver fills it in.")

tab_up, tab_cam, tab_type = st.tabs(["📷 Upload photo", "🎥 Camera", "⌨️ Type a grid"])

with tab_up:
    up = st.file_uploader("Sudoku photo", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if up is None:
        st.caption("No photo handy? Try one of the samples in `assets/`.")
    else:
        try:
            photo_flow(decode(up), "upload")
        except ValueError as exc:
            st.error(str(exc))

with tab_cam:
    shot = st.camera_input("Point the camera at a Sudoku and take a picture")
    if shot is not None:
        photo_flow(decode(shot), "camera")
    st.caption("For live video with the solution overlaid in real time, run `python realtime.py` locally.")

with tab_type:
    st.markdown("Type the puzzle into the grid, or paste 81 characters (digits, with `0` or `.` for blanks).")
    text = st.text_input("Paste a puzzle", placeholder=EXAMPLE)
    start = [[0] * 9 for _ in range(9)]
    if text.strip():
        try:
            start = parse_grid(text)
        except SudokuError as exc:
            st.error(str(exc))
    df = grid_editor(start, key=f"typed_{hash(text)}")
    if st.button("Solve", type="primary"):
        try:
            show_solution(df_to_grid(df))
        except SudokuError as exc:
            st.error(str(exc))

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Find the board**: adaptive threshold, largest square contour, perspective warp.\n"
        "2. **Read digits**: grid lines removed with morphology, each cell's digit is centred "
        "and classified by a CNN (ONNX Runtime, all cells in one batch).\n"
        "3. **Solve**: bitmask constraint propagation with minimum-remaining-values backtracking.\n"
        "4. **Draw**: the answer is warped back onto your photo."
    )
    st.markdown("Accuracy and speed numbers: see `results/RESULTS.md`.")
