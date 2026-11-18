"""Digit recognition with the trained CNN.

Uses ONNX Runtime when the exported ``.onnx`` model is present (small,
fast to load, no TensorFlow needed) and falls back to Keras otherwise.
All digits of a board are classified in ONE batched call instead of 81
separate ``model.predict`` calls as in the original notebook.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_ONNX = MODELS_DIR / "digit_cnn.onnx"
DEFAULT_KERAS = MODELS_DIR / "digit_cnn.keras"


class DigitRecognizer:
    """Classifies 28x28 digit images (white digit on black, values 0-1) as 1-9."""

    def __init__(self, model_path: Optional[str | os.PathLike] = None):
        path = Path(model_path) if model_path else (DEFAULT_ONNX if DEFAULT_ONNX.exists() else DEFAULT_KERAS)
        if not path.exists():
            raise FileNotFoundError(f"Digit model not found at {path}. Run training/train_digits.py first.")
        self.path = path
        if path.suffix == ".onnx":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            self._sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            self._input = self._sess.get_inputs()[0].name
            self.backend = "onnxruntime"
        else:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import keras

            self._model = keras.models.load_model(path, compile=False)
            self.backend = "keras"

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[..., None]
        if len(batch) == 0:
            return np.zeros((0, 9), np.float32)
        if self.backend == "onnxruntime":
            return self._sess.run(None, {self._input: batch})[0]
        return np.asarray(self._model(batch, training=False))

    def predict(self, batch: np.ndarray):
        """Return (digits 1-9, confidence) arrays."""
        p = self.predict_proba(batch)
        return p.argmax(axis=1) + 1, p.max(axis=1)


@lru_cache(maxsize=4)
def get_recognizer(model_path: Optional[str] = None) -> DigitRecognizer:
    """Load the model once per process and reuse it."""
    return DigitRecognizer(model_path)
