"""Minimal model protocol shared by every predictor (baselines and, later, M1/M2/meta)."""
from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd

class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Model": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # shape (n, 3)
