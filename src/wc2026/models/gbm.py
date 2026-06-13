"""M2 — gradient-boosting match model (sklearn HistGradientBoosting; NaN-native, no libomp)."""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

def _enough_for_cv(y: np.ndarray, cv: int = 3) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return len(counts) >= 2 and counts.min() >= cv

class GBMModel:
    def __init__(self, features: List[str], calibrate: bool = True,
                 random_state: int = 0, cv: int = 3):
        self.features = features
        self.calibrate = calibrate
        self.random_state = random_state
        self.cv = cv
        self._clf = None

    def fit(self, X: pd.DataFrame, y) -> "GBMModel":
        y = np.asarray(y)
        base = HistGradientBoostingClassifier(random_state=self.random_state)
        if self.calibrate and _enough_for_cv(y, self.cv):
            self._clf = CalibratedClassifierCV(base, method="isotonic", cv=self.cv)
        else:
            self._clf = base
        self._clf.fit(X[self.features].to_numpy(dtype=float), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._clf.predict_proba(X[self.features].to_numpy(dtype=float))
        out = np.zeros((len(X), 3))
        for col, cls in enumerate(self._clf.classes_):
            out[:, int(cls)] = raw[:, col]
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s
