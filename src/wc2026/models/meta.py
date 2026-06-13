"""Meta-learner: stacks base-model probability vectors via multinomial logistic regression."""
from __future__ import annotations
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression

class MetaStacker:
    def __init__(self, max_iter: int = 2000, C: float = 1.0):
        self.max_iter = max_iter
        self.C = C
        self._clf = None

    @staticmethod
    def _stack(base_probs_list: List[np.ndarray]) -> np.ndarray:
        return np.hstack([np.asarray(b, dtype=float) for b in base_probs_list])

    def fit(self, base_probs_list: List[np.ndarray], y) -> "MetaStacker":
        X = self._stack(base_probs_list)
        self._clf = LogisticRegression(max_iter=self.max_iter, C=self.C)
        self._clf.fit(X, np.asarray(y))
        return self

    def predict_proba(self, base_probs_list: List[np.ndarray]) -> np.ndarray:
        X = self._stack(base_probs_list)
        raw = self._clf.predict_proba(X)
        out = np.zeros((len(X), 3))
        for col, cls in enumerate(self._clf.classes_):
            out[:, int(cls)] = raw[:, col]
        s = out.sum(axis=1, keepdims=True); s[s == 0] = 1.0
        return out / s
