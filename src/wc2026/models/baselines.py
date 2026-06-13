"""Naive baselines the real models must beat."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CLASSES = [0, 1, 2]  # H, D, A

class UniformBaseline:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "UniformBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 3), 1.0 / 3.0)

class EloLogisticBaseline:
    """Multinomial logistic of ordinal result on elo_diff. A real, fitted Elo baseline."""
    def __init__(self, feature: str = "elo_diff"):
        self.feature = feature
        # NOTE: scikit-learn >= 1.7 removed the `multi_class` kwarg; multinomial
        # (softmax) is now the default and only behaviour for multiclass logistic
        # regression with the lbfgs solver, so this is equivalent to the plan's
        # LogisticRegression(multi_class="multinomial", max_iter=1000).
        self._clf = LogisticRegression(max_iter=1000)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EloLogisticBaseline":
        self._clf.fit(X[[self.feature]].to_numpy(), np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._clf.predict_proba(X[[self.feature]].to_numpy())
        # reindex columns to canonical [0,1,2] order regardless of classes seen in training
        out = np.zeros((len(X), 3))
        for col, cls in enumerate(self._clf.classes_):
            out[:, int(cls)] = raw[:, col]
        # renormalise in case a class was unseen in training
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums
