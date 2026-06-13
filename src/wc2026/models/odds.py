"""M3 — market odds → de-vigged implied probabilities (proportional normalisation)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def implied_probs(decimal_odds) -> np.ndarray:
    """[home, draw, away] decimal odds -> de-vigged probabilities summing to 1."""
    raw = 1.0 / np.asarray(decimal_odds, dtype=float)
    return raw / raw.sum()

class OddsModel:
    """Reads odds_home/odds_draw/odds_away columns. fit() is a no-op (market is the model)."""
    def fit(self, X: pd.DataFrame, y=None) -> "OddsModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = X[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
        raw = 1.0 / cols
        return raw / raw.sum(axis=1, keepdims=True)
