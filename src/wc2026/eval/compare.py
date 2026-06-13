"""Heterogeneous-model comparison: pass the full frame to each model; each reads what it needs."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from .scoring import rps, brier, log_loss
from .calibration import ece

def predict_frame(model, frame: pd.DataFrame) -> np.ndarray:
    """Call model.predict_proba on the full frame (model selects its own columns)."""
    return model.predict_proba(frame)

def score_predictions(P: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        "rps": float(np.mean(rps(P, y))),
        "brier": float(np.mean(brier(P, y))),
        "log_loss": float(np.mean(log_loss(P, y))),
        "ece_home": float(ece(P[:, 0], (np.asarray(y) == 0).astype(int))),
    }
