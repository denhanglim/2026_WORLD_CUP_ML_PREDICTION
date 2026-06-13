"""Proper scoring rules for ordinal 3-class match forecasts [p_home, p_draw, p_away]."""
from __future__ import annotations
import numpy as np

N_CLASSES = 3

def one_hot(idx: int) -> np.ndarray:
    v = np.zeros(N_CLASSES)
    v[int(idx)] = 1.0
    return v

def _as_batch(probs, y):
    P = np.asarray(probs, dtype=float)
    single = P.ndim == 1
    if single:
        P = P[None, :]
        y = np.array([y])
    else:
        y = np.asarray(y)
    return P, y, single

def rps(probs, y):
    """Ranked Probability Score (ordinal). Lower is better. Scalar or per-sample array."""
    P, yy, single = _as_batch(probs, y)
    O = np.zeros_like(P)
    O[np.arange(len(yy)), yy] = 1.0
    cum_p = np.cumsum(P, axis=1)
    cum_o = np.cumsum(O, axis=1)
    # sum over the first (r-1) thresholds, normalised by (r-1)
    val = np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1) / (N_CLASSES - 1)
    return val[0] if single else val

def brier(probs, y):
    """Multiclass Brier score. Lower is better."""
    P, yy, single = _as_batch(probs, y)
    O = np.zeros_like(P)
    O[np.arange(len(yy)), yy] = 1.0
    val = np.sum((P - O) ** 2, axis=1)
    return val[0] if single else val

def log_loss(probs, y, eps: float = 1e-15):
    """Negative log-likelihood of the realised class. Lower is better."""
    P, yy, single = _as_batch(probs, y)
    p_true = np.clip(P[np.arange(len(yy)), yy], eps, 1.0)
    val = -np.log(p_true)
    return val[0] if single else val
