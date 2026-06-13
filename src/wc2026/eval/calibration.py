"""Binary calibration on the home-win event."""
from __future__ import annotations
import numpy as np
import pandas as pd

def reliability_bins(p_home, y_home, n_bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(y_home, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; clip 1.0 into the last bin
    idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        conf = float(p[mask].mean()) if count else float("nan")
        acc = float(y[mask].mean()) if count else float("nan")
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1],
                     "count": count, "confidence": conf, "accuracy": acc})
    return pd.DataFrame(rows)

def ece(p_home, y_home, n_bins: int = 10) -> float:
    """Expected Calibration Error: sample-weighted mean |accuracy - confidence|."""
    bins = reliability_bins(p_home, y_home, n_bins)
    n = int(bins["count"].sum())
    if n == 0:
        return float("nan")
    nonempty = bins[bins["count"] > 0]
    gap = (nonempty["accuracy"] - nonempty["confidence"]).abs()
    return float((nonempty["count"] / n * gap).sum())
