"""Walk-forward backtest harness. Trains only on the past; scores the future."""
from __future__ import annotations
from typing import Callable, List, Sequence
import numpy as np
import pandas as pd
from .scoring import rps, brier, log_loss
from .calibration import ece

def walk_forward(feats: pd.DataFrame, cutoffs: Sequence[pd.Timestamp],
                 model_factory: Callable[[], "object"],
                 features: List[str]) -> pd.DataFrame:
    """For each cutoff: train on date < cutoff, evaluate on [cutoff, next_cutoff).

    `feats` needs columns: `date`, `result`, and every name in `features`.
    `model_factory` returns a fresh unfitted model each call (fit/predict_proba).
    """
    cutoffs = list(cutoffs)
    rows = []
    for i, cut in enumerate(cutoffs):
        hi = cutoffs[i + 1] if i + 1 < len(cutoffs) else feats["date"].max() + pd.Timedelta(days=1)
        train = feats[feats["date"] < cut]
        evalw = feats[(feats["date"] >= cut) & (feats["date"] < hi)]
        if len(train) == 0 or len(evalw) == 0:
            continue
        model = model_factory()
        model.fit(train[features], train["result"].to_numpy())
        P = model.predict_proba(evalw[features])
        y = evalw["result"].to_numpy()
        rows.append({
            "fold": i,
            "cutoff": cut,
            "n_train": len(train),
            "n_eval": len(evalw),
            "rps": float(np.mean(rps(P, y))),
            "brier": float(np.mean(brier(P, y))),
            "log_loss": float(np.mean(log_loss(P, y))),
            "ece_home": float(ece(P[:, 0], (y == 0).astype(int))),
        })
    return pd.DataFrame(rows)
