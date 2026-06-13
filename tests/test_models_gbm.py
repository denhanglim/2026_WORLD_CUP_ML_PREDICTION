import numpy as np
import pandas as pd
from wc2026.models.gbm import GBMModel
from wc2026.models.baselines import EloLogisticBaseline
from wc2026.eval.scoring import rps

def _nonlinear(n=2000, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    # outcome depends on INTERACTION a*b (linear model on a,b alone can't capture it)
    score = a * b
    p_home = 1 / (1 + np.exp(-2 * score))
    u = rng.uniform(0, 1, n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.18, 1, 2))
    X = pd.DataFrame({"a": a, "b": b, "elo_diff": a * 200})
    return X, y

def test_gbm_probs_valid():
    X, y = _nonlinear()
    P = GBMModel(features=["a", "b"]).fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= 0).all()

def test_gbm_handles_nan_features():
    X, y = _nonlinear()
    X.loc[X.index[:50], "a"] = np.nan       # inject missing values
    P = GBMModel(features=["a", "b"]).fit(X, y).predict_proba(X)
    assert np.isfinite(P).all()

def test_gbm_beats_linear_on_interaction_data():
    X, y = _nonlinear()
    split = 1500
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    gbm = GBMModel(features=["a", "b"]).fit(Xtr, ytr)
    lin = EloLogisticBaseline(feature="elo_diff").fit(Xtr, ytr)
    rps_gbm = np.mean(rps(gbm.predict_proba(Xte), yte))
    rps_lin = np.mean(rps(lin.predict_proba(Xte), yte))
    assert rps_gbm < rps_lin   # GBM captures the interaction the linear baseline cannot
