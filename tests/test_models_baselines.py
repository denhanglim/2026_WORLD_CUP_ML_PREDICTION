import numpy as np
import pandas as pd
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

def _toy():
    # elo_diff strongly separates outcomes: big positive => home win, near 0 => draw-ish
    rng = np.random.default_rng(1)
    n = 600
    elo = rng.normal(0, 200, n)
    # latent: home win prob rises with elo_diff
    p_home = 1 / (1 + np.exp(-elo / 120))
    u = rng.uniform(0, 1, n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.2, 1, 2))
    X = pd.DataFrame({"elo_diff": elo})
    return X, y

def test_uniform_outputs_thirds():
    X, y = _toy()
    P = UniformBaseline().fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.allclose(P, 1/3)

def test_elo_logistic_probabilities_valid():
    X, y = _toy()
    P = EloLogisticBaseline().fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= 0).all()

def test_elo_logistic_favours_higher_elo():
    X, y = _toy()
    model = EloLogisticBaseline().fit(X, y)
    big_home = model.predict_proba(pd.DataFrame({"elo_diff": [400.0]}))[0]
    big_away = model.predict_proba(pd.DataFrame({"elo_diff": [-400.0]}))[0]
    assert big_home[0] > big_home[2]   # home favoured when elo_diff large +
    assert big_away[2] > big_away[0]   # away favoured when elo_diff large -
