import numpy as np
from wc2026.models.odds import implied_probs, OddsModel

def test_implied_probs_fair_odds():
    # fair decimal odds [2.0, 4.0, 4.0] -> [0.5, 0.25, 0.25] (no overround)
    p = implied_probs([2.0, 4.0, 4.0])
    assert np.allclose(p, [0.5, 0.25, 0.25])

def test_implied_probs_removes_overround():
    p = implied_probs([1.9, 3.5, 4.0])   # raw inverses sum > 1 (bookmaker margin)
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()

def test_odds_model_predicts_from_columns():
    import pandas as pd
    df = pd.DataFrame({"odds_home": [2.0, 4.0], "odds_draw": [4.0, 4.0], "odds_away": [4.0, 1.5]})
    P = OddsModel().fit(df, None).predict_proba(df)
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0, 0] > P[0, 2]    # row 0: home favoured
    assert P[1, 2] > P[1, 0]    # row 1: away favoured
