import warnings
import numpy as np
import pandas as pd
import pytest
from wc2026.models.m1_poisson import M1PoissonModel

warnings.filterwarnings("ignore")

def _toy_matches(seed=0):
    """4 teams, clear strength order A>B>C>D, two yearly periods. Strong teams score more."""
    rng = np.random.default_rng(seed)
    strength = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0}
    rows = []
    teams = list(strength)
    for year in (2018, 2019):
        for _ in range(60):
            h, a = rng.choice(teams, 2, replace=False)
            lam_h = np.exp(0.2 + strength[h] - 0.5 * strength[a])
            lam_a = np.exp(0.2 + strength[a] - 0.5 * strength[h])
            rows.append({
                "date": pd.Timestamp(f"{year}-06-01"),
                "home_team": h, "away_team": a,
                "home_score": int(rng.poisson(lam_h)), "away_score": int(rng.poisson(lam_a)),
                "neutral": True,
            })
    return pd.DataFrame(rows)

def test_predict_proba_is_valid_distribution():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    pred = pd.DataFrame([{"home_team": "A", "away_team": "D", "neutral": True,
                          "date": pd.Timestamp("2019-06-01")}])
    P = m.predict_proba(pred)
    assert P.shape == (1, 3)
    assert np.isclose(P.sum(), 1.0, atol=1e-6)
    assert (P >= 0).all()

def test_stronger_team_favoured():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    pred = pd.DataFrame([
        {"home_team": "A", "away_team": "D", "neutral": True, "date": pd.Timestamp("2019-06-01")},
        {"home_team": "D", "away_team": "A", "neutral": True, "date": pd.Timestamp("2019-06-01")},
    ])
    P = m.predict_proba(pred)
    assert P[0, 0] > P[0, 2]    # A (home) beats D more likely than loses
    assert P[1, 2] > P[1, 0]    # D home vs A: away (A) win more likely

def test_unknown_team_gets_neutral_strength():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    pred = pd.DataFrame([{"home_team": "A", "away_team": "ZZZ_UNKNOWN", "neutral": True,
                          "date": pd.Timestamp("2019-06-01")}])
    P = m.predict_proba(pred)
    assert np.isclose(P.sum(), 1.0, atol=1e-6)   # does not crash; valid probs

def test_sample_strengths_shape():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    draws = m.sample_strengths(5)
    # dict with att/def arrays shaped (n_draws, n_teams, n_periods)
    assert draws["att"].shape[0] == 5
    assert draws["att"].shape[1] == m.n_teams

@pytest.mark.slow
def test_advi_smoke():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="advi", advi_iter=2000, draws=50).fit(df)
    pred = pd.DataFrame([{"home_team": "A", "away_team": "B", "neutral": True,
                          "date": pd.Timestamp("2019-06-01")}])
    P = m.predict_proba(pred)
    assert np.isclose(P.sum(), 1.0, atol=1e-6)
