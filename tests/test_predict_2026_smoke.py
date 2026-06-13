# tests/test_predict_2026_smoke.py
from scripts.predict_2026 import run_prediction

def test_run_prediction_smoke():
    table = run_prediction("data/raw/results.csv", inference="map",
                           n_draws=1, n_sims=50, window_years=8)
    assert len(table) == 48
    assert abs(table["p_win"].sum() - 1.0) < 1e-9
    # a recognised contender should appear in the top half
    top = set(table.head(24)["team"])
    assert any(t in top for t in ["Spain", "France", "Argentina", "Brazil", "England"])
