import numpy as np
import pandas as pd
from wc2026.eval.backtest import walk_forward
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

def _separable_feats(n=1200):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    elo = rng.normal(0, 250, n)
    p_home = 1 / (1 + np.exp(-elo / 100))
    u = rng.uniform(0, 1, n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.18, 1, 2))
    return pd.DataFrame({"date": dates, "elo_diff": elo, "result": y})

def test_walk_forward_runs_and_reports_metrics():
    feats = _separable_feats()
    cutoffs = [pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")]
    res = walk_forward(feats, cutoffs, lambda: EloLogisticBaseline(), features=["elo_diff"])
    assert {"fold", "n_eval", "rps", "brier", "log_loss", "ece_home"}.issubset(res.columns)
    assert (res["n_eval"] > 0).all()

def test_elo_baseline_beats_uniform_on_rps():
    feats = _separable_feats()
    cutoffs = [pd.Timestamp("2018-01-01")]
    elo_res = walk_forward(feats, cutoffs, lambda: EloLogisticBaseline(), features=["elo_diff"])
    uni_res = walk_forward(feats, cutoffs, lambda: UniformBaseline(), features=["elo_diff"])
    assert elo_res["rps"].mean() < uni_res["rps"].mean()
