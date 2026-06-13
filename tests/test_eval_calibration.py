import numpy as np
from wc2026.eval.calibration import reliability_bins, ece

def test_perfectly_calibrated_has_near_zero_ece():
    rng = np.random.default_rng(0)
    n = 20000
    p_home = rng.uniform(0, 1, n)
    # generate outcomes with TRUE prob = p_home  => perfectly calibrated
    y_home = (rng.uniform(0, 1, n) < p_home).astype(int)
    assert ece(p_home, y_home, n_bins=10) < 0.02

def test_miscalibrated_has_high_ece():
    n = 10000
    p_home = np.full(n, 0.9)   # always claims 90%
    y_home = np.zeros(n, int)  # but home never wins
    assert ece(p_home, y_home, n_bins=10) > 0.8

def test_reliability_bins_shape_and_content():
    p = np.array([0.05, 0.15, 0.95])
    y = np.array([0, 0, 1])
    bins = reliability_bins(p, y, n_bins=10)
    assert set(["bin_lo", "bin_hi", "count", "confidence", "accuracy"]).issubset(bins.columns)
    assert bins["count"].sum() == 3
