import math
import numpy as np
from wc2026.eval.scoring import rps, brier, log_loss, one_hot

def test_one_hot():
    assert one_hot(0).tolist() == [1.0, 0.0, 0.0]
    assert one_hot(2).tolist() == [0.0, 0.0, 1.0]

def test_rps_perfect_is_zero():
    p = np.array([1.0, 0.0, 0.0])
    assert rps(p, 0) == 0.0

def test_rps_uniform_home_outcome():
    # uniform forecast, home win: RPS = 5/18 ≈ 0.27778
    p = np.array([1/3, 1/3, 1/3])
    assert math.isclose(rps(p, 0), 5/18, rel_tol=1e-9)

def test_brier_uniform_home_outcome():
    p = np.array([1/3, 1/3, 1/3])
    # (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 6/9
    assert math.isclose(brier(p, 0), 6/9, rel_tol=1e-9)

def test_log_loss_uniform():
    p = np.array([1/3, 1/3, 1/3])
    assert math.isclose(log_loss(p, 0), -math.log(1/3), rel_tol=1e-9)

def test_log_loss_clips_zero():
    p = np.array([0.0, 0.5, 0.5])
    # must not be inf — clipped
    assert log_loss(p, 0) < 50.0

def test_batch_mean_helpers():
    P = np.array([[1/3, 1/3, 1/3], [1.0, 0.0, 0.0]])
    y = np.array([0, 0])
    assert np.allclose(rps(P, y), [5/18, 0.0])
    assert np.allclose(brier(P, y), [6/9, 0.0])
