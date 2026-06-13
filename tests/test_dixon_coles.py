import math
import numpy as np
from wc2026.models.dixon_coles import dc_tau

def test_tau_low_scores_exact():
    lh, la, rho = 1.3, 1.1, 0.05
    assert math.isclose(dc_tau(0, 0, lh, la, rho), 1 - lh * la * rho)
    assert math.isclose(dc_tau(0, 1, lh, la, rho), 1 + lh * rho)
    assert math.isclose(dc_tau(1, 0, lh, la, rho), 1 + la * rho)
    assert math.isclose(dc_tau(1, 1, lh, la, rho), 1 - rho)

def test_tau_high_scores_is_one():
    assert dc_tau(2, 0, 1.3, 1.1, 0.05) == 1.0
    assert dc_tau(3, 2, 1.3, 1.1, 0.05) == 1.0
    assert dc_tau(0, 2, 1.3, 1.1, 0.05) == 1.0

def test_tau_positive_for_small_rho():
    # for plausible rates and small |rho|, tau stays positive (valid likelihood)
    for hg, ag in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert dc_tau(hg, ag, 2.0, 2.0, 0.05) > 0
