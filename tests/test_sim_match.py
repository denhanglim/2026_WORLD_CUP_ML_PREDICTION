# tests/test_sim_match.py
import numpy as np
from wc2026.sim.match import StrengthDraw, sample_score, knockout_winner

def _draw():
    teams = ["Strong", "Weak"]
    att = np.array([1.5, -1.0]); dfn = np.array([1.0, -0.8])
    return StrengthDraw(teams=teams, att=att, dfn=dfn, home_adv=0.25, intercept=0.1)

def test_rates_favour_strong_team():
    d = _draw()
    lam_s, lam_w = d.rates("Strong", "Weak", host_home=False, host_away=False)
    assert lam_s > lam_w

def test_sample_score_returns_nonneg_ints():
    d = _draw()
    rng = np.random.default_rng(0)
    hg, ag = sample_score(d, "Strong", "Weak", rng)
    assert hg >= 0 and ag >= 0 and isinstance(hg, (int, np.integer))

def test_strong_team_wins_most_knockouts():
    d = _draw()
    rng = np.random.default_rng(1)
    wins = sum(knockout_winner(d, "Strong", "Weak", rng) == "Strong" for _ in range(400))
    assert wins > 300       # strong should win ~>75%

def test_knockout_never_returns_draw():
    d = _draw()
    rng = np.random.default_rng(2)
    for _ in range(50):
        w = knockout_winner(d, "Strong", "Weak", rng)
        assert w in ("Strong", "Weak")

def test_unknown_team_uses_zero_strength():
    d = _draw()
    lam_a, lam_b = d.rates("Strong", "ZZZ", host_home=False, host_away=False)
    assert lam_a > 0 and lam_b > 0   # does not crash
