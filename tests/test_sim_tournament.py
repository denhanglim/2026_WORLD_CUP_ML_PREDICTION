# tests/test_sim_tournament.py
import numpy as np
from wc2026.sim.tournament import simulate_tournament, StrengthSamples

def _toy_world():
    # 12 groups x 4 teams = 48; team "Tg<k>" with k=0..3, k=0 strongest in each group
    groups = {}
    teams = []
    for gi in range(12):
        g = chr(ord("A") + gi)
        members = [f"{g}{k}" for k in range(4)]
        groups[g] = members; teams += members
    from itertools import combinations
    fixtures = []
    for g, ts in groups.items():
        for h, a in combinations(ts, 2):
            fixtures.append({"group": g, "home": h, "away": a,
                             "home_score": None, "away_score": None, "neutral": True})
    # one strength draw: index-0 team in each group is strong
    strength = {t: (1.6 if t.endswith("0") else 0.8 if t.endswith("1")
                    else 0.0 if t.endswith("2") else -1.2) for t in teams}
    att = np.array([[strength[t] for t in teams]])        # (1 draw, n_teams)
    samples = StrengthSamples(teams=teams, att=att, dfn=att.copy(),
                              home_adv=np.array([0.0]), intercept=np.array([0.2]))
    return groups, fixtures, samples

def test_probabilities_sum_and_rank():
    groups, fixtures, samples = _toy_world()
    table = simulate_tournament(groups, fixtures, samples, n_sims=400, seed=0)
    assert {"team", "p_win", "p_final", "p_sf", "p_qf", "p_r16", "p_qualify"}.issubset(table.columns)
    assert len(table) == 48
    assert np.isclose(table["p_win"].sum(), 1.0, atol=1e-9)   # exactly one champion per sim
    assert (table["p_win"] >= 0).all()
    # monotonicity: reaching an earlier round is at least as likely as winning
    assert (table["p_qualify"] >= table["p_r16"] - 1e-9).all()
    assert (table["p_r16"] >= table["p_win"] - 1e-9).all()
    # the strong teams (suffix 0) should dominate the top of the win table
    top12 = set(table.sort_values("p_win", ascending=False).head(12)["team"])
    assert sum(t.endswith("0") for t in top12) >= 8

def test_more_sims_is_deterministic_given_seed():
    groups, fixtures, samples = _toy_world()
    t1 = simulate_tournament(groups, fixtures, samples, n_sims=200, seed=42)
    t2 = simulate_tournament(groups, fixtures, samples, n_sims=200, seed=42)
    assert t1.equals(t2)
