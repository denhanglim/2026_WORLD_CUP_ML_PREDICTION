# tests/test_sim_group_stage.py
import numpy as np
from wc2026.sim.match import StrengthDraw
from wc2026.sim.group_stage import simulate_groups, QualifierResult

def _draw(teams, strengths):
    att = np.array([strengths[t] for t in teams]); dfn = att.copy()
    return StrengthDraw(teams=teams, att=att, dfn=dfn, home_adv=0.0, intercept=0.2)

def _two_groups():
    # 8 teams, 2 groups of 4, clear strength order within each
    groups = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    fixtures = []
    from itertools import combinations
    for g, ts in groups.items():
        for h, a in combinations(ts, 2):
            fixtures.append({"group": g, "home": h, "away": a,
                             "home_score": None, "away_score": None, "neutral": True})
    return groups, fixtures

def test_qualifiers_counts():
    groups, fx = _two_groups()
    teams = [t for ts in groups.values() for t in ts]
    strengths = {t: (1.5 if t.endswith("1") else 1.0 if t.endswith("2") else
                     0.0 if t.endswith("3") else -1.5) for t in teams}
    rng = np.random.default_rng(0)
    res = simulate_groups(groups, fx, _draw(teams, strengths), rng,
                          n_third_qualify=2)   # 2 groups -> take 2 best thirds for test
    assert isinstance(res, QualifierResult)
    assert len(res.winners) == 2 and len(res.runners_up) == 2
    assert len(res.best_thirds) == 2
    # strongest team in each group should usually top it
    assert res.winners["A"] in ("A1", "A2")

def test_played_results_are_respected():
    groups, fx = _two_groups()
    teams = [t for ts in groups.values() for t in ts]
    # force A4 to have beaten A1 5-0 (played)
    for f in fx:
        if f["home"] == "A1" and f["away"] == "A4":
            f["home_score"], f["away_score"] = 0, 5
    strengths = {t: 1.0 for t in teams}      # equal strengths
    rng = np.random.default_rng(3)
    res = simulate_groups(groups, fx, _draw(teams, strengths), rng, n_third_qualify=2)
    # A4 banked +5 GD from the fixed result; with equal strengths it should rank highly
    assert "A4" in (res.winners["A"], res.runners_up["A"]) or "A4" in res.best_thirds
