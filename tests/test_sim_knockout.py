# tests/test_sim_knockout.py
import numpy as np
from wc2026.sim.match import StrengthDraw
from wc2026.sim.knockout import resolve_r32, simulate_knockout, ROUNDS

def _draw(teams):
    # team i strength decreasing with index; team 0 strongest
    att = np.array([2.0 - 0.1 * i for i in range(len(teams))])
    return StrengthDraw(teams=teams, att=att, dfn=att.copy(), home_adv=0.0, intercept=0.2)

def _full_qual():
    groups = [chr(ord("A") + i) for i in range(12)]
    winners = {g: f"W{g}" for g in groups}
    runners = {g: f"R{g}" for g in groups}
    thirds = [f"T{i}" for i in range(8)]
    return winners, runners, thirds

def test_resolve_r32_gives_32_teams():
    winners, runners, thirds = _full_qual()
    pairs = resolve_r32(winners, runners, thirds)
    assert len(pairs) == 16
    flat = [t for p in pairs for t in p]
    assert len(flat) == 32 and len(set(flat)) == 32

def test_simulate_knockout_returns_champion_and_progress():
    winners, runners, thirds = _full_qual()
    pairs = resolve_r32(winners, runners, thirds)
    teams = [t for p in pairs for t in p]
    rng = np.random.default_rng(0)
    champion, furthest = simulate_knockout(pairs, _draw(teams), rng)
    assert champion in teams
    assert furthest[champion] == "champion"
    # every R32 team has a recorded furthest round
    assert all(t in furthest for t in teams)
    assert set(furthest.values()).issubset(set(ROUNDS) | {"champion"})
