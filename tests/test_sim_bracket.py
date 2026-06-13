# tests/test_sim_bracket.py
import pandas as pd
from wc2026.data.results import load_results
from wc2026.sim.bracket import derive_groups, wc2026_group_fixtures, KNOCKOUT_R32_SLOTS

def _wc26():
    raw = pd.read_csv("data/raw/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    wc = raw[(raw.tournament == "FIFA World Cup") & (raw.date >= "2026-01-01")]
    return wc

def test_derive_12_groups_of_four():
    fx = wc2026_group_fixtures(_wc26())
    groups = derive_groups(fx)
    assert len(groups) == 12
    assert all(len(teams) == 4 for teams in groups.values())
    all_teams = [t for ts in groups.values() for t in ts]
    assert len(all_teams) == 48 and len(set(all_teams)) == 48

def test_group_labels_are_A_to_L():
    groups = derive_groups(wc2026_group_fixtures(_wc26()))
    assert sorted(groups.keys()) == [chr(c) for c in range(ord("A"), ord("L") + 1)]

def test_each_group_has_six_fixtures():
    fx = wc2026_group_fixtures(_wc26())
    groups = derive_groups(fx)
    # 6 unique pairings per group of 4
    from itertools import combinations
    for label, teams in groups.items():
        pairs = set(frozenset(p) for p in combinations(teams, 2))
        got = set(frozenset((f["home"], f["away"])) for f in fx
                  if f["home"] in teams and f["away"] in teams)
        assert got == pairs

def test_r32_slots_well_formed():
    # 16 R32 matches, 32 slots; uses 12 winners + 12 runners-up + 8 thirds = 32
    assert len(KNOCKOUT_R32_SLOTS) == 16
    flat = [s for pair in KNOCKOUT_R32_SLOTS for s in pair]
    assert len(flat) == 32
    kinds = [s[0] for s in flat]
    assert kinds.count("W") == 12 and kinds.count("R") == 12 and kinds.count("T") == 8
