import math
import numpy as np
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine, mov_multiplier, K_BY_TOURNAMENT

FIX = "tests/fixtures/mini_results.csv"

def test_mov_multiplier():
    assert mov_multiplier(0) == 1.0
    assert mov_multiplier(1) == 1.0
    assert mov_multiplier(2) == 1.5
    assert mov_multiplier(3) == (11 + 3) / 8
    assert mov_multiplier(5) == (11 + 5) / 8

def test_single_match_update_is_zero_sum_and_correct():
    eng = EloEngine(base=1500.0, home_adv=100.0)
    # Alpha (home) beats Delta 3-0 in a Friendly (K=20), non-neutral
    pre_h, pre_a = eng.rate_match("Alpha", "Delta", 3, 0, "Friendly", neutral=False)
    assert pre_h == 1500.0 and pre_a == 1500.0
    # E_home = 1/(1+10^-(100/400)) ; G for gd=3 = 14/8 ; K=20, W=1
    E = 1.0 / (1.0 + 10 ** (-(100.0) / 400.0))
    G = (11 + 3) / 8
    delta = 20 * G * (1 - E)
    assert math.isclose(eng.rating("Alpha"), 1500.0 + delta, rel_tol=1e-9)
    assert math.isclose(eng.rating("Delta"), 1500.0 - delta, rel_tol=1e-9)

def test_neutral_removes_home_advantage():
    eng = EloEngine(base=1500.0, home_adv=100.0)
    pre_h, pre_a = eng.rate_match("Alpha", "Gamma", 1, 1, "Friendly", neutral=True)
    # draw between equals on neutral ground => no rating change
    assert math.isclose(eng.rating("Alpha"), 1500.0, rel_tol=1e-9)
    assert math.isclose(eng.rating("Gamma"), 1500.0, rel_tol=1e-9)

def test_prematch_ratings_column_is_point_in_time():
    df = load_results(FIX)
    eng = EloEngine()
    out = eng.run(df)
    # first Alpha match: pre-match elo must be the base (no prior games)
    first = out.iloc[0]
    assert first["home_elo_pre"] == 1500.0 and first["away_elo_pre"] == 1500.0
    # Alpha should end clearly strongest, Delta weakest
    finals = eng.ratings()
    assert finals["Alpha"] == max(finals.values())
    assert finals["Delta"] == min(finals.values())

def test_run_does_not_mutate_input():
    df = load_results(FIX)
    cols_before = list(df.columns)
    EloEngine().run(df)
    assert list(df.columns) == cols_before  # no new columns leaked onto caller's df
