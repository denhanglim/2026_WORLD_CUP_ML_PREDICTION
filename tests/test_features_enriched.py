import numpy as np
import pandas as pd
from wc2026.data.results import load_results
from wc2026.data.rankings import load_rankings, attach_fifa_rank
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features, FEATURE_COLUMNS

FIX = "tests/fixtures/mini_results.csv"

def test_new_feature_columns_present():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df)
    for col in ["home_rest_days", "away_rest_days", "rest_days_diff",
                "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]:
        assert col in feats.columns
        assert col in FEATURE_COLUMNS
    # metadata carried through
    assert "date" in feats.columns and "tournament" in feats.columns

def test_rest_days_first_match_is_nan_then_positive():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df)
    # Alpha's first match (2018-01-01) -> no prior match -> NaN rest
    first_alpha = feats.iloc[0]
    assert np.isnan(first_alpha["home_rest_days"])
    # Alpha vs Beta 2018-03-01: Alpha last played 2018-01-01 -> 59 days
    ab = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert ab["home_rest_days"] == 59.0

def test_rolling_goals_point_in_time():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df)
    # Alpha vs Beta 2018-03-01: Alpha has one prior match (3-0 vs Delta) -> gf_avg=3, ga_avg=0
    ab = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert ab["home_gf_avg"] == 3.0 and ab["home_ga_avg"] == 0.0

def test_fifa_rank_diff_when_available():
    df = load_results(FIX)
    df = attach_fifa_rank(df, load_rankings("tests/fixtures/mini_rankings.csv"))
    df = EloEngine().run(df)
    feats = build_features(df)
    assert "fifa_rank_diff" in feats.columns
    # Alpha(1) vs Beta(2) on 2018-03-01 -> diff = away_rank - home_rank = 2 - 1 = 1
    ab = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert ab["fifa_rank_diff"] == 1.0
