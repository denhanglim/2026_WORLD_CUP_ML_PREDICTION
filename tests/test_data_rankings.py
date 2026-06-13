import pandas as pd
from wc2026.data.results import load_results
from wc2026.data.rankings import load_rankings, attach_fifa_rank

def test_load_rankings_normalises():
    r = load_rankings("tests/fixtures/mini_rankings.csv")
    assert {"team", "rank_date", "rank"}.issubset(r.columns)
    assert pd.api.types.is_datetime64_any_dtype(r["rank_date"])

def test_attach_is_point_in_time_backward():
    df = load_results("tests/fixtures/mini_results.csv")
    r = load_rankings("tests/fixtures/mini_rankings.csv")
    out = attach_fifa_rank(df, r)
    assert "home_fifa_rank" in out.columns and "away_fifa_rank" in out.columns
    # Alpha vs Beta on 2018-03-01 -> must use the 2018-01-15 ranking (Alpha=1, Beta=2),
    # NOT the future 2018-05-15 one.
    row = out[(out.home_team == "Alpha") & (out.away_team == "Beta")].iloc[0]
    assert row["home_fifa_rank"] == 1 and row["away_fifa_rank"] == 2
    # A match after 2018-05-15 uses the newer ranking: Gamma vs Beta 2018-08-01 -> Beta=3
    row2 = out[(out.home_team == "Gamma") & (out.away_team == "Beta")].iloc[0]
    assert row2["away_fifa_rank"] == 3

def test_attach_missing_team_yields_nan():
    df = load_results("tests/fixtures/mini_results.csv")
    r = load_rankings("tests/fixtures/mini_rankings.csv")
    r = r[r.team != "Delta"]  # drop Delta's rankings
    out = attach_fifa_rank(df, r)
    assert out[(out.home_team == "Alpha") & (out.away_team == "Delta")]["away_fifa_rank"].isna().all()
