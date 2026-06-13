import pandas as pd
from wc2026.data.results import load_results

FIX = "tests/fixtures/mini_results.csv"

def test_load_results_schema_and_types():
    df = load_results(FIX)
    # required tidy columns present
    for col in ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral", "match_id", "result"]:
        assert col in df.columns
    # date parsed, sorted ascending, stable match_id
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing
    assert df["match_id"].is_unique
    assert df["neutral"].dtype == bool

def test_result_encoding():
    df = load_results(FIX)
    row = df[(df.home_team == "Alpha") & (df.away_team == "Delta")].iloc[0]
    assert row["result"] == 0      # home win
    drawrow = df[(df.home_team == "Beta") & (df.away_team == "Gamma")].iloc[0]
    assert drawrow["result"] == 1  # draw
    awayrow = df[(df.home_team == "Beta") & (df.away_team == "Alpha")].iloc[0]
    assert awayrow["result"] == 2  # away win
