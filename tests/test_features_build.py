from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features, FEATURE_COLUMNS

FIX = "tests/fixtures/mini_results.csv"

def test_feature_columns_present():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df, form_window=5)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    assert len(feats) == len(df)

def test_elo_diff_includes_home_adv_when_not_neutral():
    df = EloEngine(home_adv=100.0).run(load_results(FIX))
    feats = build_features(df, form_window=5)
    row = feats.iloc[0]  # Alpha vs Delta, non-neutral, both 1500 pre
    # elo_diff = (home_elo_pre + home_adv*not_neutral) - away_elo_pre
    assert row["elo_diff"] == (1500.0 + 100.0) - 1500.0

def test_form_starts_at_zero_then_accumulates():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df, form_window=5)
    first = feats.iloc[0]
    assert first["home_form_pts"] == 0.0 and first["away_form_pts"] == 0.0
    # Alpha's 3rd match (Alpha vs Beta, 2018-03-01): Alpha already beat Delta (3 pts)
    alpha_beta = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert alpha_beta["home_form_pts"] == 3.0
