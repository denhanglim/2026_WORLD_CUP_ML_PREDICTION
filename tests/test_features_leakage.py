import numpy as np
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features, FEATURE_COLUMNS

FIX = "tests/fixtures/mini_results.csv"

def test_features_are_point_in_time():
    df_full = load_results(FIX)
    k = 4
    df_trunc = df_full.iloc[:k].reset_index(drop=True)
    full = build_features(EloEngine().run(df_full)).iloc[:k].reset_index(drop=True)
    trunc = build_features(EloEngine().run(df_trunc)).reset_index(drop=True)
    cols = [c for c in FEATURE_COLUMNS if c != "neutral"]
    # NaN-aware comparison (rest_days is NaN for first appearances)
    pd.testing.assert_frame_equal(full[cols], trunc[cols], check_dtype=False)
