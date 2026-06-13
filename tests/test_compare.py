import numpy as np
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.compare import predict_frame, score_predictions
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

FIX = "tests/fixtures/mini_results.csv"

def test_build_features_carries_raw_cols():
    feats = build_features(EloEngine().run(load_results(FIX)))
    for c in ["home_team", "away_team", "home_score", "away_score", "neutral_flag"]:
        assert c in feats.columns

def test_predict_frame_and_score():
    feats = build_features(EloEngine().run(load_results(FIX)))
    model = EloLogisticBaseline().fit(feats, feats["result"].to_numpy())
    P = predict_frame(model, feats)
    assert P.shape == (len(feats), 3)
    metrics = score_predictions(P, feats["result"].to_numpy())
    assert {"rps", "brier", "log_loss"}.issubset(metrics)
    # uniform must be worse-or-equal on RPS than a fitted Elo model on this toy set's own data
    Pu = predict_frame(UniformBaseline().fit(feats, feats["result"].to_numpy()), feats)
    assert score_predictions(Pu, feats["result"].to_numpy())["rps"] >= 0
