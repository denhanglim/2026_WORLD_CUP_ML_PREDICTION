"""End-to-end Phase-1 chain: load -> Elo -> features -> walk-forward baselines."""
from __future__ import annotations
from typing import List
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.backtest import walk_forward
from wc2026.models.baselines import EloLogisticBaseline

def run_phase1(results_path: str, cutoffs: List[str]) -> pd.DataFrame:
    df = load_results(results_path)
    df = EloEngine().run(df)
    feats = build_features(df)
    feats["date"] = df["date"].values
    cuts = [pd.Timestamp(c) for c in cutoffs]
    return walk_forward(feats, cuts, lambda: EloLogisticBaseline(), features=["elo_diff"])

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/mini_results.csv"
    print(run_phase1(path, cutoffs=["2018-05-01"]).to_string(index=False))
