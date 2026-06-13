"""Real-data World Cup backtest for the naive baselines.
Usage: python scripts/baseline_report.py [data/raw/results.csv]"""
from __future__ import annotations
import sys
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.backtest import walk_forward
from wc2026.tournaments import wc_cutoffs, is_world_cup
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

BASELINE_FEATURES = ["elo_diff"]

def build(results_path: str) -> pd.DataFrame:
    df = EloEngine().run(load_results(results_path))
    return build_features(df)

def report(results_path: str) -> pd.DataFrame:
    feats = build(results_path)
    cuts = wc_cutoffs()
    out = []
    for name, factory in [("uniform", lambda: UniformBaseline()),
                          ("elo_logistic", lambda: EloLogisticBaseline())]:
        res = walk_forward(feats, cuts, factory, features=BASELINE_FEATURES,
                           eval_filter=is_world_cup)
        res.insert(0, "model", name)
        out.append(res)
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    tbl = report(path)
    print(tbl.to_string(index=False))
    print("\nMean RPS by model:")
    print(tbl.groupby("model")["rps"].mean().to_string())
