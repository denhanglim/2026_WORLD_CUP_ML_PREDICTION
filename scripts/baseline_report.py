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

GBM_FEATURES = ["elo_diff", "home_form_pts", "away_form_pts", "rest_days_diff",
                "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]

def report_models(results_path: str, cutoffs_override=None) -> pd.DataFrame:
    from wc2026.models.gbm import GBMModel
    feats = build(results_path)
    cuts = [pd.Timestamp(c) for c in cutoffs_override] if cutoffs_override else wc_cutoffs()
    # On the tiny fixture there may be no WC matches in-window; fall back to no filter there.
    use_filter = is_world_cup if cutoffs_override is None else None
    specs = [
        ("uniform", lambda: UniformBaseline(), BASELINE_FEATURES),
        ("elo_logistic", lambda: EloLogisticBaseline(), BASELINE_FEATURES),
        ("gbm", lambda: GBMModel(features=GBM_FEATURES), GBM_FEATURES),
    ]
    out = []
    for name, factory, feature_list in specs:
        res = walk_forward(feats, cuts, factory, features=feature_list, eval_filter=use_filter)
        if len(res):
            res.insert(0, "model", name)
            out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["model"])

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    tbl = report_models(path)
    print(tbl.to_string(index=False))
    print("\nMean RPS by model:")
    print(tbl.groupby("model")["rps"].mean().to_string())
