"""Phase 2c+2d real-data backtest: M1 (Bayesian) + M2 (GBM) + Elo baseline + meta-learner.

Heavy: M1 fits via ADVI per fold. Run manually:
    python scripts/full_backtest.py data/raw/results.csv
The smoke entrypoint run_fold_smoke() uses the tiny fixture + M1 MAP for a fast CI check.
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.compare import predict_frame, score_predictions
from wc2026.tournaments import wc_cutoffs, is_world_cup
from wc2026.models.baselines import EloLogisticBaseline
from wc2026.models.gbm import GBMModel
from wc2026.models.m1_poisson import M1PoissonModel

GBM_FEATURES = ["elo_diff", "home_form_pts", "away_form_pts", "rest_days_diff",
                "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]

def _m1_frame(feats: pd.DataFrame) -> pd.DataFrame:
    """M1 needs a 'neutral' bool column; build_features carries it as 'neutral_flag'."""
    f = feats.copy()
    f["neutral"] = f["neutral_flag"].astype(bool)
    return f

def _fit_predict_bases(train, evalw, m1_inference="advi", m1_kwargs=None):
    m1_kwargs = m1_kwargs or {}
    elo = EloLogisticBaseline().fit(train, train["result"].to_numpy())
    gbm = GBMModel(features=GBM_FEATURES).fit(train, train["result"].to_numpy())
    m1 = M1PoissonModel(inference=m1_inference, **m1_kwargs).fit(_m1_frame(train))
    P = {
        "elo_logistic": predict_frame(elo, evalw),
        "gbm": predict_frame(gbm, evalw),
        "m1_poisson": m1.predict_proba(_m1_frame(evalw)),
    }
    return P, (elo, gbm, m1)

def _fold(feats, cut, hi, m1_inference="advi", m1_kwargs=None):
    train = feats[feats["date"] < cut]
    evalw = feats[(feats["date"] >= cut) & (feats["date"] < hi)]
    evalw = evalw[is_world_cup(evalw)]
    if len(train) == 0 or len(evalw) == 0:
        return None
    # inner split of train for meta out-of-fold predictions
    inner_cut = train["date"].quantile(0.8)
    base_tr = train[train["date"] < inner_cut]
    meta_tr = train[train["date"] >= inner_cut]
    rows = []
    if len(base_tr) and len(meta_tr):
        Pin, _ = _fit_predict_bases(base_tr, meta_tr, m1_inference, m1_kwargs)
        meta = __import__("wc2026.models.meta", fromlist=["MetaStacker"]).MetaStacker()
        order = ["elo_logistic", "gbm", "m1_poisson"]
        meta.fit([Pin[k] for k in order], meta_tr["result"].to_numpy())
    else:
        meta, order = None, ["elo_logistic", "gbm", "m1_poisson"]
    Pev, _ = _fit_predict_bases(train, evalw, m1_inference, m1_kwargs)
    y = evalw["result"].to_numpy()
    for name in order:
        rows.append({"model": name, "cutoff": cut, "n_eval": len(evalw), **score_predictions(Pev[name], y)})
    if meta is not None:
        Pmeta = meta.predict_proba([Pev[k] for k in order])
        rows.append({"model": "meta", "cutoff": cut, "n_eval": len(evalw), **score_predictions(Pmeta, y)})
    return pd.DataFrame(rows)

def run_full(results_path: str, m1_window_years: int = 12) -> pd.DataFrame:
    feats = build_features(EloEngine().run(load_results(results_path)))
    cuts = wc_cutoffs()
    out = []
    for i, cut in enumerate(cuts):
        hi = cuts[i + 1] if i + 1 < len(cuts) else feats["date"].max() + pd.Timedelta(days=1)
        res = _fold(feats, cut, hi, m1_inference="advi",
                    m1_kwargs={"window_years": m1_window_years, "draws": 200})
        if res is not None:
            out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["model"])

def run_fold_smoke(results_path: str) -> pd.DataFrame:
    """Fast CI path: tiny fixture, M1 MAP, single synthetic fold (no WC filter)."""
    feats = build_features(EloEngine().run(load_results(results_path)))
    cut = feats["date"].quantile(0.5)
    hi = feats["date"].max() + pd.Timedelta(days=1)
    train = feats[feats["date"] < cut]
    evalw = feats[(feats["date"] >= cut) & (feats["date"] < hi)]
    P, _ = _fit_predict_bases(train, evalw, m1_inference="map")
    meta = __import__("wc2026.models.meta", fromlist=["MetaStacker"]).MetaStacker()
    order = ["elo_logistic", "gbm", "m1_poisson"]
    meta.fit([P[k] for k in order], evalw["result"].to_numpy())  # smoke only (in-sample)
    y = evalw["result"].to_numpy()
    rows = [{"model": k, "rps": score_predictions(P[k], y)["rps"]} for k in order]
    Pmeta = meta.predict_proba([P[k] for k in order])
    rows.append({"model": "meta", "rps": score_predictions(Pmeta, y)["rps"]})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    tbl = run_full(path)
    print(tbl.to_string(index=False))
    print("\nMean RPS by model:")
    print(tbl.groupby("model")["rps"].mean().sort_values().to_string())
