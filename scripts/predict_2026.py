"""Headline output: 2026 World Cup win probabilities via posterior-propagated Monte Carlo.

Real run (heavy, M1 via NUTS):
    python scripts/predict_2026.py data/raw/results.csv
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from wc2026.data.results import load_results
from wc2026.models.m1_poisson import M1PoissonModel
from wc2026.sim.bracket import wc2026_group_fixtures, derive_groups
from wc2026.sim.group_stage import simulate_groups  # noqa: F401 (kept for parity/debug)
from wc2026.sim.tournament import simulate_tournament, StrengthSamples

def _wc26_matches(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df[(raw_df["tournament"] == "FIFA World Cup") & (raw_df["date"] >= "2026-01-01")]

def _m1_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = df[["date", "home_team", "away_team", "home_score", "away_score", "neutral"]].copy()
    return f

def run_prediction(results_path: str, inference: str = "nuts", n_draws: int = 1000,
                   n_sims: int = 20000, window_years: int = 12, seed: int = 0) -> pd.DataFrame:
    df = load_results(results_path)                      # played matches only (NaN dropped)
    raw = pd.read_csv(results_path); raw["date"] = pd.to_datetime(raw["date"])
    wc26 = _wc26_matches(raw)
    fixtures = wc2026_group_fixtures(wc26)
    groups = derive_groups(fixtures)
    # attach the group label onto each fixture for the group simulator
    team_group = {t: g for g, ts in groups.items() for t in ts}
    for f in fixtures:
        f["group"] = team_group[f["home"]]

    m1 = M1PoissonModel(inference=inference, window_years=window_years,
                        draws=n_draws, random_seed=seed).fit(_m1_frame(df))
    s = m1.sample_strengths(n_draws)
    last = s["last_period"]
    att = s["att"][:, :, last]      # (n_draws, n_teams)
    dfn = s["def"][:, :, last]
    samples = StrengthSamples(teams=s["teams"], att=att, dfn=dfn,
                              home_adv=np.asarray(s["home_adv"]).ravel(),
                              intercept=np.asarray(s["intercept"]).ravel())
    return simulate_tournament(groups, fixtures, samples, n_sims=n_sims, seed=seed)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    table = run_prediction(path)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== 2026 World Cup — title & round probabilities (top 20) ===")
    print(table.head(20).to_string(index=False))
    print(f"\nfield: {len(table)} teams | sum p_win = {table['p_win'].sum():.4f}")
