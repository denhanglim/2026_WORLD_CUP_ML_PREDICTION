"""Point-in-time match features. Single chronological pass => no future leakage."""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Deque, Dict
import numpy as np
import pandas as pd

# Model features only (NOT metadata like date/tournament).
FEATURE_COLUMNS = [
    "elo_diff", "home_elo_pre", "away_elo_pre",
    "home_form_pts", "away_form_pts",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "neutral",
]

def _points(team: str, home: str, away: str, result: int) -> float:
    if result == 1:
        return 1.0
    if (result == 0 and team == home) or (result == 2 and team == away):
        return 3.0
    return 0.0

def _avg(dq: Deque[float]) -> float:
    return float(np.mean(dq)) if len(dq) else 0.0

def build_features(df: pd.DataFrame, form_window: int = 5,
                   home_adv: float = 100.0) -> pd.DataFrame:
    """Feature frame aligned 1:1 with `df` (requires `home_elo_pre`/`away_elo_pre`).
    All rolling/lagged features use prior matches only. If `home_fifa_rank`/`away_fifa_rank`
    columns exist, adds `fifa_rank_diff`."""
    form: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    gf: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    ga: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    last_date: Dict[str, pd.Timestamp] = {}
    has_rank = "home_fifa_rank" in df.columns and "away_fifa_rank" in df.columns
    rows = []
    for r in df.itertuples(index=False):
        not_neutral = 0.0 if r.neutral else 1.0
        h_rest = (r.date - last_date[r.home_team]).days if r.home_team in last_date else np.nan
        a_rest = (r.date - last_date[r.away_team]).days if r.away_team in last_date else np.nan
        rec = {
            "elo_diff": (r.home_elo_pre + home_adv * not_neutral) - r.away_elo_pre,
            "home_elo_pre": r.home_elo_pre,
            "away_elo_pre": r.away_elo_pre,
            "home_form_pts": sum(form[r.home_team]),
            "away_form_pts": sum(form[r.away_team]),
            "home_rest_days": float(h_rest),
            "away_rest_days": float(a_rest),
            "rest_days_diff": float(h_rest - a_rest) if not (np.isnan(h_rest) or np.isnan(a_rest)) else np.nan,
            "home_gf_avg": _avg(gf[r.home_team]),
            "home_ga_avg": _avg(ga[r.home_team]),
            "away_gf_avg": _avg(gf[r.away_team]),
            "away_ga_avg": _avg(ga[r.away_team]),
            "neutral": bool(r.neutral),
        }
        if has_rank:
            hr, ar = getattr(r, "home_fifa_rank"), getattr(r, "away_fifa_rank")
            rec["fifa_rank_diff"] = (float(ar) - float(hr)) if pd.notna(hr) and pd.notna(ar) else np.nan
        rows.append(rec)
        # update AFTER recording (point-in-time)
        form[r.home_team].append(_points(r.home_team, r.home_team, r.away_team, r.result))
        form[r.away_team].append(_points(r.away_team, r.home_team, r.away_team, r.result))
        gf[r.home_team].append(float(r.home_score)); ga[r.home_team].append(float(r.away_score))
        gf[r.away_team].append(float(r.away_score)); ga[r.away_team].append(float(r.home_score))
        last_date[r.home_team] = r.date
        last_date[r.away_team] = r.date
    feats = pd.DataFrame(rows, index=df.index)
    feats["match_id"] = df["match_id"].values
    feats["result"] = df["result"].values
    feats["date"] = df["date"].values
    feats["tournament"] = df["tournament"].values
    return feats
