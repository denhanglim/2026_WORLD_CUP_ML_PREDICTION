"""Point-in-time match features. Single chronological pass => no future leakage."""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Deque, Dict
import pandas as pd

FEATURE_COLUMNS = ["elo_diff", "home_elo_pre", "away_elo_pre",
                   "home_form_pts", "away_form_pts", "neutral"]

def _points(team: str, home: str, away: str, result: int) -> float:
    # result: 0=home win, 1=draw, 2=away win
    if result == 1:
        return 1.0
    if (result == 0 and team == home) or (result == 2 and team == away):
        return 3.0
    return 0.0

def build_features(df: pd.DataFrame, form_window: int = 5,
                   home_adv: float = 100.0) -> pd.DataFrame:
    """Return a feature frame aligned 1:1 with `df` rows (requires `home_elo_pre`,
    `away_elo_pre` columns from EloEngine.run). Form is computed from prior matches only."""
    recent: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    rows = []
    for r in df.itertuples(index=False):
        home_form = sum(recent[r.home_team])
        away_form = sum(recent[r.away_team])
        not_neutral = 0.0 if r.neutral else 1.0
        rows.append({
            "elo_diff": (r.home_elo_pre + home_adv * not_neutral) - r.away_elo_pre,
            "home_elo_pre": r.home_elo_pre,
            "away_elo_pre": r.away_elo_pre,
            "home_form_pts": home_form,
            "away_form_pts": away_form,
            "neutral": bool(r.neutral),
        })
        # update AFTER recording (point-in-time)
        recent[r.home_team].append(_points(r.home_team, r.home_team, r.away_team, r.result))
        recent[r.away_team].append(_points(r.away_team, r.home_team, r.away_team, r.result))
    feats = pd.DataFrame(rows, index=df.index)
    feats["match_id"] = df["match_id"].values
    feats["result"] = df["result"].values
    return feats
