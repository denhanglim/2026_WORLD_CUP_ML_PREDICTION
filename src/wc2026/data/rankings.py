"""Optional FIFA-rankings loader + point-in-time (backward) join onto matches."""
from __future__ import annotations
import pandas as pd

def load_rankings(path: str) -> pd.DataFrame:
    """Normalise a FIFA rankings CSV to columns: team, rank_date, rank."""
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "team": raw["country_full"],
        "rank_date": pd.to_datetime(raw["rank_date"]),
        "rank": raw["rank"].astype(int),
    })
    return out.sort_values("rank_date", kind="stable").reset_index(drop=True)

def _asof_rank(matches: pd.DataFrame, rankings: pd.DataFrame, team_col: str) -> pd.Series:
    left = matches[["date", team_col]].copy()
    left["_row"] = range(len(left))
    left = left.sort_values("date", kind="stable")
    merged = pd.merge_asof(
        left, rankings.rename(columns={"team": team_col, "rank_date": "date"}),
        on="date", by=team_col, direction="backward",
    )
    return merged.sort_values("_row")["rank"].reset_index(drop=True)

def attach_fifa_rank(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Add home_fifa_rank / away_fifa_rank using the latest ranking on/before each match date.
    Teams with no prior ranking get NaN."""
    out = matches.copy()
    out["home_fifa_rank"] = _asof_rank(matches, rankings, "home_team").values
    out["away_fifa_rank"] = _asof_rank(matches, rankings, "away_team").values
    return out
