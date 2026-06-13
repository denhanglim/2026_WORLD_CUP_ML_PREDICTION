"""Load and tidy the international football results CSV (martj42 schema)."""
from __future__ import annotations
import pandas as pd

def _encode_result(home: int, away: int) -> int:
    if home > away:
        return 0   # HOME win
    if home == away:
        return 1   # DRAW
    return 2       # AWAY win

def load_results(path: str) -> pd.DataFrame:
    """Read the results CSV into a tidy, chronologically sorted match table.

    Adds: parsed `date`, boolean `neutral`, integer `result` (0=H,1=D,2=A),
    and a stable `match_id` assigned in chronological order.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    # martj42 stores neutral as the strings "True"/"False" or booleans
    df["neutral"] = df["neutral"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    ).fillna(False).astype(bool)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["result"] = [
        _encode_result(h, a) for h, a in zip(df["home_score"], df["away_score"])
    ]
    df["match_id"] = df.index.astype(int)
    return df
