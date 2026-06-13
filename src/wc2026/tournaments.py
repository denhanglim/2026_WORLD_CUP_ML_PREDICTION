"""World Cup edition dates (group-stage opener) used as walk-forward cutoffs."""
from __future__ import annotations
import pandas as pd

WORLD_CUP_OPENERS = {2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20"}

def wc_cutoffs() -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in WORLD_CUP_OPENERS.values()]

def is_world_cup(df: pd.DataFrame) -> pd.Series:
    return df["tournament"] == "FIFA World Cup"
