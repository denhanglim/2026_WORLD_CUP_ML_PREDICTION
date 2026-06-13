"""Discretize match dates into ordered period buckets for dynamic team strengths."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

def assign_periods(dates, freq: str = "Y") -> Tuple[np.ndarray, int]:
    """Map each date to an integer period code (0-based, in calendar order).
    `freq`: pandas period alias — "Y" yearly, "Q" quarterly, "2Y" two-yearly, etc.
    Returns (codes aligned to input order, n_periods)."""
    d = pd.to_datetime(pd.Series(list(dates)).reset_index(drop=True))
    per = d.dt.to_period(freq)
    codes, uniques = pd.factorize(per, sort=True)   # sorted => chronological codes
    return np.asarray(codes), int(len(uniques))
