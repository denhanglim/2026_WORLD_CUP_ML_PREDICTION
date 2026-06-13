"""Self-computed football Elo. Chronological single pass; emits pre-match ratings.

K (match importance) weights are our own documented choice, not a third-party feed, so the
ratings are fully reproducible from the results table alone.
"""
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd

K_BY_TOURNAMENT: Dict[str, float] = {
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "African Cup of Nations": 50.0,
    "Friendly": 20.0,
}
K_DEFAULT = 30.0

def k_for(tournament: str) -> float:
    return K_BY_TOURNAMENT.get(tournament, K_DEFAULT)

def mov_multiplier(goal_diff: int) -> float:
    """Margin-of-victory multiplier (World Football Elo style)."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0

def _expected_home(r_home: float, r_away: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((r_home + home_adv) - r_away) / 400.0))

class EloEngine:
    def __init__(self, base: float = 1500.0, home_adv: float = 100.0):
        self.base = base
        self.home_adv = home_adv
        self._r: Dict[str, float] = {}

    def rating(self, team: str) -> float:
        return self._r.get(team, self.base)

    def ratings(self) -> Dict[str, float]:
        return dict(self._r)

    def rate_match(self, home: str, away: str, hs: int, as_: int,
                   tournament: str, neutral: bool) -> Tuple[float, float]:
        """Apply one match. Returns (home_elo_pre, away_elo_pre) — ratings BEFORE update."""
        r_h, r_a = self.rating(home), self.rating(away)
        adv = 0.0 if neutral else self.home_adv
        e_h = _expected_home(r_h, r_a, adv)
        if hs > as_:
            w_h = 1.0
        elif hs == as_:
            w_h = 0.5
        else:
            w_h = 0.0
        k = k_for(tournament)
        g = mov_multiplier(hs - as_)
        change = k * g * (w_h - e_h)
        self._r[home] = r_h + change
        self._r[away] = r_a - change
        return r_h, r_a

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a chronologically sorted match table; return a copy with pre-match
        rating columns `home_elo_pre`, `away_elo_pre`. Does not mutate the input."""
        home_pre, away_pre = [], []
        for row in df.itertuples(index=False):
            ph, pa = self.rate_match(row.home_team, row.away_team,
                                     int(row.home_score), int(row.away_score),
                                     row.tournament, bool(row.neutral))
            home_pre.append(ph)
            away_pre.append(pa)
        out = df.copy()
        out["home_elo_pre"] = home_pre
        out["away_elo_pre"] = away_pre
        return out
