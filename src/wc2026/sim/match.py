"""Sample match outcomes from one M1 posterior strength draw."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class StrengthDraw:
    teams: List[str]
    att: np.ndarray          # (n_teams,) attack at the prediction period
    dfn: np.ndarray          # (n_teams,) defense
    home_adv: float
    intercept: float

    def __post_init__(self):
        self._idx: Dict[str, int] = {t: i for i, t in enumerate(self.teams)}

    def _a(self, team):
        i = self._idx.get(team); return 0.0 if i is None else float(self.att[i])
    def _d(self, team):
        i = self._idx.get(team); return 0.0 if i is None else float(self.dfn[i])

    def rates(self, home: str, away: str, host_home: bool, host_away: bool) -> Tuple[float, float]:
        adv_h = self.home_adv if host_home else 0.0
        adv_a = self.home_adv if host_away else 0.0
        lam_h = np.exp(self.intercept + adv_h + self._a(home) - self._d(away))
        lam_a = np.exp(self.intercept + adv_a + self._a(away) - self._d(home))
        return float(lam_h), float(lam_a)

def sample_score(draw: StrengthDraw, home: str, away: str, rng: np.random.Generator,
                 host_home: bool = False, host_away: bool = False) -> Tuple[int, int]:
    lam_h, lam_a = draw.rates(home, away, host_home, host_away)
    return int(rng.poisson(lam_h)), int(rng.poisson(lam_a))

def knockout_winner(draw: StrengthDraw, home: str, away: str, rng: np.random.Generator,
                    host_home: bool = False, host_away: bool = False) -> str:
    hg, ag = sample_score(draw, home, away, rng, host_home, host_away)
    if hg != ag:
        return home if hg > ag else away
    # extra time: 30 mins at ~1/3 of the 90-min rate
    lam_h, lam_a = draw.rates(home, away, host_home, host_away)
    eg_h, eg_a = int(rng.poisson(lam_h / 3.0)), int(rng.poisson(lam_a / 3.0))
    if eg_h != eg_a:
        return home if eg_h > eg_a else away
    # penalties: strength-weighted coin flip
    p_home = lam_h / (lam_h + lam_a) if (lam_h + lam_a) > 0 else 0.5
    return home if rng.random() < p_home else away
