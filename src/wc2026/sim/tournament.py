"""Monte Carlo orchestrator: posterior-propagated correlated tournament simulation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from .match import StrengthDraw
from .group_stage import simulate_groups
from .knockout import resolve_r32, simulate_knockout

@dataclass
class StrengthSamples:
    teams: List[str]
    att: np.ndarray          # (n_draws, n_teams)
    dfn: np.ndarray          # (n_draws, n_teams)
    home_adv: np.ndarray     # (n_draws,)
    intercept: np.ndarray    # (n_draws,)

    @property
    def n_draws(self) -> int:
        return self.att.shape[0]

    def draw(self, i: int) -> StrengthDraw:
        j = i % self.n_draws
        return StrengthDraw(teams=self.teams, att=self.att[j], dfn=self.dfn[j],
                            home_adv=float(self.home_adv[j % len(self.home_adv)]),
                            intercept=float(self.intercept[j % len(self.intercept)]))

# furthest-round label (from knockout.py) -> which round milestones it satisfies.
# Keys MUST match knockout.ROUNDS + {"champion"} exactly; a team's label is the round
# it last played in, so the milestones are every round it *advanced past* plus that one.
_PROGRESS_TO_FLAGS = {
    "r32":      set(),
    "r16":      {"r16"},
    "qf":       {"r16", "qf"},
    "sf":       {"r16", "qf", "sf"},
    "final":    {"r16", "qf", "sf", "final"},
    "champion": {"r16", "qf", "sf", "final", "win"},
}

def simulate_tournament(groups: Dict[str, List[str]], fixtures: List[dict],
                        samples: StrengthSamples, n_sims: int = 20000,
                        seed: int = 0) -> pd.DataFrame:
    teams = [t for ts in groups.values() for t in ts]
    n_third = 8
    counts = {t: {"win": 0, "final": 0, "sf": 0, "qf": 0, "r16": 0, "qualify": 0} for t in teams}
    rng = np.random.default_rng(seed)
    for s in range(n_sims):
        d = samples.draw(s)
        qual = simulate_groups(groups, fixtures, d, rng, n_third_qualify=n_third)
        qualifiers = set(qual.winners.values()) | set(qual.runners_up.values()) | set(qual.best_thirds)
        for t in qualifiers:
            counts[t]["qualify"] += 1
        pairs = resolve_r32(qual.winners, qual.runners_up, qual.best_thirds)
        champion, furthest = simulate_knockout(pairs, d, rng)
        for t, label in furthest.items():
            flags = _PROGRESS_TO_FLAGS[label]
            if "r16" in flags: counts[t]["r16"] += 1
            if "qf" in flags: counts[t]["qf"] += 1
            if "sf" in flags: counts[t]["sf"] += 1
            if "final" in flags: counts[t]["final"] += 1
            if "win" in flags: counts[t]["win"] += 1
    rows = []
    for t in teams:
        c = counts[t]
        rows.append({"team": t,
                     "p_win": c["win"] / n_sims, "p_final": c["final"] / n_sims,
                     "p_sf": c["sf"] / n_sims, "p_qf": c["qf"] / n_sims,
                     "p_r16": c["r16"] / n_sims, "p_qualify": c["qualify"] / n_sims})
    return pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)
