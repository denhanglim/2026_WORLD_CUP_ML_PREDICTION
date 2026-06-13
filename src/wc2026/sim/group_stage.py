"""Simulate group stage: standings with FIFA-style tiebreakers + qualifier selection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from .match import StrengthDraw, sample_score
from .bracket import HOSTS

@dataclass
class QualifierResult:
    winners: Dict[str, str]        # group -> team
    runners_up: Dict[str, str]     # group -> team
    best_thirds: List[str]         # the qualifying third-placed teams
    standings: Dict[str, List[str]]  # group -> ranked team list

def _rank_key(stats, rng):
    # higher points, GD, GF better; random jitter breaks remaining ties
    return (stats["pts"], stats["gd"], stats["gf"], rng.random())

def simulate_groups(groups: Dict[str, List[str]], fixtures: List[dict],
                    draw: StrengthDraw, rng: np.random.Generator,
                    n_third_qualify: int = 8) -> QualifierResult:
    stats = {t: {"pts": 0, "gd": 0, "gf": 0} for ts in groups.values() for t in ts}
    for f in fixtures:
        h, a = f["home"], f["away"]
        if f.get("home_score") is not None and f.get("away_score") is not None:
            hg, ag = f["home_score"], f["away_score"]
        else:
            hg, ag = sample_score(draw, h, a, rng,
                                  host_home=(h in HOSTS), host_away=(a in HOSTS))
        stats[h]["gf"] += hg; stats[a]["gf"] += ag
        stats[h]["gd"] += hg - ag; stats[a]["gd"] += ag - hg
        if hg > ag: stats[h]["pts"] += 3
        elif hg < ag: stats[a]["pts"] += 3
        else: stats[h]["pts"] += 1; stats[a]["pts"] += 1

    winners, runners, thirds_pool, standings = {}, {}, [], {}
    for g, ts in groups.items():
        ranked = sorted(ts, key=lambda t: _rank_key(stats[t], rng), reverse=True)
        standings[g] = ranked
        winners[g], runners[g] = ranked[0], ranked[1]
        thirds_pool.append((ranked[2], stats[ranked[2]]))
    thirds_ranked = sorted(thirds_pool, key=lambda x: _rank_key(x[1], rng), reverse=True)
    best_thirds = [t for t, _ in thirds_ranked[:n_third_qualify]]
    return QualifierResult(winners, runners, best_thirds, standings)
