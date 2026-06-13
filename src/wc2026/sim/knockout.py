"""Single-elimination knockout from R32 to the final.

Furthest-round labels are the *name of the round a team last played in*, drawn from
``ROUNDS`` (``r32``, ``r16``, ``qf``, ``sf``, ``final``) plus ``"champion"``:

- a team knocked out in a round carries that round's name as its furthest label
  (e.g. lost in the round of 16 -> ``"r16"``);
- the beaten finalist's furthest label is ``"final"`` (they played the final);
- the title-winner's label is ``"champion"``.

This vocabulary is shared with ``tournament.py``'s ``_PROGRESS_TO_FLAGS`` aggregator;
the two MUST agree on these exact strings.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from .match import StrengthDraw, knockout_winner
from .bracket import KNOCKOUT_R32_SLOTS, HOSTS

ROUNDS = ["r32", "r16", "qf", "sf", "final"]      # round a team last played in
# The round a winner advances to play next (None when winning the final -> champion).
_NEXT_ROUND = {"r32": "r16", "r16": "qf", "qf": "sf", "sf": "final", "final": None}

def resolve_r32(winners: Dict[str, str], runners_up: Dict[str, str],
                thirds: List[str]) -> List[Tuple[str, str]]:
    def slot(s):
        kind, key = s
        if kind == "W": return winners[key]
        if kind == "R": return runners_up[key]
        return thirds[key]          # ("T", rank)
    return [(slot(a), slot(b)) for a, b in KNOCKOUT_R32_SLOTS]

def simulate_knockout(r32_pairs: List[Tuple[str, str]], draw: StrengthDraw,
                      rng: np.random.Generator) -> Tuple[str, Dict[str, str]]:
    furthest: Dict[str, str] = {}
    teams = [t for p in r32_pairs for t in p]
    for t in teams:
        furthest[t] = "r32"          # everyone has at least played the round of 32
    pairs = list(r32_pairs)
    for rnd in ROUNDS:
        winners_round = []
        for h, a in pairs:
            w = knockout_winner(draw, h, a, rng,
                                host_home=(h in HOSTS), host_away=(a in HOSTS))
            loser = a if w == h else h
            # The loser's furthest round is the one they just lost in (already set to `rnd`).
            furthest[loser] = rnd
            nxt = _NEXT_ROUND[rnd]
            if nxt is None:
                furthest[w] = "champion"      # won the final
            else:
                furthest[w] = nxt             # advances to play the next round
            winners_round.append(w)
        if rnd == "final":
            return winners_round[0], furthest
        # pair up winners for the next round (adjacent pairing down the bracket tree)
        pairs = [(winners_round[i], winners_round[i + 1]) for i in range(0, len(winners_round), 2)]
    raise RuntimeError("unreachable")
