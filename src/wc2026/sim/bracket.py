"""Derive the 2026 group structure from the real fixture list + encode the knockout tree."""
from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd

HOSTS = {"United States", "Canada", "Mexico"}

def wc2026_group_fixtures(wc26_matches: pd.DataFrame) -> List[dict]:
    """All 72 group-stage fixtures as dicts: home, away, home_score, away_score (NaN if unplayed),
    neutral. (The knockout matches are not in the dataset; only group fixtures exist.)"""
    fx = []
    for r in wc26_matches.itertuples(index=False):
        fx.append({
            "home": r.home_team, "away": r.away_team,
            "home_score": (None if pd.isna(r.home_score) else int(r.home_score)),
            "away_score": (None if pd.isna(r.away_score) else int(r.away_score)),
            "neutral": bool(r.neutral),
        })
    return fx

def derive_groups(fixtures: List[dict]) -> Dict[str, List[str]]:
    """Connected components of the group-stage graph => the 12 groups. Labelled A–L by the
    alphabetically-first team in each component."""
    adj: Dict[str, set] = {}
    for f in fixtures:
        adj.setdefault(f["home"], set()).add(f["away"])
        adj.setdefault(f["away"], set()).add(f["home"])
    seen, comps = set(), []
    for team in adj:
        if team in seen:
            continue
        stack, comp = [team], []
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t); comp.append(t)
            stack.extend(adj[t] - seen)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: c[0])           # deterministic order by first team
    return {chr(ord("A") + i): comp for i, comp in enumerate(comps)}

# Knockout structure. Slots: ("W", group) winner, ("R", group) runner-up, ("T", rank) third.
# 16 R32 matches. The 8 thirds fill the ("T", 0..7) slots in ranked order (approximation of
# FIFA's exact combination table — documented).
KNOCKOUT_R32_SLOTS: List[Tuple[tuple, tuple]] = [
    (("W", "A"), ("T", 0)), (("R", "C"), ("R", "D")),
    (("W", "B"), ("T", 1)), (("R", "A"), ("R", "B")),
    (("W", "C"), ("T", 2)), (("W", "E"), ("R", "F")),
    (("W", "D"), ("T", 3)), (("W", "G"), ("R", "H")),
    (("W", "F"), ("T", 4)), (("R", "E"), ("R", "G")),
    (("W", "H"), ("T", 5)), (("W", "I"), ("R", "J")),
    (("W", "J"), ("T", 6)), (("W", "K"), ("R", "L")),
    (("W", "L"), ("T", 7)), (("R", "I"), ("R", "K")),
]
