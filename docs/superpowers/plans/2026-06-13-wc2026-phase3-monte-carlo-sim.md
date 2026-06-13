# WC-2026 Predictor — Phase 3 (Monte Carlo Tournament Simulator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulate the real 48-team 2026 World Cup tens of thousands of times — propagating M1's posterior team-strength uncertainty as *correlated* draws — to produce the headline output: each team's probability of winning the title (and reaching each round).

**Architecture:** Builds on Phases 1–2c/2d. New pieces in `src/wc2026/sim/`: `bracket.py` (derive the 12 groups from the real fixture graph + encode the knockout structure), `match.py` (sample a scoreline / knockout winner from M1 strengths), `group_stage.py` (round-robin standings with FIFA tiebreakers + qualifier selection incl. 8 best third-placed), `knockout.py` (single-elim progression with extra time + penalties), and `tournament.py` (the Monte Carlo orchestrator that draws an M1 posterior sample per simulation and runs the whole bracket under it). Plus `scripts/predict_2026.py` — the heavy real run (fit M1 via NUTS on all data to June 2026, simulate, print the win-probability table).

**Why M1 is the in-sim engine (not the meta-learner):** M1 predicts any hypothetical pairing directly from latent team strengths and — critically — exposes a *posterior sampler*. Drawing one strength sample per simulation makes a team's quality consistent across all its matches in that sim (correlated outcomes), which is the methodologically correct way to capture tail behaviour. The meta-learner needs match-context features (form, rest) that are undefined for hypothetical future knockout pairings, so it stays the per-match accuracy benchmark from Phase 2, not the simulator. (Phase 2 showed M1 alone trails Elo on point RPS but provides the posterior + calibration the sim needs.)

**Tech Stack:** Existing — numpy, pandas, pymc/arviz (M1), pytest. The heavy run uses M1 NUTS.

---

## Conventions & honest approximations (stated up front)

- Outcome convention unchanged. Goals sampled as independent Poisson from M1 rate draws
  (the Dixon–Coles correction is a small low-score adjustment to the *likelihood*; we do not
  apply it when sampling scorelines — standard practice, noted).
- **Group tiebreakers:** points → goal difference → goals for → random. (FIFA's full chain adds
  head-to-head then fair-play then lots; head-to-head/fair-play are approximated by the random
  tiebreak — documented; second-order effect on title odds.)
- **R32 third-placed slotting:** the 8 best third-placed teams fill 8 designated R32 slots by a
  fixed documented rule, not FIFA's exact combination table. Affects specific early matchups,
  not materially the champion distribution for contenders — documented.
- **Host advantage:** USA, Canada, Mexico receive M1's `home_adv` in their own matches; all other
  matches are neutral. Documented.
- **Correlated propagation:** each simulation draws ONE M1 posterior sample of all team strengths,
  then plays the entire tournament under it (plus fresh per-match Poisson randomness). Posterior
  draws are cycled across the N simulations.

**Branch:** create `feature/phase3-monte-carlo` off `feature/phase2cd-bayes-meta`.

---

### Task 1: Derive the 2026 bracket (groups from fixtures + knockout structure)

**Files:**
- Create: `src/wc2026/sim/__init__.py`, `src/wc2026/sim/bracket.py`
- Test: `tests/test_sim_bracket.py`

**Design:** the 72 group fixtures form 12 disjoint complete graphs (K4). Connected components of
the "plays-in-group-stage" graph recover the 12 groups. Group labels are assigned A–L by sorted
order of each group's alphabetically-first team (deterministic). The knockout structure is a fixed
binary tree over 32 slots; R32 slots reference group positions (`W`=winner, `R`=runner-up,
`T`=third) — the 8 thirds fill the 8 `T` slots in ranked order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_bracket.py
import pandas as pd
from wc2026.data.results import load_results
from wc2026.sim.bracket import derive_groups, wc2026_group_fixtures, KNOCKOUT_R32_SLOTS

def _wc26():
    raw = pd.read_csv("data/raw/results.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    wc = raw[(raw.tournament == "FIFA World Cup") & (raw.date >= "2026-01-01")]
    return wc

def test_derive_12_groups_of_four():
    fx = wc2026_group_fixtures(_wc26())
    groups = derive_groups(fx)
    assert len(groups) == 12
    assert all(len(teams) == 4 for teams in groups.values())
    all_teams = [t for ts in groups.values() for t in ts]
    assert len(all_teams) == 48 and len(set(all_teams)) == 48

def test_group_labels_are_A_to_L():
    groups = derive_groups(wc2026_group_fixtures(_wc26()))
    assert sorted(groups.keys()) == [chr(c) for c in range(ord("A"), ord("L") + 1)]

def test_each_group_has_six_fixtures():
    fx = wc2026_group_fixtures(_wc26())
    groups = derive_groups(fx)
    # 6 unique pairings per group of 4
    from itertools import combinations
    for label, teams in groups.items():
        pairs = set(frozenset(p) for p in combinations(teams, 2))
        got = set(frozenset((f["home"], f["away"])) for f in fx
                  if f["home"] in teams and f["away"] in teams)
        assert got == pairs

def test_r32_slots_well_formed():
    # 16 R32 matches, 32 slots; uses 12 winners + 12 runners-up + 8 thirds = 32
    assert len(KNOCKOUT_R32_SLOTS) == 16
    flat = [s for pair in KNOCKOUT_R32_SLOTS for s in pair]
    assert len(flat) == 32
    kinds = [s[0] for s in flat]
    assert kinds.count("W") == 12 and kinds.count("R") == 12 and kinds.count("T") == 8
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sim_bracket.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.sim.bracket'`

- [ ] **Step 3: Write the implementation**

```python
# src/wc2026/sim/__init__.py
"""Monte Carlo tournament simulation for the 2026 World Cup."""
```

```python
# src/wc2026/sim/bracket.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sim_bracket.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/sim/__init__.py src/wc2026/sim/bracket.py tests/test_sim_bracket.py
git commit -m "feat(sim): derive 2026 groups from fixtures + knockout bracket structure"
```

---

### Task 2: Match-outcome sampler from M1 strengths

**Files:**
- Create: `src/wc2026/sim/match.py`
- Test: `tests/test_sim_match.py`

**Design:** given one posterior strength draw (att/def vectors indexed by team, plus home_adv,
intercept), compute the two Poisson rates for a pairing and sample a scoreline. `sample_score`
returns `(home_goals, away_goals)`; `knockout_winner` resolves a tie via extra time
(reduced-rate Poisson) then a strength-weighted penalty shootout. A `StrengthDraw` dataclass wraps
one posterior sample for clean indexing by team name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_match.py
import numpy as np
from wc2026.sim.match import StrengthDraw, sample_score, knockout_winner

def _draw():
    teams = ["Strong", "Weak"]
    att = np.array([1.5, -1.0]); dfn = np.array([1.0, -0.8])
    return StrengthDraw(teams=teams, att=att, dfn=dfn, home_adv=0.25, intercept=0.1)

def test_rates_favour_strong_team():
    d = _draw()
    lam_s, lam_w = d.rates("Strong", "Weak", host_home=False, host_away=False)
    assert lam_s > lam_w

def test_sample_score_returns_nonneg_ints():
    d = _draw()
    rng = np.random.default_rng(0)
    hg, ag = sample_score(d, "Strong", "Weak", rng)
    assert hg >= 0 and ag >= 0 and isinstance(hg, (int, np.integer))

def test_strong_team_wins_most_knockouts():
    d = _draw()
    rng = np.random.default_rng(1)
    wins = sum(knockout_winner(d, "Strong", "Weak", rng) == "Strong" for _ in range(400))
    assert wins > 300       # strong should win ~>75%

def test_knockout_never_returns_draw():
    d = _draw()
    rng = np.random.default_rng(2)
    for _ in range(50):
        w = knockout_winner(d, "Strong", "Weak", rng)
        assert w in ("Strong", "Weak")

def test_unknown_team_uses_zero_strength():
    d = _draw()
    lam_a, lam_b = d.rates("Strong", "ZZZ", host_home=False, host_away=False)
    assert lam_a > 0 and lam_b > 0   # does not crash
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sim_match.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.sim.match'`

- [ ] **Step 3: Write the implementation**

```python
# src/wc2026/sim/match.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sim_match.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/sim/match.py tests/test_sim_match.py
git commit -m "feat(sim): match scoreline + knockout-winner sampler from M1 strengths"
```

---

### Task 3: Group-stage simulator (standings + qualifiers)

**Files:**
- Create: `src/wc2026/sim/group_stage.py`
- Test: `tests/test_sim_group_stage.py`

**Design:** simulate every group fixture (use the real played scoreline if present, else sample),
accumulate points/GD/GF, rank within group (points→GD→GF→random), then select the 24 group
qualifiers (top 2 each) + the 8 best third-placed (rank all 12 thirds the same way, take top 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_group_stage.py
import numpy as np
from wc2026.sim.match import StrengthDraw
from wc2026.sim.group_stage import simulate_groups, QualifierResult

def _draw(teams, strengths):
    att = np.array([strengths[t] for t in teams]); dfn = att.copy()
    return StrengthDraw(teams=teams, att=att, dfn=dfn, home_adv=0.0, intercept=0.2)

def _two_groups():
    # 8 teams, 2 groups of 4, clear strength order within each
    groups = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    fixtures = []
    from itertools import combinations
    for g, ts in groups.items():
        for h, a in combinations(ts, 2):
            fixtures.append({"group": g, "home": h, "away": a,
                             "home_score": None, "away_score": None, "neutral": True})
    return groups, fixtures

def test_qualifiers_counts():
    groups, fx = _two_groups()
    teams = [t for ts in groups.values() for t in ts]
    strengths = {t: (1.5 if t.endswith("1") else 1.0 if t.endswith("2") else
                     0.0 if t.endswith("3") else -1.5) for t in teams}
    rng = np.random.default_rng(0)
    res = simulate_groups(groups, fx, _draw(teams, strengths), rng,
                          n_third_qualify=2)   # 2 groups -> take 2 best thirds for test
    assert isinstance(res, QualifierResult)
    assert len(res.winners) == 2 and len(res.runners_up) == 2
    assert len(res.best_thirds) == 2
    # strongest team in each group should usually top it
    assert res.winners["A"] in ("A1", "A2")

def test_played_results_are_respected():
    groups, fx = _two_groups()
    teams = [t for ts in groups.values() for t in ts]
    # force A4 to have beaten A1 5-0 (played)
    for f in fx:
        if f["home"] == "A1" and f["away"] == "A4":
            f["home_score"], f["away_score"] = 0, 5
    strengths = {t: 1.0 for t in teams}      # equal strengths
    rng = np.random.default_rng(3)
    res = simulate_groups(groups, fx, _draw(teams, strengths), rng, n_third_qualify=2)
    # A4 banked +5 GD from the fixed result; with equal strengths it should rank highly
    assert "A4" in (res.winners["A"], res.runners_up["A"]) or "A4" in res.best_thirds
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sim_group_stage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.sim.group_stage'`

- [ ] **Step 3: Write the implementation**

```python
# src/wc2026/sim/group_stage.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sim_group_stage.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/sim/group_stage.py tests/test_sim_group_stage.py
git commit -m "feat(sim): group-stage standings, tiebreakers, qualifier selection"
```

---

### Task 4: Knockout simulator

**Files:**
- Create: `src/wc2026/sim/knockout.py`
- Test: `tests/test_sim_knockout.py`

**Design:** resolve R32 slot references to actual teams (winners/runners by group, thirds by
rank), then play a single-elimination tree R32→R16→QF→SF→Final using `knockout_winner`. Returns
the champion plus the furthest round each team reached.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_knockout.py
import numpy as np
from wc2026.sim.match import StrengthDraw
from wc2026.sim.knockout import resolve_r32, simulate_knockout, ROUNDS

def _draw(teams):
    # team i strength decreasing with index; team 0 strongest
    att = np.array([2.0 - 0.1 * i for i in range(len(teams))])
    return StrengthDraw(teams=teams, att=att, dfn=att.copy(), home_adv=0.0, intercept=0.2)

def _full_qual():
    groups = [chr(ord("A") + i) for i in range(12)]
    winners = {g: f"W{g}" for g in groups}
    runners = {g: f"R{g}" for g in groups}
    thirds = [f"T{i}" for i in range(8)]
    return winners, runners, thirds

def test_resolve_r32_gives_32_teams():
    winners, runners, thirds = _full_qual()
    pairs = resolve_r32(winners, runners, thirds)
    assert len(pairs) == 16
    flat = [t for p in pairs for t in p]
    assert len(flat) == 32 and len(set(flat)) == 32

def test_simulate_knockout_returns_champion_and_progress():
    winners, runners, thirds = _full_qual()
    pairs = resolve_r32(winners, runners, thirds)
    teams = [t for p in pairs for t in p]
    rng = np.random.default_rng(0)
    champion, furthest = simulate_knockout(pairs, _draw(teams), rng)
    assert champion in teams
    assert furthest[champion] == "champion"
    # every R32 team has a recorded furthest round
    assert all(t in furthest for t in teams)
    assert set(furthest.values()).issubset(set(ROUNDS) | {"champion"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sim_knockout.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.sim.knockout'`

- [ ] **Step 3: Write the implementation**

```python
# src/wc2026/sim/knockout.py
"""Single-elimination knockout from R32 to the final."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from .match import StrengthDraw, knockout_winner
from .bracket import KNOCKOUT_R32_SLOTS, HOSTS

ROUNDS = ["r32", "r16", "qf", "sf", "final"]      # furthest-reached labels (pre-champion)
_REACHED = {"r32": "reached_r16", "r16": "reached_qf", "qf": "reached_sf",
            "sf": "reached_final", "final": "runner_up"}

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
        furthest[t] = "reached_r32"
    pairs = list(r32_pairs)
    for rnd in ROUNDS:
        winners_round = []
        for h, a in pairs:
            w = knockout_winner(draw, h, a, rng,
                                host_home=(h in HOSTS), host_away=(a in HOSTS))
            loser = a if w == h else h
            if rnd != "final":
                furthest[w] = _REACHED[rnd]      # winner advanced past this round
            furthest[loser] = _REACHED[rnd] if rnd != "final" else "runner_up"
            winners_round.append(w)
        if rnd == "final":
            champion = winners_round[0]
            furthest[champion] = "champion"
            return champion, furthest
        # pair up winners for the next round (adjacent pairing down the bracket tree)
        pairs = [(winners_round[i], winners_round[i + 1]) for i in range(0, len(winners_round), 2)]
    raise RuntimeError("unreachable")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sim_knockout.py -q`
Expected: PASS (2 passed)

> Note: the furthest-round bookkeeping uses labels like `reached_r16` (= won R32, so reached the
> Round of 16). The Task-5 aggregator maps these to "reached round X" probabilities. Verify the
> label semantics in the test match the aggregator's expectations.

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/sim/knockout.py tests/test_sim_knockout.py
git commit -m "feat(sim): single-elimination knockout R32->final with progress tracking"
```

---

### Task 5: Tournament Monte Carlo orchestrator

**Files:**
- Create: `src/wc2026/sim/tournament.py`
- Test: `tests/test_sim_tournament.py`

**Design:** `simulate_tournament(groups, fixtures, strength_draws, n_sims, seed)` runs `n_sims`
simulations. Each sim draws ONE posterior strength sample (cycled from `strength_draws`),
simulates groups then knockout under that sample, and records the champion + furthest round per
team. Aggregates into a tidy DataFrame: per team `p_win`, `p_final`, `p_sf`, `p_qf`, `p_r16`,
`p_qualify`, sorted by `p_win`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_tournament.py
import numpy as np
from wc2026.sim.tournament import simulate_tournament, StrengthSamples

def _toy_world():
    # 12 groups x 4 teams = 48; team "Tg<k>" with k=0..3, k=0 strongest in each group
    groups = {}
    teams = []
    for gi in range(12):
        g = chr(ord("A") + gi)
        members = [f"{g}{k}" for k in range(4)]
        groups[g] = members; teams += members
    from itertools import combinations
    fixtures = []
    for g, ts in groups.items():
        for h, a in combinations(ts, 2):
            fixtures.append({"group": g, "home": h, "away": a,
                             "home_score": None, "away_score": None, "neutral": True})
    # one strength draw: index-0 team in each group is strong
    strength = {t: (1.6 if t.endswith("0") else 0.8 if t.endswith("1")
                    else 0.0 if t.endswith("2") else -1.2) for t in teams}
    att = np.array([[strength[t] for t in teams]])        # (1 draw, n_teams)
    samples = StrengthSamples(teams=teams, att=att, dfn=att.copy(),
                              home_adv=np.array([0.0]), intercept=np.array([0.2]))
    return groups, fixtures, samples

def test_probabilities_sum_and_rank():
    groups, fixtures, samples = _toy_world()
    table = simulate_tournament(groups, fixtures, samples, n_sims=400, seed=0)
    assert {"team", "p_win", "p_final", "p_sf", "p_qf", "p_r16", "p_qualify"}.issubset(table.columns)
    assert len(table) == 48
    assert np.isclose(table["p_win"].sum(), 1.0, atol=1e-9)   # exactly one champion per sim
    assert (table["p_win"] >= 0).all()
    # monotonicity: reaching an earlier round is at least as likely as winning
    assert (table["p_qualify"] >= table["p_r16"] - 1e-9).all()
    assert (table["p_r16"] >= table["p_win"] - 1e-9).all()
    # the strong teams (suffix 0) should dominate the top of the win table
    top12 = set(table.sort_values("p_win", ascending=False).head(12)["team"])
    assert sum(t.endswith("0") for t in top12) >= 8

def test_more_sims_is_deterministic_given_seed():
    groups, fixtures, samples = _toy_world()
    t1 = simulate_tournament(groups, fixtures, samples, n_sims=200, seed=42)
    t2 = simulate_tournament(groups, fixtures, samples, n_sims=200, seed=42)
    assert t1.equals(t2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sim_tournament.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.sim.tournament'`

- [ ] **Step 3: Write the implementation**

```python
# src/wc2026/sim/tournament.py
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

_PROGRESS_TO_FLAGS = {
    # furthest label -> which round milestones it satisfies
    "reached_r32":   set(),
    "reached_r16":   {"r16"},
    "reached_qf":    {"r16", "qf"},
    "reached_sf":    {"r16", "qf", "sf"},
    "runner_up":     {"r16", "qf", "sf", "final"},
    "champion":      {"r16", "qf", "sf", "final", "win"},
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sim_tournament.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the FULL fast suite (no regression)**

Run: `pytest -q -m "not slow"`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/wc2026/sim/tournament.py tests/test_sim_tournament.py
git commit -m "feat(sim): Monte Carlo tournament orchestrator with posterior propagation"
```

---

### Task 6: The real 2026 prediction (HEAVY — integration, run/monitored separately)

**Files:**
- Create: `scripts/predict_2026.py`
- Test: `tests/test_predict_2026_smoke.py` (M1 MAP + tiny n_sims → fast)

**Design:** load the real data, fit M1 on all matches up to the cutoff (NUTS for the real run),
pull `n_draws` posterior strength samples at the latest period, build the 2026 bracket from the
real fixtures, simulate `n_sims` tournaments, print the win-probability table (top 20) + reach-
round columns.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_predict_2026_smoke.py
from scripts.predict_2026 import run_prediction

def test_run_prediction_smoke():
    table = run_prediction("data/raw/results.csv", inference="map",
                           n_draws=1, n_sims=50, window_years=8)
    assert len(table) == 48
    assert abs(table["p_win"].sum() - 1.0) < 1e-9
    # a recognised contender should appear in the top half
    top = set(table.head(24)["team"])
    assert any(t in top for t in ["Spain", "France", "Argentina", "Brazil", "England"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_predict_2026_smoke.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_prediction'`

- [ ] **Step 3: Write the script**

```python
# scripts/predict_2026.py
"""Headline output: 2026 World Cup win probabilities via posterior-propagated Monte Carlo.

Real run (heavy, M1 via NUTS):
    python scripts/predict_2026.py data/raw/results.csv
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from wc2026.data.results import load_results
from wc2026.models.m1_poisson import M1PoissonModel
from wc2026.sim.bracket import wc2026_group_fixtures, derive_groups
from wc2026.sim.group_stage import simulate_groups  # noqa: F401 (kept for parity/debug)
from wc2026.sim.tournament import simulate_tournament, StrengthSamples

def _wc26_matches(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df[(raw_df["tournament"] == "FIFA World Cup") & (raw_df["date"] >= "2026-01-01")]

def _m1_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = df[["date", "home_team", "away_team", "home_score", "away_score", "neutral"]].copy()
    return f

def run_prediction(results_path: str, inference: str = "nuts", n_draws: int = 1000,
                   n_sims: int = 20000, window_years: int = 12, seed: int = 0) -> pd.DataFrame:
    df = load_results(results_path)                      # played matches only (NaN dropped)
    raw = pd.read_csv(results_path); raw["date"] = pd.to_datetime(raw["date"])
    wc26 = _wc26_matches(raw)
    fixtures = wc2026_group_fixtures(wc26)
    groups = derive_groups(fixtures)
    # attach the group label onto each fixture for the group simulator
    team_group = {t: g for g, ts in groups.items() for t in ts}
    for f in fixtures:
        f["group"] = team_group[f["home"]]

    m1 = M1PoissonModel(inference=inference, window_years=window_years,
                        draws=n_draws, random_seed=seed).fit(_m1_frame(df))
    s = m1.sample_strengths(n_draws)
    last = s["last_period"]
    att = s["att"][:, :, last]      # (n_draws, n_teams)
    dfn = s["def"][:, :, last]
    samples = StrengthSamples(teams=s["teams"], att=att, dfn=dfn,
                              home_adv=np.asarray(s["home_adv"]).ravel(),
                              intercept=np.asarray(s["intercept"]).ravel())
    return simulate_tournament(groups, fixtures, samples, n_sims=n_sims, seed=seed)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    table = run_prediction(path)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== 2026 World Cup — title & round probabilities (top 20) ===")
    print(table.head(20).to_string(index=False))
    print(f"\nfield: {len(table)} teams | sum p_win = {table['p_win'].sum():.4f}")
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/test_predict_2026_smoke.py -q`
Expected: PASS (1 passed) — M1 MAP + 50 sims, a few seconds. Numbers are rough (single MAP point,
tiny n_sims) but the full real chain composes.

- [ ] **Step 5: Run the FULL fast suite**

Run: `pytest -q -m "not slow"`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add scripts/predict_2026.py tests/test_predict_2026_smoke.py
git commit -m "feat(sim): 2026 win-probability prediction script + smoke"
```

- [ ] **Step 7: HEAVY real run (manual / monitored — NOT a unit test)**

Run: `python scripts/predict_2026.py data/raw/results.csv`
Expected: M1 fits via NUTS on ~last-12-years of data, ~1000 posterior draws, 20k tournament
simulations, then prints the top-20 title-probability table with reach-round columns. **This is
THE headline deliverable.** Capture the table. Honest expectation: the favourites' title
probabilities should land in single-to-low-double digits (no team much above ~15–20%), reflecting
genuine tournament variance — if one team is wildly dominant, inspect for a bug (strength scale,
host advantage, or period indexing).

---

## Phase 3 Done = Definition

- `pytest -q -m "not slow"` green across all test files (Phases 1–3).
- Groups derived from the real fixtures (12×4=48); knockout simulates R32→final.
- `simulate_tournament` returns a 48-team table with `p_win` summing to 1.0 and monotone
  reach-round probabilities.
- `python scripts/predict_2026.py data/raw/results.csv` produces the 2026 win-probability table
  (headline result captured to `docs/results/`).

**Deferred to Phase 4:** live re-simulation after each matchday (re-fit + re-run conditioned on
new real results) → title-odds timeline; the Awwwards web dashboard; wiring real odds (M3) as a
benchmark overlay.

---

## Self-Review

**Spec coverage (this slice):**
- Spec §5 real 48-team bracket (12 groups, 8 best thirds, R32→final) → Tasks 1, 3, 4. ✔
- Spec §5 FIFA tiebreakers → Task 3 (points→GD→GF→random; head-to-head/fair-play approximated,
  documented). ✔
- Spec §5 posterior-propagated *correlated* Monte Carlo (one strength draw per sim, whole
  tournament under it) → Task 5 (`StrengthSamples.draw` + `simulate_tournament`). ✔
- Spec §5 extra time + penalties → Task 2 (`knockout_winner`). ✔
- Spec §5 outputs: P(win), P(final/SF/QF/R16), P(qualify) → Task 5 aggregation. ✔
- Spec §0 honesty (favourite ~20–25% max) → Task 6 sanity expectation + documented approximations. ✔

**Placeholder scan:** no TBD/TODO. All sim modules fully coded with deterministic-seed tests
asserting exact properties (p_win sums to 1, monotone reach-round, strong teams dominate). Heavy
run (Task 6 Step 7) explicitly marked manual/integration.

**Type consistency:** `StrengthDraw` (single sample) vs `StrengthSamples` (batch of draws) are
distinct and used consistently — `StrengthSamples.draw(i)` yields a `StrengthDraw`. Furthest-round
labels (`reached_r16` … `champion`) are defined in `knockout.py` and mapped in `tournament.py`'s
`_PROGRESS_TO_FLAGS` — the Task-4 note flags verifying these align. `derive_groups` returns
`{label: [teams]}`, consumed by `simulate_groups` and `predict_2026.py` identically. M1's
`sample_strengths` dict keys (`att`, `def`, `home_adv`, `intercept`, `last_period`, `teams`) are
consumed exactly in `predict_2026.py`.
