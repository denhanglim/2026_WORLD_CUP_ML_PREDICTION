# WC-2026 Predictor — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer — a reproducible data pipeline, a self-computed Elo rating engine, point-in-time (leakage-safe) features, naive baseline predictors, and a walk-forward backtest harness with proper scoring rules (RPS/Brier/log-loss) and calibration metrics — so every later model is measured against honest baselines from day one.

**Architecture:** A `wc2026` Python package under `src/`. Data flows: cached raw CSV → tidy match table (`data/`) → chronological single-pass Elo engine (`elo/`) emitting *pre-match* ratings → point-in-time feature matrix (`features/`) → baseline models (`models/`) → walk-forward backtest harness (`eval/`) scoring with RPS/Brier/log-loss + binary calibration on the home-win event. Single-pass chronological construction is the leakage-safety guarantee. Everything is TDD with in-repo fixture data (no network in tests).

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn (baseline logistic), pytest. No heavy modelling deps yet (PyMC/xgboost arrive in Phase 2).

---

## Outcome encoding convention (used everywhere)

Match result is a 3-class **ordinal** variable ordered by goal difference:

```
index 0 = HOME win  (H)
index 1 = DRAW      (D)
index 2 = AWAY win  (A)
```

This ordering (H < D < A) is monotone in the home-relative outcome, so the Ranked Probability
Score is well-defined. Probability vectors are always `[p_home, p_draw, p_away]` summing to 1.

---

## File Structure

```
worldcup-2026/
  pyproject.toml                     # deps + pytest config
  README.md
  src/wc2026/
    __init__.py
    data/
      __init__.py
      results.py                     # load + tidy international results CSV → DataFrame
    elo/
      __init__.py
      engine.py                      # chronological Elo; emits pre-match ratings + finals
    features/
      __init__.py
      build.py                       # point-in-time feature matrix (elo + form)
    models/
      __init__.py
      base.py                        # Model protocol (fit/predict_proba)
      baselines.py                   # UniformBaseline, EloLogisticBaseline
    eval/
      __init__.py
      scoring.py                     # rps, brier, log_loss (+ result_to_index, one_hot)
      calibration.py                 # reliability_bins, ece (binary on home-win)
      backtest.py                    # walk_forward harness
  tests/
    fixtures/mini_results.csv        # tiny handcrafted match set for deterministic tests
    test_data_results.py
    test_elo_engine.py
    test_features_build.py
    test_features_leakage.py
    test_eval_scoring.py
    test_eval_calibration.py
    test_models_baselines.py
    test_backtest.py
  data/raw/                          # cached real data (gitignored, fetched out-of-band)
```

Each file has one responsibility. `scoring.py` and `calibration.py` are pure functions (no
state) and are the most reused — they get the most test coverage.

---

### Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `README.md`, `src/wc2026/__init__.py`, and empty `__init__.py` in each subpackage (`data`, `elo`, `features`, `models`, `eval`).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "wc2026"
version = "0.1.0"
description = "FIFA World Cup 2026 winner predictor — foundation layer"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
  "scikit-learn>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package `__init__.py` files**

```bash
mkdir -p src/wc2026/data src/wc2026/elo src/wc2026/features src/wc2026/models src/wc2026/eval tests/fixtures
touch src/wc2026/__init__.py src/wc2026/data/__init__.py src/wc2026/elo/__init__.py \
      src/wc2026/features/__init__.py src/wc2026/models/__init__.py src/wc2026/eval/__init__.py
```

- [ ] **Step 3: Create `README.md`**

```markdown
# worldcup-2026

PhD-grade FIFA World Cup 2026 winner predictor. See `docs/superpowers/specs/` for the design
and `docs/superpowers/plans/` for the build plan.

## Dev setup
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
```

- [ ] **Step 4: Create the test fixture `tests/fixtures/mini_results.csv`**

A handcrafted set where "Alpha" is clearly the strongest team and "Delta" the weakest, plus a
draw, used across multiple tests. Columns match the martj42 schema.

```csv
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2018-01-01,Alpha,Delta,3,0,Friendly,TownA,Aland,False
2018-02-01,Beta,Gamma,1,1,Friendly,TownB,Bland,False
2018-03-01,Alpha,Beta,2,0,Friendly,TownA,Aland,False
2018-04-01,Gamma,Delta,2,1,FIFA World Cup qualification,TownC,Cland,False
2018-05-01,Delta,Beta,0,2,Friendly,TownD,Dland,False
2018-06-01,Alpha,Gamma,4,0,FIFA World Cup,TownN,Nland,True
2018-07-01,Beta,Alpha,0,1,Friendly,TownB,Bland,False
2018-08-01,Gamma,Beta,1,2,Friendly,TownC,Cland,False
```

- [ ] **Step 5: Install and verify the toolchain**

Run:
```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest -q
```
Expected: pytest runs and reports "no tests ran" (exit 5) — confirms install + discovery work.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md src tests/fixtures/mini_results.csv
git commit -m "chore: scaffold wc2026 package, deps, and test fixture"
```

---

### Task 1: Data loader (tidy match table)

**Files:**
- Create: `src/wc2026/data/results.py`
- Test: `tests/test_data_results.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_results.py
import pandas as pd
from wc2026.data.results import load_results

FIX = "tests/fixtures/mini_results.csv"

def test_load_results_schema_and_types():
    df = load_results(FIX)
    # required tidy columns present
    for col in ["date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral", "match_id", "result"]:
        assert col in df.columns
    # date parsed, sorted ascending, stable match_id
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing
    assert df["match_id"].is_unique
    assert df["neutral"].dtype == bool

def test_result_encoding():
    df = load_results(FIX)
    row = df[(df.home_team == "Alpha") & (df.away_team == "Delta")].iloc[0]
    assert row["result"] == 0      # home win
    drawrow = df[(df.home_team == "Beta") & (df.away_team == "Gamma")].iloc[0]
    assert drawrow["result"] == 1  # draw
    awayrow = df[(df.home_team == "Beta") & (df.away_team == "Alpha")].iloc[0]
    assert awayrow["result"] == 2  # away win
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_results.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.data.results'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/data/results.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_results.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/data/results.py tests/test_data_results.py
git commit -m "feat(data): tidy international results loader with ordinal result encoding"
```

---

### Task 2: Elo rating engine (chronological, pre-match ratings)

**Files:**
- Create: `src/wc2026/elo/engine.py`
- Test: `tests/test_elo_engine.py`

**Design:** single chronological pass. For each match, record the *pre-match* ratings
(point-in-time safe), then apply the update. Expected score uses home advantage `H` (zeroed on
neutral grounds). Update is zero-sum: `R' = R + K·G·(W − E)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elo_engine.py
import math
import numpy as np
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine, mov_multiplier, K_BY_TOURNAMENT

FIX = "tests/fixtures/mini_results.csv"

def test_mov_multiplier():
    assert mov_multiplier(0) == 1.0
    assert mov_multiplier(1) == 1.0
    assert mov_multiplier(2) == 1.5
    assert mov_multiplier(3) == (11 + 3) / 8
    assert mov_multiplier(5) == (11 + 5) / 8

def test_single_match_update_is_zero_sum_and_correct():
    eng = EloEngine(base=1500.0, home_adv=100.0)
    # Alpha (home) beats Delta 3-0 in a Friendly (K=20), non-neutral
    pre_h, pre_a = eng.rate_match("Alpha", "Delta", 3, 0, "Friendly", neutral=False)
    assert pre_h == 1500.0 and pre_a == 1500.0
    # E_home = 1/(1+10^-(100/400)) ; G for gd=3 = 14/8 ; K=20, W=1
    E = 1.0 / (1.0 + 10 ** (-(100.0) / 400.0))
    G = (11 + 3) / 8
    delta = 20 * G * (1 - E)
    assert math.isclose(eng.rating("Alpha"), 1500.0 + delta, rel_tol=1e-9)
    assert math.isclose(eng.rating("Delta"), 1500.0 - delta, rel_tol=1e-9)

def test_neutral_removes_home_advantage():
    eng = EloEngine(base=1500.0, home_adv=100.0)
    pre_h, pre_a = eng.rate_match("Alpha", "Gamma", 1, 1, "Friendly", neutral=True)
    # draw between equals on neutral ground => no rating change
    assert math.isclose(eng.rating("Alpha"), 1500.0, rel_tol=1e-9)
    assert math.isclose(eng.rating("Gamma"), 1500.0, rel_tol=1e-9)

def test_prematch_ratings_column_is_point_in_time():
    df = load_results(FIX)
    eng = EloEngine()
    out = eng.run(df)
    # first Alpha match: pre-match elo must be the base (no prior games)
    first = out.iloc[0]
    assert first["home_elo_pre"] == 1500.0 and first["away_elo_pre"] == 1500.0
    # Alpha should end clearly strongest, Delta weakest
    finals = eng.ratings()
    assert finals["Alpha"] == max(finals.values())
    assert finals["Delta"] == min(finals.values())

def test_run_does_not_mutate_input():
    df = load_results(FIX)
    cols_before = list(df.columns)
    EloEngine().run(df)
    assert list(df.columns) == cols_before  # no new columns leaked onto caller's df
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_elo_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.elo.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/elo/engine.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_elo_engine.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/elo/engine.py tests/test_elo_engine.py
git commit -m "feat(elo): chronological Elo engine with MOV, home advantage, pre-match ratings"
```

---

### Task 3: Point-in-time feature builder

**Files:**
- Create: `src/wc2026/features/build.py`
- Test: `tests/test_features_build.py`

**Design:** one chronological pass that consumes the Elo engine's `*_elo_pre` columns and adds
recent-form features computed from matches strictly *before* the current row. Form points use
W=3 / D=1 / L=0 over the last up-to-`window` matches per team.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_build.py
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features, FEATURE_COLUMNS

FIX = "tests/fixtures/mini_results.csv"

def test_feature_columns_present():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df, form_window=5)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    assert len(feats) == len(df)

def test_elo_diff_includes_home_adv_when_not_neutral():
    df = EloEngine(home_adv=100.0).run(load_results(FIX))
    feats = build_features(df, form_window=5)
    row = feats.iloc[0]  # Alpha vs Delta, non-neutral, both 1500 pre
    # elo_diff = (home_elo_pre + home_adv*not_neutral) - away_elo_pre
    assert row["elo_diff"] == (1500.0 + 100.0) - 1500.0

def test_form_starts_at_zero_then_accumulates():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df, form_window=5)
    first = feats.iloc[0]
    assert first["home_form_pts"] == 0.0 and first["away_form_pts"] == 0.0
    # Alpha's 3rd match (Alpha vs Beta, 2018-03-01): Alpha already beat Delta (3 pts)
    alpha_beta = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert alpha_beta["home_form_pts"] == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.features.build'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/features/build.py
"""Point-in-time match features. Single chronological pass => no future leakage."""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Deque, Dict
import pandas as pd

FEATURE_COLUMNS = ["elo_diff", "home_elo_pre", "away_elo_pre",
                   "home_form_pts", "away_form_pts", "neutral"]

def _points(team: str, home: str, away: str, result: int) -> float:
    # result: 0=home win, 1=draw, 2=away win
    if result == 1:
        return 1.0
    if (result == 0 and team == home) or (result == 2 and team == away):
        return 3.0
    return 0.0

def build_features(df: pd.DataFrame, form_window: int = 5,
                   home_adv: float = 100.0) -> pd.DataFrame:
    """Return a feature frame aligned 1:1 with `df` rows (requires `home_elo_pre`,
    `away_elo_pre` columns from EloEngine.run). Form is computed from prior matches only."""
    recent: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    rows = []
    for r in df.itertuples(index=False):
        home_form = sum(recent[r.home_team])
        away_form = sum(recent[r.away_team])
        not_neutral = 0.0 if r.neutral else 1.0
        rows.append({
            "elo_diff": (r.home_elo_pre + home_adv * not_neutral) - r.away_elo_pre,
            "home_elo_pre": r.home_elo_pre,
            "away_elo_pre": r.away_elo_pre,
            "home_form_pts": home_form,
            "away_form_pts": away_form,
            "neutral": bool(r.neutral),
        })
        # update AFTER recording (point-in-time)
        recent[r.home_team].append(_points(r.home_team, r.home_team, r.away_team, r.result))
        recent[r.away_team].append(_points(r.away_team, r.home_team, r.away_team, r.result))
    feats = pd.DataFrame(rows, index=df.index)
    feats["match_id"] = df["match_id"].values
    feats["result"] = df["result"].values
    return feats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_features_build.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/features/build.py tests/test_features_build.py
git commit -m "feat(features): point-in-time elo + form feature builder"
```

---

### Task 4: Leakage guard test (the safety property)

**Files:**
- Test: `tests/test_features_leakage.py` (no new source — this hardens Task 3)

**Property under test:** features for the first *k* matches must be identical whether computed
on the full table or on the table truncated to those first *k* rows. If any feature peeked at
future matches, truncation would change it.

- [ ] **Step 1: Write the failing-then-passing test**

```python
# tests/test_features_leakage.py
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features, FEATURE_COLUMNS

FIX = "tests/fixtures/mini_results.csv"

def test_features_are_point_in_time():
    df_full = load_results(FIX)
    k = 4
    df_trunc = df_full.iloc[:k].reset_index(drop=True)

    # IMPORTANT: Elo must also be recomputed on each slice so the comparison is fair —
    # both pipelines see only the first k matches for the truncated case.
    full = build_features(EloEngine().run(df_full)).iloc[:k].reset_index(drop=True)
    trunc = build_features(EloEngine().run(df_trunc)).reset_index(drop=True)

    cols = [c for c in FEATURE_COLUMNS if c != "neutral"]
    pd.testing.assert_frame_equal(full[cols], trunc[cols], check_dtype=False)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_features_leakage.py -q`
Expected: PASS (1 passed). If it FAILS, a feature is leaking future data — fix `build_features`
before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features_leakage.py
git commit -m "test(features): assert point-in-time leakage safety"
```

---

### Task 5: Scoring rules (RPS, Brier, log-loss)

**Files:**
- Create: `src/wc2026/eval/scoring.py`
- Test: `tests/test_eval_scoring.py`

- [ ] **Step 1: Write the failing test (with hand-computed expected values)**

```python
# tests/test_eval_scoring.py
import math
import numpy as np
from wc2026.eval.scoring import rps, brier, log_loss, one_hot

def test_one_hot():
    assert one_hot(0).tolist() == [1.0, 0.0, 0.0]
    assert one_hot(2).tolist() == [0.0, 0.0, 1.0]

def test_rps_perfect_is_zero():
    p = np.array([1.0, 0.0, 0.0])
    assert rps(p, 0) == 0.0

def test_rps_uniform_home_outcome():
    # uniform forecast, home win: RPS = 5/18 ≈ 0.27778
    p = np.array([1/3, 1/3, 1/3])
    assert math.isclose(rps(p, 0), 5/18, rel_tol=1e-9)

def test_brier_uniform_home_outcome():
    p = np.array([1/3, 1/3, 1/3])
    # (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 6/9
    assert math.isclose(brier(p, 0), 6/9, rel_tol=1e-9)

def test_log_loss_uniform():
    p = np.array([1/3, 1/3, 1/3])
    assert math.isclose(log_loss(p, 0), -math.log(1/3), rel_tol=1e-9)

def test_log_loss_clips_zero():
    p = np.array([0.0, 0.5, 0.5])
    # must not be inf — clipped
    assert log_loss(p, 0) < 50.0

def test_batch_mean_helpers():
    P = np.array([[1/3, 1/3, 1/3], [1.0, 0.0, 0.0]])
    y = np.array([0, 0])
    assert math.isclose(rps(P, y).mean() if hasattr(rps(P, y), "mean") else 0, 0)  # see note
```

> Note: the final test documents intent; implement `rps`/`brier`/`log_loss` to accept either a
> single `(3,)` vector + int, or a batch `(n,3)` + `(n,)` array, returning a scalar or `(n,)`
> array respectively. Adjust the last assertion to: `np.allclose(rps(P, y), [5/18, 0.0])`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.eval.scoring'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/eval/scoring.py
"""Proper scoring rules for ordinal 3-class match forecasts [p_home, p_draw, p_away]."""
from __future__ import annotations
import numpy as np

N_CLASSES = 3

def one_hot(idx: int) -> np.ndarray:
    v = np.zeros(N_CLASSES)
    v[int(idx)] = 1.0
    return v

def _as_batch(probs, y):
    P = np.asarray(probs, dtype=float)
    single = P.ndim == 1
    if single:
        P = P[None, :]
        y = np.array([y])
    else:
        y = np.asarray(y)
    return P, y, single

def rps(probs, y):
    """Ranked Probability Score (ordinal). Lower is better. Scalar or per-sample array."""
    P, yy, single = _as_batch(probs, y)
    O = np.zeros_like(P)
    O[np.arange(len(yy)), yy] = 1.0
    cum_p = np.cumsum(P, axis=1)
    cum_o = np.cumsum(O, axis=1)
    # sum over the first (r-1) thresholds, normalised by (r-1)
    val = np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1) / (N_CLASSES - 1)
    return val[0] if single else val

def brier(probs, y):
    """Multiclass Brier score. Lower is better."""
    P, yy, single = _as_batch(probs, y)
    O = np.zeros_like(P)
    O[np.arange(len(yy)), yy] = 1.0
    val = np.sum((P - O) ** 2, axis=1)
    return val[0] if single else val

def log_loss(probs, y, eps: float = 1e-15):
    """Negative log-likelihood of the realised class. Lower is better."""
    P, yy, single = _as_batch(probs, y)
    p_true = np.clip(P[np.arange(len(yy)), yy], eps, 1.0)
    val = -np.log(p_true)
    return val[0] if single else val
```

- [ ] **Step 4: Fix the last test assertion then run**

Edit `test_batch_mean_helpers` to:
```python
def test_batch_mean_helpers():
    P = np.array([[1/3, 1/3, 1/3], [1.0, 0.0, 0.0]])
    y = np.array([0, 0])
    assert np.allclose(rps(P, y), [5/18, 0.0])
    assert np.allclose(brier(P, y), [6/9, 0.0])
```
Run: `pytest tests/test_eval_scoring.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/eval/scoring.py tests/test_eval_scoring.py
git commit -m "feat(eval): RPS, Brier, and log-loss scoring rules (scalar + batch)"
```

---

### Task 6: Calibration (reliability bins + ECE on the home-win event)

**Files:**
- Create: `src/wc2026/eval/calibration.py`
- Test: `tests/test_eval_calibration.py`

**Scope (Phase 1):** binary calibration on the home-win event — bin the predicted `p_home`,
compare bin-mean prediction (confidence) to observed home-win frequency (accuracy). ECE is the
sample-weighted mean absolute gap. (Per-class multiclass calibration is a Phase-2 extension.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_calibration.py
import numpy as np
from wc2026.eval.calibration import reliability_bins, ece

def test_perfectly_calibrated_has_near_zero_ece():
    rng = np.random.default_rng(0)
    n = 20000
    p_home = rng.uniform(0, 1, n)
    # generate outcomes with TRUE prob = p_home  => perfectly calibrated
    y_home = (rng.uniform(0, 1, n) < p_home).astype(int)
    assert ece(p_home, y_home, n_bins=10) < 0.02

def test_miscalibrated_has_high_ece():
    n = 10000
    p_home = np.full(n, 0.9)   # always claims 90%
    y_home = np.zeros(n, int)  # but home never wins
    assert ece(p_home, y_home, n_bins=10) > 0.8

def test_reliability_bins_shape_and_content():
    p = np.array([0.05, 0.15, 0.95])
    y = np.array([0, 0, 1])
    bins = reliability_bins(p, y, n_bins=10)
    assert set(["bin_lo", "bin_hi", "count", "confidence", "accuracy"]).issubset(bins.columns)
    assert bins["count"].sum() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_calibration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.eval.calibration'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/eval/calibration.py
"""Binary calibration on the home-win event."""
from __future__ import annotations
import numpy as np
import pandas as pd

def reliability_bins(p_home, y_home, n_bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p_home, dtype=float)
    y = np.asarray(y_home, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; clip 1.0 into the last bin
    idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        conf = float(p[mask].mean()) if count else float("nan")
        acc = float(y[mask].mean()) if count else float("nan")
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1],
                     "count": count, "confidence": conf, "accuracy": acc})
    return pd.DataFrame(rows)

def ece(p_home, y_home, n_bins: int = 10) -> float:
    """Expected Calibration Error: sample-weighted mean |accuracy - confidence|."""
    bins = reliability_bins(p_home, y_home, n_bins)
    n = int(bins["count"].sum())
    if n == 0:
        return float("nan")
    nonempty = bins[bins["count"] > 0]
    gap = (nonempty["accuracy"] - nonempty["confidence"]).abs()
    return float((nonempty["count"] / n * gap).sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_calibration.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/eval/calibration.py tests/test_eval_calibration.py
git commit -m "feat(eval): reliability bins + ECE for home-win calibration"
```

---

### Task 7: Baseline models (uniform + Elo logistic)

**Files:**
- Create: `src/wc2026/models/base.py`, `src/wc2026/models/baselines.py`
- Test: `tests/test_models_baselines.py`

**Design:** a tiny `Model` protocol (`fit(X, y)` → self, `predict_proba(X)` → `(n,3)`).
`UniformBaseline` ignores input. `EloLogisticBaseline` fits multinomial logistic regression of
the ordinal result on `elo_diff` (a real, learnable Elo→3-way mapping).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_baselines.py
import numpy as np
import pandas as pd
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

def _toy():
    # elo_diff strongly separates outcomes: big positive => home win, near 0 => draw-ish
    rng = np.random.default_rng(1)
    n = 600
    elo = rng.normal(0, 200, n)
    # latent: home win prob rises with elo_diff
    p_home = 1 / (1 + np.exp(-elo / 120))
    u = rng.uniform(0, 1, n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.2, 1, 2))
    X = pd.DataFrame({"elo_diff": elo})
    return X, y

def test_uniform_outputs_thirds():
    X, y = _toy()
    P = UniformBaseline().fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.allclose(P, 1/3)

def test_elo_logistic_probabilities_valid():
    X, y = _toy()
    P = EloLogisticBaseline().fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= 0).all()

def test_elo_logistic_favours_higher_elo():
    X, y = _toy()
    model = EloLogisticBaseline().fit(X, y)
    big_home = model.predict_proba(pd.DataFrame({"elo_diff": [400.0]}))[0]
    big_away = model.predict_proba(pd.DataFrame({"elo_diff": [-400.0]}))[0]
    assert big_home[0] > big_home[2]   # home favoured when elo_diff large +
    assert big_away[2] > big_away[0]   # away favoured when elo_diff large -
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_baselines.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.models.baselines'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/models/base.py
"""Minimal model protocol shared by every predictor (baselines and, later, M1/M2/meta)."""
from __future__ import annotations
from typing import Protocol
import numpy as np
import pandas as pd

class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Model": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # shape (n, 3)
```

```python
# src/wc2026/models/baselines.py
"""Naive baselines the real models must beat."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CLASSES = [0, 1, 2]  # H, D, A

class UniformBaseline:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "UniformBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full((len(X), 3), 1.0 / 3.0)

class EloLogisticBaseline:
    """Multinomial logistic of ordinal result on elo_diff. A real, fitted Elo baseline."""
    def __init__(self, feature: str = "elo_diff"):
        self.feature = feature
        self._clf = LogisticRegression(multi_class="multinomial", max_iter=1000)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EloLogisticBaseline":
        self._clf.fit(X[[self.feature]].to_numpy(), np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._clf.predict_proba(X[[self.feature]].to_numpy())
        # reindex columns to canonical [0,1,2] order regardless of classes seen in training
        out = np.zeros((len(X), 3))
        for col, cls in enumerate(self._clf.classes_):
            out[:, int(cls)] = raw[:, col]
        # renormalise in case a class was unseen in training
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models_baselines.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/models/base.py src/wc2026/models/baselines.py tests/test_models_baselines.py
git commit -m "feat(models): Model protocol + uniform and Elo-logistic baselines"
```

---

### Task 8: Walk-forward backtest harness

**Files:**
- Create: `src/wc2026/eval/backtest.py`
- Test: `tests/test_backtest.py`

**Design:** `walk_forward(feats, cutoffs, model_factory)` — for each cutoff date, train a fresh
model on matches strictly before the cutoff and evaluate on matches in the eval window after it
(default: all matches in `[cutoff, next_cutoff)`). Returns a tidy results frame of mean
RPS/Brier/log-loss + home-win ECE per fold and overall. Proves the harness end-to-end: the Elo
baseline must beat uniform on RPS on a separable synthetic set.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest.py
import numpy as np
import pandas as pd
from wc2026.eval.backtest import walk_forward
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

def _separable_feats(n=1200):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    elo = rng.normal(0, 250, n)
    p_home = 1 / (1 + np.exp(-elo / 100))
    u = rng.uniform(0, 1, n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.18, 1, 2))
    return pd.DataFrame({"date": dates, "elo_diff": elo, "result": y})

def test_walk_forward_runs_and_reports_metrics():
    feats = _separable_feats()
    cutoffs = [pd.Timestamp("2017-01-01"), pd.Timestamp("2018-01-01")]
    res = walk_forward(feats, cutoffs, lambda: EloLogisticBaseline(), features=["elo_diff"])
    assert {"fold", "n_eval", "rps", "brier", "log_loss", "ece_home"}.issubset(res.columns)
    assert (res["n_eval"] > 0).all()

def test_elo_baseline_beats_uniform_on_rps():
    feats = _separable_feats()
    cutoffs = [pd.Timestamp("2018-01-01")]
    elo_res = walk_forward(feats, cutoffs, lambda: EloLogisticBaseline(), features=["elo_diff"])
    uni_res = walk_forward(feats, cutoffs, lambda: UniformBaseline(), features=["elo_diff"])
    assert elo_res["rps"].mean() < uni_res["rps"].mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.eval.backtest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/eval/backtest.py
"""Walk-forward backtest harness. Trains only on the past; scores the future."""
from __future__ import annotations
from typing import Callable, List, Sequence
import numpy as np
import pandas as pd
from .scoring import rps, brier, log_loss
from .calibration import ece

def walk_forward(feats: pd.DataFrame, cutoffs: Sequence[pd.Timestamp],
                 model_factory: Callable[[], "object"],
                 features: List[str]) -> pd.DataFrame:
    """For each cutoff: train on date < cutoff, evaluate on [cutoff, next_cutoff).

    `feats` needs columns: `date`, `result`, and every name in `features`.
    `model_factory` returns a fresh unfitted model each call (fit/predict_proba).
    """
    cutoffs = list(cutoffs)
    rows = []
    for i, cut in enumerate(cutoffs):
        hi = cutoffs[i + 1] if i + 1 < len(cutoffs) else feats["date"].max() + pd.Timedelta(days=1)
        train = feats[feats["date"] < cut]
        evalw = feats[(feats["date"] >= cut) & (feats["date"] < hi)]
        if len(train) == 0 or len(evalw) == 0:
            continue
        model = model_factory()
        model.fit(train[features], train["result"].to_numpy())
        P = model.predict_proba(evalw[features])
        y = evalw["result"].to_numpy()
        rows.append({
            "fold": i,
            "cutoff": cut,
            "n_train": len(train),
            "n_eval": len(evalw),
            "rps": float(np.mean(rps(P, y))),
            "brier": float(np.mean(brier(P, y))),
            "log_loss": float(np.mean(log_loss(P, y))),
            "ece_home": float(ece(P[:, 0], (y == 0).astype(int))),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: PASS — all tests across the 8 test files green.

- [ ] **Step 6: Commit**

```bash
git add src/wc2026/eval/backtest.py tests/test_backtest.py
git commit -m "feat(eval): walk-forward backtest harness with RPS/Brier/log-loss/ECE"
```

---

### Task 9: End-to-end smoke script on the fixture (integration glue)

**Files:**
- Create: `scripts/phase1_smoke.py`
- Test: `tests/test_phase1_smoke.py`

**Purpose:** prove the whole Phase-1 chain composes: load → Elo → features → walk-forward,
producing a metrics table. This is the artifact that, when pointed at the *real* downloaded
dataset, yields the first honest baseline numbers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase1_smoke.py
from scripts.phase1_smoke import run_phase1

def test_phase1_smoke_returns_metrics():
    res = run_phase1("tests/fixtures/mini_results.csv",
                     cutoffs=["2018-05-01"])
    assert "rps" in res.columns
    assert len(res) >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_phase1_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.phase1_smoke'`

- [ ] **Step 3: Write the script**

```python
# scripts/phase1_smoke.py
"""End-to-end Phase-1 chain: load -> Elo -> features -> walk-forward baselines."""
from __future__ import annotations
from typing import List
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.backtest import walk_forward
from wc2026.models.baselines import EloLogisticBaseline

def run_phase1(results_path: str, cutoffs: List[str]) -> pd.DataFrame:
    df = load_results(results_path)
    df = EloEngine().run(df)
    feats = build_features(df)
    feats["date"] = df["date"].values
    cuts = [pd.Timestamp(c) for c in cutoffs]
    return walk_forward(feats, cuts, lambda: EloLogisticBaseline(), features=["elo_diff"])

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/mini_results.csv"
    print(run_phase1(path, cutoffs=["2018-05-01"]).to_string(index=False))
```

- [ ] **Step 4: Add `scripts/__init__.py` so the test can import it**

```bash
touch scripts/__init__.py
```
And add `"."` to pytest's pythonpath in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_phase1_smoke.py -q`
Expected: PASS (1 passed)

- [ ] **Step 6: Run the full suite + the script**

Run: `pytest -q && python scripts/phase1_smoke.py`
Expected: all tests pass; the script prints a one-row metrics table.

- [ ] **Step 7: Commit**

```bash
git add scripts/phase1_smoke.py scripts/__init__.py tests/test_phase1_smoke.py pyproject.toml
git commit -m "feat: phase-1 end-to-end smoke chain (load->elo->features->backtest)"
```

---

## Phase 1 Done = Definition

- `pytest -q` green across all 9 test files.
- `python scripts/phase1_smoke.py <real-results.csv>` prints baseline RPS/Brier/log-loss/ECE.
- Leakage guard passes (point-in-time features proven).
- Naive Elo logistic beats uniform on RPS in the harness.

**Out of scope for Phase 1 (deferred to later plans):** M1 Bayesian model, M2 GBM, M3 odds,
meta-learner, the Monte Carlo tournament simulator, the live-updating layer, the web dashboard,
and downloading/wiring the real datasets (the loader takes any path; fetching real data is a
one-line script wired at the start of Phase 2).

---

## Self-Review

**Spec coverage (Phase-1 slice of the spec):**
- Spec §2 data layer → Task 1 (loader) + Task 9 (real-data wiring noted as deferred). ✔
- Spec §2 computed Elo → Task 2. ✔
- Spec §3 point-in-time features → Tasks 3 + 4 (leakage guard). ✔
- Spec §6 RPS/Brier/log-loss + calibration → Tasks 5 + 6. ✔
- Spec §6 naive baselines (Elo-only, uniform) + walk-forward → Tasks 7 + 8. ✔
- Spec §4/§5/§7/§8 (M1/M2/meta, sim, live, web) → explicitly deferred to later phase plans. ✔ (by design — foundation-first)

**Placeholder scan:** no TBD/TODO; every code step shows complete runnable code. ✔

**Type consistency:** `[p_home, p_draw, p_away]` order and `result ∈ {0,1,2}` (H,D,A) used
identically across scoring, calibration, baselines, backtest. `Model.fit/predict_proba`
signature consistent in `base.py`, baselines, and harness. `EloEngine.run` adds
`home_elo_pre`/`away_elo_pre`, consumed under those exact names in `features/build.py`. ✔
