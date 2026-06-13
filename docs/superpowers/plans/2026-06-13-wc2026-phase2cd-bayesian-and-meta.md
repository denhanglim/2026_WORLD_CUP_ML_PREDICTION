# WC-2026 Predictor — Phase 2c+2d (Bayesian M1 + Odds + Meta-Learner) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the statistical flagship and the stacker — a dynamic hierarchical Poisson goal model (M1, PyMC) with time-varying team strengths and a Dixon–Coles low-score correction; a market-odds de-vig utility (M3); and a meta-learner that stacks base-model probability vectors — then compare M1/M2/M3/meta against the baselines on the real World Cup backtest.

**Architecture:** Builds on Phase 1 + 2a/2b. New pieces: `models/dixon_coles.py` (pure τ correction), `models/periods.py` (time discretization), `models/m1_poisson.py` (PyMC dynamic hierarchical Poisson; configurable MAP/ADVI/NUTS inference; exposes a posterior strength sampler for the Phase-3 simulator), `models/odds.py` (de-vig → implied probabilities), `models/meta.py` (multinomial-logistic stacker over base probability vectors), plus `eval/compare.py` (a heterogeneous-model comparison helper that passes the *full* feature frame so goal-based models like M1 can read team/score columns) and `scripts/full_backtest.py` (the heavy real-data integration run). The existing `walk_forward` harness is left untouched so the 43 green tests stay green.

**Tech Stack:** Existing + `pymc` 6.0.1, `arviz` 1.2.0 (verified to install and sample — MAP and NUTS both work on this box), `scipy` (already present via sklearn).

---

## Conventions carried forward (do not change)

- Outcome encoding `result ∈ {0=HOME,1=DRAW,2=AWAY}`; probability vectors `[p_home,p_draw,p_away]`.
- Goal-based models need raw match columns (`home_team, away_team, home_score, away_score, date, neutral`). Task 4 makes `build_features` carry these as metadata so every model can read from one frame.
- Unit tests must be FAST and reliable: M1 unit tests use `inference="map"` on a tiny synthetic dataset (sub-second). One smoke test exercises `"advi"`. The heavy real fit (NUTS/ADVI on the full dataset) is an integration run in Task 7, NOT a unit test.

**Branch:** create `feature/phase2cd-bayes-meta` off `feature/phase2ab-data-gbm`.

---

### Task 1: Dependencies + Dixon–Coles correction (pure function)

**Files:**
- Modify: `pyproject.toml` (add `pymc`, `arviz`)
- Create: `src/wc2026/models/dixon_coles.py`
- Test: `tests/test_dixon_coles.py`

- [ ] **Step 1: Add deps to `pyproject.toml`** (in `[project].dependencies`)

```toml
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
  "scikit-learn>=1.3",
  "scipy>=1.10",
  "pymc>=6.0",
  "arviz>=1.0",
]
```
Then: `pip install -e ".[dev]"` (pymc/arviz already installed in the venv; this records them).

- [ ] **Step 2: Write the failing test** (Dixon–Coles τ has exact closed-form values)

```python
# tests/test_dixon_coles.py
import math
import numpy as np
from wc2026.models.dixon_coles import dc_tau

def test_tau_low_scores_exact():
    lh, la, rho = 1.3, 1.1, 0.05
    assert math.isclose(dc_tau(0, 0, lh, la, rho), 1 - lh * la * rho)
    assert math.isclose(dc_tau(0, 1, lh, la, rho), 1 + lh * rho)
    assert math.isclose(dc_tau(1, 0, lh, la, rho), 1 + la * rho)
    assert math.isclose(dc_tau(1, 1, lh, la, rho), 1 - rho)

def test_tau_high_scores_is_one():
    assert dc_tau(2, 0, 1.3, 1.1, 0.05) == 1.0
    assert dc_tau(3, 2, 1.3, 1.1, 0.05) == 1.0
    assert dc_tau(0, 2, 1.3, 1.1, 0.05) == 1.0

def test_tau_positive_for_small_rho():
    # for plausible rates and small |rho|, tau stays positive (valid likelihood)
    for hg, ag in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert dc_tau(hg, ag, 2.0, 2.0, 0.05) > 0
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/test_dixon_coles.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.models.dixon_coles'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/wc2026/models/dixon_coles.py
"""Dixon & Coles (1997) low-score dependence correction for independent Poisson goals."""
from __future__ import annotations

def dc_tau(home_goals: int, away_goals: int, lambda_home: float,
          lambda_away: float, rho: float) -> float:
    """Multiplicative correction τ applied to the independent-Poisson joint pmf.
    Adjusts only the four low-score cells; returns 1.0 elsewhere."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_dixon_coles.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/wc2026/models/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat(models): Dixon-Coles low-score correction + pymc/arviz deps"
```

---

### Task 2: Time-period discretization

**Files:**
- Create: `src/wc2026/models/periods.py`
- Test: `tests/test_periods.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_periods.py
import numpy as np
import pandas as pd
from wc2026.models.periods import assign_periods

def test_yearly_periods_are_chronological_codes():
    dates = pd.to_datetime(["2010-03-01", "2010-09-01", "2011-01-01", "2012-06-01"])
    codes, n = assign_periods(dates, freq="Y")
    assert n == 3
    assert codes.tolist() == [0, 0, 1, 2]   # 2010,2010 -> 0 ; 2011 -> 1 ; 2012 -> 2

def test_unsorted_input_still_maps_by_calendar():
    dates = pd.to_datetime(["2012-06-01", "2010-03-01", "2011-01-01"])
    codes, n = assign_periods(dates, freq="Y")
    assert n == 3
    assert codes.tolist() == [2, 0, 1]      # period code reflects calendar order, not row order
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_periods.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.models.periods'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/models/periods.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_periods.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/models/periods.py tests/test_periods.py
git commit -m "feat(models): yearly/quarterly time-period discretization"
```

---

### Task 3: M1 — dynamic hierarchical Poisson (PyMC)

**Files:**
- Create: `src/wc2026/models/m1_poisson.py`
- Test: `tests/test_m1_poisson.py`

**Design notes:**
- Latent per-team `attack[team, period]` and `defense[team, period]`, evolving as a Gaussian
  random walk across periods (dynamic state-space), 0-centered hierarchical priors (soft
  identifiability against the intercept).
- Log goal rates: `log λ_home = intercept + home_adv·(1−neutral) + att[home,p] − def[away,p]`;
  `log λ_away = intercept + att[away,p] − def[home,p]`.
- Independent Poisson goals + Dixon–Coles correction added as a `pm.Potential` (log τ summed
  over matches; τ from the same closed form as `dc_tau`, vectorized with precomputed score masks).
- Inference is configurable: `"map"` (fast, deterministic-ish — used by unit tests), `"advi"`
  (variational — used by the real backtest), `"nuts"` (gold — used for the final 2026 fit).
- `predict_proba(df)` integrates the (DC-corrected) bivariate score grid up to `max_goals`,
  averaging over posterior draws. Teams unseen in training get strength 0 (the prior mean).
- `sample_strengths(n)` exposes posterior draws for the Phase-3 correlated Monte Carlo.

- [ ] **Step 1: Write the failing test** (tiny synthetic, MAP inference → fast + reliable)

```python
# tests/test_m1_poisson.py
import warnings
import numpy as np
import pandas as pd
import pytest
from wc2026.models.m1_poisson import M1PoissonModel

warnings.filterwarnings("ignore")

def _toy_matches(seed=0):
    """4 teams, clear strength order A>B>C>D, two yearly periods. Strong teams score more."""
    rng = np.random.default_rng(seed)
    strength = {"A": 2.0, "B": 1.0, "C": 0.0, "D": -1.0}
    rows = []
    teams = list(strength)
    for year in (2018, 2019):
        for _ in range(60):
            h, a = rng.choice(teams, 2, replace=False)
            lam_h = np.exp(0.2 + strength[h] - 0.5 * strength[a])
            lam_a = np.exp(0.2 + strength[a] - 0.5 * strength[h])
            rows.append({
                "date": pd.Timestamp(f"{year}-06-01"),
                "home_team": h, "away_team": a,
                "home_score": int(rng.poisson(lam_h)), "away_score": int(rng.poisson(lam_a)),
                "neutral": True,
            })
    return pd.DataFrame(rows)

def test_predict_proba_is_valid_distribution():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    pred = pd.DataFrame([{"home_team": "A", "away_team": "D", "neutral": True,
                          "date": pd.Timestamp("2019-06-01")}])
    P = m.predict_proba(pred)
    assert P.shape == (1, 3)
    assert np.isclose(P.sum(), 1.0, atol=1e-6)
    assert (P >= 0).all()

def test_stronger_team_favoured():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    pred = pd.DataFrame([
        {"home_team": "A", "away_team": "D", "neutral": True, "date": pd.Timestamp("2019-06-01")},
        {"home_team": "D", "away_team": "A", "neutral": True, "date": pd.Timestamp("2019-06-01")},
    ])
    P = m.predict_proba(pred)
    assert P[0, 0] > P[0, 2]    # A (home) beats D more likely than loses
    assert P[1, 2] > P[1, 0]    # D home vs A: away (A) win more likely

def test_unknown_team_gets_neutral_strength():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    pred = pd.DataFrame([{"home_team": "A", "away_team": "ZZZ_UNKNOWN", "neutral": True,
                          "date": pd.Timestamp("2019-06-01")}])
    P = m.predict_proba(pred)
    assert np.isclose(P.sum(), 1.0, atol=1e-6)   # does not crash; valid probs

def test_sample_strengths_shape():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="map").fit(df)
    draws = m.sample_strengths(5)
    # dict with att/def arrays shaped (n_draws, n_teams, n_periods)
    assert draws["att"].shape[0] == 5
    assert draws["att"].shape[1] == m.n_teams

@pytest.mark.slow
def test_advi_smoke():
    df = _toy_matches()
    m = M1PoissonModel(period_freq="Y", inference="advi", advi_iter=2000, draws=50).fit(df)
    pred = pd.DataFrame([{"home_team": "A", "away_team": "B", "neutral": True,
                          "date": pd.Timestamp("2019-06-01")}])
    P = m.predict_proba(pred)
    assert np.isclose(P.sum(), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_m1_poisson.py -q -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.models.m1_poisson'`

- [ ] **Step 3: Register the `slow` marker** in `pyproject.toml` (`[tool.pytest.ini_options]`)

```toml
[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
markers = ["slow: long-running (variational/MCMC) tests"]
```

- [ ] **Step 4: Write the implementation** (reference — adjust internals to make the tests pass; the tests define correctness)

```python
# src/wc2026/models/m1_poisson.py
"""M1 — dynamic hierarchical Poisson goal model with Dixon-Coles correction (PyMC).

Latent team attack/defense strengths evolve as a Gaussian random walk across yearly periods.
Goals are independent Poisson with a Dixon-Coles low-score correction (pm.Potential).
Inference: MAP (fast), ADVI (variational), or NUTS (gold). predict_proba integrates the
DC-corrected bivariate score grid, averaged over posterior draws.
"""
from __future__ import annotations
from typing import Dict, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson
from .periods import assign_periods

class M1PoissonModel:
    def __init__(self, period_freq: str = "Y", inference: str = "map",
                 draws: int = 300, tune: int = 500, chains: int = 2,
                 advi_iter: int = 20000, window_years: Optional[int] = None,
                 max_goals: int = 10, random_seed: int = 0):
        self.period_freq = period_freq
        self.inference = inference
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.advi_iter = advi_iter
        self.window_years = window_years
        self.max_goals = max_goals
        self.random_seed = random_seed
        self._idx: Dict[str, int] = {}
        self.n_teams = 0
        self.n_periods = 0
        self._last_period = 0
        self._point = None
        self._idata = None

    # ---- fitting -------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y=None) -> "M1PoissonModel":
        import pymc as pm
        import pytensor.tensor as pt

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"])
        if self.window_years is not None:
            cutoff = data["date"].max() - pd.DateOffset(years=self.window_years)
            data = data[data["date"] >= cutoff]
        data = data.sort_values("date", kind="stable").reset_index(drop=True)

        teams = sorted(set(data["home_team"]) | set(data["away_team"]))
        self._idx = {t: i for i, t in enumerate(teams)}
        self.n_teams = len(teams)
        period, self.n_periods = assign_periods(data["date"], self.period_freq)
        self._last_period = self.n_periods - 1

        hi = data["home_team"].map(self._idx).to_numpy()
        ai = data["away_team"].map(self._idx).to_numpy()
        hg = data["home_score"].to_numpy(dtype=int)
        ag = data["away_score"].to_numpy(dtype=int)
        neutral = data["neutral"].astype(float).to_numpy()

        # precompute Dixon-Coles low-score masks (constants)
        m00 = ((hg == 0) & (ag == 0)).astype(float)
        m01 = ((hg == 0) & (ag == 1)).astype(float)
        m10 = ((hg == 1) & (ag == 0)).astype(float)
        m11 = ((hg == 1) & (ag == 1)).astype(float)
        m_other = 1.0 - (m00 + m01 + m10 + m11)

        T, P = self.n_teams, self.n_periods
        with pm.Model() as model:
            sigma_att0 = pm.HalfNormal("sigma_att0", 1.0)
            sigma_def0 = pm.HalfNormal("sigma_def0", 1.0)
            sigma_evo = pm.HalfNormal("sigma_evo", 0.15)
            home_adv = pm.Normal("home_adv", 0.25, 0.5)
            intercept = pm.Normal("intercept", 0.0, 1.0)
            rho = pm.Normal("rho", 0.0, 0.1)

            att0 = pm.Normal("att0", 0.0, sigma_att0, shape=T)
            def0 = pm.Normal("def0", 0.0, sigma_def0, shape=T)
            if P > 1:
                att_step = pm.Normal("att_step", 0.0, sigma_evo, shape=(T, P - 1))
                def_step = pm.Normal("def_step", 0.0, sigma_evo, shape=(T, P - 1))
                att = pt.concatenate([att0[:, None], att0[:, None] + pt.cumsum(att_step, axis=1)], axis=1)
                dfn = pt.concatenate([def0[:, None], def0[:, None] + pt.cumsum(def_step, axis=1)], axis=1)
            else:
                att = att0[:, None]
                dfn = def0[:, None]
            att = pm.Deterministic("att", att)   # (T, P)
            dfn = pm.Deterministic("def", dfn)   # (T, P)

            ha = att[hi, period]; hd = dfn[hi, period]
            aa = att[ai, period]; ad = dfn[ai, period]
            lam_h = pt.exp(intercept + home_adv * (1.0 - neutral) + ha - ad)
            lam_a = pt.exp(intercept + aa - hd)

            pm.Poisson("hg_obs", mu=lam_h, observed=hg)
            pm.Poisson("ag_obs", mu=lam_a, observed=ag)

            tau = (m00 * (1 - lam_h * lam_a * rho) + m01 * (1 + lam_h * rho)
                   + m10 * (1 + lam_a * rho) + m11 * (1 - rho) + m_other)
            pm.Potential("dc", pt.sum(pt.log(pt.clip(tau, 1e-9, np.inf))))

            if self.inference == "map":
                self._point = pm.find_MAP(progressbar=False)
            elif self.inference == "advi":
                approx = pm.fit(self.advi_iter, method="advi", progressbar=False,
                                random_seed=self.random_seed)
                self._idata = approx.sample(self.draws)
            elif self.inference == "nuts":
                self._idata = pm.sample(self.draws, tune=self.tune, chains=self.chains,
                                        progressbar=False, random_seed=self.random_seed,
                                        compute_convergence_checks=False)
            else:
                raise ValueError(f"unknown inference: {self.inference}")
        return self

    # ---- posterior access ----------------------------------------------------
    def _params(self):
        """Return (att, dfn, home_adv, intercept, rho) with att/dfn shaped (S, T, P)."""
        if self._point is not None:
            return (self._point["att"][None], self._point["def"][None],
                    np.array([float(self._point["home_adv"])]),
                    np.array([float(self._point["intercept"])]),
                    np.array([float(self._point["rho"])]))
        post = self._idata.posterior
        def stack(name):
            arr = post[name].stack(s=("chain", "draw"))
            # move the stacked 's' axis to the front
            return np.moveaxis(arr.values, -1, 0)
        att = stack("att"); dfn = stack("def")
        ha = post["home_adv"].stack(s=("chain", "draw")).values
        ic = post["intercept"].stack(s=("chain", "draw")).values
        rh = post["rho"].stack(s=("chain", "draw")).values
        return att, dfn, ha, ic, rh

    def sample_strengths(self, n: int) -> Dict[str, np.ndarray]:
        """Up to `n` posterior draws of team strengths for the Phase-3 simulator."""
        att, dfn, ha, ic, rh = self._params()
        S = att.shape[0]
        take = min(n, S) if S > 1 else n
        idx = (np.arange(take) % S)
        return {"att": att[idx], "def": dfn[idx], "home_adv": ha[idx % len(ha)],
                "intercept": ic[idx % len(ic)], "rho": rh[idx % len(rh)],
                "teams": list(self._idx.keys()), "last_period": self._last_period}

    # ---- prediction ----------------------------------------------------------
    def _match_probs(self, lam_h: float, lam_a: float, rho: float) -> np.ndarray:
        g = np.arange(self.max_goals + 1)
        ph = poisson.pmf(g, lam_h); pa = poisson.pmf(g, lam_a)
        M = np.outer(ph, pa)               # rows = home goals, cols = away goals
        M[0, 0] *= (1 - lam_h * lam_a * rho)
        M[0, 1] *= (1 + lam_h * rho)
        M[1, 0] *= (1 + lam_a * rho)
        M[1, 1] *= (1 - rho)
        M = np.clip(M, 0.0, None)
        s = M.sum()
        if s <= 0:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        M /= s
        home = np.tril(M, -1).sum()        # home goals > away goals
        draw = np.trace(M)
        away = np.triu(M, 1).sum()
        return np.array([home, draw, away])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        att, dfn, ha, ic, rh = self._params()
        S = att.shape[0]
        p = self._last_period
        hi = df["home_team"].map(self._idx)
        ai = df["away_team"].map(self._idx)
        neutral = df["neutral"].astype(float).to_numpy()
        out = np.zeros((len(df), 3))
        for m in range(len(df)):
            h = hi.iloc[m]; a = ai.iloc[m]
            acc = np.zeros(3)
            for s in range(S):
                ah = att[s, int(h), p] if pd.notna(h) else 0.0
                dh = dfn[s, int(h), p] if pd.notna(h) else 0.0
                aa = att[s, int(a), p] if pd.notna(a) else 0.0
                da = dfn[s, int(a), p] if pd.notna(a) else 0.0
                lam_h = np.exp(ic[s] + ha[s] * (1.0 - neutral[m]) + ah - da)
                lam_a = np.exp(ic[s] + aa - dh)
                acc += self._match_probs(float(lam_h), float(lam_a), float(rh[s]))
            out[m] = acc / S
        return out
```

- [ ] **Step 5: Run the fast tests** (exclude the slow ADVI smoke)

Run: `pytest tests/test_m1_poisson.py -q -m "not slow"`
Expected: PASS (5 passed). If MAP struggles to favour the strong team, increase toy data per
period or check the index/period wiring — do NOT weaken the assertions.

- [ ] **Step 6: Run the slow ADVI smoke once to confirm variational path works**

Run: `pytest tests/test_m1_poisson.py -q -m slow`
Expected: PASS (1 passed) within ~tens of seconds.

- [ ] **Step 7: Commit**

```bash
git add src/wc2026/models/m1_poisson.py tests/test_m1_poisson.py pyproject.toml
git commit -m "feat(models): M1 dynamic hierarchical Poisson with Dixon-Coles (PyMC)"
```

---

### Task 4: Carry raw match columns + heterogeneous comparison helper

**Files:**
- Modify: `src/wc2026/features/build.py` (carry raw match cols as metadata — additive)
- Create: `src/wc2026/eval/compare.py`
- Test: `tests/test_compare.py`

**Why:** M1 reads team/score columns, baselines/GBM read engineered features. A single comparison
helper passes the *whole* frame to every model so each reads what it needs. The existing
`walk_forward` is left untouched.

- [ ] **Step 1: Add raw-column passthrough to `build_features`** (append before `return feats`)

```python
    # raw match columns carried as metadata so goal-based models (M1) can read them
    for col in ["home_team", "away_team", "home_score", "away_score"]:
        feats[col] = df[col].values
    feats["neutral_flag"] = df["neutral"].values   # bool; 'neutral' feature col already exists
```
> Note: `build_features` already emits a `neutral` feature column (float-ish bool). We add the
> raw cols and a clearly-named `neutral_flag` to avoid colliding with the existing `neutral`
> entry in `FEATURE_COLUMNS`. M1's input frames use `neutral` (bool) — when calling M1 from the
> comparison helper, pass `neutral` derived from `neutral_flag`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_compare.py
import numpy as np
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.compare import predict_frame, score_predictions
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

FIX = "tests/fixtures/mini_results.csv"

def test_build_features_carries_raw_cols():
    feats = build_features(EloEngine().run(load_results(FIX)))
    for c in ["home_team", "away_team", "home_score", "away_score", "neutral_flag"]:
        assert c in feats.columns

def test_predict_frame_and_score():
    feats = build_features(EloEngine().run(load_results(FIX)))
    model = EloLogisticBaseline().fit(feats, feats["result"].to_numpy())
    P = predict_frame(model, feats)
    assert P.shape == (len(feats), 3)
    metrics = score_predictions(P, feats["result"].to_numpy())
    assert {"rps", "brier", "log_loss"}.issubset(metrics)
    # uniform must be worse-or-equal on RPS than a fitted Elo model on this toy set's own data
    Pu = predict_frame(UniformBaseline().fit(feats, feats["result"].to_numpy()), feats)
    assert score_predictions(Pu, feats["result"].to_numpy())["rps"] >= 0
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/test_compare.py -q`
Expected: FAIL — missing `wc2026.eval.compare` and/or missing raw columns.

- [ ] **Step 4: Write `compare.py`**

```python
# src/wc2026/eval/compare.py
"""Heterogeneous-model comparison: pass the full frame to each model; each reads what it needs."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from .scoring import rps, brier, log_loss
from .calibration import ece

def predict_frame(model, frame: pd.DataFrame) -> np.ndarray:
    """Call model.predict_proba on the full frame (model selects its own columns)."""
    return model.predict_proba(frame)

def score_predictions(P: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        "rps": float(np.mean(rps(P, y))),
        "brier": float(np.mean(brier(P, y))),
        "log_loss": float(np.mean(log_loss(P, y))),
        "ece_home": float(ece(P[:, 0], (np.asarray(y) == 0).astype(int))),
    }
```

- [ ] **Step 5: Run the comparison test + the full feature tests (no regression)**

Run: `pytest tests/test_compare.py tests/test_features_build.py tests/test_features_enriched.py tests/test_features_leakage.py -q`
Expected: PASS. (Raw cols are additive; leakage guard compares only `FEATURE_COLUMNS`.)

- [ ] **Step 6: Commit**

```bash
git add src/wc2026/features/build.py src/wc2026/eval/compare.py tests/test_compare.py
git commit -m "feat(eval): carry raw match cols + heterogeneous model comparison helper"
```

---

### Task 5: M3 — market-odds de-vig

**Files:**
- Create: `src/wc2026/models/odds.py`
- Test: `tests/test_odds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_odds.py
import numpy as np
from wc2026.models.odds import implied_probs, OddsModel

def test_implied_probs_fair_odds():
    # fair decimal odds [2.0, 4.0, 4.0] -> [0.5, 0.25, 0.25] (no overround)
    p = implied_probs([2.0, 4.0, 4.0])
    assert np.allclose(p, [0.5, 0.25, 0.25])

def test_implied_probs_removes_overround():
    p = implied_probs([1.9, 3.5, 4.0])   # raw inverses sum > 1 (bookmaker margin)
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()

def test_odds_model_predicts_from_columns():
    import pandas as pd
    df = pd.DataFrame({"odds_home": [2.0, 4.0], "odds_draw": [4.0, 4.0], "odds_away": [4.0, 1.5]})
    P = OddsModel().fit(df, None).predict_proba(df)
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0, 0] > P[0, 2]    # row 0: home favoured
    assert P[1, 2] > P[1, 0]    # row 1: away favoured
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_odds.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.models.odds'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/models/odds.py
"""M3 — market odds → de-vigged implied probabilities (proportional normalisation)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def implied_probs(decimal_odds) -> np.ndarray:
    """[home, draw, away] decimal odds -> de-vigged probabilities summing to 1."""
    raw = 1.0 / np.asarray(decimal_odds, dtype=float)
    return raw / raw.sum()

class OddsModel:
    """Reads odds_home/odds_draw/odds_away columns. fit() is a no-op (market is the model)."""
    def fit(self, X: pd.DataFrame, y=None) -> "OddsModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = X[["odds_home", "odds_draw", "odds_away"]].to_numpy(dtype=float)
        raw = 1.0 / cols
        return raw / raw.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_odds.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/models/odds.py tests/test_odds.py
git commit -m "feat(models): M3 market-odds de-vig to implied probabilities"
```

---

### Task 6: Meta-learner (stacker)

**Files:**
- Create: `src/wc2026/models/meta.py`
- Test: `tests/test_meta.py`

**Design:** `MetaStacker.fit(base_probs_list, y)` horizontally stacks the base models'
`(n,3)` probability matrices into an `(n, 3k)` feature matrix and fits a multinomial logistic
regression to the true outcome. `predict_proba(base_probs_list)` applies it. The meta must learn
to down-weight a useless base model: with one good base and one pure-noise base, the meta's RPS
must be ≤ the noise base's and close to the good base's.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meta.py
import numpy as np
from wc2026.models.meta import MetaStacker
from wc2026.eval.scoring import rps

def _good_and_noise(n=1500, seed=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    # good base: high prob on the true class
    good = np.full((n, 3), 0.1); good[np.arange(n), y] = 0.8
    good = good / good.sum(axis=1, keepdims=True)
    # noise base: random simplex
    noise = rng.dirichlet([1, 1, 1], size=n)
    return good, noise, y

def test_meta_outputs_valid_probs():
    good, noise, y = _good_and_noise()
    meta = MetaStacker().fit([good, noise], y)
    P = meta.predict_proba([good, noise])
    assert P.shape == (len(y), 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)

def test_meta_ignores_noise_base():
    good, noise, y = _good_and_noise()
    tr = slice(0, 1000); te = slice(1000, None)
    meta = MetaStacker().fit([good[tr], noise[tr]], y[tr])
    P = meta.predict_proba([good[te], noise[te]])
    rps_meta = np.mean(rps(P, y[te]))
    rps_good = np.mean(rps(good[te], y[te]))
    rps_noise = np.mean(rps(noise[te], y[te]))
    assert rps_meta < rps_noise                 # learned to discard noise
    assert rps_meta <= rps_good + 0.02           # nearly as good as the good base
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_meta.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.models.meta'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/models/meta.py
"""Meta-learner: stacks base-model probability vectors via multinomial logistic regression."""
from __future__ import annotations
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression

class MetaStacker:
    def __init__(self, max_iter: int = 2000, C: float = 1.0):
        self.max_iter = max_iter
        self.C = C
        self._clf = None

    @staticmethod
    def _stack(base_probs_list: List[np.ndarray]) -> np.ndarray:
        return np.hstack([np.asarray(b, dtype=float) for b in base_probs_list])

    def fit(self, base_probs_list: List[np.ndarray], y) -> "MetaStacker":
        X = self._stack(base_probs_list)
        self._clf = LogisticRegression(max_iter=self.max_iter, C=self.C)
        self._clf.fit(X, np.asarray(y))
        return self

    def predict_proba(self, base_probs_list: List[np.ndarray]) -> np.ndarray:
        X = self._stack(base_probs_list)
        raw = self._clf.predict_proba(X)
        out = np.zeros((len(X), 3))
        for col, cls in enumerate(self._clf.classes_):
            out[:, int(cls)] = raw[:, col]
        s = out.sum(axis=1, keepdims=True); s[s == 0] = 1.0
        return out / s
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_meta.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the FULL fast suite (no regression, exclude slow)**

Run: `pytest -q -m "not slow"`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/wc2026/models/meta.py tests/test_meta.py
git commit -m "feat(models): meta-learner stacker over base probability vectors"
```

---

### Task 7: Real-data full backtest (HEAVY — integration, run/monitored separately)

**Files:**
- Create: `scripts/full_backtest.py`
- Test: `tests/test_full_backtest_smoke.py` (uses tiny fixture + MAP, fast)

**Design:** for each World Cup fold (train on `date < cutoff`, eval = that edition's WC matches):
fit M1 (ADVI, windowed), fit Elo-logistic and GBM, generate eval predictions for each; build a
meta-training set from an inner time-split of the training data (out-of-fold base predictions);
fit the meta on those; predict eval; score M1/M2/Elo/meta with RPS/Brier/log-loss/ECE. Odds (M3)
are included only if `odds_home/draw/away` columns are present (they are not in the free dataset,
so M3 is skipped by default and noted). Print a comparison table.

- [ ] **Step 1: Write the smoke test** (tiny fixture, M1 MAP → fast)

```python
# tests/test_full_backtest_smoke.py
from scripts.full_backtest import run_fold_smoke

def test_run_fold_smoke_returns_table():
    tbl = run_fold_smoke("tests/fixtures/mini_results.csv")
    assert "model" in tbl.columns and "rps" in tbl.columns
    assert set(tbl["model"]) >= {"elo_logistic", "m1_poisson", "meta"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_full_backtest_smoke.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_fold_smoke'`

- [ ] **Step 3: Write the script**

```python
# scripts/full_backtest.py
"""Phase 2c+2d real-data backtest: M1 (Bayesian) + M2 (GBM) + Elo baseline + meta-learner.

Heavy: M1 fits via ADVI per fold. Run manually:
    python scripts/full_backtest.py data/raw/results.csv
The smoke entrypoint run_fold_smoke() uses the tiny fixture + M1 MAP for a fast CI check.
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.compare import predict_frame, score_predictions
from wc2026.tournaments import wc_cutoffs, is_world_cup
from wc2026.models.baselines import EloLogisticBaseline
from wc2026.models.gbm import GBMModel
from wc2026.models.m1_poisson import M1PoissonModel

GBM_FEATURES = ["elo_diff", "home_form_pts", "away_form_pts", "rest_days_diff",
                "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]

def _m1_frame(feats: pd.DataFrame) -> pd.DataFrame:
    """M1 needs a 'neutral' bool column; build_features carries it as 'neutral_flag'."""
    f = feats.copy()
    f["neutral"] = f["neutral_flag"].astype(bool)
    return f

def _fit_predict_bases(train, evalw, m1_inference="advi", m1_kwargs=None):
    m1_kwargs = m1_kwargs or {}
    elo = EloLogisticBaseline().fit(train, train["result"].to_numpy())
    gbm = GBMModel(features=GBM_FEATURES).fit(train, train["result"].to_numpy())
    m1 = M1PoissonModel(inference=m1_inference, **m1_kwargs).fit(_m1_frame(train))
    P = {
        "elo_logistic": predict_frame(elo, evalw),
        "gbm": predict_frame(gbm, evalw),
        "m1_poisson": m1.predict_proba(_m1_frame(evalw)),
    }
    return P, (elo, gbm, m1)

def _fold(feats, cut, hi, m1_inference="advi", m1_kwargs=None):
    train = feats[feats["date"] < cut]
    evalw = feats[(feats["date"] >= cut) & (feats["date"] < hi)]
    evalw = evalw[is_world_cup(evalw)]
    if len(train) == 0 or len(evalw) == 0:
        return None
    # inner split of train for meta out-of-fold predictions
    inner_cut = train["date"].quantile(0.8)
    base_tr = train[train["date"] < inner_cut]
    meta_tr = train[train["date"] >= inner_cut]
    rows = []
    if len(base_tr) and len(meta_tr):
        Pin, _ = _fit_predict_bases(base_tr, meta_tr, m1_inference, m1_kwargs)
        meta = __import__("wc2026.models.meta", fromlist=["MetaStacker"]).MetaStacker()
        order = ["elo_logistic", "gbm", "m1_poisson"]
        meta.fit([Pin[k] for k in order], meta_tr["result"].to_numpy())
    else:
        meta, order = None, ["elo_logistic", "gbm", "m1_poisson"]
    Pev, _ = _fit_predict_bases(train, evalw, m1_inference, m1_kwargs)
    y = evalw["result"].to_numpy()
    for name in order:
        rows.append({"model": name, "cutoff": cut, "n_eval": len(evalw), **score_predictions(Pev[name], y)})
    if meta is not None:
        Pmeta = meta.predict_proba([Pev[k] for k in order])
        rows.append({"model": "meta", "cutoff": cut, "n_eval": len(evalw), **score_predictions(Pmeta, y)})
    return pd.DataFrame(rows)

def run_full(results_path: str, m1_window_years: int = 12) -> pd.DataFrame:
    feats = build_features(EloEngine().run(load_results(results_path)))
    cuts = wc_cutoffs()
    out = []
    for i, cut in enumerate(cuts):
        hi = cuts[i + 1] if i + 1 < len(cuts) else feats["date"].max() + pd.Timedelta(days=1)
        res = _fold(feats, cut, hi, m1_inference="advi",
                    m1_kwargs={"window_years": m1_window_years, "draws": 200})
        if res is not None:
            out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["model"])

def run_fold_smoke(results_path: str) -> pd.DataFrame:
    """Fast CI path: tiny fixture, M1 MAP, single synthetic fold (no WC filter)."""
    feats = build_features(EloEngine().run(load_results(results_path)))
    cut = feats["date"].quantile(0.5)
    hi = feats["date"].max() + pd.Timedelta(days=1)
    train = feats[feats["date"] < cut]
    evalw = feats[(feats["date"] >= cut) & (feats["date"] < hi)]
    P, _ = _fit_predict_bases(train, evalw, m1_inference="map")
    meta = __import__("wc2026.models.meta", fromlist=["MetaStacker"]).MetaStacker()
    order = ["elo_logistic", "gbm", "m1_poisson"]
    meta.fit([P[k] for k in order], evalw["result"].to_numpy())  # smoke only (in-sample)
    y = evalw["result"].to_numpy()
    rows = [{"model": k, "rps": score_predictions(P[k], y)["rps"]} for k in order]
    Pmeta = meta.predict_proba([P[k] for k in order])
    rows.append({"model": "meta", "rps": score_predictions(Pmeta, y)["rps"]})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    tbl = run_full(path)
    print(tbl.to_string(index=False))
    print("\nMean RPS by model:")
    print(tbl.groupby("model")["rps"].mean().sort_values().to_string())
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest tests/test_full_backtest_smoke.py -q`
Expected: PASS (1 passed). The tiny fixture has only 4 teams/8 matches, so numbers are
meaningless — this only proves the M1→meta orchestration composes.

- [ ] **Step 5: Run the FULL fast suite**

Run: `pytest -q -m "not slow"`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add scripts/full_backtest.py tests/test_full_backtest_smoke.py
git commit -m "feat(eval): full M1/M2/Elo/meta real-data backtest harness + smoke"
```

- [ ] **Step 7: HEAVY real run (manual / monitored — NOT a unit test)**

Run: `python scripts/full_backtest.py data/raw/results.csv`
Expected: a per-World-Cup comparison of `elo_logistic` / `gbm` / `m1_poisson` / `meta` with
RPS/Brier/log-loss/ECE, and a mean-RPS-by-model summary. This is the headline Phase-2 result.
**Capture the table.** Honest expectation: M1 (a goal-level model) and especially the `meta`
(which can blend M1's structure with Elo's linear signal) are the candidates to finally beat the
~0.202 Elo-logistic mean RPS. If the meta does not beat Elo, report it honestly — it is a real
finding about international-match predictability, and it informs Phase-3 model selection.

---

## Phase 2c+2d Done = Definition

- `pytest -q -m "not slow"` green across all test files (Phase 1 + 2a/2b + 2c/2d).
- `pytest -q -m slow` green (ADVI smoke).
- M1 fits via MAP (unit) and ADVI (real); exposes `sample_strengths()` for Phase-3.
- M3 de-vig + meta-learner unit-tested.
- `python scripts/full_backtest.py data/raw/results.csv` produces the M1/M2/Elo/meta real
  World Cup comparison table (headline result captured).

**Deferred to Phase 3:** the Monte Carlo tournament simulator (uses `M1.sample_strengths()` for
posterior-propagated, correlated draws over the real 48-team bracket) → win probabilities.
Phase 4: live updating + web dashboard. M3 stays inert until a real odds source is wired
(current 2026 outright odds for the live layer / benchmark).

---

## Self-Review

**Spec coverage (this slice):**
- Spec §4 M1 dynamic state-space hierarchical bivariate Poisson + Dixon-Coles + time-decay →
  Tasks 1, 2, 3 (random-walk dynamics = the state-space; DC correction = Task 1; hierarchical
  0-centered priors; "bivariate" realised as independent Poisson + DC correction, the standard
  tractable form — noted in the spec). ✔
- Spec §4 M3 odds de-vig → Task 5. ✔
- Spec §4 meta-learner stacking, walk-forward, no leakage → Task 6 + Task 7 (inner out-of-fold
  split for meta training). ✔
- Spec §5 posterior sampler for correlated simulation → `M1.sample_strengths()` (Task 3),
  consumed in Phase 3. ✔
- Spec §6 backtest comparison across M1/M2/M3/meta/baselines with RPS/calibration → Task 7. ✔

**Placeholder scan:** no TBD/TODO. Pure functions (`dc_tau`, `assign_periods`, `implied_probs`,
`MetaStacker`) are fully coded with exact-value tests. M1 is a complete reference implementation;
the tests define correctness and the plan instructs adjusting internals (not assertions) if MAP
needs help. Heavy runs (Task 3 Step 6, Task 7 Step 7) are explicitly marked manual/integration.

**Type consistency:** every model exposes `predict_proba(frame) -> (n,3)` in canonical
`[p_home,p_draw,p_away]` order; class-column reindexing to `[0,1,2]` is applied in GBM, meta, and
baselines. M1 reads `home_team/away_team/neutral/date` (+ scores for fit); `compare.predict_frame`
passes the full frame so heterogeneous models coexist. `MetaStacker.fit/predict_proba` take a
*list of base prob matrices* (distinct from the frame-based models) — this asymmetry is contained
in `full_backtest.py`, which assembles the base-prob list before calling the meta. `build_features`
additions (`home_team`, `away_team`, `home_score`, `away_score`, `neutral_flag`) are metadata, not
in `FEATURE_COLUMNS`, so the leakage guard and Phase-1/2 consumers are unaffected.
