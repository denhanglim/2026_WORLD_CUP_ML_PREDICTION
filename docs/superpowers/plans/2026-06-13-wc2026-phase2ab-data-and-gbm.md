# WC-2026 Predictor — Phase 2a+2b (Real Data + GBM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the real international-results dataset into the Phase-1 pipeline, enrich the feature set, run the first honest World Cup backtest (2010/2014/2018/2022) with the naive baselines, then add and validate the first *learned* model — a calibrated gradient-boosting classifier (M2) that must beat the Elo-logistic baseline on real World Cup matches.

**Architecture:** Builds directly on the committed Phase-1 package `wc2026`. Adds: a network data fetcher (injectable for tests), an optional FIFA-rankings point-in-time join, richer point-in-time features (rest days, rolling goals, FIFA-rank diff), a World-Cup-restricted evaluation filter in the backtest harness, a real-data baseline+model comparison report, and `models/gbm.py` (M2) built on scikit-learn's `HistGradientBoostingClassifier` with isotonic calibration. No new heavy dependencies — `HistGradientBoostingClassifier` is pure sklearn (already a dep), handles NaN natively, and avoids the macOS libomp requirement that XGBoost/LightGBM impose.

**Tech Stack:** Existing — Python 3.12 venv, pandas, numpy, scikit-learn, pytest. Network access (stdlib `urllib`) for the one-time real-data download.

---

## Conventions carried from Phase 1 (do not change)

- Outcome encoding: `result ∈ {0=HOME, 1=DRAW, 2=AWAY}`, probabilities `[p_home, p_draw, p_away]`.
- `Model` protocol: `fit(X: DataFrame, y: ndarray) -> self`, `predict_proba(X) -> (n,3)`.
- Existing committed APIs you will reuse (already on `main`/branch, real code):
  - `wc2026.data.results.load_results(path) -> DataFrame` (cols incl. `date, home_team, away_team, home_score, away_score, tournament, country, neutral, result, match_id`).
  - `wc2026.elo.engine.EloEngine().run(df) -> df + [home_elo_pre, away_elo_pre]`.
  - `wc2026.features.build.build_features(df, form_window=5, home_adv=100.0) -> feats` with `FEATURE_COLUMNS` and metadata cols `match_id, result`.
  - `wc2026.eval.backtest.walk_forward(feats, cutoffs, model_factory, features) -> DataFrame`.
  - `wc2026.models.baselines.UniformBaseline`, `EloLogisticBaseline`.

**Branch:** continue on `feature/phase1-foundation` is wrong — create a NEW branch `feature/phase2ab-data-gbm` off `feature/phase1-foundation` (Phase 2 depends on Phase 1 code that is not yet merged to main).

---

### Task 1: Real-data fetcher (injectable, network-isolated in tests)

**Files:**
- Create: `src/wc2026/data/fetch.py`
- Test: `tests/test_data_fetch.py`

- [ ] **Step 1: Write the failing test** (no real network — inject a fake fetcher returning the fixture bytes)

```python
# tests/test_data_fetch.py
from pathlib import Path
from wc2026.data.fetch import download_results, RESULTS_URL
from wc2026.data.results import load_results

def test_download_writes_file_and_is_loadable(tmp_path):
    fixture_bytes = Path("tests/fixtures/mini_results.csv").read_bytes()
    calls = {}
    def fake_fetch(url):
        calls["url"] = url
        return fixture_bytes
    dest = tmp_path / "raw" / "results.csv"
    out = download_results(str(dest), fetch=fake_fetch)
    assert Path(out).exists()
    assert calls["url"] == RESULTS_URL          # default URL used
    df = load_results(out)                       # downloaded file is loadable
    assert len(df) == 8

def test_download_respects_custom_url(tmp_path):
    seen = {}
    download_results(str(tmp_path / "r.csv"), url="http://x/y.csv",
                     fetch=lambda u: seen.setdefault("u", u) or b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n2020-01-01,A,B,1,0,Friendly,c,d,False\n")
    assert seen["u"] == "http://x/y.csv"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_fetch.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.data.fetch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/data/fetch.py
"""One-time real-data acquisition. Network call is injectable so tests stay offline."""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import urllib.request

# martj42/international_results — canonical free dataset (1872–present).
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (trusted URL)
        return resp.read()

def download_results(dest: str, url: str = RESULTS_URL, *,
                     fetch: Callable[[str], bytes] = _http_get) -> str:
    """Download the results CSV to `dest`. `fetch` is injectable for testing.
    Returns the path written."""
    data = fetch(url)
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_fetch.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Actually fetch the real data (integration, manual — not a unit test)**

Run:
```bash
python -c "from wc2026.data.fetch import download_results; print(download_results('data/raw/results.csv'))"
python -c "from wc2026.data.results import load_results; df=load_results('data/raw/results.csv'); print(len(df), 'matches', df['date'].min(), '->', df['date'].max())"
```
Expected: prints a path, then tens of thousands of matches spanning 1872 → 2026.
**If the URL 404s**, the default branch may be `main` not `master` — retry with
`url="https://raw.githubusercontent.com/martj42/international_results/main/results.csv"`, and
if that works, update `RESULTS_URL` in `fetch.py` accordingly (commit the fix). `data/raw/` is
gitignored, so the CSV itself is not committed.

- [ ] **Step 6: Commit**

```bash
git add src/wc2026/data/fetch.py tests/test_data_fetch.py
git commit -m "feat(data): injectable real-results fetcher (martj42 dataset)"
```

---

### Task 2: Optional FIFA-rankings point-in-time join

**Files:**
- Create: `src/wc2026/data/rankings.py`
- Test: `tests/test_data_rankings.py`
- Create fixture: `tests/fixtures/mini_rankings.csv`

**Design:** load a rankings table (`team, rank_date, rank`), then attach each match's home/away
FIFA rank *as of the most recent ranking on or before the match date* using `merge_asof`
(backward). The whole feature is optional: if no rankings file is supplied, the join is skipped
and downstream code imputes (XGBoost/HistGBM handle NaN natively).

- [ ] **Step 1: Create the fixture**

```csv
rank,country_full,country_abrv,total_points,rank_date
1,Alpha,ALP,1800,2018-01-15
2,Beta,BET,1700,2018-01-15
3,Gamma,GAM,1500,2018-01-15
4,Delta,DEL,1400,2018-01-15
1,Alpha,ALP,1820,2018-05-15
3,Beta,BET,1650,2018-05-15
2,Gamma,GAM,1660,2018-05-15
4,Delta,DEL,1390,2018-05-15
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_data_rankings.py
import pandas as pd
from wc2026.data.results import load_results
from wc2026.data.rankings import load_rankings, attach_fifa_rank

def test_load_rankings_normalises():
    r = load_rankings("tests/fixtures/mini_rankings.csv")
    assert {"team", "rank_date", "rank"}.issubset(r.columns)
    assert pd.api.types.is_datetime64_any_dtype(r["rank_date"])

def test_attach_is_point_in_time_backward():
    df = load_results("tests/fixtures/mini_results.csv")
    r = load_rankings("tests/fixtures/mini_rankings.csv")
    out = attach_fifa_rank(df, r)
    assert "home_fifa_rank" in out.columns and "away_fifa_rank" in out.columns
    # Alpha vs Beta on 2018-03-01 -> must use the 2018-01-15 ranking (Alpha=1, Beta=2),
    # NOT the future 2018-05-15 one.
    row = out[(out.home_team == "Alpha") & (out.away_team == "Beta")].iloc[0]
    assert row["home_fifa_rank"] == 1 and row["away_fifa_rank"] == 2
    # A match after 2018-05-15 uses the newer ranking: Gamma vs Beta 2018-08-01 -> Beta=3
    row2 = out[(out.home_team == "Gamma") & (out.away_team == "Beta")].iloc[0]
    assert row2["away_fifa_rank"] == 3

def test_attach_missing_team_yields_nan():
    df = load_results("tests/fixtures/mini_results.csv")
    r = load_rankings("tests/fixtures/mini_rankings.csv")
    r = r[r.team != "Delta"]  # drop Delta's rankings
    out = attach_fifa_rank(df, r)
    assert out[(out.home_team == "Alpha") & (out.away_team == "Delta")]["away_fifa_rank"].isna().all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_data_rankings.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.data.rankings'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/wc2026/data/rankings.py
"""Optional FIFA-rankings loader + point-in-time (backward) join onto matches."""
from __future__ import annotations
import pandas as pd

def load_rankings(path: str) -> pd.DataFrame:
    """Normalise a FIFA rankings CSV to columns: team, rank_date, rank."""
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "team": raw["country_full"],
        "rank_date": pd.to_datetime(raw["rank_date"]),
        "rank": raw["rank"].astype(int),
    })
    return out.sort_values("rank_date", kind="stable").reset_index(drop=True)

def _asof_rank(matches: pd.DataFrame, rankings: pd.DataFrame, team_col: str) -> pd.Series:
    left = matches[["date", team_col]].copy()
    left["_row"] = range(len(left))
    left = left.sort_values("date", kind="stable")
    merged = pd.merge_asof(
        left, rankings.rename(columns={"team": team_col, "rank_date": "date"}),
        on="date", by=team_col, direction="backward",
    )
    return merged.sort_values("_row")["rank"].reset_index(drop=True)

def attach_fifa_rank(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Add home_fifa_rank / away_fifa_rank using the latest ranking on/before each match date.
    Teams with no prior ranking get NaN."""
    out = matches.copy()
    out["home_fifa_rank"] = _asof_rank(matches, rankings, "home_team").values
    out["away_fifa_rank"] = _asof_rank(matches, rankings, "away_team").values
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_data_rankings.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/wc2026/data/rankings.py tests/test_data_rankings.py tests/fixtures/mini_rankings.csv
git commit -m "feat(data): optional point-in-time FIFA-ranking join"
```

---

### Task 3: Enrich point-in-time features (rest days, rolling goals, rank diff)

**Files:**
- Modify: `src/wc2026/features/build.py`
- Test: `tests/test_features_enriched.py`

**Design:** extend the single chronological pass to also emit, per match (point-in-time):
`home_rest_days`, `away_rest_days`, `rest_days_diff`; `home_gf_avg`, `home_ga_avg`,
`away_gf_avg`, `away_ga_avg` (rolling means over the last `form_window` matches); and, if the
input frame carries `home_fifa_rank`/`away_fifa_rank` (from Task 2), `fifa_rank_diff`
(= away_rank − home_rank, so larger = home stronger; NaN-safe). Also carry `date` and
`tournament` through as metadata columns (needed by the WC-filtered backtest). `FEATURE_COLUMNS`
remains *model features only* (date/tournament are metadata, not features).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_enriched.py
import numpy as np
import pandas as pd
from wc2026.data.results import load_results
from wc2026.data.rankings import load_rankings, attach_fifa_rank
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features, FEATURE_COLUMNS

FIX = "tests/fixtures/mini_results.csv"

def test_new_feature_columns_present():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df)
    for col in ["home_rest_days", "away_rest_days", "rest_days_diff",
                "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]:
        assert col in feats.columns
        assert col in FEATURE_COLUMNS
    # metadata carried through
    assert "date" in feats.columns and "tournament" in feats.columns

def test_rest_days_first_match_is_nan_then_positive():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df)
    # Alpha's first match (2018-01-01) -> no prior match -> NaN rest
    first_alpha = feats.iloc[0]
    assert np.isnan(first_alpha["home_rest_days"])
    # Alpha vs Beta 2018-03-01: Alpha last played 2018-01-01 -> 59 days
    ab = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert ab["home_rest_days"] == 59.0

def test_rolling_goals_point_in_time():
    df = EloEngine().run(load_results(FIX))
    feats = build_features(df)
    # Alpha vs Beta 2018-03-01: Alpha has one prior match (3-0 vs Delta) -> gf_avg=3, ga_avg=0
    ab = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert ab["home_gf_avg"] == 3.0 and ab["home_ga_avg"] == 0.0

def test_fifa_rank_diff_when_available():
    df = load_results(FIX)
    df = attach_fifa_rank(df, load_rankings("tests/fixtures/mini_rankings.csv"))
    df = EloEngine().run(df)
    feats = build_features(df)
    assert "fifa_rank_diff" in feats.columns
    # Alpha(1) vs Beta(2) on 2018-03-01 -> diff = away_rank - home_rank = 2 - 1 = 1
    ab = feats[(df.home_team == "Alpha") & (df.away_team == "Beta")].iloc[0]
    assert ab["fifa_rank_diff"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features_enriched.py -q`
Expected: FAIL — KeyError / missing columns.

- [ ] **Step 3: Rewrite `build_features`** (full replacement of the file)

```python
# src/wc2026/features/build.py
"""Point-in-time match features. Single chronological pass => no future leakage."""
from __future__ import annotations
from collections import defaultdict, deque
from typing import Deque, Dict
import numpy as np
import pandas as pd

# Model features only (NOT metadata like date/tournament).
FEATURE_COLUMNS = [
    "elo_diff", "home_elo_pre", "away_elo_pre",
    "home_form_pts", "away_form_pts",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "neutral",
]

def _points(team: str, home: str, away: str, result: int) -> float:
    if result == 1:
        return 1.0
    if (result == 0 and team == home) or (result == 2 and team == away):
        return 3.0
    return 0.0

def _avg(dq: Deque[float]) -> float:
    return float(np.mean(dq)) if len(dq) else 0.0

def build_features(df: pd.DataFrame, form_window: int = 5,
                   home_adv: float = 100.0) -> pd.DataFrame:
    """Feature frame aligned 1:1 with `df` (requires `home_elo_pre`/`away_elo_pre`).
    All rolling/lagged features use prior matches only. If `home_fifa_rank`/`away_fifa_rank`
    columns exist, adds `fifa_rank_diff`."""
    form: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    gf: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    ga: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=form_window))
    last_date: Dict[str, pd.Timestamp] = {}
    has_rank = "home_fifa_rank" in df.columns and "away_fifa_rank" in df.columns
    rows = []
    for r in df.itertuples(index=False):
        not_neutral = 0.0 if r.neutral else 1.0
        h_rest = (r.date - last_date[r.home_team]).days if r.home_team in last_date else np.nan
        a_rest = (r.date - last_date[r.away_team]).days if r.away_team in last_date else np.nan
        rec = {
            "elo_diff": (r.home_elo_pre + home_adv * not_neutral) - r.away_elo_pre,
            "home_elo_pre": r.home_elo_pre,
            "away_elo_pre": r.away_elo_pre,
            "home_form_pts": sum(form[r.home_team]),
            "away_form_pts": sum(form[r.away_team]),
            "home_rest_days": float(h_rest),
            "away_rest_days": float(a_rest),
            "rest_days_diff": float(h_rest - a_rest) if not (np.isnan(h_rest) or np.isnan(a_rest)) else np.nan,
            "home_gf_avg": _avg(gf[r.home_team]),
            "home_ga_avg": _avg(ga[r.home_team]),
            "away_gf_avg": _avg(gf[r.away_team]),
            "away_ga_avg": _avg(ga[r.away_team]),
            "neutral": bool(r.neutral),
        }
        if has_rank:
            hr, ar = getattr(r, "home_fifa_rank"), getattr(r, "away_fifa_rank")
            rec["fifa_rank_diff"] = (float(ar) - float(hr)) if pd.notna(hr) and pd.notna(ar) else np.nan
        rows.append(rec)
        # update AFTER recording (point-in-time)
        form[r.home_team].append(_points(r.home_team, r.home_team, r.away_team, r.result))
        form[r.away_team].append(_points(r.away_team, r.home_team, r.away_team, r.result))
        gf[r.home_team].append(float(r.home_score)); ga[r.home_team].append(float(r.away_score))
        gf[r.away_team].append(float(r.away_score)); ga[r.away_team].append(float(r.home_score))
        last_date[r.home_team] = r.date
        last_date[r.away_team] = r.date
    feats = pd.DataFrame(rows, index=df.index)
    feats["match_id"] = df["match_id"].values
    feats["result"] = df["result"].values
    feats["date"] = df["date"].values
    feats["tournament"] = df["tournament"].values
    if has_rank and "fifa_rank_diff" not in FEATURE_COLUMNS:
        # keep FEATURE_COLUMNS stable across calls; expose via the returned frame only
        pass
    return feats
```

> Note: `fifa_rank_diff` is intentionally NOT added to the module-level `FEATURE_COLUMNS`
> (it is only present when rankings were joined). Models that want it pass it explicitly in
> their `features=[...]` list. The enriched test asserts the *column exists in the frame*, which
> it does. **Adjust** `test_new_feature_columns_present` to not require `fifa_rank_diff` in
> `FEATURE_COLUMNS` (it only checks the always-present new columns there — already the case).

- [ ] **Step 4: Run the enriched test + the Phase-1 tests to confirm no regression**

Run: `pytest tests/test_features_enriched.py tests/test_features_build.py tests/test_features_leakage.py -q`
Expected: PASS. If the Phase-1 leakage test fails because it now compares more columns, see Task 4.

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/features/build.py tests/test_features_enriched.py
git commit -m "feat(features): rest days, rolling goals, optional FIFA-rank diff (point-in-time)"
```

---

### Task 4: Extend the leakage guard to the new features

**Files:**
- Modify: `tests/test_features_leakage.py`

**Why:** the safety property must cover the new lagged features too. Update the guard to compare
all numeric `FEATURE_COLUMNS` under truncation.

- [ ] **Step 1: Replace the leakage test body**

```python
# tests/test_features_leakage.py
import numpy as np
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features, FEATURE_COLUMNS

FIX = "tests/fixtures/mini_results.csv"

def test_features_are_point_in_time():
    df_full = load_results(FIX)
    k = 4
    df_trunc = df_full.iloc[:k].reset_index(drop=True)
    full = build_features(EloEngine().run(df_full)).iloc[:k].reset_index(drop=True)
    trunc = build_features(EloEngine().run(df_trunc)).reset_index(drop=True)
    cols = [c for c in FEATURE_COLUMNS if c != "neutral"]
    # NaN-aware comparison (rest_days is NaN for first appearances)
    pd.testing.assert_frame_equal(full[cols], trunc[cols], check_dtype=False)
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_features_leakage.py -q`
Expected: PASS (1 passed). A failure here means a new feature leaks future data — fix
`build_features`, do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_features_leakage.py
git commit -m "test(features): extend leakage guard to enriched features"
```

---

### Task 5: World-Cup-restricted backtest + real baseline report

**Files:**
- Modify: `src/wc2026/eval/backtest.py` (add optional `eval_filter`)
- Create: `src/wc2026/tournaments.py`
- Create: `scripts/baseline_report.py`
- Test: `tests/test_backtest_filter.py`, `tests/test_tournaments.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tournaments.py
import pandas as pd
from wc2026.tournaments import wc_cutoffs, is_world_cup

def test_wc_cutoffs_are_sorted_timestamps():
    cuts = wc_cutoffs()
    assert all(isinstance(c, pd.Timestamp) for c in cuts)
    assert cuts == sorted(cuts)
    assert len(cuts) == 4

def test_is_world_cup_mask():
    df = pd.DataFrame({"tournament": ["FIFA World Cup", "Friendly", "FIFA World Cup qualification"]})
    assert is_world_cup(df).tolist() == [True, False, False]
```

```python
# tests/test_backtest_filter.py
import numpy as np
import pandas as pd
from wc2026.eval.backtest import walk_forward
from wc2026.models.baselines import UniformBaseline

def test_eval_filter_restricts_eval_rows():
    n = 400
    dates = pd.date_range("2016-01-01", periods=n, freq="W")
    df = pd.DataFrame({
        "date": dates,
        "elo_diff": np.zeros(n),
        "result": np.zeros(n, int),
        "tournament": ["Friendly"] * n,
    })
    # mark 10 rows after the cutoff as World Cup
    df.loc[df.index[-10:], "tournament"] = "FIFA World Cup"
    cut = pd.Timestamp(dates[-20])
    res = walk_forward(df, [cut], lambda: UniformBaseline(), features=["elo_diff"],
                       eval_filter=lambda d: d["tournament"] == "FIFA World Cup")
    assert int(res.iloc[0]["n_eval"]) == 10
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_tournaments.py tests/test_backtest_filter.py -q`
Expected: FAIL — missing module `wc2026.tournaments` and `walk_forward()` has no `eval_filter`.

- [ ] **Step 3: Add `eval_filter` to the harness** (edit `walk_forward` signature + eval slice)

```python
# in src/wc2026/eval/backtest.py — replace the function signature and the eval-window slice
def walk_forward(feats, cutoffs, model_factory, features, eval_filter=None):
    """For each cutoff: train on date < cutoff, evaluate on [cutoff, next_cutoff).
    If `eval_filter` is given (callable df->bool mask), the eval set is further restricted
    (e.g. to World Cup matches only)."""
    cutoffs = list(cutoffs)
    rows = []
    for i, cut in enumerate(cutoffs):
        hi = cutoffs[i + 1] if i + 1 < len(cutoffs) else feats["date"].max() + pd.Timedelta(days=1)
        train = feats[feats["date"] < cut]
        evalw = feats[(feats["date"] >= cut) & (feats["date"] < hi)]
        if eval_filter is not None:
            evalw = evalw[eval_filter(evalw)]
        if len(train) == 0 or len(evalw) == 0:
            continue
        model = model_factory()
        model.fit(train[features], train["result"].to_numpy())
        P = model.predict_proba(evalw[features])
        y = evalw["result"].to_numpy()
        rows.append({
            "fold": i, "cutoff": cut, "n_train": len(train), "n_eval": len(evalw),
            "rps": float(np.mean(rps(P, y))),
            "brier": float(np.mean(brier(P, y))),
            "log_loss": float(np.mean(log_loss(P, y))),
            "ece_home": float(ece(P[:, 0], (y == 0).astype(int))),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Create `tournaments.py`**

```python
# src/wc2026/tournaments.py
"""World Cup edition dates (group-stage opener) used as walk-forward cutoffs."""
from __future__ import annotations
import pandas as pd

WORLD_CUP_OPENERS = {2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20"}

def wc_cutoffs() -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in WORLD_CUP_OPENERS.values()]

def is_world_cup(df: pd.DataFrame) -> pd.Series:
    return df["tournament"] == "FIFA World Cup"
```

- [ ] **Step 5: Run the two tests**

Run: `pytest tests/test_tournaments.py tests/test_backtest_filter.py -q`
Expected: PASS (3 passed total).

- [ ] **Step 6: Create the real-data baseline report script**

```python
# scripts/baseline_report.py
"""Real-data World Cup backtest for the naive baselines.
Usage: python scripts/baseline_report.py [data/raw/results.csv]"""
from __future__ import annotations
import sys
import pandas as pd
from wc2026.data.results import load_results
from wc2026.elo.engine import EloEngine
from wc2026.features.build import build_features
from wc2026.eval.backtest import walk_forward
from wc2026.tournaments import wc_cutoffs, is_world_cup
from wc2026.models.baselines import UniformBaseline, EloLogisticBaseline

BASELINE_FEATURES = ["elo_diff"]

def build(results_path: str) -> pd.DataFrame:
    df = EloEngine().run(load_results(results_path))
    return build_features(df)

def report(results_path: str) -> pd.DataFrame:
    feats = build(results_path)
    cuts = wc_cutoffs()
    out = []
    for name, factory in [("uniform", lambda: UniformBaseline()),
                          ("elo_logistic", lambda: EloLogisticBaseline())]:
        res = walk_forward(feats, cuts, factory, features=BASELINE_FEATURES,
                           eval_filter=is_world_cup)
        res.insert(0, "model", name)
        out.append(res)
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    tbl = report(path)
    print(tbl.to_string(index=False))
    print("\nMean RPS by model:")
    print(tbl.groupby("model")["rps"].mean().to_string())
```

- [ ] **Step 7: Run it on the REAL data (integration)**

Run: `python scripts/baseline_report.py data/raw/results.csv`
Expected: a per-World-Cup table (folds 0–3) for `uniform` and `elo_logistic`, and a mean-RPS
summary where **`elo_logistic` mean RPS < `uniform` mean RPS** (Elo beats coin-flip). Paste this
table into your final report. Typical World Cup match RPS is roughly ~0.19–0.21 for a decent
model and ~0.23 for uniform — exact numbers depend on the data snapshot.

- [ ] **Step 8: Commit**

```bash
git add src/wc2026/eval/backtest.py src/wc2026/tournaments.py scripts/baseline_report.py \
        tests/test_tournaments.py tests/test_backtest_filter.py
git commit -m "feat(eval): WC-restricted backtest filter + real-data baseline report"
```

---

### Task 6: M2 — calibrated gradient-boosting model

**Files:**
- Create: `src/wc2026/models/gbm.py`
- Test: `tests/test_models_gbm.py`

**Design:** `GBMModel` wraps sklearn `HistGradientBoostingClassifier` (NaN-native, no libomp).
Optional isotonic calibration via `CalibratedClassifierCV` when there are enough samples per
class for the internal CV; otherwise fits the raw classifier. Conforms to the `Model` protocol
and reindexes class columns to canonical `[0,1,2]` order.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_gbm.py
import numpy as np
import pandas as pd
from wc2026.models.gbm import GBMModel
from wc2026.models.baselines import EloLogisticBaseline
from wc2026.eval.scoring import rps

def _nonlinear(n=2000, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    # outcome depends on INTERACTION a*b (linear model on a,b alone can't capture it)
    score = a * b
    p_home = 1 / (1 + np.exp(-2 * score))
    u = rng.uniform(0, 1, n)
    y = np.where(u < p_home * 0.8, 0, np.where(u < p_home * 0.8 + 0.18, 1, 2))
    X = pd.DataFrame({"a": a, "b": b, "elo_diff": a * 200})
    return X, y

def test_gbm_probs_valid():
    X, y = _nonlinear()
    P = GBMModel(features=["a", "b"]).fit(X, y).predict_proba(X)
    assert P.shape == (len(X), 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= 0).all()

def test_gbm_handles_nan_features():
    X, y = _nonlinear()
    X.loc[X.index[:50], "a"] = np.nan       # inject missing values
    P = GBMModel(features=["a", "b"]).fit(X, y).predict_proba(X)
    assert np.isfinite(P).all()

def test_gbm_beats_linear_on_interaction_data():
    X, y = _nonlinear()
    split = 1500
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    gbm = GBMModel(features=["a", "b"]).fit(Xtr, ytr)
    lin = EloLogisticBaseline(feature="elo_diff").fit(Xtr, ytr)
    rps_gbm = np.mean(rps(gbm.predict_proba(Xte), yte))
    rps_lin = np.mean(rps(lin.predict_proba(Xte), yte))
    assert rps_gbm < rps_lin   # GBM captures the interaction the linear baseline cannot
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_models_gbm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'wc2026.models.gbm'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/wc2026/models/gbm.py
"""M2 — gradient-boosting match model (sklearn HistGradientBoosting; NaN-native, no libomp)."""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

def _enough_for_cv(y: np.ndarray, cv: int = 3) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return len(counts) >= 2 and counts.min() >= cv

class GBMModel:
    def __init__(self, features: List[str], calibrate: bool = True,
                 random_state: int = 0, cv: int = 3):
        self.features = features
        self.calibrate = calibrate
        self.random_state = random_state
        self.cv = cv
        self._clf = None

    def fit(self, X: pd.DataFrame, y) -> "GBMModel":
        y = np.asarray(y)
        base = HistGradientBoostingClassifier(random_state=self.random_state)
        if self.calibrate and _enough_for_cv(y, self.cv):
            self._clf = CalibratedClassifierCV(base, method="isotonic", cv=self.cv)
        else:
            self._clf = base
        self._clf.fit(X[self.features].to_numpy(dtype=float), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._clf.predict_proba(X[self.features].to_numpy(dtype=float))
        out = np.zeros((len(X), 3))
        for col, cls in enumerate(self._clf.classes_):
            out[:, int(cls)] = raw[:, col]
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_models_gbm.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/wc2026/models/gbm.py tests/test_models_gbm.py
git commit -m "feat(models): M2 calibrated gradient-boosting match model"
```

---

### Task 7: Add M2 to the real-data comparison report

**Files:**
- Modify: `scripts/baseline_report.py` → rename concept to a model-comparison report (keep file name; add GBM)
- Test: `tests/test_report_includes_gbm.py`

**Design:** extend the report to also run the GBM over a richer feature list, so the output table
compares uniform vs elo_logistic vs gbm on the same World Cup folds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_includes_gbm.py
from scripts.baseline_report import report_models

def test_report_includes_three_models():
    tbl = report_models("tests/fixtures/mini_results.csv",
                        cutoffs_override=["2018-05-01"])
    assert set(tbl["model"].unique()) >= {"uniform", "elo_logistic", "gbm"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_report_includes_gbm.py -q`
Expected: FAIL — `ImportError: cannot import name 'report_models'`

- [ ] **Step 3: Extend `scripts/baseline_report.py`** (add `report_models`, keep `report`)

```python
# append to scripts/baseline_report.py (above the __main__ block)
GBM_FEATURES = ["elo_diff", "home_form_pts", "away_form_pts", "rest_days_diff",
                "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg"]

def report_models(results_path: str, cutoffs_override=None) -> pd.DataFrame:
    from wc2026.models.gbm import GBMModel
    feats = build(results_path)
    cuts = [pd.Timestamp(c) for c in cutoffs_override] if cutoffs_override else wc_cutoffs()
    # On the tiny fixture there may be no WC matches in-window; fall back to no filter there.
    use_filter = is_world_cup if cutoffs_override is None else None
    specs = [
        ("uniform", lambda: UniformBaseline(), BASELINE_FEATURES),
        ("elo_logistic", lambda: EloLogisticBaseline(), BASELINE_FEATURES),
        ("gbm", lambda: GBMModel(features=GBM_FEATURES), GBM_FEATURES),
    ]
    out = []
    for name, factory, feature_list in specs:
        res = walk_forward(feats, cuts, factory, features=feature_list, eval_filter=use_filter)
        if len(res):
            res.insert(0, "model", name)
            out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["model"])
```

And update the `__main__` block to call `report_models`:
```python
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    tbl = report_models(path)
    print(tbl.to_string(index=False))
    print("\nMean RPS by model:")
    print(tbl.groupby("model")["rps"].mean().to_string())
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_report_includes_gbm.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the FULL suite**

Run: `pytest -q`
Expected: all tests green (Phase 1 + Phase 2a/2b).

- [ ] **Step 6: Run the real comparison + capture numbers**

Run: `python scripts/baseline_report.py data/raw/results.csv`
Expected: a table comparing uniform / elo_logistic / gbm across the 4 World Cups, plus mean RPS.
**Capture this output** — it is the first real model comparison and goes in the report. The bar:
`gbm` mean RPS should be ≤ `elo_logistic` mean RPS (if it isn't, note it honestly — on sparse WC
data a richer model can overfit; that is a real finding, not a failure to hide).

- [ ] **Step 7: Commit**

```bash
git add scripts/baseline_report.py tests/test_report_includes_gbm.py
git commit -m "feat(eval): model comparison report incl. GBM on real World Cup backtest"
```

---

## Phase 2a+2b Done = Definition

- `pytest -q` green across all test files (Phase 1 + new).
- Real martj42 dataset downloaded to `data/raw/` (gitignored) and loadable.
- `python scripts/baseline_report.py data/raw/results.csv` prints a real World-Cup backtest
  comparing uniform / elo_logistic / gbm with RPS / Brier / log-loss / ECE per edition.
- Leakage guard still passes over the enriched feature set.
- Elo-logistic beats uniform on real WC RPS; GBM result reported honestly vs elo_logistic.

**Deferred to the next plan (Phase 2c+2d):** M1 dynamic state-space hierarchical bivariate
Poisson (PyMC), M3 odds ingestion, the meta-learner stacker, and the full M1/M2/M3/meta
comparison. (Phase 3 = Monte Carlo tournament sim; Phase 4 = live + web.)

---

## Self-Review

**Spec coverage (this slice):**
- Spec §2 real data (martj42) + FIFA rankings → Tasks 1, 2. ✔
- Spec §3 richer point-in-time features → Task 3, leakage-guarded Task 4. ✔
- Spec §6 backtest on WC2010–22 with RPS/calibration → Tasks 5, 7. ✔
- Spec §4 M2 gradient boosting (calibrated) → Task 6. ✔
- Spec §4 M1/M3/meta → explicitly deferred to Phase 2c+2d. ✔ (by design)

**Placeholder scan:** none — every code step is complete, runnable code. The two integration
steps that hit the network/real data (Task 1 Step 5, Task 5 Step 7, Task 7 Step 6) are clearly
marked as manual integration runs, not unit tests, with explicit expected output and a documented
URL-branch fallback.

**Type consistency:** `Model.fit/predict_proba` honored by `GBMModel` (matches `UniformBaseline`
/`EloLogisticBaseline`). `walk_forward`'s new `eval_filter` is keyword-only-optional →
backward-compatible with every Phase-1 call. `build_features` adds columns but does not remove
any → Phase-1 consumers unaffected; `FEATURE_COLUMNS` extended (model features), `date`/
`tournament` carried as metadata. Canonical `[0,1,2]` / `[p_home,p_draw,p_away]` order preserved
across `gbm.py`, baselines, scoring.
