# Design Spec — `worldcup-2026`: FIFA World Cup 2026 Winner Predictor

- **Date:** 2026-06-13
- **Author:** Dennis Limbu (with Claude Code)
- **Status:** Approved design — ready for implementation planning
- **Repo:** `~/Developer/worldcup-2026` (off Google Drive, private git)

---

## 0. Thesis & success criteria

A single-elimination tournament is **high variance**. Even the true strongest team rarely
exceeds ~20–25% probability of lifting the trophy, and bookmaker markets are extremely hard
to beat on raw point accuracy. This project does **not** claim to "know" the winner.

Success is defined by four properties, not by picking the champion:

1. **Calibration** — when the model says 18%, that class of events happens ~18% of the time
   (measured by reliability diagrams + Expected Calibration Error).
2. **Uncertainty-awareness** — team-strength uncertainty is *propagated* into the tournament
   simulation, not collapsed to point estimates.
3. **Interpretability** — latent attack/defense strengths (and their posteriors) are
   inspectable; we can explain *why* a team is favoured.
4. **Liveness** — title probabilities update as real 2026 results arrive, producing a
   title-odds timeline.

**Quantitative bar (hard):** on a walk-forward backtest (WC 2010/2014/2018/2022) the
meta-model must achieve a Ranked Probability Score (RPS) **strictly better than naive
baselines** (Elo-only, uniform), with calibration ECE on backtest match predictions ≤ ~0.05.

**Quantitative bar (stretch, data-permitting):** where de-vigged historical bookmaker odds
are obtainable for free, the meta-model's RPS should be **at least competitive** with them.
This is a stretch goal, not a gate — free historical *international* match odds are sparse, so
the odds benchmark may only be available for current/outright 2026 markets, not the full
backtest.

### The 48-team novelty (load-bearing honesty)

2026 is the **first 48-team World Cup**: 12 groups of 4, the 8 best third-placed teams join
the 24 group qualifiers → 32 teams into a Round-of-32 knockout. There is **zero historical
precedent** for this structure. Therefore:

- The **match-outcome model** trains on ~150 years of international results.
- The **tournament structure** (group format, tiebreaks, R32 bracket mapping) is **encoded
  from FIFA's published 2026 rules**, not learned.

This boundary is explicit everywhere: we never claim to have backtested the bracket itself,
only the match model that feeds it.

---

## 1. Architecture

Five isolated, independently-testable layers feeding a simulation engine, then three outputs:

```
DATA → FEATURES → [ M1 Bayes | M2 GBM | M3 Odds ] → META-LEARNER → MONTE CARLO SIM → { EVAL, LIVE, WEB }
```

Interface contracts between layers:
- **DATA → FEATURES:** tidy match dataframe (one row per historical match, point-in-time safe).
- **FEATURES → MODELS:** feature matrix `X` + targets (result W/D/L, goals home/away).
- **MODELS → META:** each model exposes `predict(matchup, date) → predictive distribution`
  over scorelines / {W,D,L}.
- **META → SIM:** a single calibrated `match_distribution(team_a, team_b, neutral, date)` plus,
  for M1, a sampler that draws *team-strength posterior samples* (for correlated simulation).
- **SIM → outputs:** arrays of simulated tournament outcomes → aggregated probabilities.

Each unit can be swapped or re-run without touching the others.

---

## 2. Data layer (free / reproducible only)

Decision: **free public data + free-tier odds, degrade gracefully.** Prefer data we can
*derive ourselves* over scraped data. Everything cached to disk with a snapshot date.

| Source | Use | Notes |
|---|---|---|
| Kaggle `martj42/international-football-results` (1872–2026, ~45k matches) | The spine: scores, neutral-venue flag, tournament type | Free CSV |
| **Computed Elo** (derived from results) | Feature + naive baseline | Importance-weighted K, margin-of-victory, home advantage. Reproducible, no scraping |
| FIFA rankings (free historical Kaggle dataset) | Feature | Point-in-time joined |
| Squad market values (free Transfermarkt snapshots) | Feature | **Best-effort**; degrades to absent if unavailable. No hard dependency (ToS-brittle) |
| Market odds — current 2026 outright + free-tier match-odds API (e.g. the-odds-api, 500 req/mo) | **M3** benchmark + feature | De-vigged → implied probs. Falls back to outright-only if free tier exhausted |
| 2026 fixtures/draw + free live-score API | SIM bracket + LIVE layer | Public fixtures; live scores for updating |

**Reproducibility:** all raw pulls cached under `data/raw/<source>/<snapshot-date>/`; a
`data/build.py` materialises the tidy match table deterministically from cache.

---

## 3. Feature layer

Per match at time `t` (teams A vs B), constructed **strictly point-in-time** (only data before
`t` — no leakage):

- Elo difference (our computed Elo)
- FIFA-rank difference
- Time-weighted recent form (points + goals over last N, opponent-strength-adjusted)
- Rolling goals for/against
- Rest days since last match
- Host-nation / neutral-venue / venue / altitude flags
- Confederation, intra/inter-confederation flag
- Squad-value ratio + average age (if available)
- Head-to-head history

Output: `features/build.py → X` (feature matrix) with a leakage-guard test asserting every
feature for match `t` uses only rows with date `< t`.

---

## 4. Model layer

### M1 — Bayesian hierarchical **dynamic** bivariate Poisson (statistics flagship)

Latent per-team attack `αᵢ(t)` and defense `βᵢ(t)`. Goals:

```
λ_scored = exp( μ + home_adv·[venue ≠ neutral] + α_attacker(t) − β_defender(t) )
```

Four design elements:

1. **Dynamic state-space strengths** — `αᵢ(t) = αᵢ(t−1) + εᵢ`, `εᵢ ~ Normal(0, σ_evo)`
   (random walk). Captures rising/declining teams and squad turnover. *This is the chosen M1
   target (over static + time-decay).*
2. **Dixon–Coles low-score correction** (τ) — corrects the false independence assumption on
   0-0/1-0/0-1/1-1 scorelines.
3. **Hierarchical partial pooling** — `αᵢ, βᵢ ~ Normal(0, σ)` shrinks small-sample minnows
   toward the global mean, preventing wild estimates from teams with few matches.
4. **Neutral-venue handling** — home advantage zeroed for neutral games (most WC matches),
   which matters greatly for international football.

Fit in **PyMC** (fallback **cmdstanpy**) → **full posterior** over `α, β, μ, home_adv, σ_evo`.
Exposes both a point predictive distribution and a **posterior sampler** for correlated
simulation (§5).

> Implementation de-risking note (for the plan, not a scope change): a static + time-decay
> variant is built first internally to validate the full pipeline end-to-end, then upgraded
> to the dynamic state-space form. The **spec target is dynamic.**

### M2 — Gradient boosting (XGBoost / LightGBM)

On the §3 feature matrix. Predicts W/D/L (multiclass) and goals (twin Poisson regressors).
Captures nonlinear interactions Poisson/Elo miss (form × rest × squad value). **Isotonic
calibration** applied on a held-out fold (GBMs are miscalibrated out of the box).

### M3 — Market odds

De-vigged bookmaker implied probabilities. The benchmark every model must beat-or-match, and
a feature into the meta-learner.

### Meta-learner (stacking)

Combines M1 + M2 (+ M3) via **walk-forward-trained** logistic / Dirichlet regression (or
RPS-weighted Bayesian model averaging). Trained **only on out-of-sample base predictions** to
avoid leakage. Output: one calibrated predictive distribution per matchup.

---

## 5. Simulation layer (methodological crux)

Encode the **actual 2026 bracket**:

- **Group stage:** 12 groups of 4, round-robin. Full FIFA tiebreak chain: points → goal
  difference → goals for → head-to-head (points, GD, GF among tied) → fair-play → drawing of
  lots.
- **Qualification:** top 2 of each group (24) + **8 best third-placed teams** → 32.
- **Knockout:** FIFA's predefined R32 mapping → R32 → R16 → QF → SF → Final, with extra time
  + a **penalty-shootout model** (skill-adjusted, near-coinflip).

**The PhD move — correlated simulation via posterior propagation:** for each of **N ≥ 50,000**
simulations, first **draw a single posterior sample of all team strengths from M1**, then
simulate the *entire* tournament under that draw. This makes match outcomes **correlated**
(if a team is strong in a draw, it is strong in *every* match that draw), capturing tail
behaviour that the naive "sample each match independently from fixed probabilities" approach
systematically underestimates.

Outputs: P(win title), P(reach Final / SF / QF / R16), per-group qualification probabilities,
expected finish, and "path difficulty" per team.

---

## 6. Evaluation layer (rigor)

- **Walk-forward backtest** on WC **2010 / 2014 / 2018 / 2022** — only pre-tournament data
  available at each cutoff. (Euros optionally added for more folds.)
- **Proper scoring rules:** Ranked Probability Score (**RPS** — ordinal-aware, the football
  standard) as primary; log-loss + Brier secondary.
- **Comparison set:** meta vs M1 vs M2 vs M3(odds) vs naive (Elo-only, uniform).
- **Calibration:** reliability diagrams + ECE on backtest match predictions.
- **Tournament-level:** report how the pre-tournament title distribution matched reality
  across the 4 backtested WCs — **stated honestly as n=4, high variance.**

---

## 7. Live-updating layer

After each 2026 matchday: ingest real results → update M1 posterior + recompute Elo/form →
re-run Monte Carlo on the **remaining** bracket conditioned on actual standings → emit an
updated title-odds **timeline**. Refresh can be scheduled during the tournament window
(June 11 – July 19, 2026).

---

## 8. Deliverables & repo layout

```
~/Developer/worldcup-2026/
  data/        # loaders, cache, tidy match-table builder
  features/    # point-in-time feature construction + leakage guard
  models/      # m1_bayes/ m2_gbm/ m3_odds/ meta/
  sim/         # tournament engine + posterior-propagated Monte Carlo
  eval/        # backtest harness, RPS/Brier/log-loss, calibration plots
  live/        # 2026 result ingestion + incremental update pipeline
  web/         # Awwwards-grade static dashboard (built from sim outputs JSON)
  docs/        # this spec + plan + REPORT.md
  REPORT.md    # methodology (coaching-style), results, plots, final win-prob table
```

- **`REPORT.md`** explains each technique in coaching tone (user is learning the stats), plus
  results, calibration plots, and the final win-probability table.
- **Dashboard** (`web/`): win-probability leaderboard, interactive bracket simulator,
  title-odds timeline, calibration plots. Static site built from simulation-output JSON; UI
  design to be brainstormed separately when reached.

**Stack:** Python — PyMC / cmdstanpy, xgboost/lightgbm, arviz, numpy, pandas, matplotlib.
Web stack TBD at the web-design stage.

---

## 9. Build phasing (foundation first)

One spec (this doc) for the whole vision; build and validate **phase by phase**, each proven
before the next, because everything rests on the match model.

1. **Phase 1 — Foundation:** data loaders + computed Elo + point-in-time features + the
   **backtest harness** (RPS / Brier / log-loss / calibration). *Nothing predicts well until
   this is correct.* Delivers measurable baselines (Elo-only, uniform) before any heavy model.
2. **Phase 2 — Models:** M1 (static → dynamic state-space), then M2 GBM, then M3 odds, then
   meta-learner. Each validated against the Phase-1 harness.
3. **Phase 3 — Simulation:** tournament engine + posterior-propagated Monte Carlo. Produces
   the first real win-probability table.
4. **Phase 4 — Live + Web:** updating pipeline + dashboard.

---

## 10. Risks & honest caveats

- **Beating the market is hard.** Realistic edge over de-vigged odds is small; value is in
  calibration, uncertainty propagation, interpretability, and the live layer — not raw accuracy.
- **48-team format has no precedent** — bracket logic is rule-encoded, not validated on history.
- **Small-sample minnows** — hierarchical shrinkage handles this, but their uncertainty is wide
  and will show as wide intervals.
- **Free-data fragility** — odds free tier / Transfermarkt snapshots may degrade; the design
  fails gracefully (features drop, M3 falls back to outright-only) rather than breaking.
- **Tournament variance** — even a perfect model gives the favourite ~20–25%; communicated
  prominently in the report and dashboard so results aren't over-read.
```
