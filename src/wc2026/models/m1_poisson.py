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
