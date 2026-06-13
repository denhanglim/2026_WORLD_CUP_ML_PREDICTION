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
