"""Export the 2026 prediction to web/data/predictions.json for the dashboard.

Runs the real M1 (NUTS) + Monte Carlo prediction, attaches group + flag metadata,
and writes a single JSON the static site consumes. Deterministic (seed=0).

    python scripts/export_web_data.py
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from scripts.predict_2026 import run_prediction, _wc26_matches
from wc2026.sim.bracket import wc2026_group_fixtures, derive_groups

# ISO-3166 alpha-2 for each 2026 qualifier (England/Scotland handled specially).
ISO2 = {
    "Algeria": "DZ", "Argentina": "AR", "Australia": "AU", "Austria": "AT", "Belgium": "BE",
    "Bosnia and Herzegovina": "BA", "Brazil": "BR", "Canada": "CA", "Cape Verde": "CV",
    "Colombia": "CO", "Croatia": "HR", "Curaçao": "CW", "Czech Republic": "CZ", "DR Congo": "CD",
    "Ecuador": "EC", "Egypt": "EG", "France": "FR", "Germany": "DE", "Ghana": "GH", "Haiti": "HT",
    "Iran": "IR", "Iraq": "IQ", "Ivory Coast": "CI", "Japan": "JP", "Jordan": "JO", "Mexico": "MX",
    "Morocco": "MA", "Netherlands": "NL", "New Zealand": "NZ", "Norway": "NO", "Panama": "PA",
    "Paraguay": "PY", "Portugal": "PT", "Qatar": "QA", "Saudi Arabia": "SA", "Senegal": "SN",
    "South Africa": "ZA", "South Korea": "KR", "Spain": "ES", "Sweden": "SE", "Switzerland": "CH",
    "Tunisia": "TN", "Turkey": "TR", "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ",
}
SPECIAL_FLAG = {"England": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
                "Scotland": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F"}

def _flag(team: str) -> str:
    if team in SPECIAL_FLAG:
        return SPECIAL_FLAG[team]
    code = ISO2.get(team)
    if not code:
        return "⚽"  # fallback: football
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code)

def main(results_path: str = "data/raw/results.csv") -> Path:
    table = run_prediction(results_path, inference="nuts", n_draws=1000,
                           n_sims=20000, window_years=12, seed=0)
    raw = pd.read_csv(results_path); raw["date"] = pd.to_datetime(raw["date"])
    groups = derive_groups(wc2026_group_fixtures(_wc26_matches(raw)))
    team_group = {t: g for g, ts in groups.items() for t in ts}

    teams = []
    for rank, row in enumerate(table.itertuples(index=False), start=1):
        teams.append({
            "rank": rank,
            "team": row.team,
            "code": ISO2.get(row.team, ""),
            "flag": _flag(row.team),
            "group": team_group.get(row.team, "?"),
            "p_win": round(float(row.p_win), 4),
            "p_final": round(float(row.p_final), 4),
            "p_sf": round(float(row.p_sf), 4),
            "p_qf": round(float(row.p_qf), 4),
            "p_r16": round(float(row.p_r16), 4),
            "p_qualify": round(float(row.p_qualify), 4),
        })
    payload = {
        "meta": {
            "title": "2026 World Cup — Title Probabilities",
            "method": "M1 dynamic hierarchical Poisson (NUTS) → 20,000 posterior-propagated Monte Carlo simulations",
            "as_of": "2026-06-13",
            "n_sims": 20000,
            "n_teams": len(teams),
            "favourite": teams[0]["team"],
            "favourite_p": teams[0]["p_win"],
            "caveat": "A ~1-in-5 favourite means ~4-in-5 against. Single-elimination variance dominates — this is a calibrated probability spread across ~10 contenders, not a prediction of one winner.",
        },
        "groups": groups,
        "teams": teams,
    }
    out = Path("web/data/predictions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out

if __name__ == "__main__":
    p = main()
    data = json.loads(p.read_text())
    print("wrote", p, "|", len(data["teams"]), "teams | top:",
          data["teams"][0]["team"], data["teams"][0]["p_win"])
