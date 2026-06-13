from scripts.full_backtest import run_fold_smoke

def test_run_fold_smoke_returns_table():
    tbl = run_fold_smoke("tests/fixtures/mini_results.csv")
    assert "model" in tbl.columns and "rps" in tbl.columns
    assert set(tbl["model"]) >= {"elo_logistic", "m1_poisson", "meta"}
