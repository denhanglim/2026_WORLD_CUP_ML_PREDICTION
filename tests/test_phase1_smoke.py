from scripts.phase1_smoke import run_phase1

def test_phase1_smoke_returns_metrics():
    res = run_phase1("tests/fixtures/mini_results.csv",
                     cutoffs=["2018-05-01"])
    assert "rps" in res.columns
    assert len(res) >= 1
