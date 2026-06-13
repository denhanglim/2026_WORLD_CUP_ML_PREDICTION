from scripts.baseline_report import report_models

def test_report_includes_three_models():
    tbl = report_models("tests/fixtures/mini_results.csv",
                        cutoffs_override=["2018-05-01"])
    assert set(tbl["model"].unique()) >= {"uniform", "elo_logistic", "gbm"}
