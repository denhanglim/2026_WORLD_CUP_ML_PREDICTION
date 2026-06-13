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
