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
