import numpy as np
import pandas as pd
from wc2026.models.periods import assign_periods

def test_yearly_periods_are_chronological_codes():
    dates = pd.to_datetime(["2010-03-01", "2010-09-01", "2011-01-01", "2012-06-01"])
    codes, n = assign_periods(dates, freq="Y")
    assert n == 3
    assert codes.tolist() == [0, 0, 1, 2]   # 2010,2010 -> 0 ; 2011 -> 1 ; 2012 -> 2

def test_unsorted_input_still_maps_by_calendar():
    dates = pd.to_datetime(["2012-06-01", "2010-03-01", "2011-01-01"])
    codes, n = assign_periods(dates, freq="Y")
    assert n == 3
    assert codes.tolist() == [2, 0, 1]      # period code reflects calendar order, not row order
