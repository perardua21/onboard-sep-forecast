# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from oracle.crater import CGY_DAY_TO_UGY_MIN, dose_1min


def test_minute_dose_keeps_hourly_mean():
    idx = pd.date_range("2012-03-07 00:00", periods=120, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    sec = pd.DataFrame({"S1": rng.uniform(10, 20, 120), "S2": np.linspace(50, 500, 120)}, index=idx)
    hourly = pd.DataFrame({"d1_2": [1.0, 5.0]},
                          index=pd.date_range("2012-03-07", periods=2, freq="1h", tz="UTC"))
    d = dose_1min(sec, hourly)
    assert np.allclose(d.resample("1h").mean().to_numpy(), [1.0, 5.0])
    assert d.iloc[59] > d.iloc[0]                 # 시간 안의 상승을 따라간다


def test_unit_conversion():
    assert np.isclose(1.0 * CGY_DAY_TO_UGY_MIN, 6.944, atol=1e-3)
