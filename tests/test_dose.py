# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from oracle.dose import dose_rate, stopping_power_water


def test_stopping_power_matches_pstar():
    # PSTAR (NIST) 물: 10 MeV 45.67, 30 MeV 18.76, 100 MeV 7.289 MeV cm2/g
    s = stopping_power_water([10, 30, 100])
    assert np.allclose(s, [45.67, 18.76, 7.289], rtol=0.02)


def test_dose_scales_with_flux_and_cuts_low_energy():
    idx = pd.date_range("2020-01-01", periods=3, freq="5min", tz="UTC")
    base = {"p_gt1": 100.0, "p_gt5": 10.0, "p_gt10": 5.0, "p_gt30": 1.0,
            "p_gt50": 0.5, "p_gt60": 0.4, "p_gt100": 0.2}
    df = pd.DataFrame([base, {k: 10 * v for k, v in base.items()}, base], index=idx)
    d = dose_rate(df, e_min=13)
    assert np.isclose(d.iloc[1], 10 * d.iloc[0])
    assert dose_rate(df, e_min=1).iloc[0] > d.iloc[0]
