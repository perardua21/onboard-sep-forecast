# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from oracle.datasets import Standardizer, make_windows, split_report, time_split
from oracle.ingest import add_xray_flag, align, quality_control


def _F(n=100, c=3):
    idx = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(np.arange(n * c, dtype="float32").reshape(n, c), index=idx)


def test_window_shape_and_alignment():
    F = _F(100, 3)
    X, idx = make_windows(F, window_min=60, cadence=5, stride=1)
    assert X.shape == (100 - 12 + 1, 12, 3)
    assert idx[0] == F.index[11]
    assert np.allclose(X[0, -1], F.to_numpy()[11])       # 창의 끝이 그 시각


def test_time_split_is_ordered_and_disjoint():
    F = _F(1000)
    tr, va, te = time_split(F.index, "2020-01-01 12:00", "2020-01-02 00:00")
    assert tr.sum() and va.sum() and te.sum()
    assert not (tr & va).any() and not (va & te).any()
    assert F.index[tr].max() < F.index[va].min() < F.index[te].min()


def test_standardizer_fits_on_train_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(5, 2, (50, 10, 3)).astype("float32")
    Xte = rng.normal(50, 2, (20, 10, 3)).astype("float32")
    sc = Standardizer().fit(Xtr)
    assert abs(sc.transform(Xtr).mean()) < 0.1
    assert sc.transform(Xte).mean() > 5                  # test 는 그대로 치우쳐야 한다


def test_align_resamples_and_dedups():
    idx = pd.date_range("2020-01-01", periods=10, freq="1min", tz="UTC")
    d = pd.DataFrame({"xrs_long": 1.0}, index=idx)
    d = pd.concat([d, d.iloc[:3]])                       # 중복
    out = align(d, 5)
    assert out.index.is_monotonic_increasing
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("5min")).all()


def test_quality_control_keeps_nan_not_zero():
    idx = pd.date_range("2020-01-01", periods=5, freq="5min", tz="UTC")
    d = pd.DataFrame({"xrs_long": [1.0, -1e31, 2.0, -5.0, 3.0]}, index=idx)
    out = quality_control(d)
    assert out["xrs_long"].isna().sum() == 2
    assert (out["xrs_long"].dropna() > 0).all()


def test_add_xray_flag_marks_missing():
    idx = pd.date_range("2020-01-01", periods=5, freq="5min", tz="UTC")
    d = pd.DataFrame({"xrs_long": [1.0, np.nan, 1.0, 1.0, 1.0],
                      "xrs_short": 1.0}, index=idx)
    out = add_xray_flag(d)
    assert out["xray_ok"].tolist() == [1.0, 0.0, 1.0, 1.0, 1.0]


def test_locked_split_boundaries():
    """운용 요구: 학습 2010-2019 / 검증 2020-2022 / 잠금 시험 2023-2026-10-22"""
    idx = pd.date_range("2009-01-01", "2026-12-31", freq="30D", tz="UTC")
    tr, va, te = time_split(idx, "2019-12-31", "2022-12-31",
                            "2010-01-01", "2026-10-22")
    assert idx[tr].min().year == 2010 and idx[tr].max().year == 2019
    assert idx[va].min().year == 2020 and idx[va].max().year == 2022
    assert idx[te].min().year == 2023
    assert idx[te].max() <= pd.Timestamp("2026-10-22", tz="UTC")
    assert not (tr & va).any() and not (va & te).any()
    assert not tr[idx < pd.Timestamp("2010-01-01", tz="UTC")].any()


def test_split_report_names_ranges():
    idx = pd.date_range("2010-01-01", "2026-01-01", freq="30D", tz="UTC")
    tr, va, te = time_split(idx, "2019-12-31", "2022-12-31", "2010-01-01", "2026-10-22")
    r = split_report(idx, tr, va, te)
    assert "학습" in r and "잠금 시험" in r and "2023" in r["잠금 시험"]


def test_split_end_dates_include_whole_day():
    idx = pd.date_range("2019-12-30", "2023-01-02", freq="5min", tz="UTC")
    tr, va, te = time_split(idx, "2019-12-31", "2022-12-31", "2010-01-01", "2023-01-01")
    assert idx[tr].max() == pd.Timestamp("2019-12-31 23:55", tz="UTC")
    assert idx[va].min() == pd.Timestamp("2020-01-01 00:00", tz="UTC")
    assert idx[va].max() == pd.Timestamp("2022-12-31 23:55", tz="UTC")
    assert idx[te].min() == pd.Timestamp("2023-01-01 00:00", tz="UTC")
    assert idx[te].max() == pd.Timestamp("2023-01-01 23:55", tz="UTC")
