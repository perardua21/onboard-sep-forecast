# -*- coding: utf-8 -*-
"""사건 정의 - 설계 기준 + [26] SPHINX (연속 3점 / <5 pfu 3시간)."""
import numpy as np
import pandas as pd
import pytest

from oracle.labeling import detect_events, make_labels, make_labels_multi, ongoing_mask


@pytest.fixture
def series():
    idx = pd.date_range("2020-01-01", periods=24 * 60 // 5 * 10, freq="5min", tz="UTC")
    return pd.Series(0.05, index=idx)


def test_three_consecutive_points_make_an_event(series):
    s = series.copy()
    s.iloc[100:160] = 50.0
    ev = detect_events(s)
    assert len(ev) == 1
    assert ev[0][0] == s.index[100]


def test_two_points_is_not_an_event(series):
    s = series.copy()
    s.iloc[100:102] = 50.0                       # 2점 - 연속 3점 미만
    assert detect_events(s) == []


def test_exactly_three_points_is_an_event(series):
    s = series.copy()
    s.iloc[100:103] = 50.0
    assert len(detect_events(s)) == 1


def test_event_ends_after_three_hours_below_5pfu(series):
    s = series.copy()
    s.iloc[100:160] = 50.0
    (onset, end), = detect_events(s)
    assert end == s.index[160]                   # 미달 구간이 시작된 시각


def test_reflare_after_recovery_is_a_separate_event(series):
    """병합 기본값 0. 5 pfu 미만이 3시간 이어졌으면 해제된 것이므로 새 사건이다."""
    s = series.copy()
    s.iloc[100:160] = 50.0
    s.iloc[300:360] = 50.0                       # 약 11.7시간 뒤
    assert len(detect_events(s)) == 2


def test_reflare_merges_when_merge_hours_set(series):
    s = series.copy()
    s.iloc[100:160] = 50.0
    s.iloc[300:360] = 50.0
    assert len(detect_events(s, merge_hours=24.0)) == 1


def test_above_threshold_but_above_5pfu_stays_one_event(series):
    """10 pfu 밑으로 내려가도 5 pfu 위면 해제가 아니다."""
    s = series.copy()
    s.iloc[100:160] = 50.0
    s.iloc[160:400] = 7.0                        # 5~10 pfu 사이
    s.iloc[400:460] = 50.0
    assert len(detect_events(s)) == 1


def test_labels_are_before_onset_only(series):
    s = series.copy()
    s.iloc[200:260] = 50.0
    ev = detect_events(s)
    y = make_labels(s.index, ev, horizon_min=60)
    assert y[199] == 1                           # onset 직전
    assert y[200] == 0                           # onset 시점은 음성
    assert y[200 - 13] == 0                      # 65분 전 - 창 밖
    assert y.sum() == 12                         # 60분 / 5분


def test_dual_horizon_labels(series):
    s = series.copy()
    s.iloc[200:260] = 50.0
    ev = detect_events(s)
    Y = make_labels_multi(s.index, ev, (30, 60))
    assert set(Y) == {30, 60}
    assert Y[30].sum() == 6 and Y[60].sum() == 12
    assert (Y[60][Y[30] == 1] == 1).all()        # 30분 양성은 60분에서도 양성


def test_ongoing_mask_covers_event(series):
    s = series.copy()
    s.iloc[200:260] = 50.0
    m = ongoing_mask(s.index, detect_events(s))
    assert m[210] and not m[190]


def test_no_event_returns_zero_labels(series):
    ev = detect_events(series)
    assert ev == []
    assert make_labels(series.index, ev).sum() == 0
