# -*- coding: utf-8 -*-
"""지표 - 운용 요구이 공개하라고 정한 항목을 모두 검사한다."""
import numpy as np
import pandas as pd
import pytest

from oracle.alarm_episodes import cluster, persist
from oracle.delivery import duty, receipt_lead
from oracle.evaluate import (block_bootstrap, brier_score, brier_skill_score,
                             event_metrics, reliability, verdict, wilson_interval)


@pytest.fixture
def idx():
    return pd.date_range("2020-01-01", periods=2000, freq="5min", tz="UTC")


def test_cluster_merges_nearby_alarms(idx):
    a = pd.Series(False, index=idx)
    a.iloc[100:112] = True
    a.iloc[120:124] = True                       # 40분 뒤 → 같은 묶음
    a.iloc[900:904] = True
    assert len(cluster(a, gap_hours=3.0)) == 2


def test_persist_requires_duration(idx):
    raw = pd.Series(False, index=idx)
    raw.iloc[100:102] = True
    assert not persist(raw, 15, 5).any()
    raw.iloc[200:206] = True
    assert persist(raw, 15, 5).any()


def test_perfect_alarm_scores_one(idx):
    onset = idx[500]
    events = [(onset, idx[560])]
    a = pd.Series(False, index=idx)
    a.iloc[494:500] = True                       # onset 30분 전부터
    m = event_metrics(a, events)
    assert m["POD"] == 1.0 and m["FAR"] == 0.0 and m["CSI"] == 1.0
    assert m["lead_median_min"] == pytest.approx(30.0, abs=1e-6)


def test_no_alarm_scores_zero(idx):
    m = event_metrics(pd.Series(False, index=idx), [(idx[500], idx[560])])
    assert m["POD"] == 0.0
    assert np.isnan(m["FAR"])
    assert m["CSI"] == 0.0


def test_unrelated_alarms_are_false(idx):
    events = [(idx[500], idx[560])]
    a = pd.Series(False, index=idx)
    a.iloc[10:16] = True
    a.iloc[1000:1006] = True
    m = event_metrics(a, events)
    assert m["FAR"] == 1.0 and m["n_alarms"] == 2


def test_csi_counts_misses_and_false_alarms(idx):
    """사건 2건 중 1건 탐지, 오경보 1건 → CSI = 1/(1+1+1)"""
    events = [(idx[500], idx[520]), (idx[1200], idx[1220])]
    a = pd.Series(False, index=idx)
    a.iloc[494:500] = True                       # 첫 사건 탐지
    a.iloc[50:56] = True                         # 오경보
    m = event_metrics(a, events)
    assert m["POD"] == 0.5
    assert m["CSI"] == pytest.approx(1 / 3)


def test_min_lead_filters_late_detection(idx):
    events = [(idx[500], idx[560])]
    a = pd.Series(False, index=idx)
    a.iloc[498:500] = True                       # 선행 10분
    assert event_metrics(a, events)["POD"] == 1.0
    assert event_metrics(a, events, min_lead_min=30.0)["POD"] == 0.0


def test_lead_percentile_is_reported(idx):
    events = [(idx[500], idx[520]), (idx[1200], idx[1220])]
    a = pd.Series(False, index=idx)
    a.iloc[488:500] = True                       # 선행 60분
    a.iloc[1198:1200] = True                     # 선행 10분
    m = event_metrics(a, events)
    assert m["lead_p10_min"] <= m["lead_median_min"] <= m["lead_mean_min"] + 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = wilson_interval(7, 10)
    assert lo < 0.7 < hi and 0.0 <= lo and hi <= 1.0


def test_brier_rewards_calibration():
    y = np.array([0, 0, 1, 1])
    assert brier_score(np.array([0.0, 0.0, 1.0, 1.0]), y) == 0.0
    assert brier_score(np.array([1.0, 1.0, 0.0, 0.0]), y) == 1.0


def test_brier_skill_is_zero_for_climatology():
    y = np.array([0, 0, 0, 1])
    assert brier_skill_score(np.full(4, y.mean()), y) == pytest.approx(0.0, abs=1e-9)


def test_reliability_bins_sum_to_sample_count():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    r = reliability(p, y, n_bins=10)
    assert r["n"].sum() == 500
    assert len(r) == 10


def test_block_bootstrap_contains_point_estimate(idx):
    events = [(idx[100 + 200 * k], idx[110 + 200 * k]) for k in range(8)]
    a = pd.Series(False, index=idx)
    for k in range(6):                           # 8건 중 6건 탐지
        a.iloc[94 + 200 * k: 100 + 200 * k] = True
    m = event_metrics(a, events)
    b = block_bootstrap(m, n_boot=300, mode="event", seed=1)
    lo, hi = b["POD_ci"]
    assert lo <= m["POD"] <= hi


def test_period_block_bootstrap_runs(idx):
    events = [(idx[100 + 200 * k], idx[110 + 200 * k]) for k in range(8)]
    a = pd.Series(False, index=idx)
    for k in range(6):
        a.iloc[94 + 200 * k: 100 + 200 * k] = True
    b = block_bootstrap(event_metrics(a, events), n_boot=200,
                        mode="period", block_days=2, seed=1)
    assert 0.0 <= b["POD_ci"][0] <= b["POD_ci"][1] <= 1.0


# 수신 선행시간
def test_duty_matches_proposal():
    """설계 기준: 12.43 / 117.8 ≈ 10.6%"""
    assert duty() == pytest.approx(0.1055, abs=0.002)


def test_three_satellites_match_revision_ceiling():
    """운용 요구: 3기 균등 배치 시 직접 가시 합 상한 31.6%"""
    assert duty(n_sat=3) == pytest.approx(0.316, abs=0.003)


def test_receipt_lead_never_exceeds_onboard_lead():
    d = receipt_lead([60.0, 90.0], n_samples=200, seed=0)
    assert max(d["recv_leads_min"]) <= 90.0


def test_short_onboard_lead_is_often_undelivered():
    """기내 선행시간이 가시 공백(105분)보다 짧으면 못 받는 경우가 생긴다."""
    d = receipt_lead([20.0] * 5, n_samples=500, seed=0)
    assert 0.0 < d["delivered_frac"] < 1.0


def test_more_satellites_improve_delivery():
    a = receipt_lead([45.0] * 5, n_sat=1, n_samples=500, seed=0)
    b = receipt_lead([45.0] * 5, n_sat=3, n_samples=500, seed=0)
    assert b["delivered_frac"] > a["delivered_frac"]


# 성공 기준 판정
class _Cfg:
    def get(self, k, d=None):
        return {"alarm.target_pod": 0.70, "alarm.target_far": 0.35,
                "alarm.target_lead_min": 30.0, "alarm.target_pod_min": 0.50,
                "alarm.target_pod_ext": 0.80}.get(k, d)


def test_verdict_levels():
    base = {"POD": 0.75, "FAR": 0.30, "lead_mean_min": 40.0}
    assert verdict(base, _Cfg())["판정"] == "완전 성공"
    assert verdict({**base, "POD": 0.85}, _Cfg())["판정"] == "확장 성공"
    assert verdict({**base, "FAR": 0.50}, _Cfg())["판정"] == "최소 성공"
    assert verdict({**base, "POD": 0.40}, _Cfg())["판정"] == "미달"
    assert verdict({**base, "lead_mean_min": 10.0}, _Cfg())["판정"] == "최소 성공"
