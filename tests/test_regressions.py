# -*- coding: utf-8 -*-
"""코드 검토에서 찾은 버그들. 다시 생기면 여기서 걸린다."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from oracle import baselines, ingest
from oracle.evaluate import block_bootstrap, event_metrics, select_threshold
from oracle.labeling import detect_events
from oracle.train import harmonize


def _csv(path, months_vals):
    hdr = "time_tag,ZPGT10E_QUAL_FLAG,ZPGT10E,ZPGT10W_QUAL_FLAG,ZPGT10W\n"
    for m, val in months_vals:
        idx = pd.date_range(f"2015-{m:02d}-01", periods=12, freq="5min")
        rows = "".join(f"{t:%Y-%m-%d %H:%M:%S}.000,0,{val},0,{val}\n" for t in idx)
        (path / f"g15_epead_cpflux_5m_2015{m:02d}01.csv").write_text("h\ndata:\n" + hdr + rows)


def test_ingest_keeps_every_file_of_same_kind():
    """열 방향 병합으로 첫 파일만 남던 버그."""
    d = Path(tempfile.mkdtemp())
    _csv(d, [(1, 1.0), (2, 2.0), (3, 3.0)])
    df = ingest.build_dataset(d, d / "o.csv", 5)
    s = df["p_gt10"].dropna()
    assert sorted(s.unique().tolist()) == [1.0, 2.0, 3.0]


def test_ingest_ignores_1min_legacy_files():
    assert ingest.read_any(Path("g15_epead_e13ew_1m_20170901_20170930.csv")) is None
    assert ingest.read_any(Path("g15_epead_e13ew_1m_20170901_20170930_science_v1.0.0.csv")) is None


def test_event_end_not_delayed_by_gap():
    idx = pd.date_range("2020-01-01", periods=2000, freq="5min", tz="UTC")
    s = pd.Series(0.05, index=idx)
    s.iloc[100:160] = 50.0
    (_, e0), = detect_events(s.copy())
    s.iloc[180:186] = np.nan
    (_, e1), = detect_events(s)
    assert e0 == e1


def test_chained_false_alarms_are_not_merged_into_a_hit():
    """3시간 묶음 간격 때문에 오경보가 사라지고 선행시간이 750분이 되던 버그."""
    idx = pd.date_range("2020-01-01", periods=3000, freq="5min", tz="UTC")
    onset = idx[1500]
    a = pd.Series(False, index=idx)
    for k in range(1500 - 150, 1494, 24):
        a.iloc[k:k + 2] = True
    a.iloc[1494:1500] = True
    m = event_metrics(a, [(onset, idx[1560])])
    assert m["leads_min"] == [30.0]
    assert m["n_false"] == 6


def test_period_bootstrap_resamples_lead():
    idx = pd.date_range("2020-01-01", periods=6000, freq="5min", tz="UTC")
    ev = [(idx[300 + 600 * k], idx[310 + 600 * k]) for k in range(9)]
    a = pd.Series(False, index=idx)
    for k, lead in enumerate(range(2, 20, 2)):
        a.iloc[300 + 600 * k - lead:300 + 600 * k] = True
    b = block_bootstrap(event_metrics(a, ev), 300, mode="period", block_days=2, seed=0)
    lo, hi = b["lead_mean_ci"]
    assert hi - lo > 1.0


def test_threshold_is_chosen_on_event_level_far():
    idx = pd.date_range("2020-01-01", periods=4000, freq="5min", tz="UTC")
    ev = [(idx[1000], idx[1010]), (idx[3000], idx[3010])]
    p = np.full(len(idx), 0.05)
    p[990:1000] = 0.9                 # 사건 1 앞
    p[2990:3000] = 0.9                # 사건 2 앞
    p[500:504] = 0.6                  # 오경보 후보
    th, info = select_threshold(p, idx, ev, np.ones(len(idx), bool), 0.35,
                                persist_points=2, horizon_min=60, verbose=False)
    assert 0.6 < th <= 0.9
    assert info["POD"] == 1.0 and info["FAR"] == 0.0


def test_threshold_warns_when_target_unreachable():
    idx = pd.date_range("2020-01-01", periods=2000, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    p = rng.random(len(idx))
    th, info = select_threshold(p, idx, [(idx[1000], idx[1010])], np.ones(len(idx), bool),
                                0.01, persist_points=1, horizon_min=30, verbose=False)
    assert "없음" in info["note"]


def _two_eras():
    idx = pd.date_range("2018-01-01", periods=1000, freq="5min", tz="UTC")
    df = pd.DataFrame({c: 1.0 for c in ("xrs_long", "e_gt0p8", "e_gt2", "e_dif_hi", "p_gt500")},
                      index=idx)
    old = np.arange(len(idx)) < 600
    df.loc[old, ["e_dif_hi", "p_gt500"]] = np.nan       # 구세대에 없음
    df.loc[~old, ["e_gt0p8"]] = np.nan                  # 신세대에 없음
    return df, old


def test_harmonize_drops_channels_missing_in_one_era():
    df, old = _two_eras()
    keep, _ = harmonize(df, ["xrs_long", "e_gt0p8", "e_gt2", "e_dif_hi", "p_gt500", "xray_ok"],
                        {"학습": old, "시험": ~old}, 0.8, verbose=False)
    assert set(keep) == {"xrs_long", "e_gt2", "xray_ok"}


def test_electron_gate_channel_exists_in_both_eras():
    """>0.8 MeV 고정이면 신세대 구간에서 전자 게이트가 영영 안 울리던 버그."""
    df, old = _two_eras()
    assert baselines.pick_electron_channel(df, {"학습": old, "시험": ~old}) == "e_gt2"


def test_always_on_alarm_is_not_rewarded():
    """경보를 계속 켜 두면 묶음 하나가 사건을 품어 탐지율 100%·오경보 0% 가 되던 구멍."""
    idx = pd.date_range("2020-01-01", periods=2000, freq="5min", tz="UTC")
    a = pd.Series(True, index=idx)
    m = event_metrics(a, [(idx[1000], idx[1010])])
    assert m["POD"] == 1.0
    assert m["FAR"] > 0.9


def test_early_continuous_warning_keeps_its_lead_without_false_alarm():
    """X선을 보고 57분 전에 켜서 사건까지 유지 → 선행 57분, 오경보 0 (30분 출력 기준)."""
    idx = pd.date_range("2020-01-01", periods=600, freq="1min", tz="UTC")
    a = pd.Series(False, index=idx)
    a.iloc[300 - 57:300] = True
    m = event_metrics(a, [(idx[300], idx[310])])
    assert m["leads_min"] == [57.0]
    assert m["n_false"] == 0
