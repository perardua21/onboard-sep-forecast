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


def test_sgps_integral_matches_power_law():
    """SGPS 에는 ≥10 MeV 적분이 없어 미분 채널을 적분해 만든다. 해석해와 비교."""
    from oracle.ingest import integrate_spectrum
    e = np.array([1380, 2090, 2780, 4700, 8020, 16460, 31420, 54500,
                  90800, 108500, 126100, 196600, 333400.])
    thr = [1, 5, 10, 30, 50, 60, 100]
    for g in (2.0, 3.0, 4.0):
        got = integrate_spectrum((1e6 * e ** -g)[None, :], e, thr, 500.0)[0]
        E = np.array(thr) * 1000.0
        exact = 1e6 / (g - 1) * (E ** (1 - g) - 500e3 ** (1 - g))
        assert np.allclose(got, exact, rtol=0.01)


def test_nc_time_decodes_j2000_seconds():
    from oracle.ingest import _nc_time

    class V:
        def __init__(self, v): self.values = v
    ds = type("D", (), {})()
    ds.variables = {"L2_SciData_TimeStamp": None}
    secs = (pd.Timestamp("2021-10-28", tz="UTC") - pd.Timestamp("2000-01-01 12:00", tz="UTC")).total_seconds()
    ds.__class__.__getitem__ = lambda self, k: V(np.array([secs, secs + 300.0]))
    t = _nc_time(ds)
    assert t[0] == pd.Timestamp("2021-10-28", tz="UTC")


def test_policy_satellite_file_is_read_first():
    from oracle.ingest import _policy_rank
    assert _policy_rank(Path("g15_epead_e13ew_5m_20170901_20170930.csv")) == 0
    assert _policy_rank(Path("sci_mpsh-l2-avg5m_g16_d20170910_v2-0-3.nc")) == 2
    assert _policy_rank(Path("sci_sgps-l2-avg5m_g16_d20211028_v2-0-0.nc")) == 0


def test_ace_hapi_reads_header_and_fill():
    """ACE 는 HAPI CSV 로 받는다. 열 이름과 결측값을 머리말에서 읽는다."""
    import json
    meta = {"HAPI": "2.0", "parameters": [
        {"name": "Time", "type": "isotime"},
        {"name": "DE3", "type": "double", "units": "1/(cm^2 s sr MeV)", "fill": "-1.0E31"},
        {"name": "DE4", "type": "double", "units": "1/(cm^2 s sr MeV)", "fill": "-1.0E31"}]}
    head = "\n".join("#" + l for l in json.dumps(meta, indent=1).splitlines())
    rows = "2017-09-10T00:00:00.000Z,12.5,3.1\n2017-09-10T00:05:00.000Z,-1.0E31,3.4\n"
    p = Path(tempfile.mkdtemp()) / "ac_h1_epm_201709.csv"
    p.write_text(head + "\n" + rows)
    d = ingest.read_any(p)
    assert list(d.columns) == ["e_dif_lo", "e_dif_hi"]
    assert np.isnan(d["e_dif_lo"].iloc[1]) and d["e_dif_hi"].iloc[1] == 3.4
    assert d.index[0] == pd.Timestamp("2017-09-10", tz="UTC")


def test_absent_channel_is_nan_so_harmonize_and_gate_skip_it():
    """없는 채널을 0 으로 채워서 정합을 통과하고 전자 게이트가 빈 채널을 고르던 버그."""
    from oracle.channels import fill_missing
    idx = pd.date_range("2015-01-01", periods=100, freq="5min", tz="UTC")
    df = fill_missing(pd.DataFrame({"p_gt10": 1.0, "e_gt2": 5.0}, index=idx))
    assert df["e_dif_hi"].isna().all() and (df["e_dif_hi_ok"] == 0).all()
    m = np.ones(len(df), bool)
    keep, _ = harmonize(df, ["p_gt10", "e_gt2", "e_dif_hi", "dose_main", "xray_ok"],
                        {"a": m}, 0.8, verbose=False)
    assert "e_dif_hi" not in keep and "dose_main" not in keep
    assert baselines.pick_electron_channel(df, {"a": m}) == "e_gt2"


def test_lagged_background_catches_gradual_electron_rise():
    """2021-10-28 처럼 한 시간 넘게 천천히 오르면 직전 1시간 배경으로는 ×10 이 안 된다."""
    idx = pd.date_range("2021-10-28", periods=288, freq="5min", tz="UTC")
    e = np.full(len(idx), 34.0)
    e[192:216] = np.geomspace(34, 1100, 24)      # 16:00~18:00 두 시간에 걸쳐 30배
    e[216:] = 1100
    df = pd.DataFrame({"e_dif_hi": e}, index=idx)
    assert not baselines.electron_only(df, channel="e_dif_hi").any()
    assert baselines.electron_only(df, channel="e_dif_hi", background_lag_min=60).any()


def test_relative_proton_gate_ignores_high_quiet_floor():
    """신세대 바닥값(약 1 pfu)에서 '≥1 pfu' 가 계속 켜지는 문제."""
    idx = pd.date_range("2021-10-28", periods=288, freq="5min", tz="UTC")
    p = np.full(len(idx), 1.1)                    # 조용한데 1 pfu 위
    p[200:] = 15.0                                # 사건
    df = pd.DataFrame({"p_gt10": p}, index=idx)
    absolute = baselines.proton_rise(df)
    relative = baselines.proton_rise(df, mode="relative")
    assert absolute[:190].mean() > 0.9           # 원안은 조용한 시간 내내 켜짐
    assert not relative[:190].any()              # 배경 대비는 꺼져 있음
    assert relative[205:].any()
