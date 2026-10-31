# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from oracle import dosecast as dc


def test_targets_look_forward_only():
    idx = pd.date_range("2024-01-01", periods=30, freq="5min", tz="UTC")
    d = pd.Series(np.arange(1, 31, dtype=float), index=idx)
    T = dc.targets(d, idx)
    assert np.isclose(10 ** T["y10"].iloc[0], 3.0)         # 10분 = 2칸 뒤
    assert np.isclose(10 ** T["y60"].iloc[0], 13.0)
    assert T["y60"].iloc[-12:].isna().all()


def test_features_use_past_only():
    idx = pd.date_range("2024-01-01", periods=40, freq="5min", tz="UTC")
    g = pd.DataFrame({"p_gt10": 1.0, "xrs_long": 1e-6}, index=idx)
    d = pd.Series(1.0, index=idx)
    d.iloc[30:] = 100.0                                    # 30번째부터 오름
    F = dc.features(g, d, d)
    assert F["meas"].iloc[29] == 0.0                       # 오르기 전 값은 그대로
    assert F["meas_d5"].iloc[29] == 0.0


def test_calib_factor_is_median_ratio():
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    meas = pd.Series([2, 4, 8, 16, 0.1], index=idx, dtype=float)
    calc = meas * 8.0
    assert np.isclose(dc.calib_factor(calc, meas), 8.0)


def test_sequence_index_past_only_and_padded():
    idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
    pos = dc.sequence_index(idx, steps=4)
    assert list(pos[10]) == [7, 8, 9, 10]                  # 오래된 것부터, 끝이 t
    assert list(pos[0]) == [0, 0, 0, 0]                    # 앞이 비면 t 로 채움
    assert list(pos[2]) == [0, 0, 1, 2]
    idx1 = pd.date_range("2024-01-01", periods=30, freq="1min", tz="UTC")
    assert list(dc.sequence_index(idx1, steps=3)[20]) == [10, 15, 20]


def test_tcn_path_plumbing():
    """torch 없이 TCN 경로의 입력 모양과 지속성 잔차 복원만 본다."""
    from oracle import models as M
    seen = {}

    def fake_fit(X, y, **kw):
        seen["X"], seen["loss"] = X.shape, kw.get("loss")
        return "m"

    def fake_predict(model, X, raw=False):
        return np.zeros((len(X), 1), np.float32)

    old = M.fit_nn, M.predict_nn
    M.fit_nn, M.predict_nn = fake_fit, fake_predict
    try:
        idx = pd.date_range("2024-01-01", periods=400, freq="5min", tz="UTC")
        rng = np.random.default_rng(0)
        meas = np.log10(1 + np.arange(400.0))
        S = pd.DataFrame({"eid": np.repeat(np.arange(8), 50), "rise": False,
                          "meas_raw": 10 ** meas, "calc_raw": 10 ** meas * 8}, index=idx)
        for h in dc.HORIZONS:
            S[f"y{h}"] = meas + 0.01
        F = pd.DataFrame({"meas": meas, "a": rng.normal(size=400)}, index=idx)
        P = dc.cross_validate(S, F, k=4, inner_k=2, models=["TCN(60분 창)"])
    finally:
        M.fit_nn, M.predict_nn = old
    assert seen["X"][1:] == (dc.SEQ_STEPS, 2) and seen["loss"] == "mse"
    assert np.allclose(P["yhat"], P["meas"])               # 잔차 0 → 지속성과 같다


def test_alarm_episodes_groups_runs():
    idx = pd.date_range("2024-01-01", periods=40, freq="5min", tz="UTC")
    meas = np.full(40, 0.0)
    meas[20:] = 1.5                                        # 20번째에 10 µGy/min 넘음
    alarm = np.zeros(40, bool)
    alarm[[3, 4, 5]] = True                                # 오경보 묶음 1개
    alarm[[16, 17, 18, 19]] = True                         # 맞은 묶음 1개
    P = pd.DataFrame({"h": 10, "model": "m", "eid": 0, "meas": meas,
                      "alarm10": alarm}, index=idx)
    E = dc.alarm_episodes(P, 10.0).iloc[0]
    assert E["경보 묶음"] == 2 and E["맞음"] == 1 and E["잡은 사건"] == "1/1"


def test_seed_spread_formats_range():
    a = pd.DataFrame({"범위": ["60분"], "모델": ["m"], "RMSE": [0.15], "경보 묶음": [30.0]})
    b = pd.DataFrame({"범위": ["60분"], "모델": ["m"], "RMSE": [0.17], "경보 묶음": [30.0]})
    T = dc.seed_spread([a, b]).iloc[0]
    assert T["seed 수"] == 2 and T["RMSE"] == "0.160 [0.150~0.170]" and T["경보 묶음"] == "30"
