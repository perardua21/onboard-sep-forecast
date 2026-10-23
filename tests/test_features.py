# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from oracle.channels import (CHANNEL_SETS, DOSE, ELECTRON, FUSED17, LABEL_CHANNEL,
                             PROTON, XRAY_FLAG, XRS, fill_missing, model_input)
from oracle.features import build_features, mask_channels, mask_xray, window_summary


def _df(n=600):
    idx = pd.date_range("2020-01-01", periods=n, freq="5min", tz="UTC")
    d = pd.DataFrame({"xrs_long": 1e-8, "xrs_short": 1e-9}, index=idx)
    for c in ELECTRON:
        d[c] = 1e2
    for c in PROTON:
        d[c] = 0.05
    for c in DOSE:
        d[c] = 60.0
    d[XRAY_FLAG] = 1.0
    return fill_missing(d)


def test_fused_set_is_seventeen_channels():
    """설계 기준: X선 2 + 전자 5 + 양성자 8 + 선량률 2 = 17"""
    assert len(FUSED17) == 17
    assert len(XRS) == 2 and len(ELECTRON) == 5
    assert len(PROTON) == 8 and len(DOSE) == 2


def test_ablation_sets_exist():
    for k in ("fused17", "xray_only", "electron_only", "no_proton", "no_dose"):
        assert k in CHANNEL_SETS
    assert not any(c.startswith("p_gt") for c in CHANNEL_SETS["no_proton"])
    assert LABEL_CHANNEL in CHANNEL_SETS["fused17"]       # 과거값은 입력으로 허용


def test_missing_dose_is_flagged_not_hidden():
    idx = pd.date_range("2020-01-01", periods=10, freq="5min", tz="UTC")
    d = pd.DataFrame({c: 1.0 for c in XRS + ELECTRON + PROTON}, index=idx)
    out = fill_missing(d)
    for c in DOSE:
        assert (out[c + "_ok"] == 0).all()


def test_flag_passes_through_untransformed():
    F = build_features(_df(), channels=model_input("fused17"))
    assert XRAY_FLAG in F.columns
    assert set(np.unique(F[XRAY_FLAG])) <= {0.0, 1.0}


def test_ratio_is_zero_on_flat_background():
    F = build_features(_df(), channels=XRS).dropna()
    assert abs(F["xrs_long__ratio"]).max() < 1e-4


def test_rise_rate_responds_to_step():
    d = _df()
    d.iloc[300:, d.columns.get_loc("xrs_long")] = 1e-5
    F = build_features(d, channels=XRS)
    assert F["xrs_long__d10"].max() > 0.1


def test_mask_xray_sets_flag_zero():
    d = _df()
    where = np.zeros(len(d), bool); where[100:200] = True
    out = mask_xray(d, where)
    assert (out.loc[out.index[100:200], XRAY_FLAG] == 0).all()
    assert (out.loc[out.index[0:100], XRAY_FLAG] == 1).all()


def test_mask_channels_replaces_with_median():
    d = _df()
    d.iloc[:, d.columns.get_loc("e_gt2")] = np.linspace(1, 100, len(d))
    where = np.zeros(len(d), bool); where[10:20] = True
    out = mask_channels(d, ["e_gt2"], where)
    assert np.allclose(out["e_gt2"].iloc[10:20], d["e_gt2"].median())


def test_window_summary_column_count():
    F = build_features(_df(), channels=XRS)
    S = window_summary(F)
    n_log = sum(c.endswith("__log") for c in F.columns)
    assert S.shape[1] == n_log * 2 + (F.shape[1] - n_log) * 3
    assert (S.dtypes == "float32").all()


def test_nn_feature_count_keeps_params_in_budget():
    """설계 기준: 파라미터 5~15만, 20 MFLOPs/추론.

    torch 없이도 검산되도록 순수 산식(tcn_budget)으로 확인한다.
    """
    from oracle.models import tcn_budget
    F = build_features(_df(), channels=model_input("fused17"),
                       derive=("log", "ratio"))
    n_feat = F.shape[1]
    assert n_feat == 17 * 2 + 3            # 17채널 × (log, ratio) + 플래그 3

    b = tcn_budget(n_feat, hidden=48, n_blocks=3, kernel_size=5,
                   n_heads=2, window_steps=12)
    assert 50_000 <= b["params"] <= 150_000
    assert b["flops_m"] <= 20.0
    assert b["onboard_max_steps"] >= 60     # 60분 창을 1분 간격 이상으로는 넣을 수 있다
