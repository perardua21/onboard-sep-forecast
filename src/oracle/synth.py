# -*- coding: utf-8 -*-
"""합성 자료 생성기 - 실제 자료를 받기 전에 파이프라인을 검증한다.

실제 SEP 의 순서(X선 → 전자 → 양성자)를 흉내 낸다. 물리 모사가 목적이 아니라
라벨·지표·학습 코드가 올바른지 확인하는 용도다.

**합성 자료 결과는 보고에 쓰지 않는다.** 선행 신호를 규칙적으로 심어 둔
가짜 자료라 성능이 실제보다 훨씬 좋게 나온다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import ALL_COLUMNS, PROTON, ELECTRON, DOSE, XRAY_FLAG

BASE = {
    "xrs_long": 1e-8, "xrs_short": 1e-9,
    "e_gt0p8": 1e2, "e_gt2": 3e1, "e_gt4": 1e1,
    "e_dif_lo": 5e2, "e_dif_hi": 5e1,
    "p_gt1": 0.3, "p_gt5": 0.1, "p_gt10": 0.05, "p_gt30": 0.02,
    "p_gt50": 0.01, "p_gt60": 0.008, "p_gt100": 0.004, "p_gt500": 0.001,
    "dose_main": 60.0, "dose_backup": 60.0,      # µSv/h, 달 표면 은하우주선 수준
}
# 에너지가 높을수록 상승폭이 작다
PRO_SCALE = dict(zip(PROTON, [1.15, 1.05, 1.00, 0.85, 0.75, 0.72, 0.60, 0.35]))
ELE_SCALE = dict(zip(ELECTRON, [1.00, 0.90, 0.80, 1.10, 0.95]))


def make_synthetic(days: int = 1200, n_events: int = 120, seed: int = 20170910,
                   weak_frac: float = 0.25, flare_only: int = 200,
                   cadence_min: int = 5, start: str = "2016-01-01"):
    """weak_frac : 선행 신호가 약한 사건 비율 (자기장 연결 불량 모사)
    flare_only  : 양성자를 동반하지 않는 플레어 수 (오경보 유발원)
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / cadence_min)
    idx = pd.date_range(start, periods=n, freq=f"{cadence_min}min", tz="UTC")
    t = np.arange(n)
    log = {c: np.full(n, np.log10(BASE[c])) for c in ALL_COLUMNS}
    for c in ALL_COLUMNS:
        log[c] += (0.25 * np.sin(2 * np.pi * t / (27 * 24 * 60 / cadence_min)
                                 + rng.uniform(0, 6.28))
                   + rng.normal(0, 0.06, n))

    def bump(arr, start_i, amp, rise_min, decay_min):
        s = int(start_i)
        rise = max(1, int(rise_min / cadence_min))
        decay = max(1.0, decay_min / cadence_min)
        end = int(min(n, s + rise + decay * 5))
        if s >= n or end <= s:
            return
        k = np.arange(s, end) - s
        arr[s:end] += np.where(k < rise, amp * k / rise,
                               amp * np.exp(-(k - rise) / decay))

    per_day = 24 * 60 / cadence_min
    lo, hi = int(2 * per_day), int(n - 3 * per_day)
    on = np.sort(rng.choice(np.arange(lo, hi), size=min(n_events, hi - lo), replace=False))
    on = on[np.insert(np.diff(on) > 3 * per_day, 0, True)]
    truth = []
    for o in on:
        weak = rng.random() < weak_frac
        amp_p = rng.uniform(2.6, 4.2)                  # 배경 0.05 pfu → 10 pfu 는 +2.3 dex
        if not weak:
            lx, le = rng.uniform(35, 75), rng.uniform(15, 45)
            bump(log["xrs_long"], o - lx / cadence_min, rng.uniform(2.6, 4.0), 8, 120)
            bump(log["xrs_short"], o - lx / cadence_min, rng.uniform(3.0, 4.4), 6, 90)
            amp_e = rng.uniform(1.2, 2.4)
            for c, sc in ELE_SCALE.items():
                bump(log[c], o - le / cadence_min, amp_e * sc, 20, 300)
        else:
            amp_e = rng.uniform(0.3, 0.6)
            for c, sc in ELE_SCALE.items():
                bump(log[c], o - 1, amp_e * sc, 10, 120)
        rise_p, dec_p = rng.uniform(20, 90), rng.uniform(600, 2000)
        for c, sc in PRO_SCALE.items():
            bump(log[c], o, amp_p * sc, rise_p, dec_p)
        for c in DOSE:                           # 선량률은 양성자를 따라 오른다
            bump(log[c], o, amp_p * 0.45, rise_p, dec_p)
        truth.append((idx[int(o)], bool(weak)))

    for o in rng.choice(np.arange(10, n - 10), size=min(flare_only, n - 20), replace=False):
        bump(log["xrs_long"], o, rng.uniform(2.2, 3.6), 8, 100)
        bump(log["xrs_short"], o, rng.uniform(2.6, 4.0), 6, 80)
        if rng.random() < 0.5:
            amp_e = rng.uniform(1.1, 1.6)
            for c, sc in ELE_SCALE.items():
                bump(log[c], o + 2, amp_e * sc, 20, 200)

    df = pd.DataFrame({c: 10.0 ** log[c] for c in ALL_COLUMNS}, index=idx)
    df[XRAY_FLAG] = 1.0
    for c in DOSE:                               # 합성 자료에는 선량계가 있다
        df[c + "_ok"] = 1
    return df, truth
