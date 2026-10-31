# -*- coding: utf-8 -*-
"""GOES 양성자 적분 플럭스로 흡수선량률을 계산한다 (계산 선량).

실측 선량(LRO/CRaTER 등)과 맞춰 보정하기 전의 1차 근사다.

  적분 채널 차이 → 에너지 구간별 입자 수
  구간 대표 에너지에서 물의 질량 저지능 S(E) (Bethe 식)
  선량률 = 4π · Σ J_구간 · S(E) · 1.602e-10  [Gy/s]

4π 는 등방 입사를 가정한 점 표적의 전방향 플럭스다. 차폐는 최소 에너지
e_min 으로만 근사한다 (그보다 낮은 양성자는 차폐에 막힌다고 본다).
CRaTER 끝판은 약 13 MeV 아래를 막는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MEV_PER_G_TO_GY = 1.602176634e-10       # (MeV/g) → Gy
MP = 938.272                            # 양성자 정지질량 (MeV)
ME = 0.51099895                         # 전자 정지질량 (MeV)
K = 0.307075                            # MeV cm^2 / mol
Z_A_WATER = 0.5551                      # 물의 Z/A
I_WATER = 75e-6                         # 평균 들뜸 에너지 (MeV)

#: 적분 채널과 문턱 에너지 (MeV)
INTEGRAL = [("p_gt1", 1), ("p_gt5", 5), ("p_gt10", 10), ("p_gt30", 30),
            ("p_gt50", 50), ("p_gt60", 60), ("p_gt100", 100)]
#: 100 MeV 위는 p_gt500 이 없으면 한 구간으로 본다
TOP_E = 300.0


def stopping_power_water(e_mev) -> np.ndarray:
    """양성자 질량 저지능 (MeV cm^2/g), Bethe 식. 수 MeV 이상에서 PSTAR 와 몇 % 이내."""
    e = np.asarray(e_mev, dtype=float)
    gamma = 1 + e / MP
    beta2 = 1 - 1 / gamma ** 2
    tmax = 2 * ME * beta2 * gamma ** 2 / (1 + 2 * gamma * ME / MP + (ME / MP) ** 2)
    arg = 2 * ME * beta2 * gamma ** 2 * tmax / I_WATER ** 2
    return K * Z_A_WATER / beta2 * (0.5 * np.log(arg) - beta2)


def _bins(df: pd.DataFrame, e_min: float):
    """(하한, 상한, 구간 플럭스 열) 목록. e_min 아래 구간은 빼고 걸친 구간은 잘라 비례 배분."""
    have = [(c, e) for c, e in INTEGRAL if c in df.columns]
    out = []
    for (c_lo, e_lo), nxt in zip(have, have[1:] + [(None, TOP_E)]):
        c_hi, e_hi = nxt
        flux = df[c_lo] - (df[c_hi] if c_hi else 0.0)
        if e_hi <= e_min:
            continue
        if e_lo < e_min:
            # 구간 안에서 거듭제곱 분포라 보고 e_min 위쪽만 남긴다 (로그 간격 비례)
            frac = np.log(e_hi / e_min) / np.log(e_hi / e_lo)
            flux, e_lo = flux * frac, e_min
        out.append((e_lo, e_hi, flux.clip(lower=0)))
    if not out and "p_gt100" in df.columns:
        out.append((max(100.0, e_min), TOP_E, df["p_gt100"].clip(lower=0)))
    return out


def dose_rate(df: pd.DataFrame, e_min: float = 13.0, solid_angle: float = 4 * np.pi) -> pd.Series:
    """흡수선량률 (µGy/min, 물). 입력은 적분 플럭스 (pfu = /cm2/s/sr)."""
    total = 0.0
    for e_lo, e_hi, flux in _bins(df, e_min):
        e_mid = np.sqrt(e_lo * e_hi)
        total = total + flux * float(stopping_power_water(e_mid))
    gy_s = solid_angle * total * MEV_PER_G_TO_GY
    return (gy_s * 60 * 1e6).rename("dose_calc_uGy_min")
