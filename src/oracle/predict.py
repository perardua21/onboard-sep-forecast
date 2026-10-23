# -*- coding: utf-8 -*-
"""확률 → 경보.

설계 기준 T2: 모델 A 확률 P(30분 내 ≥10 pfu) ≥ 0.7 이 **2회 연속 추론**.

기내 추론 주기는 10초이므로 2회 연속은 20초다. 지상 재현은 5분 격자 자료로
하므로 같은 "2점"이 10분이 되어 훨씬 보수적이다. 이 차이는 숨기지 않고
표에 적는다 (지상 재현의 선행시간은 기내보다 최대 한 격자만큼 짧게 나온다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_alarm(prob, index, threshold: float, persist_points: int = 2) -> pd.Series:
    """확률이 임계값 이상인 상태가 persist_points 연속이면 경보."""
    s = pd.Series(np.asarray(prob, dtype=float), index=index)
    raw = (s >= threshold).fillna(False)
    n = max(1, int(persist_points))
    return raw.rolling(n, min_periods=n).min().fillna(0).astype(bool)


def persist_minutes(persist_points: int, cadence_min: int) -> float:
    return persist_points * cadence_min
