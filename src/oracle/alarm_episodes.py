# -*- coding: utf-8 -*-
"""경보 묶음(alarm episode). 묶지 않으면 오경보율이 의미를 잃는다.

한 번 틀린 경보가 분 단위로 수백 건으로 계산되기 때문이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def persist(raw: pd.Series, minutes: int, cadence: int = 5) -> pd.Series:
    """경보 지속 조건. 한두 점 튐으로 경보가 깜빡이는 것을 막는다."""
    n = max(1, int(round(minutes / cadence)))
    return (raw.fillna(False).rolling(n, min_periods=n)
            .min().fillna(0).astype(bool))


def cluster(alarm: pd.Series, gap_hours: float = 3.0):
    """연속 경보를 묶는다 → [(시작, 끝), ...]. gap_hours 이상 꺼지면 새 묶음."""
    a = alarm.fillna(False).to_numpy(dtype=bool)
    on = np.flatnonzero(a)
    if on.size == 0:
        return []
    idx = alarm.index
    t = np.asarray((idx - idx[0]) / pd.Timedelta(seconds=1), dtype=float)
    prev, nxt = on[:-1], on[1:]
    merge = (nxt == prev + 1) | ((t[nxt - 1] - t[prev]) < gap_hours * 3600.0)
    starts = np.concatenate([[on[0]], nxt[~merge]])
    ends = np.concatenate([prev[~merge], [on[-1]]])
    return list(zip(idx[starts], idx[ends]))
