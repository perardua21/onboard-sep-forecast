# -*- coding: utf-8 -*-
"""경보 묶음(alarm episode). 묶지 않으면 오경보율이 의미를 잃는다.

한 번 틀린 경보가 분 단위로 수백 건으로 계산되기 때문이다.
"""
from __future__ import annotations

import pandas as pd


def persist(raw: pd.Series, minutes: int, cadence: int = 5) -> pd.Series:
    """경보 지속 조건. 한두 점 튐으로 경보가 깜빡이는 것을 막는다."""
    n = max(1, int(round(minutes / cadence)))
    return (raw.fillna(False).rolling(n, min_periods=n)
            .min().fillna(0).astype(bool))


def cluster(alarm: pd.Series, gap_hours: float = 3.0):
    """연속 경보를 gap_hours 기준으로 하나로 묶는다 → [(시작, 끝), ...]"""
    a = alarm.fillna(False).to_numpy(dtype=bool)
    idx = alarm.index
    out, i, n = [], 0, len(a)
    while i < n:
        if not a[i]:
            i += 1
            continue
        j, last = i, i
        while j < n:
            if a[j]:
                last, j = j, j + 1
                continue
            if (idx[j] - idx[last]).total_seconds() / 3600.0 >= gap_hours:
                break
            j += 1
        out.append((idx[i], idx[last]))
        i = j
    return out
