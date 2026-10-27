# -*- coding: utf-8 -*-
"""사건 정의와 라벨 생성.

설계 기준 + SPHINX 를 그대로 옮긴다.
  시작: ≥10 MeV 적분 플럭스가 10 pfu 이상인 점이 연속 3점
  종료: ≥10 MeV 가 5 pfu 미만인 상태가 연속 3시간
docs/spec.md 참조.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import S1_PFU, RECOVERY_PFU


def detect_events(p10: pd.Series,
                  threshold_pfu: float = S1_PFU,
                  sustain_points: int = 3,
                  end_threshold_pfu: float = RECOVERY_PFU,
                  end_hours: float = 3.0,
                  merge_hours: float = 0.0):
    """≥10 MeV 적분 플럭스에서 SEP 사건의 (시작, 종료)를 찾는다.

    시작 시각은 **연속 3점을 만족시킨 첫 점**, 즉 3점 구간의 첫 점이다.
    경보는 이 시각 이전에 나와야 하므로(AI-01), 시작 시각을 늦게 잡으면
    선행시간이 부풀려진다. 가장 이른 점을 쓴다.
    """
    s = p10.astype(float)
    idx = s.index
    n = len(idx)
    if n < sustain_points:
        return []

    v = s.to_numpy()
    hi = (s >= threshold_pfu).fillna(False).to_numpy()
    end_td = pd.Timedelta(hours=end_hours)

    events = []
    i = 0
    while i <= n - sustain_points:
        if not hi[i:i + sustain_points].all():
            i += 1
            continue
        onset = idx[i]

        # 종료: end_threshold 미만이 end_hours 동안 이어짐.
        # 결측은 미달 구간을 끊지도 늘리지도 않는다. 결측 30분에 종료가
        # 몇 시간씩 밀리면 그 뒤 구간이 통째로 평가에서 빠진다.
        j = i + sustain_points
        run_start = None
        end_at = idx[-1]
        while j < n:
            x = v[j]
            if x != x:                      # NaN
                j += 1
                continue
            if x < end_threshold_pfu:
                if run_start is None:
                    run_start = idx[j]
                if idx[j] - run_start >= end_td - (idx[1] - idx[0]):
                    end_at = run_start      # 미달 구간이 시작된 시각
                    break
            else:
                run_start = None
            j += 1
        events.append((onset, end_at))
        i = j + 1

    if merge_hours and merge_hours > 0:
        merged = []
        for o, e in events:
            if merged and (o - merged[-1][1]).total_seconds() / 3600.0 < merge_hours:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((o, e))
        events = merged
    return events


def make_labels(index: pd.DatetimeIndex, events, horizon_min: int = 30) -> np.ndarray:
    """y[t] = 1 ⟺ (t, t+horizon] 안에 onset 이 있다."""
    y = np.zeros(len(index), dtype=np.int8)
    if not events:
        return y
    onsets = pd.DatetimeIndex([o for o, _ in events])
    lo = index.searchsorted(onsets - pd.Timedelta(minutes=horizon_min), side="left")
    hi = index.searchsorted(onsets, side="left")
    for a, b in zip(lo, hi):
        y[a:b] = 1
    return y


def make_labels_multi(index: pd.DatetimeIndex, events, horizons_min=(30, 60)):
    """모델 A 는 출력이 두 개다. {30: y30, 60: y60}"""
    return {int(h): make_labels(index, events, int(h)) for h in horizons_min}


def ongoing_mask(index: pd.DatetimeIndex, events) -> np.ndarray:
    """사건 진행 중 구간. 예측이 아니라 관측이므로 학습·평가에서 제외한다."""
    m = np.zeros(len(index), dtype=bool)
    for o, e in events:
        m[index.searchsorted(o, "left"): index.searchsorted(e, "right")] = True
    return m


def events_to_frame(events) -> pd.DataFrame:
    return pd.DataFrame(
        [{"event_id": i + 1, "onset": o, "end": e,
          "duration_h": round((e - o).total_seconds() / 3600.0, 2)}
         for i, (o, e) in enumerate(events)])
