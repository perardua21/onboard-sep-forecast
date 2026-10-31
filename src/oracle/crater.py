# -*- coding: utf-8 -*-
"""LRO/CRaTER 실측 선량률 (달 궤도).

  iSWA 시간별 선량률  : 검출기별 cGy/day. 물 환산(×1.333)과 달 표면 고도 보정이
                        들어간 값. 2009-06 ~ 현재.
  secondary 1초 카운터 : 검출기별 singles (회/초).

시간별 값만으로는 10분·30분 예측의 정답이 될 수 없다. 그래서 같은 검출기의
1초 계수율로 시간 안의 변화를 나눈다.

    1분 선량률 = 그 시간의 선량률 × (그 1분 계수율 / 그 시간 평균 계수율)

한 시간 안에서는 입자 하나가 남기는 평균 에너지가 일정하다고 가정한다.
이렇게 만든 1분 값의 시간 평균은 원래 시간별 값과 같다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CGY_DAY_TO_UGY_MIN = 1e4 / 1440.0
SEC_COLS = (["sec", "fr", "time"] + [f"c{i}" for i in range(4, 21)]
            + [f"S{i}" for i in range(1, 7)] + ["GOOD", "REJECT", "TOTAL", "mx", "my", "mz"])
#: iSWA 선량 열 → 그 선량을 만드는 singles
PAIRS = {"d1_2": ("S1", "S2"), "d3_4": ("S3", "S4"), "d5_6": ("S5", "S6")}


def read_iswa(paths) -> pd.DataFrame:
    """iSWA CRaTER CSV 들 → 시간별 선량률 (µGy/min, 정시 기준)."""
    parts = []
    for p in sorted(Path(x) for x in paths):
        d = pd.read_csv(p)
        if d.empty:
            continue
        d["Time"] = pd.to_datetime(d["Time"], utc=True)
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts).drop_duplicates("Time").set_index("Time").sort_index()
    out = pd.DataFrame(index=d.index.floor("h"))
    for c in d.columns:
        if c.endswith("dose_rate"):
            out[c.replace("_dose_rate", "")] = d[c].to_numpy(dtype=float) * CGY_DAY_TO_UGY_MIN
    out = out[~out.index.duplicated(keep="first")]
    return out.where(out >= 0)


def read_sec(path) -> pd.DataFrame:
    """secondary 1초 카운터 → 1분 평균 계수율 (회/초)."""
    d = pd.read_csv(path, header=None, names=SEC_COLS, skipinitialspace=True,
                    usecols=["time"] + [f"S{i}" for i in range(1, 7)] + ["GOOD", "TOTAL"])
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.set_index("time").sort_index()
    d = d.where(d < 99999)                    # 5자리 계수기 포화는 버린다
    return d.resample("1min").mean()


def dose_1min(sec_1min: pd.DataFrame, hourly: pd.DataFrame, pair: str = "d1_2",
              max_jump: float = 3.0) -> pd.Series:
    """시간별 선량률을 1분 계수율로 나눠 1분 선량률 (µGy/min).

    시간마다 '계수 하나당 선량' k = 시간 선량률 / 시간 평균 계수율 을 구한다.
    k 는 스펙트럼이 천천히 바뀌므로 시간끼리 부드럽게 이어져야 한다. 앞뒤 5시간
    중앙값에서 max_jump 배 넘게 벗어난 시간은 시간별 값이 잘못된 것으로 보고
    (예: 2012-03-08 12시, 계수는 그대로인데 선량만 1/17) 중앙값 k 를 쓴다.
    """
    a, b = PAIRS[pair]
    counts = (sec_1min[a] + sec_1min[b]).astype(float)
    hour = counts.index.floor("h")
    mean_hour = counts.groupby(hour).mean()
    k = hourly[pair].reindex(mean_hour.index) / mean_hour
    k_med = k.rolling(5, center=True, min_periods=2).median()
    bad = (np.abs(np.log10(k / k_med)) > np.log10(max_jump)).fillna(False)
    k = k.where(~bad, k_med)
    out = counts.to_numpy() * k.reindex(hour).to_numpy()
    return pd.Series(out, index=counts.index, name=f"dose_{pair}_uGy_min")


def build(iswa_dir: Path, sec_dir: Path, pair: str = "d1_2") -> tuple[pd.Series, pd.DataFrame]:
    """1분 선량률 (카운터가 있는 날만)과 시간별 선량률 전체."""
    hourly = read_iswa(Path(iswa_dir).glob("CRaTER_[0-9]*.csv"))
    secs = sorted(Path(sec_dir).glob("CRAT_L2_SEC_*.TAB"))
    if not secs:
        return pd.Series(dtype=float), hourly
    one = pd.concat([read_sec(p) for p in secs]).sort_index()
    one = one[~one.index.duplicated(keep="first")]
    return dose_1min(one, hourly, pair), hourly
