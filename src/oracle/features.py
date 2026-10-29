# -*- coding: utf-8 -*-
"""특징 생성. 로그 변환, 배경 대비 상승비, 상승률.

절대값이 아니라 배경 대비 비율을 주 특징으로 쓴다. 위성 세대가 바뀌어
절대 보정이 달라져도 비율은 상대적으로 덜 흔들린다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .channels import model_input, XRAY_FLAG, XRS

FLOOR = 1e-12


def _is_flag(c: str) -> bool:
    """플래그는 변환하지 않고 그대로 통과시킨다."""
    return c == XRAY_FLAG or c.endswith("_ok")


def _steps(minutes: float, cadence: int, at_least: int = 1) -> int:
    return max(at_least, int(round(minutes / cadence)))


def _long_baseline(lg: pd.Series, cadence: int, days: float) -> pd.Series:
    """지난 days 일 중앙값 (과거만). 1시간 중앙값으로 줄여서 계산.

    GOES-15 와 GOES-16/18 의 평상시 수준 차이를 줄이려고 넣었다.
    logreg 에서 오히려 나빠져 기본값으로는 안 쓴다.
    """
    hourly = lg.resample("1h").median()
    nh = max(24, int(round(days * 24)))
    base = hourly.rolling(nh, min_periods=24).median().shift(1)
    return base.reindex(lg.index, method="ffill")


def build_features(df: pd.DataFrame, cadence: int = 5,
                   background_min: int = 60,
                   rise_min=(10, 30),
                   channels=None,
                   derive=("log", "ratio", "rise"),
                   base_days: float = 27.0) -> pd.DataFrame:
    """입력 채널 → 특징 행렬. 플래그 채널은 변환하지 않고 통과시킨다.

    derive 로 파생 특징을 고른다.
      신경망(TCN)  : ("log", "ratio") - 상승률은 팽창 합성곱이 직접 배운다.
                     17채널 × 2 = 34 + 플래그로 두면 파라미터가 예산 안에 든다.
      평탄 모델     : ("log", "ratio", "rise") - 창을 접어 쓰므로 상승률을 미리 준다.
      "base"        : 지난 base_days 일 중앙값 대비 (_long_baseline)
    """
    channels = list(channels or model_input("fused17"))
    nb = _steps(background_min, cadence, 4)
    out: dict[str, pd.Series] = {}
    for c in channels:
        if c not in df.columns:
            raise ValueError(f"입력 채널 없음: {c}")
        if _is_flag(c):
            out[c] = df[c].astype("float32").fillna(0.0)
            continue
        x = df[c].astype("float32").clip(lower=FLOOR)
        lg = np.log10(x)
        bg = lg.rolling(nb, min_periods=max(3, nb // 4)).median()
        if "log" in derive:
            out[f"{c}__log"] = lg
        if "base" in derive:
            out[f"{c}__base"] = (lg - _long_baseline(lg, cadence, base_days)).astype("float32")
        if "ratio" in derive:
            out[f"{c}__ratio"] = (lg - bg).astype("float32")
        if "rise" in derive:
            for r in rise_min:
                k = _steps(r, cadence)
                out[f"{c}__d{r}"] = (lg.diff(k) / r).astype("float32")
    F = pd.DataFrame(out, index=df.index).astype("float32")
    return F.replace([np.inf, -np.inf], np.nan)


def window_summary(F: pd.DataFrame, window_min: int = 60, cadence: int = 5) -> pd.DataFrame:
    """창을 요약 통계로 접는다 (트리 모델용).

    log 계열은 마지막값·최댓값, 나머지는 평균까지. 전 채널에 네 통계를
    다 붙이면 열이 네 배가 되어 메모리가 터진다.
    """
    n = _steps(window_min, cadence, 3)
    mp = max(3, n // 4)
    agg: dict[str, pd.Series] = {}
    for c in F.columns:
        s = F[c]
        agg[f"{c}|last"] = s
        agg[f"{c}|max"] = s.rolling(n, min_periods=mp).max().astype("float32")
        if not c.endswith("__log"):
            agg[f"{c}|mean"] = s.rolling(n, min_periods=mp).mean().astype("float32")
    return pd.DataFrame(agg, index=F.index).astype("float32")


def mask_xray(df: pd.DataFrame, where: np.ndarray, cadence: int = 5) -> pd.DataFrame:
    """X선을 배경값으로 고정하고 xray_ok 를 0 으로 둔다.

    식 구간 모사(EXP-07)와 학습 증강에 함께 쓴다. 값을 0 이나 NaN 으로
    두지 않는 이유는, 실제 식 구간에서도 검출기는 암전류 수준의 배경을
    계속 내놓기 때문이다.
    """
    out = df.copy()
    nb = max(4, int(round(24 * 60 / cadence)))
    for c in XRS:
        if c not in out.columns:
            continue
        bg = out[c].rolling(nb, min_periods=nb // 4).median()
        out.loc[where, c] = bg[where]
    if XRAY_FLAG in out.columns:
        out.loc[where, XRAY_FLAG] = 0.0
    return out


def mask_channels(df: pd.DataFrame, cols, where: np.ndarray) -> pd.DataFrame:
    """임의 채널을 배경값으로 고정한다 (EXP-08 전자 결손)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        bg = out[c].median()
        out.loc[where, c] = bg
    return out
