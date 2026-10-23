# -*- coding: utf-8 -*-
"""창 생성과 split. scaler 는 train 에만 fit 한다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(F: pd.DataFrame, window_min: int = 60, cadence: int = 5,
                 stride: int = 1):
    """(n, steps, n_feat) 텐서와 그 끝 시각 인덱스."""
    A = F.to_numpy(dtype=np.float32)
    n, c = A.shape
    w = max(2, int(round(window_min / cadence)))
    if n < w:
        return np.empty((0, w, c), np.float32), F.index[:0]
    ends = np.arange(w - 1, n, stride)
    out = np.empty((len(ends), w, c), np.float32)
    for k, i in enumerate(ends):
        out[k] = A[i - w + 1: i + 1]
    return out, F.index[ends]


def time_split(index: pd.DatetimeIndex, train_end: str, valid_end: str,
               train_start: str | None = None, test_end: str | None = None):
    """시간순 분할. 창을 만든 뒤 무작위로 섞지 않는다.

    운용 요구
      학습 2010-01-01 ~ 2019-12-31
      검증 2020-01-01 ~ 2022-12-31
      잠금 시험 2023-01-01 ~ 2026-10-22
    범위 밖(시험 끝 이후)은 어느 쪽에도 넣지 않는다.
    """
    def ts(x):
        return None if x is None else pd.Timestamp(x, tz="UTC")

    t0, t1, t2, t3 = ts(train_start), ts(train_end), ts(valid_end), ts(test_end)
    tr = np.asarray(index <= t1)
    if t0 is not None:
        tr &= np.asarray(index >= t0)
    va = np.asarray((index > t1) & (index <= t2))
    te = np.asarray(index > t2)
    if t3 is not None:
        te &= np.asarray(index <= t3)
    return tr, va, te


def split_report(index: pd.DatetimeIndex, tr, va, te) -> dict:
    """분할이 실제로 설정대로 잘렸는지 보고한다. 표에 그대로 넣는다."""
    def span(m):
        if not m.any():
            return "비어 있음"
        s = index[m]
        return f"{s.min():%Y-%m-%d} ~ {s.max():%Y-%m-%d} ({int(m.sum()):,}점)"
    return {"학습": span(tr), "검증": span(va), "잠금 시험": span(te)}


class Standardizer:
    """train 구간에서만 평균·표준편차를 구한다."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray):
        flat = X.reshape(-1, X.shape[-1])
        self.mu = np.nanmean(flat, axis=0).astype(np.float32)
        sd = np.nanstd(flat, axis=0).astype(np.float32)
        self.sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = (X - self.mu) / self.sd
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
