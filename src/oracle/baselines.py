# -*- coding: utf-8 -*-
"""규칙 게이트와 비교 기준선.

설계 기준 '전환 지표' 표를 그대로 옮긴다.
운용 요구에 따라 eclipse 중 전자 지속조건을 단축하는 규칙은 **삭제**했다.
이 규칙은 임계값을 상황에 따라 바꾸는 것이어서, 임계값을 잠그고 시험집합에서
재조정하지 않는다는 원칙과 충돌한다.
"""
from __future__ import annotations

import pandas as pd

from .channels import M1_WM2, EVENT_ENTRY_PFU, S1_PFU, XRAY_FLAG

#: 전자 배경 대비 비율 조건에 쓸 채널 (0.5 MeV 근처 - 설계 기준)
ELECTRON_GATE_CH = "e_gt0p8"


def _sustained(cond: pd.Series, minutes: float, cadence: int = 5) -> pd.Series:
    n = max(1, int(round(minutes / cadence)))
    return cond.fillna(False).rolling(n, min_periods=n).min().fillna(0).astype(bool)


def _recent(cond: pd.Series, minutes: float, cadence: int = 5) -> pd.Series:
    """최근 minutes 안에 한 번이라도 참이었는가.

    2종 동의를 '같은 시각에 동시에 참'으로 보면 안 된다. X선은 양성자보다
    약 57분, 전자는 약 45분 먼저 오므로 두 지표의 참 구간이
    겹치지 않는 사건이 대부분이다. 동시성이 아니라 '최근 동의'로 본다.
    """
    n = max(1, int(round(minutes / cadence)))
    return cond.fillna(False).rolling(n, min_periods=1).max().fillna(0).astype(bool)


# 개별 지표
def persistence(df: pd.DataFrame, **_) -> pd.Series:
    """경보를 내지 않는다. 비교의 바닥."""
    return pd.Series(False, index=df.index)


def xray_only(df: pd.DataFrame, cadence: int = 5, minutes: float = 2,
              threshold: float = M1_WM2, **_) -> pd.Series:
    """T1: X선 1–8 Å ≥ 1×10⁻⁵ W/m² (M1급), 2분 연속."""
    c = (df["xrs_long"] >= threshold)
    if XRAY_FLAG in df.columns:                 # 식 구간에는 X선 지표를 쓸 수 없다
        c = c & (df[XRAY_FLAG] > 0)
    return _sustained(c, minutes, cadence)


def electron_only(df: pd.DataFrame, cadence: int = 5, factor: float = 10.0,
                  minutes: float = 5, background_min: int = 60,
                  channel: str = ELECTRON_GATE_CH, **_) -> pd.Series:
    """T1: 전자 0.5 MeV 가 1시간 배경 대비 ×10, 5분 연속.

    minutes 는 식 구간에서도 5 로 둔다.
    """
    e = df[channel].astype(float).clip(lower=1e-12)
    nb = max(4, int(round(background_min / cadence)))
    bg = e.rolling(nb, min_periods=max(3, nb // 4)).median()
    return _sustained(e >= factor * bg, minutes, cadence)


def proton_rise(df: pd.DataFrame, cadence: int = 5, pfu: float = EVENT_ENTRY_PFU,
                minutes: float = 3, **_) -> pd.Series:
    """T2: ≥10 MeV 양성자 ≥1 pfu, 3분 연속. 게이트 전용 - 모델 입력이 아니다."""
    return _sustained(df["p_gt10"] >= pfu, minutes, cadence)


def s1_reached(df: pd.DataFrame, **_) -> pd.Series:
    """경보 조건 그 자체. 선행시간이 구조적으로 0 이 되는 상한 비교용."""
    return (df["p_gt10"] >= S1_PFU).fillna(False)


# 2종 이상 동의
def rule_gate(df: pd.DataFrame, cadence: int = 5, gate: dict | None = None) -> pd.Series:
    """OBC 규칙 게이트 = ORACLE 이 정지했을 때 남는 성능.

    서로 다른 센서 2종 이상이 같은 방향을 가리킬 때만 올린다.
    게이트는 학습하지 않는 임계값 비교이므로 양성자를 써도 누설이 아니다.
    모델 A 는 미래를 예측하니 못 쓰고, 게이트는 현재를 판정하니 쓸 수 있다.
    """
    g = gate or {}
    w = float(g.get("agree_window_min", 60))
    a = _recent(xray_only(df, cadence, g.get("xray_sustain_min", 2),
                          g.get("xray_m1_wm2", M1_WM2)), w, cadence).astype(int)
    b = _recent(electron_only(df, cadence, g.get("electron_ratio", 10.0),
                              g.get("electron_sustain_min", 5),
                              g.get("electron_background_min", 60)), w, cadence).astype(int)
    c = _recent(proton_rise(df, cadence, g.get("proton_pfu", EVENT_ENTRY_PFU),
                            g.get("proton_sustain_min", 3)), w, cadence).astype(int)
    need = 2 if g.get("two_of_n", True) else 1
    return (a + b + c) >= need


def rule_gate_no_xray(df: pd.DataFrame, cadence: int = 5, gate: dict | None = None) -> pd.Series:
    """X선 결손(식 구간) 시의 게이트. 전자·양성자 둘 다 필요해진다."""
    g = gate or {}
    w = float(g.get("agree_window_min", 60))
    b = _recent(electron_only(df, cadence, g.get("electron_ratio", 10.0),
                              g.get("electron_sustain_min", 5)), w, cadence).astype(int)
    c = _recent(proton_rise(df, cadence, g.get("proton_pfu", EVENT_ENTRY_PFU),
                            g.get("proton_sustain_min", 3)), w, cadence).astype(int)
    return (b + c) >= 2


REGISTRY = {
    "persistence": persistence,
    "xray_only": xray_only,
    "electron_only": electron_only,
    "proton_rise": proton_rise,
    "s1_reached": s1_reached,
    "rule_gate": rule_gate,
    "rule_gate_no_xray": rule_gate_no_xray,
}

#: 표에 쓸 한국어 이름
LABELS = {
    "persistence": "지속성 (무예보)",
    "xray_only": "단순 임계값 - X선 단독",
    "electron_only": "단순 임계값 - 전자 단독",
    "proton_rise": "단순 임계값 - 양성자 ≥1 pfu",
    "s1_reached": "경보 기준 도달 (선행시간 0)",
    "rule_gate": "OBC 규칙 게이트 (2종 동의)",
    "rule_gate_no_xray": "OBC 규칙 게이트 - X선 결손",
}
