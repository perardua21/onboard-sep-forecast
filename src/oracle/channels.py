# -*- coding: utf-8 -*-
"""채널 정의. 무엇이 입력이고 무엇이 라벨인지 한곳에서 정한다.

설계 기준 모델표: 입력 = 최근 60분 × 약 17채널 (X선 2, 전자 5, 양성자 8, 선량률 2).
docs/spec.md 참조.
"""
from __future__ import annotations

# 17채널
XRS = ["xrs_long", "xrs_short"]                     # 1–8 Å, 0.5–4 Å  (2)

ELECTRON = ["e_gt0p8", "e_gt2", "e_gt4",            # 적분       (3)
            "e_dif_lo", "e_dif_hi"]                 # 미분 대표  (2)  = 5

PROTON = ["p_gt1", "p_gt5", "p_gt10", "p_gt30",     # ≥10 MeV 가 라벨 채널
          "p_gt50", "p_gt60", "p_gt100", "p_gt500"]                # 8

DOSE = ["dose_main", "dose_backup"]                 # 주/백업 선량계      2

#: 보조 입력. 채널이 아니라 관측 가능 여부 플래그다.
XRAY_FLAG = "xray_ok"                               # 식 구간 X선 결손
PRESENCE = [c + "_ok" for c in DOSE]                # 지구 자료에는 선량률이 없다

FUSED17 = XRS + ELECTRON + PROTON + DOSE            # 2+5+8+2 = 17
AUX = [XRAY_FLAG] + PRESENCE

#: 제거 실험용 채널 집합 (R 2.4: X선 단독 / 전자 단독 / 융합)
CHANNEL_SETS = {
    "fused17":       FUSED17,
    "xray_only":     XRS,
    "electron_only": ELECTRON,
    "no_proton":     XRS + ELECTRON + DOSE,         # 선행 신호만
    "no_dose":       XRS + ELECTRON + PROTON,       # 지구 자료로 실제 채울 수 있는 15채널
}


def model_input(set_name: str = "fused17", with_aux: bool = True) -> list[str]:
    if set_name not in CHANNEL_SETS:
        raise ValueError(f"알 수 없는 채널 집합: {set_name} (가능: {list(CHANNEL_SETS)})")
    cols = list(CHANNEL_SETS[set_name])
    if with_aux:
        cols = cols + [XRAY_FLAG] + [c + "_ok" for c in DOSE if c in cols]
    return cols


#: 라벨을 만드는 채널. 절대 과거값 이외의 방식으로 입력에 들어가면 안 된다.
LABEL_CHANNEL = "p_gt10"

#: 저장 파일에 있어야 하는 전체 열
ALL_COLUMNS = FUSED17

S1_PFU = 10.0            # NOAA S1 = ≥10 MeV 적분 플럭스 10 pfu
RECOVERY_PFU = 5.0       # 설계 기준 해제 조건
EVENT_ENTRY_PFU = 1.0    # 설계 기준 T2 선제 진입
M1_WM2 = 1.0e-5          # GOES X선 M1급

#: 로그 변환 대상 (플럭스는 자릿수가 여러 개 차이난다 - 설계 기준)
LOG_CHANNELS = XRS + ELECTRON + PROTON + DOSE


def require_columns(df, cols=None):
    cols = cols or ALL_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"열 누락: {missing}")
    return df


def fill_missing(df, cols=None):
    """없는 채널은 0 으로 채우고 `<채널>_ok` = 0 으로 표시한다.

    지구 근처 자료에는 선량률 2채널이 없다. 없는 것을 있는 척하지 않고
    플래그로 남겨 학습·보고에서 구분한다.
    """
    cols = cols or ALL_COLUMNS
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
            df[c + "_ok"] = 0
        elif c + "_ok" not in df.columns and c in DOSE:
            df[c + "_ok"] = (~df[c].isna()).astype("int8")
    if XRAY_FLAG not in df.columns:
        df[XRAY_FLAG] = 1
    return df
