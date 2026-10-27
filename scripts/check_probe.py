#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""--probe 로 받은 파일이 제대로 읽히는지 사건 이틀로 확인한다.

    python scripts/check_probe.py

2017-09-10 은 구세대 EPEAD, 2021-10-28 은 신세대 SGPS 다. SGPS 에는 ≥10 MeV
적분 채널이 없어 미분 채널을 적분해 만들므로, 그 값이 상식적인지 여기서 본다.

알려진 값 (NOAA SWPC 기록)
  2017-09-10  16 UT 무렵 10 pfu 를 넘어 S2~S3 까지 오른다 (최대 1,000 pfu 대)
  2021-10-28  오후에 10 pfu 를 넘는 S1 사건 (최대 수십 pfu)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                   # noqa: E402

from oracle.ingest import build_dataset               # noqa: E402
from oracle.labeling import detect_events             # noqa: E402

DAYS = {"2017-09-10": "구세대 GOES-15 EPEAD", "2021-10-28": "신세대 GOES-16 SGPS (적분 계산값)"}
COLS = ["xrs_long", "e_gt2", "p_gt1", "p_gt10", "p_gt100", "p_gt500"]


def main():
    out = Path(tempfile.mkdtemp()) / "probe.csv"
    df = build_dataset(ROOT / "data" / "raw", out, 5)
    pd.set_option("display.width", 140)

    for day, label in DAYS.items():
        w = df.loc[day:day]
        print("\n" + "=" * 70)
        print(f"{day}  {label}   ({len(w)}점)")
        print("=" * 70)
        if w.empty:
            print("  이 날짜 자료가 없다")
            continue
        have = [c for c in COLS if c in w.columns]
        h = w[have].resample("2h").max()
        print(h.to_string(float_format=lambda x: f"{x:.3g}"))

        p10 = w["p_gt10"]
        print(f"\n  ≥10 MeV 최대 {p10.max():.3g} pfu", end="")
        above = p10[p10 >= 10]
        print(f" · 10 pfu 첫 도달 {above.index[0]:%H:%M} UTC" if len(above) else " · 10 pfu 미도달")

    print("\n" + "=" * 70)
    ev = detect_events(df["p_gt10"])
    print(f"사건 탐지 {len(ev)}건")
    for o, e in ev:
        print(f"  {o:%Y-%m-%d %H:%M} ~ {e:%Y-%m-%d %H:%M}")
    print("\n이 화면 전체를 복사해서 보내면 된다.")


if __name__ == "__main__":
    main()
