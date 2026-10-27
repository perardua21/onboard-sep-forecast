#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""--probe 로 받은 파일이 제대로 읽히는지 사건 이틀로 확인한다.

    python scripts/check_probe.py

1. 사건 이틀의 값과 10 pfu 도달 시각 (공식 시작 시각과 비교)
2. 조용한 시간의 ≥10 MeV 바닥값 — '양성자 ≥1 pfu' 게이트가 의미가 있는가
3. 같은 시각 두 기기 비교 (2017-09-10 에는 GOES-15 와 GOES-16 이 둘 다 있다)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from oracle import ingest                             # noqa: E402
from oracle.labeling import detect_events             # noqa: E402

RAW = ROOT / "data" / "raw"
#: 공식 시작 시각 (NASA SDAC Solar Proton Events 목록, SWPC 발표)
OFFICIAL = {"2017-09-05": "00:40", "2017-09-10": "16:45", "2021-10-28": "17:40"}
DAYS = {"2017-09-10": ("구세대 GOES-15 EPEAD", "14:00"),
        "2021-10-28": ("신세대 GOES-16 SGPS (적분 계산값)", "15:00")}
COLS = ["xrs_long", "e_gt2", "p_gt1", "p_gt10", "p_gt100", "p_gt500"]
GATE_PFU = 1.0


def find(pattern):
    hits = sorted(RAW.rglob(pattern))
    return hits[0] if hits else None


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    pd.set_option("display.width", 150)
    df = ingest.build_dataset(RAW, Path(tempfile.mkdtemp()) / "probe.csv", 5)

    section("1. 사건 이틀")
    for day, (label, _) in DAYS.items():
        w = df.loc[day:day]
        print(f"\n[{day}] {label}  ({len(w)}점)")
        if w.empty:
            print("  자료 없음")
            continue
        have = [c for c in COLS if c in w.columns]
        print(w[have].resample("2h").max().to_string(float_format=lambda x: f"{x:.3g}"))

    ev = detect_events(df["p_gt10"])
    print("\n사건 시작 시각 비교")
    for o, _ in ev:
        d = f"{o:%Y-%m-%d}"
        off = OFFICIAL.get(d)
        if off:
            diff = (o - pd.Timestamp(f"{d} {off}", tz="UTC")).total_seconds() / 60
            print(f"  {d}  공식 {off}  우리 {o:%H:%M}  차이 {diff:+.0f}분")
        else:
            print(f"  {d}  우리 {o:%H:%M}  (공식 기록 없음)")

    section(f"2. 조용한 시간의 ≥10 MeV 바닥값 (게이트 {GATE_PFU:g} pfu)")
    for day, (label, quiet_end) in DAYS.items():
        q = df.loc[f"{day} 00:00":f"{day} {quiet_end}", "p_gt10"].dropna()
        if q.empty:
            continue
        over = float((q >= GATE_PFU).mean())
        print(f"  {day}  중앙값 {q.median():.2f} pfu   게이트 이상인 시간 {over:.0%}")
    print("  (게이트 이상인 시간이 대부분이면 '≥1 pfu' 조건이 그 구간에서 변별력이 없다)")

    section("3. 같은 시각 두 기기 비교 (2017-09-10)")
    pairs = [
        ("X선 1–8 Å", "xrs_long",
         find("*xrsf*_g15_d20170910*.nc"), ingest.read_xrs_nc,
         find("*xrsf*_g16_d20170910*.nc"), ingest.read_xrs_nc),
        ("전자 >2 MeV", "e_gt2",
         find("*epead_e13ew_5m_20170901*.csv"), ingest.read_epead_electrons,
         find("*mpsh*_g16_d20170910*.nc"), ingest.read_mpsh_nc),
    ]
    for name, col, fa, ra, fb, rb in pairs:
        if not (fa and fb):
            print(f"  {name}: 두 기기 파일이 다 있지 않다 ({fa and fa.name} / {fb and fb.name})")
            continue
        a = ingest.align(ra(fa), 5)[col].loc["2017-09-10"]
        b = ingest.align(rb(fb), 5)[col].loc["2017-09-10"]
        j = pd.concat([a.rename("GOES15"), b.rename("GOES16")], axis=1).dropna()
        j = j[(j > 0).all(axis=1)]
        if j.empty:
            print(f"  {name}: 겹치는 유효값 없음")
            continue
        r = (j["GOES16"] / j["GOES15"])
        c = np.corrcoef(np.log10(j["GOES15"]), np.log10(j["GOES16"]))[0, 1]
        print(f"  {name:10s}  GOES16/GOES15 비 중앙값 {r.median():.3g}  "
              f"(10~90% {r.quantile(.1):.3g}~{r.quantile(.9):.3g})  로그 상관 {c:.2f}  n={len(j)}")
    print("  (비가 1 에서 멀면 기기 간 눈금 차이다. 상관이 높으면 모양은 같다)")

    print("\n이 화면 전체를 복사해서 보내면 된다.")


if __name__ == "__main__":
    main()
