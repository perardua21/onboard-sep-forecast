#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""--probe 로 받은 파일이 제대로 읽히는지 사건 이틀로 확인한다.

    python scripts/check_probe.py

1. 사건 이틀의 값과 10 pfu 도달 시각 (공식 시작 시각과 비교)
2. 조용한 시간의 ≥10 MeV 바닥값 — '양성자 ≥1 pfu' 게이트가 의미가 있는가
3. 같은 시각 두 기기 비교 (2017-09-10 에는 GOES-15 와 GOES-16 이 둘 다 있다)
4. ACE 준상대론적 전자가 양성자보다 먼저 오르는가 (선행 신호로 쓸 수 있는가)
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
    from oracle.baselines import proton_rise
    for day, (label, quiet_end) in DAYS.items():
        ww = df.loc[pd.Timestamp(day, tz="UTC") - pd.Timedelta(hours=8):f"{day} {quiet_end}"]
        if ww["p_gt10"].notna().sum() == 0:
            continue
        a = proton_rise(ww).loc[day:].mean()
        r = proton_rise(ww, mode="relative").loc[day:].mean()
        print(f"  {day}  조용한 시간 게이트 켜짐  원안 {a:.0%}  /  배경 대비 3배 {r:.0%}")

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

    section("4. ACE 전자(0.175–0.315 MeV)가 양성자보다 먼저 오르는가")
    if "e_dif_hi" not in df.columns or df["e_dif_hi"].notna().sum() == 0:
        print("  ACE 자료가 없다. python scripts\\fetch_data.py --probe 를 먼저 돌릴 것")
    else:
        from oracle.baselines import electron_only
        for o, _ in ev:
            d = f"{o:%Y-%m-%d}"
            w = df.loc[o - pd.Timedelta(hours=6): o + pd.Timedelta(hours=2)]
            e = w["e_dif_hi"].astype(float)
            if e.notna().mean() < 0.5:
                print(f"  {d}: 사건 앞뒤 ACE 결측이 많다 ({e.notna().mean():.0%} 관측)")
                continue
            bg = e.rolling(12, min_periods=3).median()
            ratio = (e / bg).loc[:o]
            def first_fire(lag):
                ww = df.loc[o - pd.Timedelta(hours=8): o + pd.Timedelta(hours=2)]
                g = electron_only(ww, cadence=5, channel="e_dif_hi",
                                  background_lag_min=lag).loc[:o]
                f = g[g].index[0] if g.any() else None
                return f"{(o - f).total_seconds()/60:.0f}분 전" if f is not None else "울리지 않음"
            print(f"  {d}  양성자 10 pfu {o:%H:%M}  ·  전자 ×10 (원안) {first_fire(0)}"
                  f"  ·  (배경 1시간 앞당김) {first_fire(60)}  ·  원안 최대 배경비 {ratio.max():.1f}배")
            print(w["e_dif_hi"].resample("1h").max().to_frame("ACE DE4 1시간 최대")
                  .T.to_string(float_format=lambda x: f"{x:.3g}"))
        print("  (전자 조건이 양성자보다 수십 분 먼저 울리면 선행 신호로 쓸 수 있다)")
        print("  '배경 1시간 앞당김' 은 참고값이다. 사건 세 개로 규칙을 고르지 않는다.")

    print("\n이 화면 전체를 복사해서 보내면 된다.")


if __name__ == "__main__":
    main()
