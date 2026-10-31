#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CRaTER 실측 선량률을 정리한다.

    python scripts/build_dose.py

  data/raw/dose/iswa/CRaTER_YYYY.csv      (fetch_dose.py --iswa)
  data/raw/dose/crater/CRAT_L2_SEC_*.TAB  (fetch_dose.py --events)
→ data/processed/dose_hourly.csv, data/processed/dose_1min.csv

그리고 GOES 로 계산한 선량률과 시간별로 견줘 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from oracle import crater                             # noqa: E402
from oracle.config import load_config                 # noqa: E402
from oracle.dose import dose_rate                     # noqa: E402
from oracle.ingest import load_dataset                # noqa: E402
from oracle.labeling import detect_events             # noqa: E402

RAW = ROOT / "data" / "raw" / "dose"
PROC = ROOT / "data" / "processed"


def main() -> int:
    one, hourly = crater.build(RAW / "iswa", RAW / "crater")
    if hourly.empty:
        print("시간별 선량률이 없다. 먼저 python scripts\\fetch_dose.py --iswa")
        return 1
    PROC.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(PROC / "dose_hourly.csv")
    print(f"시간별 선량률 {hourly.index[0]:%Y-%m-%d} ~ {hourly.index[-1]:%Y-%m-%d}  "
          f"{hourly['d1_2'].notna().sum():,}시간")
    q = hourly["d1_2"].dropna()
    print(f"  D1·2 평상시 중앙값 {q.median():.3f} µGy/min, 99.9% {q.quantile(0.999):.1f}, 최대 {q.max():.1f}")

    if len(one):
        one.to_frame().to_csv(PROC / "dose_1min.csv")
        days = one.dropna().index.normalize().unique()
        print(f"1분 선량률 {len(days)}일치  ({one.notna().sum():,}분)")
        h = one.resample("1h").mean()
        both = pd.concat([h, hourly["d1_2"]], axis=1, sort=True).dropna()
        if len(both):
            err = (both.iloc[:, 0] / both.iloc[:, 1] - 1).abs()
            print(f"  1분 값의 시간 평균 vs 원래 시간별 값: 오차 중앙값 {err.median():.2%}")
    else:
        print("1초 카운터가 없다. 1분 선량률은 python scripts\\fetch_dose.py --events 뒤에 만들어진다")

    cfg = load_config(ROOT / "configs" / "locked.yaml")
    path = cfg.path(cfg.get("data.path"))
    if not path.exists() and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    if not path.exists():
        return 0
    g = load_dataset(path, cfg.cadence)
    calc = dose_rate(g, e_min=13).resample("1h").mean()
    pair = pd.concat([calc.rename("calc"), hourly["d1_2"].rename("meas")], axis=1, sort=True).dropna()
    pair = pair[(pair > 0).all(axis=1)]
    print("\nGOES 계산 선량률 vs CRaTER 실측 (시간별, D1·2)")
    for lo in (1, 10, 100):
        s = pair[pair["meas"] >= lo]
        if len(s) < 5:
            continue
        r = s["calc"] / s["meas"]
        c = np.corrcoef(np.log10(s["calc"]), np.log10(s["meas"]))[0, 1]
        print(f"  실측 ≥{lo:>3} µGy/min  {len(s):5d}시간  계산/실측 중앙값 {r.median():.2f} "
              f"(10~90% {r.quantile(.1):.2f}~{r.quantile(.9):.2f})  로그 상관 {c:.2f}")

    print("\n사건별 실측 선량률 정점 (정점 ≥100 pfu, CRaTER 기간)")
    p10 = g["p_gt10"]
    rows = []
    for o, e in detect_events(p10):
        pk = p10.loc[o:e].max()
        if pk < 100 or o < hourly.index[0]:
            continue
        m = hourly["d1_2"].loc[o - pd.Timedelta(hours=6): e]
        if m.notna().sum() == 0:
            continue
        rows.append({"시작": f"{o:%Y-%m-%d %H:%M}", "정점 pfu": round(pk),
                     "실측 정점 µGy/min": round(m.max(), 1),
                     "정점까지 시간": f"{(m.idxmax() - o).total_seconds() / 3600:.1f}h",
                     "1분 자료": "있음" if len(one) and one.loc[o:o + pd.Timedelta(hours=6)].notna().any() else "-"})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
