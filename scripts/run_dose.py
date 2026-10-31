#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""선량률 예측 개발 실행 (10·30·60분 뒤, 사건 단위 교차검증).

    python scripts/run_dose.py
    python scripts/run_dose.py --train-on rise     # 학습도 상승 구간만
    python scripts/run_dose.py --cadence 1         # 1분 간격 (느리다)
    python scripts/run_dose.py --tcn               # TCN 회귀 추가 (torch 필요)
    python scripts/run_dose.py --tcn --seeds 0,1,2 # TCN 초기값 3개로 흔들림 확인

필요: data/processed/sep_5min.csv (GOES), data/processed/dose_1min.csv (build_dose.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                   # noqa: E402

from oracle import dosecast as dc                     # noqa: E402
from oracle.config import load_config                 # noqa: E402
from oracle.ingest import load_dataset                # noqa: E402


def seed_runs(S, F, P0, models, seeds, a, cad, out, tag):
    """TCN·GBT 만 초기값을 바꿔 다시 학습. 규칙·릿지는 seed 와 무관해 첫 결과를 쓴다."""
    again = [m for m in models if dc.MODELS[m][0] in ("tcn", "gbt")]
    tables = {thr: [dc.summary(P0, thr)] for thr in (10.0, 100.0)}
    for sd in seeds[1:]:
        print(f"\n  seed {sd} 다시 학습: {' · '.join(again)}")
        Ps = dc.cross_validate(S, F, k=a.folds, train_on=a.train_on, target_pod=a.target_pod,
                               stride=5 if cad == 1 else 1, models=again, model_seed=sd)
        Ps.to_csv(out / f"predictions_{tag}_s{sd}.csv")
        Ps = pd.concat([P0[~P0["model"].isin(again)], Ps])
        for thr in tables:
            tables[thr].append(dc.summary(Ps, thr))
    for thr, tb in tables.items():
        T = dc.seed_spread(tb)
        print(f"\n[선량 예측 4] ≥{thr:g} µGy/min, seed {len(seeds)}개 평균 [최소~최대]")
        print(T.to_string(index=False))
        T.to_csv(out / f"seeds_{thr:g}_{tag}.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", choices=["window", "rise"], default="window")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--min-peak", type=float, default=100.0)
    ap.add_argument("--cadence", type=int, choices=[1, 5], default=5)
    ap.add_argument("--target-pod", type=float, default=0.7, help="경보 조정 목표 탐지율")
    ap.add_argument("--tcn", action="store_true", help="TCN 회귀도 돌린다 (torch 필요)")
    ap.add_argument("--seeds", default="0",
                    help="모델 초기값 여러 개 (예: 0,1,2). 사건 나눔은 그대로, TCN·GBT 만 다시 학습")
    ap.add_argument("--only", default="", help="이 모델만 (쉼표로 구분, 예: 지속성,TCN(60분 창))")
    a = ap.parse_args()

    cfg = load_config(ROOT / "configs" / "locked.yaml")
    path = cfg.path(cfg.get("data.path"))
    if not path.exists() and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    goes = load_dataset(path, cfg.cadence)
    d1 = pd.read_csv(ROOT / "data" / "processed" / "dose_1min.csv", index_col=0, parse_dates=True).iloc[:, 0]

    cad = a.cadence
    S, events, meas, calc_raw = dc.build(goes, d1, cad=cad, min_peak_pfu=a.min_peak)
    F = dc.features_events(goes, meas, calc_raw, events, cad)
    n_rise = sum(e.rise_start is not None for e in events)
    print(f"사건 {len(events)}건 (정점 ≥{a.min_peak:g} pfu, 1분 실측 있음) · 표본 {len(S):,} "
          f"· 상승 구간 {int(S['rise'].sum()):,} ({n_rise}건)")
    print(f"검증: 사건 단위 {a.folds}겹 · 학습 표본 {a.train_on} · {cad}분 간격 · "
          f"경보 조정 목표 탐지율 {a.target_pod:.0%}")

    models = list(dc.DEFAULT_MODELS)
    if a.tcn:
        models.append("TCN(60분 창)")
    if a.only:
        models = [m.strip() for m in a.only.split(",")]
    print("모델: " + " · ".join(models))
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    P = dc.cross_validate(S, F, k=a.folds, train_on=a.train_on, target_pod=a.target_pod,
                          stride=5 if cad == 1 else 1, models=models, model_seed=seeds[0])
    tag = f"{a.train_on}_{cad}min" + ("_tcn" if a.tcn else "")
    out = ROOT / "outputs" / "dose"
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / f"predictions_{tag}.csv")
    if len(seeds) > 1:
        seed_runs(S, F, P, models, seeds, a, cad, out, tag)

    pd.set_option("display.width", 220)
    T = dc.score(P)
    print("\n[선량 예측 1] 정확도와 기준 넘김 예보 (표본 단위)")
    print(T.to_string(index=False))
    T.to_csv(out / f"score_{tag}.csv", index=False, encoding="utf-8-sig")
    for thr in (10.0, 100.0):
        L = dc.crossing_leads(P, thr)
        print(f"\n[선량 예측 2] 사건별 첫 ≥{thr:g} µGy/min 넘김을 미리 알렸나 (조정된 경보, 넘기 2시간 전부터)")
        print(L.to_string(index=False))
        L.to_csv(out / f"crossing_{thr:g}_{tag}.csv", index=False, encoding="utf-8-sig")
        E = dc.alarm_episodes(P, thr)
        print(f"\n[선량 예측 3] ≥{thr:g} µGy/min 경보를 묶음 단위로 (30분 안에 이어진 경보는 하나, 끝+h분 안에 넘으면 맞음)")
        print(E.to_string(index=False))
        E.to_csv(out / f"episodes_{thr:g}_{tag}.csv", index=False, encoding="utf-8-sig")
    try:
        from oracle.plots import dose_event_cases
        for h in (10, 30, 60):
            dose_event_cases(P, events, out / f"cases_{h}min_{tag}.png", h=h)
    except Exception as e:                                   # noqa: BLE001
        print(f"  그림 생략: {e}")
    print("\nRMSE 는 log10 단위. 0.3 이면 약 2배 오차다.")
    print("탐지율·오경보비는 '지금은 기준 아래인데 h분 뒤 넘는' 표본만 센다.")
    print("'조정 후' 는 학습 사건 안의 교차검증으로 경보 문턱을 낮춘 결과다 (평가 사건은 보지 않음).")
    print(f"산출물: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
