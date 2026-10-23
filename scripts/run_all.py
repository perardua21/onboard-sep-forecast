# -*- coding: utf-8 -*-
"""전체 실행 진입점.

  python scripts/run_all.py --synthetic                  # 자료 없이 파이프라인 확인
  python scripts/run_all.py --model logreg               # 실제 자료, 선형 기준선
  python scripts/run_all.py --model tcn --event 2017-09-10
  python scripts/run_all.py --ingest data/raw            # raw CSV → processed parquet
  python scripts/run_all.py --horizons                   # EXP-03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle.config import load_config                    # noqa: E402
from oracle.experiments import run_all, run_horizons     # noqa: E402
from oracle.ingest import build_dataset, load_dataset    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SEP 조기경보 모델 검증")
    ap.add_argument("--config", default=str(ROOT / "configs" / "locked.yaml"))
    ap.add_argument("--data", default=None, help="processed parquet 경로 (기본: config)")
    ap.add_argument("--ingest", default=None, help="raw 폴더 → processed parquet 생성 후 종료")
    ap.add_argument("--synthetic", action="store_true", help="합성 자료로 실행")
    ap.add_argument("--model", choices=["tcn", "cnn1d", "logreg", "gbt"], default=None)
    ap.add_argument("--event", default=None, help="대표 사건 날짜 YYYY-MM-DD")
    ap.add_argument("--horizons", action="store_true", help="EXP-03 만 실행")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = Path(a.out) if a.out else cfg.path(cfg.get("outputs.dir", "outputs"))

    if a.ingest:
        build_dataset(Path(a.ingest), cfg.path(cfg.get("data.path")), cfg.cadence)
        return

    if a.synthetic:
        from oracle.synth import make_synthetic
        print("합성 자료로 실행한다. 결과는 파이프라인 확인용이며 보고에 쓰지 않는다.\n")
        df, _ = make_synthetic(cadence_min=cfg.cadence)
        # 합성 자료는 기간이 짧으므로 분할을 자료 범위에서 70/15/15 로 잡는다
        n = len(df.index)
        cfg.raw["split"] = {
            "train_start": str(df.index[0].date()),
            "train_end": str(df.index[int(n * 0.70)].date()),
            "valid_end": str(df.index[int(n * 0.85)].date()),
            "test_end": str(df.index[-1].date()),
        }
        print(f"합성 분할: train ~{cfg.raw['split']['train_end']} / "
              f"valid ~{cfg.raw['split']['valid_end']} / test 이후\n")
        print("실제 자료에서는 configs/locked.yaml 의 "
              "2010–2019 / 2020–2022 / 2023–2026-10-22 분할이 쓰인다.\n")
    else:
        path = cfg.path(a.data or cfg.get("data.path"))
        if not path.exists() and path.with_suffix(".csv").exists():
            path = path.with_suffix(".csv")
        if not path.exists():
            raise SystemExit(
                f"자료가 없다: {path}\n"
                f"  먼저 --ingest data/raw 로 만들거나, --synthetic 로 파이프라인만 확인할 것")
        df = load_dataset(path, cfg.cadence)
        print(f"자료 {df.index[0]} ~ {df.index[-1]}  ({len(df):,} 표본 @ {cfg.cadence}분)\n")

    if a.horizons:
        print("EXP-03 예측 범위 sweep")
        run_horizons(df, cfg, out, kind=a.model)
        return

    run_all(df, cfg, out, kind=a.model, event_date=a.event)


if __name__ == "__main__":
    main()
