# -*- coding: utf-8 -*-
"""학습 - 자료 적재부터 모델·임계값까지. 결과는 Trained 하나에 담아 넘긴다.

모델 A 는 출력이 두 개다 (30분, 60분 - 설계 기준).
경보 게이트는 그중 30분 쪽을 쓴다  ≥ 0.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from. import models
from .channels import LABEL_CHANNEL, model_input, fill_missing
from .datasets import Standardizer, make_windows, time_split, split_report
from .features import build_features, mask_xray, window_summary
from .labeling import detect_events, make_labels_multi, ongoing_mask

NN_KINDS = ("tcn", "cnn1d")
FLAT_KINDS = ("logreg", "gbt")


@dataclass
class Trained:
    kind: str
    model: Any
    scaler: Standardizer | None
    thresholds: dict            # {horizon_min: threshold}
    horizons: list
    events: list
    ongoing: np.ndarray
    index: pd.DatetimeIndex
    splits: tuple
    prob: dict = field(default_factory=dict)   # {horizon_min: ndarray}
    info: dict = field(default_factory=dict)

    @property
    def primary(self) -> int:
        return self.horizons[0]

    @property
    def threshold(self) -> float:
        return self.thresholds[self.primary]


def eclipse_augment(df: pd.DataFrame, cfg, rng: np.random.Generator) -> pd.DataFrame:
    """식 패턴대로 X선을 가린 학습 자료를 만든다.

    궤도 주기에 맞춰 블록 단위로 가린다. 무작위 점을 가리면 실제 식과
    통계가 달라져 모델이 엉뚱한 패턴을 배운다.
    입력 결손만 모사하며 판정 임계값은 건드리지 않는다.
    """
    p = float(cfg.get("augment.eclipse_mask_prob", 0.394))
    block_min = float(cfg.get("augment.eclipse_block_min", 46.5))
    cad = cfg.cadence
    if p <= 0:
        return df
    n = len(df)
    blk = max(1, int(round(block_min / cad)))
    mask = np.zeros(n, dtype=bool)
    for i in range(0, n, blk):
        if rng.random() < p:
            mask[i:i + blk] = True
    return mask_xray(df, mask, cad)


def channels_for(cfg) -> list:
    return model_input(cfg.get("channels.set", "fused17"))


def prepare(df: pd.DataFrame, cfg):
    """사건·라벨·분할을 만든다. 모델 종류와 무관한 공통 단계."""
    cad = cfg.cadence
    events = detect_events(
        df[LABEL_CHANNEL],
        threshold_pfu=float(cfg.get("event.threshold_pfu", 10.0)),
        sustain_points=int(cfg.get("event.sustain_points", 3)),
        end_threshold_pfu=float(cfg.get("event.end_threshold_pfu", 5.0)),
        end_hours=float(cfg.get("event.end_hours", 3.0)),
        merge_hours=float(cfg.get("event.merge_hours", 0.0)))
    horizons = [int(h) for h in cfg.get("label.horizons_min", [30, 60])]
    primary = int(cfg.get("label.primary_horizon_min", horizons[0]))
    horizons = [primary] + [h for h in horizons if h != primary]
    Y = make_labels_multi(df.index, events, horizons)
    ong = ongoing_mask(df.index, events) if cfg.get("label.exclude_ongoing", True) \
        else np.zeros(len(df), bool)
    tr, va, te = time_split(df.index,
                            cfg.get("split.train_end"), cfg.get("split.valid_end"),
                            cfg.get("split.train_start"), cfg.get("split.test_end"))
    return events, Y, horizons, ong, (tr, va, te), cad


def _features(df, cfg, channels, kind: str = "tcn"):
    key = "features.nn_derive" if kind in NN_KINDS else "features.flat_derive"
    default = ["log", "ratio"] if kind in NN_KINDS else ["log", "ratio", "rise"]
    return build_features(df, cfg.cadence,
                          background_min=int(cfg.get("features.background_min", 60)),
                          rise_min=tuple(cfg.get("features.rise_min", [10, 30])),
                          channels=channels,
                          derive=tuple(cfg.get(key, default)))


def train(df: pd.DataFrame, cfg, channels=None, kind: str | None = None,
          verbose: bool = True) -> Trained:
    df = fill_missing(df.copy())
    channels = list(channels or channels_for(cfg))
    kind = kind or cfg.get("model.kind", "tcn")
    if kind in NN_KINDS and not models.HAS_TORCH:
        print("torch 가 없어 logreg 로 돌린다. TCN 을 쓰려면 pip install torch")
        kind = "logreg"
    seed = int(cfg.get("model.seed", 0))
    rng = np.random.default_rng(seed)

    events, Y, horizons, ong, (tr, va, te), cad = prepare(df, cfg)
    stride = max(1, int(cfg.get("model.stride_min", 15)) // cad)
    win = int(cfg.get("label.window_min", 60))
    y0 = Y[horizons[0]]
    if verbose:
        print(f"사건 {len(events)}건 · 양성({horizons[0]}분) {int(y0.sum()):,} "
              f"({y0.mean():.3%}) · 진행중 제외 {ong.mean():.2%}")
        for k, v in split_report(df.index, tr, va, te).items():
            print(f"  {k}: {v}")

    prob = {h: np.full(len(df), np.nan, np.float32) for h in horizons}

    if kind in FLAT_KINDS:
        F = _features(df, cfg, channels, kind)
        S = window_summary(F, win, cad)
        ok = S.notna().all(axis=1).to_numpy() & ~ong
        Xtr = S[ok & tr].to_numpy()[::stride]
        clf = {}
        for h in horizons:
            fit = models.fit_logreg if kind == "logreg" else models.fit_gbt
            clf[h] = fit(Xtr, Y[h][ok & tr][::stride], seed)
            prob[h][ok] = models.predict_proba(clf[h], S[ok].to_numpy())
        scaler = None
        info = {"kind": kind, "n_features": int(S.shape[1])}
    else:
        df_aug = eclipse_augment(df, cfg, rng)
        Fa = _features(df_aug, cfg, channels, kind).fillna(0.0)
        F = _features(df, cfg, channels, kind).fillna(0.0)
        Xa, idx = make_windows(Fa, win, cad, 1)
        X, _ = make_windows(F, win, cad, 1)
        pos = df.index.searchsorted(idx)
        YY = np.stack([Y[h][pos] for h in horizons], axis=1)
        oo = ong[pos]
        trw, vaw = tr[pos] & ~oo, va[pos] & ~oo
        scaler = Standardizer().fit(Xa[trw])
        clf = models.fit_nn(
            scaler.transform(Xa[trw][::stride]), YY[trw][::stride],
            n_feat=Xa.shape[2], cfg=cfg, kind=kind,
            lr=float(cfg.get("model.lr", 2e-3)),
            weight_decay=float(cfg.get("model.weight_decay", 1e-4)),
            epochs=int(cfg.get("model.epochs", 20)),
            batch_size=int(cfg.get("model.batch_size", 256)),
            seed=seed,
            Xva=scaler.transform(X[vaw]) if vaw.any() else None,
            yva=YY[vaw] if vaw.any() else None,
            verbose=verbose)
        P = models.predict_nn(clf, scaler.transform(X))
        for j, h in enumerate(horizons):
            prob[h][pos] = P[:, j]
        info = {"kind": kind, "n_features": int(F.shape[1]),
                **models.budget_report(clf, Xa.shape[1], cfg)}
        if verbose:
            print(f"  파라미터 {info['파라미터']:,} (예산 {info['파라미터 예산']}, "
                  f"충족 {info['파라미터 충족']}) · 기내 창 재표본 "
                  f"{info['기내 창 재표본 간격']}")

    # 임계값은 검증집합에서만 고른다
    th = {}
    target = float(cfg.get("alarm.target_far", 0.35))
    for h in horizons:
        sel = va & ~ong & np.isfinite(prob[h])
        if not sel.any():
            sel = tr & ~ong & np.isfinite(prob[h])
        th[h] = models.pick_threshold(prob[h][sel], Y[h][sel], target)
    if verbose:
        print("  임계값(검증집합에서 선택): "
              + ", ".join(f"{h}분 {th[h]:.3f}" for h in horizons))

    return Trained(kind, clf, scaler, th, horizons, events, ong,
                   df.index, (tr, va, te), prob,
                   {**info, "labels": Y, "split_report": split_report(df.index, tr, va, te)})


def infer(t: Trained, df: pd.DataFrame, cfg, channels=None) -> dict:
    """학습된 모델로 다른(예: 결손 모사) 자료에 추론한다. {horizon: prob}"""
    df = fill_missing(df.copy())
    channels = list(channels or channels_for(cfg))
    cad, win = cfg.cadence, int(cfg.get("label.window_min", 60))
    F = _features(df, cfg, channels, t.kind)
    out = {h: np.full(len(df), np.nan, np.float32) for h in t.horizons}
    if t.kind in FLAT_KINDS:
        S = window_summary(F, win, cad)
        ok = S.notna().all(axis=1).to_numpy()
        for h in t.horizons:
            out[h][ok] = models.predict_proba(t.model[h], S[ok].to_numpy())
    else:
        X, idx = make_windows(F.fillna(0.0), win, cad, 1)
        P = models.predict_nn(t.model, t.scaler.transform(X))
        pos = df.index.searchsorted(idx)
        for j, h in enumerate(t.horizons):
            out[h][pos] = P[:, j]
    return out
