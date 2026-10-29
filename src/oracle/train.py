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

from . import models
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
    prob: dict = field(default_factory=dict)   # {horizon_min: ndarray}, 보정 후
    info: dict = field(default_factory=dict)
    calib: dict = field(default_factory=dict)  # {horizon_min: (a, b)} Platt 계수

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


def harmonize(df_raw: pd.DataFrame, channels: list, masks: dict,
              min_frac: float = 0.8, verbose: bool = True) -> tuple[list, dict]:
    """학습·시험 구간 모두에서 충분히 관측된 채널만 남긴다.

    GOES 는 세대마다 채널 구성이 다르다. 구세대에는 >500 MeV 양성자가 없고
    신세대에는 >0.8 MeV·>4 MeV 전자가 없다. 한쪽 구간에서만 있는 채널을
    입력에 두면, 학습 때 상수였던 값이 시험 때 갑자기 실제 값이 되어
    모델이 본 적 없는 입력을 받는다.
    """
    from .channels import XRAY_FLAG
    report, keep = {}, []
    for c in channels:
        if c == XRAY_FLAG or c.endswith("_ok"):
            continue
        avail = {k: (float(df_raw.loc[m, c].notna().mean()) if c in df_raw.columns and m.any()
                     else 0.0) for k, m in masks.items()}
        report[c] = avail
        if min(avail.values()) >= min_frac:
            keep.append(c)
    flags = [f for f in channels if f == XRAY_FLAG
             or (f.endswith("_ok") and f[:-3] in keep)]
    dropped = [c for c in report if c not in keep]
    if verbose and dropped:
        print(f"  구간별 관측률 {min_frac:.0%} 미만이라 뺀 채널 {len(dropped)}개:")
        for c in dropped:
            a = report[c]
            print(f"    {c:10s} " + "  ".join(f"{k} {v:.0%}" for k, v in a.items()))
    if not keep:
        raise SystemExit("남는 입력 채널이 없다. 자료 범위와 채널 매핑을 확인할 것")
    return keep + flags, report


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
                          derive=tuple(cfg.get(key, default)),
                          base_days=float(cfg.get("features.base_days", 27)))


def cv_blocks(index: pd.DatetimeIndex, tr: np.ndarray, k: int, gap_days: float = 1.0):
    """tr 구간을 시간순 k 블록으로 → [(블록, 블록+앞뒤 gap_days), ...]

    여유 구간은 학습에서 빼서 입력 창·라벨 창이 경계를 넘지 않게 한다.
    """
    pos = np.flatnonzero(tr)
    if k < 2 or pos.size == 0:
        return []
    edges = np.linspace(0, pos.size, k + 1).astype(int)
    gap = pd.Timedelta(days=gap_days)
    out = []
    for i in range(k):
        a, b = pos[edges[i]], pos[edges[i + 1] - 1]
        blk = np.zeros(len(index), bool)
        blk[a:b + 1] = True
        lo = index.searchsorted(index[a] - gap, "left")
        hi = index.searchsorted(index[b] + gap, "right")
        ext = np.zeros(len(index), bool)
        ext[lo:hi] = True
        out.append((blk & tr, ext))
    return out


def _fit_nn(Xtr, Ytr, Xva, Yva, n_feat, cfg, kind, seed, verbose, epochs=None, snapshots=()):
    return models.fit_nn(
        Xtr, Ytr, n_feat=n_feat, cfg=cfg, kind=kind,
        lr=float(cfg.get("model.lr", 1e-3)),
        weight_decay=float(cfg.get("model.weight_decay", 1e-4)),
        epochs=int(epochs or cfg.get("model.epochs", 25)),
        batch_size=int(cfg.get("model.batch_size", 256)),
        seed=seed, Xva=Xva, yva=Yva, verbose=verbose,
        patience_max=int(cfg.get("model.patience", 5)),
        pos_weight=cfg.get("model.pos_weight", "sqrt"),
        ema_decay=float(cfg.get("model.ema_decay", 0.0)),
        cosine=bool(cfg.get("model.cosine_lr", False)),
        snapshot_epochs=snapshots)


def subsample(mask: np.ndarray, positive: np.ndarray, stride: int, keep_positive: bool = True):
    """stride 간격으로 솎되 양성은 전부 남긴다 (양성이 0.03% 뿐이라)."""
    idx = np.flatnonzero(mask)
    if stride <= 1:
        return idx
    take = np.zeros(idx.size, bool)
    take[::stride] = True
    if keep_positive:
        take |= positive[idx]
    return idx[take]


def train(df: pd.DataFrame, cfg, channels=None, kind: str | None = None,
          verbose: bool = True, select_on: str | None = None) -> Trained:
    df_raw = df
    df = fill_missing(df.copy())
    channels = list(channels or channels_for(cfg))
    kind = kind or cfg.get("model.kind", "tcn")
    if kind in NN_KINDS and not models.HAS_TORCH:
        print("torch 가 없어 logreg 로 돌린다. TCN 을 쓰려면 pip install torch")
        kind = "logreg"
    seed = int(cfg.get("model.seed", 0))
    rng = np.random.default_rng(seed)

    events, Y, horizons, ong, (tr, va, te), cad = prepare(df, cfg)
    avail = {}
    if cfg.get("channels.harmonize", True):
        masks = {"학습": tr, "시험": te} if te.any() else {"학습": tr}
        channels, avail = harmonize(df_raw, channels, masks,
                                    float(cfg.get("channels.min_available", 0.8)), verbose)
    stride = max(1, int(cfg.get("model.stride_min", 15)) // cad)
    win = int(cfg.get("label.window_min", 60))
    y0 = Y[horizons[0]]
    if verbose:
        print(f"사건 {len(events)}건 · 양성({horizons[0]}분) {int(y0.sum()):,} "
              f"({y0.mean():.3%}) · 진행중 제외 {ong.mean():.2%}")
        for k, v in split_report(df.index, tr, va, te).items():
            print(f"  {k}: {v}")

    prob = {h: np.full(len(df), np.nan, np.float32) for h in horizons}

    select_on = select_on or str(cfg.get("threshold.select_on", "cv"))
    # fit_on="train_valid": 2010-2022 로 학습하고 임계값도 같은 범위 교차검증으로.
    # 2010-2019(GOES-15) 만 쓰면 GOES-16/18 을 못 본다.
    allv = str(cfg.get("model.fit_on", "train")) == "train_valid"
    base = (tr | va) if allv else tr
    folds = cv_blocks(df.index, base, int(cfg.get("threshold.cv_folds", 4)),
                      float(cfg.get("threshold.cv_gap_days", 1))) \
        if (select_on == "cv" or allv) else []
    oof = {h: np.full(len(df), np.nan, np.float32) for h in horizons}
    keep_pos = bool(cfg.get("model.keep_all_positive", True))
    platt_on = str(cfg.get("threshold.calibrate", "platt")) == "platt"
    # 교차검증 모델끼리 출력 눈금이 달라서 블록마다 보정하고 합친다
    per_fold = platt_on and allv and bool(cfg.get("threshold.per_fold_calibration", True))
    fold_ab = {h: [] for h in horizons}
    from .evaluate import apply_platt, fit_platt

    def _store_oof(h, where_rows, raw, fit_rows):
        """블록 예측을 oof 에 저장 (per_fold 면 블록 안에서 보정)."""
        if per_fold:
            ab = fit_platt(raw[fit_rows], Y[h][where_rows][fit_rows])
            fold_ab[h].append(ab)
            raw = apply_platt(raw, ab)
        oof[h][where_rows] = raw

    if kind in FLAT_KINDS:
        F = _features(df, cfg, channels, kind)
        S = window_summary(F, win, cad)
        SV = S.to_numpy()
        ok = (S.notna().mean(axis=1) >= 0.8).to_numpy() & ~ong
        yany = np.any(np.stack([Y[h] for h in horizons]), axis=0)
        fit = models.fit_logreg if kind == "logreg" else models.fit_gbt
        itr = subsample(ok & base, yany, stride, keep_pos)
        clf = {}
        for h in horizons:
            clf[h] = fit(SV[itr], Y[h][itr], seed)
            prob[h][ok] = models.predict_proba(clf[h], SV[ok])
        for k, (blk, ext) in enumerate(folds):
            ifm = subsample(ok & base & ~ext, yany, stride, keep_pos)
            rows = np.flatnonzero(blk & ok)
            for h in horizons:
                c = fit(SV[ifm], Y[h][ifm], seed)
                _store_oof(h, rows, models.predict_proba(c, SV[rows]),
                           np.ones(rows.size, bool))
        scaler = None
        info = {"kind": kind, "n_features": int(S.shape[1]),
                "fit_on": "학습+검증" if allv else "학습"}
    else:
        df_aug = eclipse_augment(df, cfg, rng)
        Fa = _features(df_aug, cfg, channels, kind)
        F = _features(df, cfg, channels, kind)
        Xa, idx = make_windows(Fa, win, cad, 1)
        X, _ = make_windows(F, win, cad, 1)
        pos = df.index.searchsorted(idx)
        YY = np.stack([Y[h][pos] for h in horizons], axis=1)
        yw = YY.any(axis=1)
        # 학습 목표만 soft label: 30분 출력에 30~60분 앞 창을 soft 로.
        # 평가·임계값·보정은 원래 라벨(YY).
        soft = float(cfg.get("model.soft_label_weight", 0.0))
        YT = YY.astype(np.float32).copy()
        if soft > 0:
            for j, h in enumerate(horizons):
                longer = sorted(x for x in horizons if x > h)
                if longer:
                    j2 = horizons.index(longer[0])
                    YT[:, j] = np.maximum(YT[:, j], soft * YY[:, j2])
        oo = ong[pos]
        trw, vaw = tr[pos] & ~oo, va[pos] & ~oo
        aw = (trw | vaw) if allv else trw
        fixed = str(cfg.get("model.epochs_mode", "early_stop")) == "fixed"
        if fixed:
            # 검증 손실이 epoch 마다 크게 흔들려서 조기 종료 대신 고정 epoch + EMA
            n_ep = int(cfg.get("model.epochs_fixed", 12))
            # 개발 모드에서 epoch 별 중간 가중치도 평가
            snap = sorted({int(e) for e in (cfg.get("model.snapshot_epochs") or [])} | {n_ep}) \
                if cfg.get("model.snapshot_epochs") else []
            n_train = max(snap) if snap else n_ep
            if verbose:
                print(f"  {'학습+검증(2010-2022)으로' if allv else '학습으로'} {n_train} epoch 학습 "
                      f"(고정, 가중치 평균 {cfg.get('model.ema_decay', 0)})"
                      + (f" · 비교할 epoch {snap}" if snap else ""))
            scaler = Standardizer().fit(Xa[aw])
            iw = subsample(aw, yw, stride, keep_pos)
            clf = _fit_nn(scaler.transform(Xa[iw]), YT[iw], None, None,
                          Xa.shape[2], cfg, kind, seed, verbose, epochs=n_train, snapshots=snap)
        else:
            snap, n_train = [], None
            scaler = Standardizer().fit(Xa[trw])
            iw = subsample(trw, yw, stride, keep_pos)
            clf = _fit_nn(scaler.transform(Xa[iw]), YT[iw],
                          scaler.transform(X[vaw]) if vaw.any() else None,
                          YY[vaw] if vaw.any() else None,
                          Xa.shape[2], cfg, kind, seed, verbose)
            n_ep = int(getattr(clf, "best_epoch_", 0) or cfg.get("model.epochs", 25))
            if allv:
                if verbose:
                    print(f"  학습+검증(2010-2022)으로 {n_ep} epoch 다시 학습 "
                          f"(위 검증 손실이 가장 낮았던 epoch 수)")
                scaler = Standardizer().fit(Xa[aw])
                iw = subsample(aw, yw, stride, keep_pos)
                clf = _fit_nn(scaler.transform(Xa[iw]), YT[iw],
                              None, None, Xa.shape[2], cfg, kind, seed, False, epochs=n_ep)
        n_train = n_train or n_ep
        Xs = scaler.transform(X)
        snap_final = {}
        for e in snap:
            if e != n_ep:
                clf.load_state_dict(clf.snapshots_[e])
                snap_final[e] = models.predict_nn(clf, Xs)
        if snap:
            clf.load_state_dict(clf.snapshots_[n_ep])
        P = models.predict_nn(clf, Xs)
        del Xs
        for j, h in enumerate(horizons):
            prob[h][pos] = P[:, j]
        snap_fold = {e: [] for e in snap_final}
        for k, (blk, ext) in enumerate(folds):
            if verbose:
                print(f"  교차검증 {k + 1}/{len(folds)}: "
                      f"{df.index[blk][0]:%Y-%m} ~ {df.index[blk][-1]:%Y-%m} 를 빼고 학습")
            fw = aw & ~ext[pos]
            sc = Standardizer().fit(Xa[fw])
            iw = subsample(fw, yw, stride, keep_pos)
            if allv or fixed:
                m = _fit_nn(sc.transform(Xa[iw]), YT[iw], None, None,
                            Xa.shape[2], cfg, kind, seed + k + 1, False, epochs=n_train,
                            snapshots=snap)
            else:
                m = _fit_nn(sc.transform(Xa[iw]), YT[iw],
                            sc.transform(X[vaw]) if vaw.any() else None,
                            YY[vaw] if vaw.any() else None,
                            Xa.shape[2], cfg, kind, seed + k + 1, False)
            bw = blk[pos]
            Xb = sc.transform(X[bw])
            for e in snap_final:
                m.load_state_dict(m.snapshots_[e])
                snap_fold[e].append((pos[bw], models.predict_nn(m, Xb), ~oo[bw]))
            if snap:
                m.load_state_dict(m.snapshots_[n_ep])
            Pk = models.predict_nn(m, Xb)
            for j, h in enumerate(horizons):
                _store_oof(h, pos[bw], Pk[:, j], ~oo[bw])
        info = {"kind": kind, "n_features": int(F.shape[1]),
                "fit_on": "학습+검증" if allv else "학습", "epochs_used": n_ep,
                "epochs_mode": "고정+EMA" if fixed else "조기 종료",
                "_snap_fold": snap_fold,
                **models.budget_report(clf, Xa.shape[1], cfg)}
        if verbose:
            print(f"  파라미터 {info['파라미터']:,} (예산 {info['파라미터 예산']}, "
                  f"충족 {info['파라미터 충족']}) · 기내 창 재표본 "
                  f"{info['기내 창 재표본 간격']}")

    # 임계값과 확률 보정은 시험집합을 보지 않은 예측으로 정한다.
    #   select_on="cv"   : 학습 구간은 교차검증 예측, 검증 구간은 본 모델 예측
    #   select_on="valid": 검증 구간만 (사건이 적으면 불안정하다)
    from .evaluate import select_threshold
    pick_prob = {}
    for h in horizons:
        if allv:
            q = np.where(base, oof[h], np.nan).astype(np.float32)
        else:
            q = np.where(va, prob[h], np.nan).astype(np.float32)
            if folds:
                q = np.where(tr, oof[h], q).astype(np.float32)
        pick_prob[h] = q
    region = base if allv else (va | tr if folds else va)
    pick_mask = region & ~ong & np.isfinite(pick_prob[horizons[0]])
    where = ("학습+검증 교차검증" if allv else
             "학습 교차검증 + 검증" if folds else "검증")
    at = lambda e: min(df.index.searchsorted(e[0]), len(df) - 1)   # noqa: E731
    ev_pick = [e for e in events if region[at(e)]]
    if not ev_pick:
        pick_mask, where = tr & ~ong & np.isfinite(prob[horizons[0]]), "학습"
        pick_prob = {h: prob[h] for h in horizons}
        ev_pick = [e for e in events if tr[at(e)]]
        if verbose:
            print("  ! 검증 구간에 사건이 없어 학습 구간에서 임계값을 고른다")

    calib = {}
    if per_fold and all(fold_ab[h] for h in horizons):
        # 최종 모델은 블록 보정 계수의 평균
        for h in horizons:
            calib[h] = tuple(float(np.mean([ab[i] for ab in fold_ab[h]])) for i in (0, 1))
            prob[h] = apply_platt(prob[h], calib[h])
    elif platt_on:
        for h in horizons:
            m = pick_mask & np.isfinite(pick_prob[h])
            calib[h] = fit_platt(pick_prob[h][m], Y[h][m])
            prob[h] = apply_platt(prob[h], calib[h])
            pick_prob[h] = apply_platt(pick_prob[h], calib[h])

    th, th_info = {}, {}
    target = float(cfg.get("alarm.target_far", 0.35))
    target_pod = float(cfg.get("alarm.target_pod", 0.70))
    persist = int(cfg.get("gate.model_persist_points", 2))
    for h in horizons:
        th[h], th_info[h] = select_threshold(
            pick_prob[h], df.index, ev_pick, pick_mask, target,
            persist_points=persist, horizon_min=h, verbose=verbose,
            target_pod=target_pod, where=f"{where} {h}분",
            priority=str(cfg.get("threshold.priority", "far")))
        th_info[h]["where"] = where
    if verbose:
        print(f"  임계값({where}, 사건 {len(ev_pick)}건): "
              + ", ".join(f"{h}분 {th[h]:.4g} (선택 집합 탐지율 {th_info[h]['POD']:.0%}, "
                          f"오경보비 {th_info[h]['FAR']:.0%})" for h in horizons))

    # 개발 모드: epoch 별 선택 집합 성능
    snap_results = {}
    for e, parts in (info.pop("_snap_fold", {}) or {}).items():
        res = {}
        for j, h in enumerate(horizons):
            q = np.full(len(df), np.nan, np.float32)
            for rows, Pk, fitm in parts:
                raw = Pk[:, j]
                if per_fold:
                    raw = apply_platt(raw, fit_platt(raw[fitm], Y[h][rows][fitm]))
                q[rows] = raw
            m = pick_mask & np.isfinite(q)
            _, res[h] = select_threshold(
                q, df.index, ev_pick, m, target, persist_points=persist, horizon_min=h,
                verbose=False, target_pod=target_pod,
                priority=str(cfg.get("threshold.priority", "far")))
        snap_results[e] = res
    if snap_results:
        snap_results[info.get("epochs_used")] = th_info

    return Trained(kind, clf, scaler, th, horizons, events, ong,
                   df.index, (tr, va, te), prob,
                   {**info, "pick_prob": pick_prob, "pick_mask": pick_mask,
                    "pick_events": ev_pick, "labels": Y, "split_report": split_report(df.index, tr, va, te),
                    "channels": channels, "availability": avail, "threshold_info": th_info,
                    "fold_calib": fold_ab, "folds": [b for b, _ in folds],
                    "snapshot_results": snap_results},
                   calib)


def infer(t: Trained, df: pd.DataFrame, cfg, channels=None) -> dict:
    """학습된 모델로 다른(예: 결손 모사) 자료에 추론한다. {horizon: prob}"""
    df = fill_missing(df.copy())
    channels = list(channels or t.info.get("channels") or channels_for(cfg))
    cad, win = cfg.cadence, int(cfg.get("label.window_min", 60))
    F = _features(df, cfg, channels, t.kind)
    out = {h: np.full(len(df), np.nan, np.float32) for h in t.horizons}
    if t.kind in FLAT_KINDS:
        S = window_summary(F, win, cad)
        ok = (S.notna().mean(axis=1) >= 0.8).to_numpy()
        for h in t.horizons:
            out[h][ok] = models.predict_proba(t.model[h], S[ok].to_numpy())
    else:
        X, idx = make_windows(F, win, cad, 1)
        P = models.predict_nn(t.model, t.scaler.transform(X))
        pos = df.index.searchsorted(idx)
        for j, h in enumerate(t.horizons):
            out[h][pos] = P[:, j]
    if t.calib:
        from .evaluate import apply_platt
        out = {h: apply_platt(out[h], t.calib.get(h)) for h in t.horizons}
    return out
