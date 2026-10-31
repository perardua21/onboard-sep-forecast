# -*- coding: utf-8 -*-
"""선량률 예측 (10·30·60분 뒤).

입력 (시각 t 에 이미 끝난 값만)
  실측 선량률 (CRaTER D1·2) 과 그 상승률
  GOES 양성자로 계산한 선량률과 그 상승률
  GOES X선·전자·양성자, ACE 전자 (기존 입력)
출력
  log10(선량률 at t+h), h = 10, 30, 60분

표본은 100 pfu 이상 사건마다 시작 1일 전 ~ 2일 후 창에서 cad 분마다 뽑는다.
상승 구간(선량이 평상시의 3배를 넘은 뒤 정점까지)은 따로 표시해 그 구간 성능을
따로 본다. 검증은 사건 단위로 나눈다 (같은 사건이 학습과 평가에 함께 들어가지 않게).

경보: h분 뒤 예측이 기준 - δ 이상이면 알린다. δ 는 학습 사건 안에서 다시 나눈
교차검증 예측으로, 목표 탐지율을 맞추는 가장 작은 값으로 정한다 (평가 사건은 보지 않음).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dose import dose_rate
from .labeling import detect_events

HORIZONS = (10, 30, 60)
THRESHOLDS = (10.0, 100.0)                   # µGy/min
EPS = 1e-3                                   # µGy/min, 로그 바닥
GOES_CAD = 5                                 # GOES 처리 자료 간격 (분)
LOG2 = np.log10(2.0)


@dataclass
class EventWindow:
    eid: int
    onset: pd.Timestamp                      # ≥10 MeV 10 pfu 시작
    peak_pfu: float
    start: pd.Timestamp
    end: pd.Timestamp
    rise_start: pd.Timestamp | None
    rise_end: pd.Timestamp | None            # 실측 선량 정점


def _log(x):
    return np.log10(np.clip(np.asarray(x, dtype=float), EPS, None))


def measured(dose_1min: pd.Series, cad: int = 5) -> pd.Series:
    """1분 실측 → cad 분 평균. 시각 t 값은 (t-cad, t] 평균 (이미 끝난 구간)."""
    s = dose_1min.astype(float)
    if cad == 1:
        return s.rename("dose_meas")
    return s.resample(f"{cad}min", label="right", closed="right").mean().rename("dose_meas")


measured_5min = measured                     # 이전 이름


def available(x, grid: pd.DatetimeIndex, src_cad: int = GOES_CAD):
    """구간 시작 시각으로 찍힌 src_cad 분 자료를 끝난 시각 기준으로 grid 에 맞춘다."""
    y = x.shift(freq=f"{src_cad}min")
    return y.reindex(grid, method="ffill", limit=max(1, src_cad))


def find_events(goes: pd.DataFrame, dose_meas: pd.Series, min_peak_pfu: float = 100.0,
                before_h: float = 24, after_h: float = 48, quiet_q: float = 0.5,
                min_hours: float = 24) -> list[EventWindow]:
    """정점 ≥min_peak_pfu 사건 중 실측 선량이 min_hours 이상 있는 것만."""
    p10 = goes["p_gt10"]
    have = dose_meas.dropna()
    cad = max(1, int(round(pd.Series(have.index).diff().median().total_seconds() / 60))) if len(have) > 1 else 5
    out = []
    for o, e in detect_events(p10):
        pk = float(p10.loc[o:e].max())
        if pk < min_peak_pfu:
            continue
        a, b = o - pd.Timedelta(hours=before_h), o + pd.Timedelta(hours=after_h)
        seg = have.loc[a:b]
        if len(seg) * cad < min_hours * 60:
            continue
        base = seg.loc[:o].quantile(quiet_q) if len(seg.loc[:o]) else seg.min()
        after = seg.loc[o - pd.Timedelta(hours=6):]
        rs = after[after > 3 * base].index.min() if (after > 3 * base).any() else None
        re_ = after.idxmax() if len(after) else None
        if rs is not None and re_ is not None and re_ <= rs:
            rs = None
        out.append(EventWindow(len(out), o, pk, seg.index[0], seg.index[-1], rs, re_))
    return out


def calib_factor(calc: pd.Series, meas: pd.Series, min_meas: float = 1.0) -> float:
    """계산 ÷ 실측의 중앙값 (실측 ≥ min_meas 인 점만)."""
    j = pd.concat([calc, meas], axis=1, sort=True).dropna()
    j = j[(j.iloc[:, 1] >= min_meas) & (j.iloc[:, 0] > 0)]
    return float((j.iloc[:, 0] / j.iloc[:, 1]).median()) if len(j) else 1.0


def _slope(lg: pd.Series, minutes: int, cad: int) -> pd.Series:
    k = max(1, minutes // cad)
    return (lg - lg.shift(k)) / (k * cad)        # dex / min


def features(goes: pd.DataFrame, dose_meas: pd.Series, calc_raw: pd.Series,
             cad: int = 5, grid: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """시각 t 의 특징 (t 에 이미 끝난 값만).

    calc_raw 는 보정 전 계산값. 보정은 상수배라 로그에서 상수 차이일 뿐이고
    모델의 절편이 흡수한다.
    """
    grid = dose_meas.index if grid is None else grid
    F = pd.DataFrame(index=grid)
    G = available(goes, grid)
    lm = pd.Series(_log(dose_meas.reindex(grid)), index=grid)
    lm[dose_meas.reindex(grid).isna().to_numpy()] = np.nan
    lc = pd.Series(_log(available(calc_raw, grid)), index=grid)
    F["meas"] = lm
    F["calc"] = lc
    for m in (5, 10, 30, 60):
        F[f"meas_d{m}"] = _slope(lm, m, cad)
        F[f"calc_d{m}"] = _slope(lc, m, cad)
    if cad < 5:
        F["meas_d1"] = _slope(lm, 1, cad)
    F["meas_max60"] = lm.rolling(max(1, 60 // cad), min_periods=1).max()
    F["meas_minus_calc"] = lm - lc
    for c in ("xrs_long", "xrs_short", "e_gt2", "e_dif_lo", "e_dif_hi",
              "p_gt1", "p_gt5", "p_gt10", "p_gt30", "p_gt50", "p_gt100"):
        if c not in G.columns:
            continue
        lg = pd.Series(_log(G[c].clip(lower=1e-12)), index=grid)
        F[c] = lg
        F[f"{c}_d5"] = _slope(lg, 5, cad)
        F[f"{c}_d30"] = _slope(lg, 30, cad)
        bg = lg.rolling(max(4, 60 // cad), min_periods=3).median()
        F[f"{c}_r60"] = lg - bg
    return F.replace([np.inf, -np.inf], np.nan)


def targets(dose_meas: pd.Series, index: pd.DatetimeIndex, cad: int = 5) -> pd.DataFrame:
    """y_h = log10(선량률 at t+h). 끝난 구간 기준이라 t+h 값은 (t+h-cad, t+h] 평균."""
    s = dose_meas.reindex(index)
    lm = pd.Series(_log(s), index=index)
    lm[s.isna().to_numpy()] = np.nan
    return pd.DataFrame({f"y{h}": lm.shift(-(h // cad)) for h in HORIZONS}, index=index)


def event_grid(ev: EventWindow, cad: int, margin_h: float = 2.0) -> pd.DatetimeIndex:
    """사건 창 앞뒤로 margin_h 를 더한 연속 격자. 기울기·60분 통계·60분 뒤 정답이 창 끝에서도 채워진다."""
    m = pd.Timedelta(hours=margin_h)
    return pd.date_range(ev.start - m, ev.end + m, freq=f"{cad}min")


def build(goes: pd.DataFrame, dose_1min: pd.Series, cad: int = 5, min_peak_pfu: float = 100.0):
    """표본 표 (사건 창 안 cad 분마다), 사건 목록, 실측(cad 분), 계산(GOES 격자).

    전체 기간을 1분 격자로 펴면 900만 행이 넘어 메모리가 모자란다. 사건마다
    연속 격자를 따로 만들어 그 안에서만 계산한다.
    """
    meas = measured(dose_1min, cad).dropna()
    calc_raw = dose_rate(goes, e_min=13)
    events = find_events(goes, meas, min_peak_pfu)
    rows = []
    for ev in events:
        g = event_grid(ev, cad)
        part = pd.DataFrame(index=g[(g >= ev.start) & (g <= ev.end)])
        part["eid"] = ev.eid
        part["rise"] = False
        if ev.rise_start is not None:
            part.loc[ev.rise_start:ev.rise_end, "rise"] = True
        part["meas_raw"] = meas.reindex(part.index)
        part["calc_raw"] = available(calc_raw, part.index)
        rows.append(part.join(targets(meas.reindex(g), g, cad)))
    S = pd.concat(rows)
    S = S[~S.index.duplicated(keep="first")]             # 창이 겹치면 앞 사건에 둔다
    return S, events, meas, calc_raw


def features_events(goes: pd.DataFrame, meas: pd.Series, calc_raw: pd.Series,
                    events: list[EventWindow], cad: int = 5) -> pd.DataFrame:
    """사건 격자마다 features 를 계산해 창 안만 모은다 (build 와 같은 순서·중복 처리)."""
    parts = []
    for ev in events:
        g = event_grid(ev, cad)
        F = features(goes, meas.reindex(g), calc_raw, cad, grid=g)
        parts.append(F.loc[ev.start:ev.end].astype("float32"))
    F = pd.concat(parts)
    return F[~F.index.duplicated(keep="first")]


# 모델
def _fit(kind: str, X, y, seed: int = 0):
    if kind == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    elif kind == "gbt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          random_state=seed)
    else:
        raise ValueError(kind)
    return m.fit(X, y)


XRAY5 = ["meas", "meas_d5", "xrs_long", "xrs_long_d5"]
MODELS = {
    "지속성": ("rule", None),
    "상승률 외삽(30분)": ("rule", None),
    "계산 선량(보정)": ("rule", None),
    "X선 5분 상승률 + 현재 선량": ("ridge", XRAY5),
    "릿지(전체 입력)": ("ridge", "all"),
    "GBT(전체 입력)": ("gbt", "all"),
    "TCN(60분 창)": ("tcn", "all"),
}
DEFAULT_MODELS = [m for m in MODELS if MODELS[m][0] != "tcn"]


def _rule(name: str, X: pd.DataFrame, h: int, kf: float) -> np.ndarray:
    if name == "지속성":
        return X["meas"].to_numpy()
    if name == "상승률 외삽(30분)":
        return (X["meas"] + X["meas_d30"].fillna(0).clip(lower=0) * h).to_numpy()
    if name == "계산 선량(보정)":
        return (X["calc"] - np.log10(kf)).to_numpy()
    raise ValueError(name)


SEQ_STEPS = 12                               # TCN 입력: 60분, 5분 간격
TCN_CFG = {"model.channels_hidden": 32, "model.n_blocks": 3, "model.kernel_size": 3,
           "model.dropout": 0.2}
TCN_EPOCHS = 15


def sequence_index(index: pd.DatetimeIndex, steps: int = SEQ_STEPS, step_min: int = 5) -> np.ndarray:
    """(N, steps) 행 위치. 열은 t-(steps-1)·step_min ~ t, 오래된 것부터.

    창 앞이 비면 (사건 창 시작 직후) 가장 가까운 뒤 시점으로 채운다.
    """
    pos = np.empty((len(index), steps), dtype=np.int64)
    for j, k in enumerate(range(steps - 1, -1, -1)):
        pos[:, j] = index.get_indexer(index - pd.Timedelta(minutes=k * step_min))
    for j in range(steps - 2, -1, -1):
        bad = pos[:, j] < 0
        pos[bad, j] = pos[bad, j + 1]
    return pos


def _tcn(Xv, pos, y, meas, tr_idx, te_idx, seed, stride):
    """TCN 회귀. 지속성에서 벗어나는 양 (y - 현재 실측) 을 배운다."""
    from . import models as M
    mu = np.nan_to_num(np.nanmean(Xv[tr_idx], axis=0))
    sd = np.nanstd(Xv[tr_idx], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    Z = np.clip(np.nan_to_num((Xv - mu) / sd), -10, 10).astype(np.float32)
    tri = tr_idx[::stride]
    r = (y[tri] - meas[tri]).astype(np.float32)
    tri, r = tri[np.isfinite(r)], r[np.isfinite(r)]
    model = M.fit_nn(Z[pos[tri]], r, cfg=TCN_CFG, epochs=TCN_EPOCHS, seed=seed, verbose=False,
                     kind="tcn", pos_weight="none", loss="mse")
    return meas[te_idx] + M.predict_nn(model, Z[pos[te_idx]], raw=True)[:, 0]


def _predict(name, X, y, tr_idx, te_idx, h, kf, seed, stride=1, pos=None):
    kind, cols = MODELS[name]
    Xte = X.iloc[te_idx]
    if kind == "rule":
        return _rule(name, Xte, h, kf)
    if kind == "tcn":
        return _tcn(X.to_numpy(dtype=np.float32), pos, y, X["meas"].to_numpy(), tr_idx, te_idx,
                    seed, stride)
    cols = list(X.columns) if cols == "all" else cols
    sub = tr_idx[::stride]
    m = _fit(kind, X[cols].iloc[sub], y[sub], seed)
    return m.predict(Xte[cols])


def choose_delta(yhat, y, meas, thr: float, target_pod: float = 0.7,
                 grid=np.round(np.arange(0.0, 1.21, 0.05), 2)) -> float:
    """'지금 기준 아래, h분 뒤 넘음' 을 target_pod 이상 잡는 가장 작은 δ (dex)."""
    lt = np.log10(thr)
    below = meas < lt
    pos = below & (y >= lt)
    if pos.sum() == 0:
        return 0.0
    for d in grid:
        if ((yhat >= lt - d) & pos).sum() / pos.sum() >= target_pod:
            return float(d)
    return float(grid[-1])


def cross_validate(S: pd.DataFrame, F: pd.DataFrame, k: int = 7, seed: int = 0,
                   train_on: str = "window", inner_k: int = 3, target_pod: float = 0.7,
                   models=None, stride: int = 1, model_seed: int | None = None) -> pd.DataFrame:
    """사건 단위 k겹. 표본별 예측과 경보(기준별, 튜닝 전·후)를 돌려준다.

    train_on="rise" 면 학습에 상승 구간 표본만 쓴다. stride 는 학습 표본 솎기
    (1분 간격일 때 학습 시간을 줄인다). seed 는 사건 나눔, model_seed 는 모델
    초기값 (나눔은 그대로 두고 학습만 다시 할 때).
    """
    ms = seed if model_seed is None else model_seed
    models = list(models or DEFAULT_MODELS)
    eids = np.array(sorted(S["eid"].unique()))
    order = np.random.default_rng(seed).permutation(eids)
    fold_of = {e: i % k for i, e in enumerate(order)}
    eid_all = S["eid"].to_numpy()
    fold = S["eid"].map(fold_of).to_numpy()
    X = F.reindex(S.index)
    meas_all = X["meas"].to_numpy()
    rise = S["rise"].to_numpy()
    pos = sequence_index(S.index) if any(MODELS[m][0] == "tcn" for m in models) else None
    out = []
    for h in HORIZONS:
        y = S[f"y{h}"].to_numpy()
        ok = np.isfinite(y) & np.isfinite(meas_all)
        for f in range(k):
            te = ok & (fold == f)
            tr = ok & (fold != f)
            if train_on == "rise":
                tr = tr & rise
            if not te.any() or tr.sum() < 50:
                continue
            tr_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(te)
            kf = calib_factor(S["calc_raw"].iloc[tr_idx], S["meas_raw"].iloc[tr_idx])
            # 안쪽 교차검증: 학습 사건만 다시 나눠 δ 를 정한다
            tr_e = np.array(sorted(set(eid_all[tr_idx])))
            inner_of = {e: i % inner_k for i, e in enumerate(np.random.default_rng(seed + f).permutation(tr_e))}
            inner = np.array([inner_of[e] for e in eid_all[tr_idx]])
            for name in models:
                yp = _predict(name, X, y, tr_idx, te_idx, h, kf, ms, stride, pos)
                oof = np.full(len(tr_idx), np.nan)
                for j in range(inner_k):
                    a_, b_ = tr_idx[inner != j], inner == j
                    if b_.any() and len(a_) >= 50:
                        kfi = calib_factor(S["calc_raw"].iloc[a_], S["meas_raw"].iloc[a_])
                        oof[b_] = _predict(name, X, y, a_, tr_idx[b_], h, kfi, ms, stride, pos)
                row = pd.DataFrame({"h": h, "model": name, "eid": eid_all[te_idx],
                                    "rise": rise[te_idx], "meas": meas_all[te_idx], "y": y[te_idx],
                                    "yhat": np.asarray(yp, dtype=float)}, index=S.index[te_idx])
                good = np.isfinite(oof)
                for thr in THRESHOLDS:
                    d = choose_delta(oof[good], y[tr_idx][good], meas_all[tr_idx][good], thr, target_pod)
                    row[f"delta{thr:g}"] = d
                    row[f"alarm{thr:g}"] = (row["meas"] < np.log10(thr)) & (row["yhat"] >= np.log10(thr) - d)
                out.append(row)
    return pd.concat(out)


# 평가
def _pod_far(g: pd.DataFrame, thr: float, alarm_col: str | None):
    lt = np.log10(thr)
    below = g["meas"] < lt
    pos = below & (g["y"] >= lt)
    pred = g[alarm_col] if alarm_col else (below & (g["yhat"] >= lt))
    tp, fp, fn = int((pos & pred).sum()), int((~pos & pred).sum()), int((pos & ~pred).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    return pod, far


def _pct(x):
    return "-" if x != x else f"{x:.0%}"


def score(P: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """예측 범위·모델별: 로그 RMSE, 2배 안 비율, 기준 넘김 예보 (튜닝 전 → 후)."""
    rows = []
    for (h, m), g in P.groupby(["h", "model"], sort=False):
        g = g[np.isfinite(g["yhat"])]
        err = g["yhat"] - g["y"]
        r = g[g["rise"]]
        er = r["yhat"] - r["y"]
        row = {"범위": f"{h}분", "모델": m,
               "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
               "상승 RMSE": round(float(np.sqrt((er ** 2).mean())), 3) if len(r) else np.nan,
               "2배 안": _pct((err.abs() < LOG2).mean()),
               "상승 2배 안": _pct((er.abs() < LOG2).mean()) if len(r) else "-"}
        for thr in thresholds:
            p0, f0 = _pod_far(g, thr, None)
            p1, f1 = _pod_far(g, thr, f"alarm{thr:g}") if f"alarm{thr:g}" in g else (np.nan, np.nan)
            row[f"≥{thr:g} 탐지/오경보"] = f"{_pct(p0)}/{_pct(f0)}"
            row[f"≥{thr:g} 조정 후"] = f"{_pct(p1)}/{_pct(f1)}"
        rows.append(row)
    return pd.DataFrame(rows)


def crossing_leads(P: pd.DataFrame, thr: float = 10.0, lookback_h: float = 2.0) -> pd.DataFrame:
    """사건별 첫 기준 넘김을 미리 알렸나 (조정된 경보 기준).

    넘기 lookback_h 시간 전부터의 경보만 그 사건 경보로 센다. 넘지 않은 사건에서
    울린 경보는 '안 넘었는데 알림' 으로 따로 센다.
    """
    lt = np.log10(thr)
    col = f"alarm{thr:g}"
    rows = []
    for (h, m), g in P.groupby(["h", "model"], sort=False):
        for eid, ge in g.groupby("eid"):
            meas = ge["meas"]
            cross = meas[meas >= lt].index.min() if (meas >= lt).any() else None
            al = ge[ge[col]] if col in ge else ge[(ge["meas"] < lt) & (ge["yhat"] >= lt)]
            if cross is not None:
                al = al.loc[cross - pd.Timedelta(hours=lookback_h):cross]
            alarm = al.index.min() if len(al) else None
            lead = (cross - alarm).total_seconds() / 60 if cross is not None and alarm is not None else np.nan
            rows.append({"h": h, "model": m, "eid": eid, "crossed": cross is not None,
                         "alarm": alarm is not None, "lead_min": lead})
    D = pd.DataFrame(rows)
    out = []
    for (h, m), g in D.groupby(["h", "model"], sort=False):
        c = g[g["crossed"]]
        warned = c["lead_min"] > 0
        out.append({"범위": f"{h}분", "모델": m, "넘은 사건": len(c),
                    "미리 알림": _pct(warned.mean()) if len(c) else "-",
                    "선행 중앙값(분)": round(float(c.loc[warned, "lead_min"].median()), 0) if warned.any() else np.nan,
                    "안 넘은 사건": int((~g["crossed"]).sum()),
                    "그중 알림": int((~g["crossed"] & g["alarm"]).sum())})
    return pd.DataFrame(out)


def alarm_episodes(P: pd.DataFrame, thr: float = 10.0, gap_min: float = 30.0) -> pd.DataFrame:
    """경보를 묶음 단위로 센다 (표본 단위 오경보비는 같은 경보가 여러 번 세진다).

    gap_min 안에 이어진 경보 표본은 한 묶음. 묶음 시작 ~ 끝+h 안에 실측이 기준을
    넘으면 맞은 경보, 아니면 오경보. 묶음 시작 때 이미 기준 위였던 경우는 없다
    (경보는 기준 아래일 때만 울린다).
    """
    lt = np.log10(thr)
    col = f"alarm{thr:g}"
    rows = []
    for (h, m), g in P.groupby(["h", "model"], sort=False):
        n_ep = n_hit = 0
        hit_events, cross_events = set(), set()
        for eid, ge in g.groupby("eid"):
            ge = ge.sort_index()
            above = ge.index[ge["meas"] >= lt]
            if len(above):
                cross_events.add(eid)
            al = ge.index[ge[col].to_numpy(dtype=bool)]
            if not len(al):
                continue
            brk = np.r_[True, (al[1:] - al[:-1]) > pd.Timedelta(minutes=gap_min)]
            starts = al[brk]
            ends = al[np.r_[brk[1:], True]]
            for s0, e0 in zip(starts, ends):
                n_ep += 1
                w = above[(above > s0) & (above <= e0 + pd.Timedelta(minutes=h))]
                if len(w):
                    n_hit += 1
                    hit_events.add(eid)
        rows.append({"범위": f"{h}분", "모델": m, "경보 묶음": n_ep,
                     "맞음": n_hit, "오경보비": _pct((n_ep - n_hit) / n_ep) if n_ep else "-",
                     "잡은 사건": f"{len(hit_events)}/{len(cross_events)}"})
    return pd.DataFrame(rows)


def summary(P: pd.DataFrame, thr: float = 100.0) -> pd.DataFrame:
    """범위·모델별 핵심 숫자 (seed 사이 비교용, 모두 수치)."""
    L = crossing_leads(P, thr)
    E = alarm_episodes(P, thr)
    rows = []
    for (h, m), g in P.groupby(["h", "model"], sort=False):
        g = g[np.isfinite(g["yhat"])]
        e = g["yhat"] - g["y"]
        r = g[g["rise"]]
        rows.append({"범위": f"{h}분", "모델": m,
                     "RMSE": float(np.sqrt((e ** 2).mean())),
                     "상승 RMSE": float(np.sqrt(((r["yhat"] - r["y"]) ** 2).mean())) if len(r) else np.nan})
    S = pd.DataFrame(rows)
    li = L.set_index(["범위", "모델"])
    ei = E.set_index(["범위", "모델"])
    k = list(zip(S["범위"], S["모델"]))
    num = lambda x: float(str(x).rstrip("%")) if x not in ("-", None) else np.nan  # noqa: E731
    S[f"≥{thr:g} 미리 알림%"] = [num(li.loc[i, "미리 알림"]) for i in k]
    S["선행 중앙값(분)"] = [li.loc[i, "선행 중앙값(분)"] for i in k]
    S["안 넘은 사건 경보"] = [li.loc[i, "그중 알림"] for i in k]
    S["경보 묶음"] = [ei.loc[i, "경보 묶음"] for i in k]
    S["묶음 오경보비%"] = [num(ei.loc[i, "오경보비"]) for i in k]
    return S


def seed_spread(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """summary 여러 개 → 평균 [최소~최대]."""
    A = pd.concat(tables)
    cols = [c for c in A.columns if c not in ("범위", "모델")]
    rows = []
    for (h, m), g in A.groupby(["범위", "모델"], sort=False):
        row = {"범위": h, "모델": m, "seed 수": len(g)}
        for c in cols:
            v = g[c].astype(float)
            d = 3 if "RMSE" in c else 0
            f = (lambda x: f"{x:.{d}f}")
            row[c] = f(v.mean()) if len(g) == 1 or v.max() == v.min() else f"{f(v.mean())} [{f(v.min())}~{f(v.max())}]"
        rows.append(row)
    return pd.DataFrame(rows)
