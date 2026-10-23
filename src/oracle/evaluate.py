# -*- coding: utf-8 -*-
"""평가 지표.

운용 요구이 공개하라고 정한 것:
  사건 수, 선행시간 평균·중앙값·하위 10백분위, POD, FAR, CSI, Brier, 신뢰도 도표,
  사건 단위 95% 신뢰구간, 사건·기간 블록 재표본추출.
표본 단위 정확도는 쓰지 않는다 (양성 비율 0.1% 수준이라 무예보도 99%를 넘는다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .alarm_episodes import cluster


# 사건 단위
def event_metrics(alarm: pd.Series, events, grace_hours: float = 2.0,
                  min_lead_min: float = 0.0, valid_mask: pd.Series | None = None,
                  cluster_gap_hours: float = 3.0) -> dict:
    """탐지율·오경보비·CSI·선행시간.

    TP      : 경보 묶음 시작 이후 (묶음 끝 + grace) 안에 onset 이 있다
    선행시간 : onset − 묶음 시작
    탐지율   : 선행시간이 min_lead_min 이상인 사건 수 ÷ 전체 사건 수
    오경보비 : 어떤 사건과도 맺어지지 않은 묶음 수 ÷ 전체 묶음 수
    CSI     : hits / (hits + misses + false alarms)
    """
    if valid_mask is not None:
        alarm = alarm.where(valid_mask, False)
    clusters = cluster(alarm, cluster_gap_hours)
    onsets = [o for o, _ in events]

    best: dict[int, float] = {}
    best_at: dict[int, pd.Timestamp] = {}
    matched = []                       # 묶음별 사건 연결 여부
    for cs, ce in clusters:
        hit = next((k for k, o in enumerate(onsets)
                    if cs <= o <= ce + pd.Timedelta(hours=grace_hours)), None)
        matched.append(hit)
        if hit is None:
            continue
        lead = (onsets[hit] - cs).total_seconds() / 60.0
        if hit not in best or lead > best[hit]:
            best[hit], best_at[hit] = lead, cs

    n_ev = len(onsets)
    detected_flag = np.array([(k in best and best[k] >= min_lead_min)
                              for k in range(n_ev)], dtype=bool)
    leads = np.array(sorted(best.values()), dtype=float)
    hits = int(detected_flag.sum())
    false = int(sum(1 for m in matched if m is None))
    misses = n_ev - hits
    denom = hits + misses + false

    return {
        "n_events": n_ev,
        "n_alarms": len(clusters),
        "n_detected": hits,
        "n_false": false,
        "POD": hits / n_ev if n_ev else float("nan"),
        "FAR": false / len(clusters) if clusters else float("nan"),
        "CSI": hits / denom if denom else float("nan"),
        "lead_mean_min": float(leads.mean()) if leads.size else float("nan"),
        "lead_median_min": float(np.median(leads)) if leads.size else float("nan"),
        "lead_p10_min": float(np.percentile(leads, 10)) if leads.size else float("nan"),
        "leads_min": leads.tolist(),
        "detected_flag": detected_flag,
        "cluster_matched": matched,
        "alarm_times": {k: best_at[k] for k in best_at},
        "missed_events": [onsets[k] for k in range(n_ev) if not detected_flag[k]],
        "onsets": onsets,
        "clusters": clusters,
    }


# 확률 품질
def brier_score(prob, y, valid=None) -> float:
    """Brier = 평균((p − y)²). 낮을수록 좋다."""
    p = np.asarray(prob, dtype=float)
    t = np.asarray(y, dtype=float)
    if valid is not None:
        m = np.asarray(valid, dtype=bool)
        p, t = p[m], t[m]
    return float(np.mean((p - t) ** 2)) if p.size else float("nan")


def brier_skill_score(prob, y, valid=None) -> float:
    """기후값(전체 양성 비율) 대비 개선율. 0 이하면 무예보보다 못하다."""
    t = np.asarray(y, dtype=float)
    if valid is not None:
        t = t[np.asarray(valid, dtype=bool)]
    if t.size == 0:
        return float("nan")
    ref = float(np.mean((t.mean() - t) ** 2))
    return 1.0 - brier_score(prob, y, valid) / ref if ref > 0 else float("nan")


def reliability(prob, y, valid=None, n_bins: int = 10) -> pd.DataFrame:
    """신뢰도 도표용 표. 예측확률 구간별 실제 발생 빈도."""
    p = np.asarray(prob, dtype=float)
    t = np.asarray(y, dtype=float)
    if valid is not None:
        m = np.asarray(valid, dtype=bool)
        p, t = p[m], t[m]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        s = b == k
        rows.append({
            "bin_lo": edges[k], "bin_hi": edges[k + 1],
            "n": int(s.sum()),
            "pred_mean": float(p[s].mean()) if s.any() else float("nan"),
            "obs_freq": float(t[s].mean()) if s.any() else float("nan"),
        })
    return pd.DataFrame(rows)


# 신뢰구간
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """사건 수가 적을 때 탐지율에 반드시 붙인다."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def block_bootstrap(m: dict, n_boot: int = 2000, mode: str = "event",
                    block_days: int = 30, seed: int = 20170910,
                    q=(2.5, 97.5)) -> dict:
    """사건·기간 블록 재표본추출.

    시계열 점을 뽑으면 이웃 점끼리 강하게 묶여 있어 구간이 좁게 나온다.
    그래서 점이 아니라 블록을 뽑는다.

    mode='event'  : 사건과 오경보 묶음을 각각 복원추출
    mode='period' : 달력을 block_days 블록으로 자르고 블록을 복원추출
    """
    rng = np.random.default_rng(seed)
    det = np.asarray(m["detected_flag"], dtype=bool)
    unmatched = np.array([x is None for x in m["cluster_matched"]], dtype=bool)
    leads = np.asarray(m["leads_min"], dtype=float)

    if mode == "event":
        def draw():
            e = rng.integers(0, det.size, det.size) if det.size else np.array([], int)
            c = rng.integers(0, unmatched.size, unmatched.size) if unmatched.size else np.array([], int)
            return det[e], unmatched[c], leads[rng.integers(0, leads.size, leads.size)] if leads.size else leads
    else:
        ev_t = pd.DatetimeIndex(m["onsets"]) if m["onsets"] else pd.DatetimeIndex([])
        cl_t = pd.DatetimeIndex([c[0] for c in m["clusters"]]) if m["clusters"] else pd.DatetimeIndex([])
        allt = ev_t.append(cl_t)
        if len(allt) == 0:
            return {}
        t0 = allt.min().normalize()
        ev_b = ((ev_t - t0).days // block_days) if len(ev_t) else np.array([], int)
        cl_b = ((cl_t - t0).days // block_days) if len(cl_t) else np.array([], int)
        blocks = np.unique(np.concatenate([np.asarray(ev_b), np.asarray(cl_b)]))

        def draw():
            pick = rng.choice(blocks, blocks.size, replace=True)
            em = np.concatenate([np.flatnonzero(np.asarray(ev_b) == b) for b in pick]) \
                if len(ev_t) else np.array([], int)
            cm = np.concatenate([np.flatnonzero(np.asarray(cl_b) == b) for b in pick]) \
                if len(cl_t) else np.array([], int)
            lm = det[em] if em.size else np.array([], bool)
            return lm, unmatched[cm] if cm.size else np.array([], bool), leads

    pods, fars, csis, lmeans = [], [], [], []
    for _ in range(n_boot):
        d, u, l = draw()
        if d.size:
            h, mi = int(d.sum()), int((~d).sum())
            pods.append(h / d.size)
        else:
            h = mi = 0
        f = int(u.sum())
        if u.size:
            fars.append(f / u.size)           # u 는 전체 묶음, True 가 오경보
        if (h + mi + f) > 0:
            csis.append(h / (h + mi + f))
        if l.size:
            lmeans.append(float(l.mean()))

    def ci(v):
        return (float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))) if v else (float("nan"),) * 2

    return {"POD_ci": ci(pods), "FAR_ci": ci(fars),
            "CSI_ci": ci(csis), "lead_mean_ci": ci(lmeans),
            "mode": mode, "n_boot": n_boot}


# 표 출력
def _pct(x):
    return f"{x*100:.0f}%" if x == x else "-"


def _min(x):
    return f"{x:.0f}분" if x == x else "-"


def fmt_row(name: str, m: dict, note: str = "", with_ci: bool = True,
            boot: dict | None = None) -> dict:
    pod = _pct(m["POD"])
    if with_ci and m["n_events"]:
        lo, hi = wilson_interval(m["n_detected"], m["n_events"])
        pod += f" ({lo*100:.0f}–{hi*100:.0f})"
    row = {
        "방법": name,
        "사건": m["n_events"],
        "경보": m["n_alarms"],
        "탐지율": pod,
        "오경보비": _pct(m["FAR"]),
        "CSI": f"{m['CSI']:.2f}" if m["CSI"] == m["CSI"] else "-",
        "선행 평균": _min(m["lead_mean_min"]),
        "선행 중앙값": _min(m["lead_median_min"]),
        "선행 하위10%": _min(m["lead_p10_min"]),
        "비고": note,
    }
    if boot:
        lo, hi = boot["POD_ci"]
        row["탐지율 95%CI"] = f"{lo*100:.0f}–{hi*100:.0f}" if lo == lo else "-"
    return row


def verdict(m: dict, cfg) -> dict:
    """설계 기준 성공 기준 판정. 목표를 맞추려 값을 고치지 말고 결과를 그대로 적는다."""
    pod, far, lead = m["POD"], m["FAR"], m["lead_mean_min"]
    g = lambda k, d: cfg.get(k, d) if hasattr(cfg, "get") else d
    t_pod, t_far = g("alarm.target_pod", 0.70), g("alarm.target_far", 0.35)
    t_lead = g("alarm.target_lead_min", 30.0)
    t_min, t_ext = g("alarm.target_pod_min", 0.50), g("alarm.target_pod_ext", 0.80)

    if pod != pod:
        level = "판정 불가"
    elif pod >= t_ext and far <= t_far and lead >= t_lead:
        level = "확장 성공"
    elif pod >= t_pod and far <= t_far and lead >= t_lead:
        level = "완전 성공"
    elif pod >= t_min:
        level = "최소 성공"
    else:
        level = "미달"
    return {
        "판정": level,
        "탐지율": f"{_pct(pod)} / 목표 {t_pod*100:.0f}%",
        "오경보비": f"{_pct(far)} / 상한 {t_far*100:.0f}%",
        "평균 선행시간": f"{_min(lead)} / 하한 {t_lead:.0f}분",
    }
