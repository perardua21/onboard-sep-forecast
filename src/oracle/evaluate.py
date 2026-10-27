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
def event_metrics(alarm: pd.Series, events, grace_hours: float = 0.5,
                  min_lead_min: float = 0.0, valid_mask: pd.Series | None = None,
                  cluster_gap_hours: float = 0.5,
                  segment_hours: float | None = None) -> dict:
    """탐지율·오경보비·CSI·선행시간.

    경보는 "앞으로 H분 안에 사건"이라는 주장이다. 그래서
      - H분 넘게 끊긴 경보는 별개의 경보로 본다 (cluster_gap_hours = H)
      - 경보가 끝난 뒤 H분까지 온 사건은 맞힌 것으로 본다 (grace_hours = H)
      - 오래 켜진 경보는 H분마다 주장을 새로 한 것으로 쪼개 센다 (segment_hours = H)

    마지막 규칙이 없으면 경보를 계속 켜 두기만 해도 묶음 하나가 사건을
    포함해 탐지율 100%, 오경보비 0% 가 된다.

    선행시간은 사건을 맞힌 조각이 속한 연속 경보의 시작부터 잰다.
    사용자가 실제로 경보를 받기 시작한 시각이기 때문이다. 그 앞부분이
    틀린 주장이었다면 그만큼 오경보로 따로 잡힌다.

    탐지율   : 선행시간이 min_lead_min 이상인 사건 수 ÷ 전체 사건 수
    오경보비 : 사건과 맺어지지 않은 조각 수 ÷ 전체 조각 수
    CSI     : hits / (hits + misses + false alarms)
    """
    if valid_mask is not None:
        alarm = alarm.where(valid_mask, False)
    seg_td = pd.Timedelta(hours=segment_hours if segment_hours else grace_hours)
    grace = pd.Timedelta(hours=grace_hours)
    runs = cluster(alarm, cluster_gap_hours)
    segs = []                                   # (조각 시작, 조각 끝, 연속 경보 시작)
    for cs, ce in runs:
        t = cs
        while True:
            e = min(t + seg_td, ce)
            segs.append((t, e, cs))
            if e >= ce:
                break
            t = e
    onsets = [o for o, _ in events]

    best: dict[int, float] = {}
    best_at: dict[int, pd.Timestamp] = {}
    matched = []
    for ss, se, rs in segs:
        hit = next((k for k, o in enumerate(onsets) if ss <= o <= se + grace), None)
        matched.append(hit)
        if hit is None:
            continue
        lead = (onsets[hit] - rs).total_seconds() / 60.0
        if hit not in best or lead > best[hit]:
            best[hit], best_at[hit] = lead, rs
    clusters = [(ss, se) for ss, se, _ in segs]

    n_ev = len(onsets)
    detected_flag = np.array([(k in best and best[k] >= min_lead_min)
                              for k in range(n_ev)], dtype=bool)
    leads = np.array(sorted(best.values()), dtype=float)
    lead_by_event = np.array([best.get(k, np.nan) if detected_flag[k] else np.nan
                              for k in range(n_ev)], dtype=float)
    hits = int(detected_flag.sum())
    false = int(sum(1 for m in matched if m is None))
    misses = n_ev - hits
    denom = hits + misses + false

    return {
        "n_events": n_ev,
        "n_alarms": len(clusters),
        "n_runs": len(runs),
        "n_detected": hits,
        "n_false": false,
        "POD": hits / n_ev if n_ev else float("nan"),
        "FAR": false / len(clusters) if clusters else float("nan"),
        "CSI": hits / denom if denom else float("nan"),
        "lead_mean_min": float(leads.mean()) if leads.size else float("nan"),
        "lead_median_min": float(np.median(leads)) if leads.size else float("nan"),
        "lead_p10_min": float(np.percentile(leads, 10)) if leads.size else float("nan"),
        "leads_min": leads.tolist(),
        "lead_by_event": lead_by_event,
        "detected_flag": detected_flag,
        "cluster_matched": matched,
        "alarm_times": {k: best_at[k] for k in best_at},
        "missed_events": [onsets[k] for k in range(n_ev) if not detected_flag[k]],
        "onsets": onsets,
        "clusters": clusters,
    }


def select_threshold(prob, index, events, valid_mask, target_far: float = 0.35,
                     persist_points: int = 2, horizon_min: int = 30,
                     grid=None, verbose: bool = True) -> tuple[float, dict]:
    """검증집합에서 사건 단위 지표로 임계값을 고른다.

    보고하는 것이 사건 단위 탐지율·오경보비이므로 고르는 기준도 같아야 한다.
    표본 단위 오경보율로 고르면 둘이 따로 논다.

    오경보비 ≤ target 인 것 중 탐지율이 가장 높은 것, 같으면 평균 선행시간이
    긴 것. 조건을 만족하는 값이 없으면 오경보비가 가장 낮은 것을 쓰고 알린다.
    """
    from .predict import to_alarm
    grid = np.round(np.arange(0.05, 0.951, 0.025), 3) if grid is None else grid
    h = horizon_min / 60.0
    vm = pd.Series(np.asarray(valid_mask, dtype=bool), index=index)
    rows = []
    for th in grid:
        a = to_alarm(prob, index, float(th), persist_points).where(vm, False)
        m = event_metrics(a, events, grace_hours=h, cluster_gap_hours=h)
        rows.append((float(th), m["POD"], m["FAR"], m["lead_mean_min"]))
    ok = [r for r in rows if r[2] == r[2] and r[2] <= target_far and r[1] == r[1]]
    if ok:
        best = max(ok, key=lambda r: (r[1], -r[2], r[3] if r[3] == r[3] else -1, r[0]))
        note = "조건 충족"
    else:
        cand = [r for r in rows if r[2] == r[2]] or rows
        best = min(cand, key=lambda r: (r[2] if r[2] == r[2] else 9, -r[1]))
        note = f"오경보비 {target_far:.0%} 이하를 만족하는 임계값이 없음"
        if verbose:
            print(f"  ! 검증집합에서 {note}. 오경보비가 가장 낮은 {best[0]:.3f} 을 쓴다 "
                  f"(오경보비 {best[2]:.0%}, 탐지율 {best[1]:.0%})")
    return best[0], {"POD": best[1], "FAR": best[2], "lead_mean_min": best[3],
                     "note": note, "n_events": len(events)}


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
    lbe = np.asarray(m.get("lead_by_event", []), dtype=float)

    def leads_of(em):
        x = lbe[em] if em.size and lbe.size else np.array([])
        return x[np.isfinite(x)]

    if mode == "event":
        def draw():
            e = rng.integers(0, det.size, det.size) if det.size else np.array([], int)
            c = rng.integers(0, unmatched.size, unmatched.size) if unmatched.size else np.array([], int)
            return det[e], unmatched[c], leads_of(e)
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
            return lm, unmatched[cm] if cm.size else np.array([], bool), leads_of(em)

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
