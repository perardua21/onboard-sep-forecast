# -*- coding: utf-8 -*-
"""수신 선행시간.

운용 요구이 선행시간을 두 가지로 나눈다.
  기내 선행시간 = 도달시각 − 최초 유효 기내 예측시각
  수신 선행시간 = 도달시각 − 사용자 단말 최초 유효 수신시각
미수신·늦은 수신은 성공에서 제외하고 별도로 집계한다.

기내 판정이 아무리 빨라도 비콘이 닿지 않으면 사용자에게는 경보가 없다.
100 km 극궤도에서 남극 직접 가시는 한 바퀴 117.8분 중 12.43분뿐이므로, 궤도 위상에 따라 수신 선행시간이 0 이 되는 사건이 생긴다.
궤도 위상은 사건 발생 시각과 무관하므로 균등분포로 두고 몬테카를로로 분포를 낸다.
"""
from __future__ import annotations

import numpy as np

ORBIT_MIN = 117.8          # 설계 기준
VISIBLE_MIN = 12.43        # 최소 고도각 0° 기준
BEACON_S = 10.0            # AI-03


def duty(period_min: float = ORBIT_MIN, visible_min: float = VISIBLE_MIN,
         n_sat: int = 1) -> float:
    """직접 가시 점유율. n_sat 를 늘려도 1 을 넘지 않는다."""
    return min(1.0, n_sat * visible_min / period_min)


def _next_contact(wait_min: np.ndarray, period_min: float, visible_min: float,
                  n_sat: int) -> np.ndarray:
    """경보 발령 시각부터 최초 수신까지 기다리는 시간(분).

    위성 n_sat 기를 궤도 위상으로 균등 배치한다. 어느 한 기라도 보이면 수신된다.
    """
    step = period_min / n_sat
    # wait_min 은 발령 시각의 위상 (0 ≤ φ < period). 가장 가까운 창까지의 거리.
    best = np.full(wait_min.shape, np.inf)
    for j in range(n_sat):
        phase = np.mod(wait_min - j * step, period_min)
        # phase < visible → 지금 보인다 (대기 0), 아니면 다음 창까지
        d = np.where(phase < visible_min, 0.0, period_min - phase)
        best = np.minimum(best, d)
    return best


def receipt_lead(onboard_lead_min, period_min: float = ORBIT_MIN,
                 visible_min: float = VISIBLE_MIN, n_sat: int = 1,
                 beacon_s: float = BEACON_S, n_samples: int = 2000,
                 seed: int = 20170910) -> dict:
    """기내 선행시간 배열 → 수신 선행시간 분포.

    각 사건마다 궤도 위상을 균등 추첨해서 최초 수신 시각을 정한다.
    수신 선행시간 = 기내 선행시간 − (발령 시각부터 최초 수신까지 대기시간)
    0 이하면 양성자 도착 후에 닿은 것이므로 미수신으로 집계한다.
    """
    lead = np.asarray(list(onboard_lead_min), dtype=float)
    if lead.size == 0:
        return {"n": 0, "delivered_frac": float("nan"),
                "recv_mean_min": float("nan"), "recv_median_min": float("nan"),
                "recv_p10_min": float("nan"), "recv_leads_min": [],
                "duty": duty(period_min, visible_min, n_sat)}

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, period_min, size=(n_samples, lead.size))
    wait = _next_contact(phases, period_min, visible_min, n_sat)
    wait = wait + beacon_s / 60.0          # 다음 비콘 송신까지 최대 한 주기
    recv = lead[None, :] - wait            # (n_samples, n_events)

    delivered = recv > 0
    ok = recv[delivered]
    return {
        "n": int(lead.size),
        "n_samples": int(n_samples),
        "duty": duty(period_min, visible_min, n_sat),
        "delivered_frac": float(delivered.mean()),
        "recv_mean_min": float(ok.mean()) if ok.size else float("nan"),
        "recv_median_min": float(np.median(ok)) if ok.size else float("nan"),
        "recv_p10_min": float(np.percentile(ok, 10)) if ok.size else float("nan"),
        "per_event_delivered": delivered.mean(axis=0).tolist(),
        "recv_leads_min": ok.tolist(),
        "wait_mean_min": float(wait.mean()),
    }


def summary_row(name: str, d: dict, target_lead: float = 30.0) -> dict:
    f = lambda x: f"{x:.0f}분" if x == x else "-"
    ok30 = float("nan")
    if d.get("recv_leads_min"):
        a = np.asarray(d["recv_leads_min"])
        ok30 = float((a >= target_lead).sum()) / (d["n"] * d["n_samples"])
    return {
        "구분": name,
        "사건": d["n"],
        "직접 가시 점유율": f"{d['duty']*100:.1f}%",
        "수신 성공률": f"{d['delivered_frac']*100:.0f}%" if d["delivered_frac"] == d["delivered_frac"] else "-",
        "평균 대기": f(d.get("wait_mean_min", float("nan"))),
        "수신 선행 평균": f(d["recv_mean_min"]),
        "수신 선행 중앙값": f(d["recv_median_min"]),
        "수신 선행 하위10%": f(d["recv_p10_min"]),
        f"수신 선행 ≥{target_lead:.0f}분": f"{ok30*100:.0f}%" if ok30 == ok30 else "-",
    }
