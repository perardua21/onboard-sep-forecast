# -*- coding: utf-8 -*-
"""잠금 조건에 맞춘 실험 묶음.

docs/spec.md 5·6절이 요구하는 표를 그대로 만든다.
  표 1  비교군 (지속성 · 단순 임계값 · 로지스틱 회귀 · 경량 신경망)
  표 2  제거 실험 (X선 단독 / 전자 단독 / 융합 / 양성자 제외 / 선량 제외)
  표 3  예측 범위 30분 · 60분
  표 4  기내 선행시간 vs 수신 선행시간
  표 5  확률 품질 (Brier · BSS · 신뢰도 도표)
  표 6  성공 기준 판정
  표 7  연산 예산 (파라미터 · MFLOPs · 기내 창 재표본 간격)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from. import baselines, delivery, plots
from .channels import CHANNEL_SETS, ELECTRON, model_input
from .evaluate import (block_bootstrap, brier_score, brier_skill_score,
                       event_metrics, fmt_row, reliability, verdict)
from .features import mask_channels, mask_xray
from .labeling import events_to_frame
from .predict import to_alarm, persist_minutes
from .train import infer, prepare, train, channels_for

EXP_DOC = {
    "EXP-01": "지속성 기준선 - 경보를 내지 않을 때",
    "EXP-02": "단순 임계값 - X선 단독 / 전자 단독 / 양성자 단독",
    "EXP-03": "OBC 규칙 게이트 (2종 동의) - ORACLE 정지 시(L2) 성능",
    "EXP-04": "로지스틱 회귀 - 선형 기준선",
    "EXP-05": "경량 신경망 (TCN) - 모델 A",
    "EXP-06": "제거 실험 - 채널 집합별",
    "EXP-07": "X선 결손 강건성 (식 구간 모사, 점유율 39.4%)",
    "EXP-08": "전자 결손 강건성",
    "EXP-09": "수신 선행시간 - 비콘 가시 조건 반영",
}


# 공통
def _test_events(events, cfg):
    t2 = pd.Timestamp(cfg.get("split.valid_end"), tz="UTC")
    t3 = cfg.get("split.test_end")
    out = [e for e in events if e[0] > t2]
    if t3:
        t3 = pd.Timestamp(t3, tz="UTC")
        out = [e for e in out if e[0] <= t3]
    return out


def _metrics(alarm, events, ong, index, cfg, te_mask, min_lead=0.0, horizon_min=None):
    """묶음 간격과 유예를 예측 범위에 맞춘다 (evaluate.event_metrics 설명 참조)."""
    h = float(horizon_min or cfg.get("label.primary_horizon_min", 30)) / 60.0
    return event_metrics(
        alarm.where(pd.Series(te_mask, index=index), False),
        _test_events(events, cfg),
        grace_hours=h,
        min_lead_min=float(min_lead),
        valid_mask=pd.Series(~ong, index=index),
        cluster_gap_hours=h)


def _model_alarm(prob, index, th, cfg):
    return to_alarm(prob, index, th,
                    int(cfg.get("gate.model_persist_points", 2)))


def _boot(m, cfg):
    return block_bootstrap(
        m,
        n_boot=int(cfg.get("uncertainty.bootstrap_n", 2000)),
        mode=str(cfg.get("uncertainty.block", "event")),
        block_days=int(cfg.get("uncertainty.block_days", 30)),
        seed=int(cfg.get("model.seed", 0)))


def _save(df_or_dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_or_dict, pd.DataFrame):
        df_or_dict.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        path.write_text(json.dumps(df_or_dict, ensure_ascii=False,
                                   indent=2, default=str), encoding="utf-8")


def _show(title: str, rows):
    df = pd.DataFrame(rows)
    print(f"\n[{title}]")
    print(df.to_string(index=False))
    return df


# 본체
def run_all(df: pd.DataFrame, cfg, out_dir: Path, kind: str | None = None,
            event_date: str | None = None) -> dict:
    out_dir = Path(out_dir)
    met, fig = out_dir / "metrics", out_dir / "figures"
    met.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    cad = cfg.cadence
    gate = dict(cfg.get("gate", {}) or {})
    target_lead = float(cfg.get("alarm.target_lead_min", 30.0))

    t = train(df, cfg, kind=kind)
    tr, va, te = t.splits
    events, ong, index = t.events, t.ongoing, t.index
    Y = t.info["labels"]
    H = t.primary

    gate["electron_channel"] = baselines.pick_electron_channel(
        df, {"학습": tr, "시험": te} if te.any() else {"학습": tr},
        min_frac=float(cfg.get("channels.min_available", 0.8)))
    print(f"전자 게이트 채널: {gate['electron_channel']}")
    used = t.info.get("channels", [])
    print(f"모델 입력 채널 {len([c for c in used if not c.endswith('_ok') and c != 'xray_ok'])}개: "
          + ", ".join(c for c in used if not c.endswith("_ok") and c != "xray_ok"))

    ev = events_to_frame(events)
    _save(ev, met / "event_table.csv")
    te_events = _test_events(events, cfg)
    print(f"\n잠금 시험집합 사건 {len(te_events)}건 "
          f"(전체 {len(events)}건 중) · 지속조건 "
          f"{persist_minutes(int(gate.get('model_persist_points', 2)), cad):.0f}분 상당")

    out: dict = {"split": t.info.get("split_report", {}),
                 "n_events_total": len(events), "n_events_test": len(te_events)}

    # 표 1 비교군
    rows = []
    for key in ("persistence", "xray_only", "electron_only", "proton_rise",
                "rule_gate", "s1_reached"):
        fn = baselines.REGISTRY[key]
        if key.startswith("rule"):
            a = fn(df, cadence=cad, gate=gate)
        elif key == "electron_only":
            a = fn(df, cadence=cad, channel=gate["electron_channel"])
        else:
            a = fn(df, cadence=cad)
        m = _metrics(a, events, ong, index, cfg, te)
        rows.append(fmt_row(baselines.LABELS[key], m, EXP_DOC["EXP-02"]
                            if "only" in key or "rise" in key else ""))

    # 로지스틱 회귀
    probs_by_kind = {}
    if (kind or cfg.get("model.kind")) != "logreg":
        try:
            t_lr = train(df, cfg, kind="logreg", verbose=False)
            a_lr = _model_alarm(t_lr.prob[t_lr.primary], index, t_lr.threshold, cfg)
            m_lr = _metrics(a_lr, events, ong, index, cfg, te)
            rows.append(fmt_row("로지스틱 회귀", m_lr, EXP_DOC["EXP-04"]))
            probs_by_kind["logreg"] = (t_lr, m_lr)
        except Exception as e:                               # noqa: BLE001
            print(f"  로지스틱 회귀 생략: {e}")

    a_ai = _model_alarm(t.prob[H], index, t.threshold, cfg)
    m_ai = _metrics(a_ai, events, ong, index, cfg, te)
    b_ai = _boot(m_ai, cfg)
    note_ai = EXP_DOC["EXP-05"] if t.kind in ("tcn", "cnn1d") else \
        f"{t.kind} - 신경망 대신 돌린 기준선"
    rows.append(fmt_row(f"ORACLE 모델 A ({t.kind.upper()}, {H}분)", m_ai,
                        note_ai, boot=b_ai))
    out["t1_compare"] = _show("표 1  비교군 - 잠금 시험집합", rows).to_dict("records")
    _save(pd.DataFrame(rows), met / "t1_compare.csv")

    # 표 2 제거 실험
    rows2 = []
    for name in ("fused17", "no_dose", "no_proton", "xray_only", "electron_only"):
        try:
            tt = t if name == cfg.get("channels.set", "fused17") else \
                train(df, cfg, channels=model_input(name), kind=kind, verbose=False)
            aa = _model_alarm(tt.prob[tt.primary], index, tt.threshold, cfg)
            mm = _metrics(aa, events, ong, index, cfg, te)
            used = [c for c in tt.info.get("channels", [])
                    if not c.endswith("_ok") and c != "xray_ok"]
            rows2.append(fmt_row(f"{name} ({len(used)}/{len(CHANNEL_SETS[name])}채널)", mm,
                                 "실제 사용: " + ", ".join(used)))
        except Exception as e:                               # noqa: BLE001
            print(f"  제거 실험 {name} 생략: {e}")
    if rows2:
        out["t2_ablation"] = _show("표 2  제거 실험 - 입력 채널 집합", rows2).to_dict("records")
        _save(pd.DataFrame(rows2), met / "t2_ablation.csv")

    # 표 3 예측 범위
    rows3 = []
    for h in t.horizons:
        a = _model_alarm(t.prob[h], index, t.thresholds[h], cfg)
        m = _metrics(a, events, ong, index, cfg, te, horizon_min=h)
        rows3.append(fmt_row(f"{h}분 출력", m, f"임계값 {t.thresholds[h]:.3f}"))
    out["t3_horizon"] = _show("표 3  예측 범위", rows3).to_dict("records")
    _save(pd.DataFrame(rows3), met / "t3_horizon.csv")

    # 표 4 기내 vs 수신
    dcfg = cfg.get("delivery", {}) or {}
    rows4 = [{
        "구분": "기내 판정", "사건": m_ai["n_events"], "직접 가시 점유율": "-",
        "수신 성공률": "-", "평균 대기": "-",
        "수신 선행 평균": f"{m_ai['lead_mean_min']:.0f}분"
        if m_ai["lead_mean_min"] == m_ai["lead_mean_min"] else "-",
        "수신 선행 중앙값": f"{m_ai['lead_median_min']:.0f}분"
        if m_ai["lead_median_min"] == m_ai["lead_median_min"] else "-",
        "수신 선행 하위10%": f"{m_ai['lead_p10_min']:.0f}분"
        if m_ai["lead_p10_min"] == m_ai["lead_p10_min"] else "-",
        f"수신 선행 ≥{target_lead:.0f}분": "-"}]
    deliv = {}
    for nsat in (int(dcfg.get("n_satellites", 1)), 3):
        d = delivery.receipt_lead(
            m_ai["leads_min"],
            period_min=float(dcfg.get("orbit_period_min", 117.8)),
            visible_min=float(dcfg.get("visible_min", 12.43)),
            n_sat=nsat,
            beacon_s=float(cfg.get("inference.beacon_period_s", 10)),
            n_samples=int(dcfg.get("n_phase_samples", 2000)),
            seed=int(cfg.get("model.seed", 0)))
        rows4.append(delivery.summary_row(f"사용자 수신 ({nsat}기)", d, target_lead))
        deliv[nsat] = d
        if nsat == 3:
            break
    out["t4_delivery"] = _show("표 4  기내 선행시간 vs 수신 선행시간", rows4).to_dict("records")
    _save(pd.DataFrame(rows4), met / "t4_delivery.csv")

    # 표 5 확률 품질
    valid = te & ~ong & np.isfinite(t.prob[H])
    rel = reliability(t.prob[H], Y[H], valid)
    rows5 = [{"지표": "Brier 점수", "값": f"{brier_score(t.prob[H], Y[H], valid):.5f}"},
             {"지표": "Brier 기술점수 (기후값 대비)",
              "값": f"{brier_skill_score(t.prob[H], Y[H], valid):.3f}"},
             {"지표": "평가 표본 수", "값": f"{int(valid.sum()):,}"},
             {"지표": "양성 비율", "값": f"{Y[H][valid].mean():.4%}" if valid.any() else "-"}]
    out["t5_probability"] = _show("표 5  확률 품질", rows5).to_dict("records")
    _save(pd.DataFrame(rows5), met / "t5_probability.csv")
    _save(rel, met / "reliability.csv")

    # 표 6 성공 기준 판정
    v = verdict(m_ai, cfg)
    lo, hi = b_ai.get("POD_ci", (float("nan"),) * 2)
    rows6 = [{"항목": k, "결과": x} for k, x in v.items()]
    rows6.append({"항목": f"탐지율 95% 신뢰구간 ({b_ai.get('mode','event')} 블록)",
                  "결과": f"{lo*100:.0f}–{hi*100:.0f}%" if lo == lo else "-"})
    rows6.append({"항목": "선행시간 하위 10백분위",
                  "결과": f"{m_ai['lead_p10_min']:.0f}분"
                  if m_ai["lead_p10_min"] == m_ai["lead_p10_min"] else "-"})
    rows6.append({"항목": "CSI", "결과": f"{m_ai['CSI']:.2f}"
                  if m_ai["CSI"] == m_ai["CSI"] else "-"})
    if 1 in deliv:
        rows6.append({"항목": "수신 성공률 (1기, 직접 가시만)",
                      "결과": f"{deliv[1]['delivered_frac']*100:.0f}%"})
    out["t6_verdict"] = _show("표 6  성공 기준 판정 ", rows6).to_dict("records")
    _save(pd.DataFrame(rows6), met / "t6_verdict.csv")

    # 표 7 연산 예산
    budget = {k: v for k, v in t.info.items()
              if k not in ("labels", "split_report")}
    rows7 = [{"항목": k, "값": str(x)} for k, x in budget.items()]
    # 실제로 어떤 모델을 돌렸든 비행 모델(TCN)의 예산은 산식으로 항상 적는다.
    from .models import tcn_budget
    n_ch = len(CHANNEL_SETS[cfg.get("channels.set", "fused17")])
    n_feat = n_ch * len(cfg.get("features.nn_derive", ["log", "ratio"])) + 3
    tb = tcn_budget(n_feat,
                    hidden=int(cfg.get("model.channels_hidden", 32)),
                    n_blocks=int(cfg.get("model.n_blocks", 4)),
                    kernel_size=int(cfg.get("model.kernel_size", 7)),
                    n_heads=len(t.horizons),
                    window_steps=int(int(cfg.get("label.window_min", 60)) * 60
                                     // int(cfg.get("model.onboard_resample_s", 25))),
                    flops_budget_m=float(cfg.get("model.flops_budget_m", 20)))
    lo, hi = cfg.get("model.param_budget", [50000, 150000])
    rows7 += [
        {"항목": "비행 모델 입력 특징 수", "값": f"{n_feat} ({n_ch}채널 × 파생 + 플래그 3)"},
        {"항목": "비행 모델 파라미터", "값": f"{tb['params']:,}  (예산 {lo:,}~{hi:,}, "
                                   f"{'충족' if lo <= tb['params'] <= hi else '미충족'})"},
        {"항목": "기내 연산량 (재표본 창)", "값": f"{tb['flops_m']:.1f} MFLOPs / 추론"},
        {"항목": "수용영역", "값": f"{tb['receptive_field']}점 "
                                f"({'창 전체' if tb['covers_window'] else '창 일부만'})"},
        {"항목": "기내 입력 길이 상한", "값": f"{tb['onboard_max_steps']}점"},
        {"항목": "기내 60분 창 재표본 간격",
         "값": f"{int(cfg.get('label.window_min', 60))*60/max(tb['onboard_max_steps'],1):.1f} s"},
        {"항목": "센서 측정 / 추론 / 비콘 주기",
         "값": f"1 s / {cfg.get('inference.cadence_s', 10)} s / "
               f"{cfg.get('inference.beacon_period_s', 10)} s"},
    ]
    out["t7_budget"] = _show("표 7  연산 예산", rows7).to_dict("records")
    _save(pd.DataFrame(rows7), met / "t7_budget.csv")

    # 결손 강건성
    rows8 = []
    rng = np.random.default_rng(int(cfg.get("model.seed", 0)))
    blk = max(1, int(round(float(cfg.get("augment.eclipse_block_min", 46.5)) / cad)))
    mask = np.zeros(len(df), bool)
    for i in range(0, len(df), blk):
        if rng.random() < float(cfg.get("augment.eclipse_mask_prob", 0.394)):
            mask[i:i + blk] = True
    for name, note, dfx in [
            ("X선 결손 (식 39.4%)", EXP_DOC["EXP-07"], mask_xray(df, mask, cad)),
            ("전자 결손", EXP_DOC["EXP-08"], mask_channels(df, ELECTRON, mask))]:
        try:
            p = infer(t, dfx, cfg)[H]
            m = _metrics(_model_alarm(p, index, t.threshold, cfg),
                         events, ong, index, cfg, te)
            rows8.append(fmt_row(name, m, note))
        except Exception as e:                               # noqa: BLE001
            print(f"  결손 실험 {name} 생략: {e}")
    if rows8:
        rows8.insert(0, fmt_row("결손 없음", m_ai, "기준"))
        out["t8_degraded"] = _show("표 8  입력 결손 강건성", rows8).to_dict("records")
        _save(pd.DataFrame(rows8), met / "t8_degraded.csv")

    # 그림
    try:
        plots.use_korean_font()
        plots.lead_time_hist(m_ai["leads_min"], fig / "lead_time.png",
                             target_min=target_lead)
        plots.reliability_diagram(rel, fig / "reliability.png")
        if 1 in deliv:
            plots.lead_time_hist(deliv[1]["recv_leads_min"],
                                 fig / "lead_time_received.png",
                                 target_min=target_lead,
                                 title="수신 선행시간 분포 (1기, 직접 가시)")
        if event_date:
            want = pd.Timestamp(event_date, tz="UTC")
            near = min(te_events, key=lambda e: abs(e[0] - want), default=None)
            if near is not None:
                plots.event_timeseries(df, t.prob[H], t.threshold, near[0],
                                       fig / "event_case.png")
    except Exception as e:                                   # noqa: BLE001
        print(f"  그림 생략: {e}")

    _save(out, met / "summary.json")
    print(f"\n산출물: {met} / {fig}")
    return out


def run_horizons(df: pd.DataFrame, cfg, out_dir: Path,
                 horizons=(15, 30, 60, 120), kind: str | None = None) -> pd.DataFrame:
    """예측 범위를 바꿔 가며 선행시간과 오경보비의 교환을 본다."""
    out_dir = Path(out_dir)
    rows = []
    for h in horizons:
        cfg.raw.setdefault("label", {})["horizons_min"] = [int(h)]
        cfg.raw["label"]["primary_horizon_min"] = int(h)
        t = train(df, cfg, kind=kind, verbose=False)
        a = _model_alarm(t.prob[h], t.index, t.threshold, cfg)
        m = _metrics(a, t.events, t.ongoing, t.index, cfg, t.splits[2], horizon_min=h)
        rows.append(fmt_row(f"{h}분", m, f"임계값 {t.threshold:.3f}"))
    df_out = _show("예측 범위 sweep", rows)
    _save(df_out, out_dir / "metrics" / "horizons.csv")
    return df_out
